# ==============================
# train.py — CV-based training
# ==============================

from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, recall_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# ------------------------------
# Paths
# ------------------------------
MODELS_DIR = Path("models")
PIPELINE_PATH = MODELS_DIR / "pipeline.pkl"
META_PATH = MODELS_DIR / "meta.pkl"


# ============================================================
# Helper functions
# ============================================================

def _pick_scaler(scaler_type: str):
    if scaler_type.lower() == "standard":
        return StandardScaler()
    if scaler_type.lower() == "minmax":
        return MinMaxScaler()
    raise ValueError("scaler_type must be 'standard' or 'minmax'")


def _metric_name(select_by: str) -> str:
    # Fixed: Added comma between "recall" and "accuracy"
    if select_by in {"roc_auc", "accuracy", "f1", "recall"}:
        return select_by
    raise ValueError("select_by must be 'roc_auc', 'accuracy', 'f1' or 'recall'")


def tune_threshold(
    pipeline: Pipeline,
    X_val: pd.DataFrame,
    y_val: pd.Series,
    metric: str = "f1",
    thresholds: np.ndarray | None = None,
) -> Dict[str, float]:

    if thresholds is None:
        thresholds = np.linspace(0.2, 0.8, 61)

    proba = pipeline.predict_proba(X_val)[:, 1]

    best_t, best_score = 0.5, -np.inf

    for t in thresholds:
        preds = (proba >= t).astype(int)

        score = (
            f1_score(y_val, preds)
            if metric == "f1"
            else accuracy_score(y_val, preds)
        )

        if score > best_score:
            best_score = score
            best_t = t

    return {
        "best_threshold": float(best_t),
        "best_score": float(best_score),
        "metric": metric,
    }


def extract_logreg_coefficients(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:

    model = pipeline.named_steps["model"]

    return (
        pd.DataFrame({
            "feature": feature_names,
            "coefficient": model.coef_[0],
        })
        .assign(abs_coef=lambda d: d.coefficient.abs())
        .sort_values("abs_coef", ascending=False)
    )


def extract_rf_importance(
    pipeline: Pipeline,
    feature_names: list[str],
) -> pd.DataFrame:

    model = pipeline.named_steps["model"]

    return (
        pd.DataFrame({
            "feature": feature_names,
            "importance": model.feature_importances_,
        })
        .sort_values("importance", ascending=False)
    )


# ============================================================
# Main training function
# ============================================================

def train_and_save_model_cv(
    train_df: pd.DataFrame,
    *,
    random_state: int = 42,
    scaler_type: str = "standard",
    select_by: str = "roc_auc",
    cv_splits: int = 5,
    save_paths: Tuple[Path, Path] = (PIPELINE_PATH, META_PATH),
) -> Dict[str, Any]:

    if "Outcome" not in train_df.columns:
        raise ValueError("Expected target column 'Outcome'")

    # ------------------------------
    # Prepare Training Data
    # ------------------------------
    y_train = train_df["Outcome"].astype(int)
    X_train = train_df.drop(columns=["Outcome", "subject_id"], errors="ignore")

    if X_train.isna().any().any():
        raise ValueError("X_train contains NaNs")

    metric = _metric_name(select_by)
    scaler = _pick_scaler(scaler_type)

    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    # ------------------------------
    # Candidate models
    # ------------------------------
    candidates = {
        "logreg": (
            Pipeline([
                ("scaler", scaler),
                ("model", LogisticRegression(max_iter=3000)),
            ]),
            {
                "model__C": [0.01, 0.1, 1, 10],
                "model__penalty": ["l1", "l2"],
                "model__solver": ["liblinear", "saga"],
                "model__class_weight": [None, "balanced"],
            },
        ),

        "svm_rbf": (
            Pipeline([
                ("scaler", scaler),
                ("model", SVC(kernel="rbf", probability=True)),
            ]),
            {
                "model__C": [1, 10, 100],
                "model__gamma": ["scale", 0.01, 0.001],
                "model__class_weight": [None, "balanced"],
            },
        ),

        "random_forest": (
            Pipeline([
                ("scaler", "passthrough"),
                ("model", RandomForestClassifier(random_state=random_state)),
            ]),
            {
                "model__n_estimators": [200, 400],
                "model__max_depth": [None, 10, 20],
                "model__min_samples_leaf": [1, 4, 8],
                "model__class_weight": ["balanced"],
            },
        ),
    }

    scoring = metric
    per_model_best = {}
    fitted_estimators = {}

    # ------------------------------
    # CV + GridSearch
    # ------------------------------
    for name, (pipe, grid) in candidates.items():

        search = GridSearchCV(
            pipe,
            grid,
            scoring=scoring,
            cv=cv,
            n_jobs=-1,
            refit=True,
        )

        search.fit(X_train, y_train)

        per_model_best[name] = {
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
        }

        fitted_estimators[name] = search.best_estimator_

    # ------------------------------
    # Select best model
    # ------------------------------
    best_model_name = max(
        per_model_best,
        key=lambda k: per_model_best[k]["best_cv_score"],
    )

    best_pipeline = fitted_estimators[best_model_name]
    best_cv_score = per_model_best[best_model_name]["best_cv_score"]

    # ------------------------------
    # Threshold tuning (Internal split)
    # ------------------------------
    # We split the training data internally just to find the best threshold
    X_train_sub, X_val, y_train_sub, y_val = train_test_split(
        X_train,
        y_train,
        test_size=0.25,
        stratify=y_train,
        random_state=random_state,
    )

    best_pipeline.fit(X_train_sub, y_train_sub)

    threshold_info = tune_threshold(
        best_pipeline,
        X_val,
        y_val,
        metric="f1",
    )

    # ------------------------------
    # CRITICAL: Refit on FULL training data
    # ------------------------------
    best_pipeline.fit(X_train, y_train)

    # ------------------------------
    # Feature importance
    # ------------------------------
    feature_info = None

    if best_model_name == "logreg":
        feature_info = extract_logreg_coefficients(
            best_pipeline,
            list(X_train.columns),
        )

    elif best_model_name == "random_forest":
        feature_info = extract_rf_importance(
            best_pipeline,
            list(X_train.columns),
        )

    # ------------------------------
    # Save artifacts
    # ------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    pipeline_path, meta_path = save_paths

    joblib.dump(best_pipeline, pipeline_path)

    meta = {
        "best_model_name": best_model_name,
        "selected_metric": metric,
        "best_cv_score": best_cv_score,
        "test_scores": None,  # Evaluation is done in the notebook now
        "threshold": threshold_info,
        "feature_columns": list(X_train.columns),
        "feature_importance": (
            feature_info.to_dict(orient="records")
            if feature_info is not None
            else None
        ),
        "per_model_best": per_model_best,
        "scaler_type": scaler_type,
        "cv_splits": cv_splits,
        "random_state": random_state,
    }

    joblib.dump(meta, meta_path)

    return {
        "best_model_name": best_model_name,
        "best_cv_score": best_cv_score,
        "threshold": threshold_info,
        "pipeline_path": str(pipeline_path),
        "meta_path": str(meta_path),
    }


# ============================================================
# Entry point
# ============================================================

if __name__ == "__main__":

    # Load the specific training split created by your other script
    train_path = Path("data/train_split.csv")
    
    print(f"Loading training data: {train_path}")
    train_df = pd.read_csv(train_path)

    print("Training (CV-based)...")
    results = train_and_save_model_cv(train_df, select_by="f1")

    print("\nTraining complete ✅")
    print("Best model:", results["best_model_name"])
    print("Best CV score:", results["best_cv_score"])
    print("Threshold:", results["threshold"])
    print("Saved pipeline to:", results["pipeline_path"])