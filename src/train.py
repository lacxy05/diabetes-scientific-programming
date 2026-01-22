from __future__ import annotations

# --------------------------------------------------
# Standard library imports
# --------------------------------------------------
from pathlib import Path
from typing import Dict, Any, Tuple

# --------------------------------------------------
# Third-party libraries
# --------------------------------------------------
import joblib
import pandas as pd

# --------------------------------------------------
# Scikit-learn imports
# --------------------------------------------------
from sklearn.model_selection import (
    train_test_split,
    StratifiedKFold,
    GridSearchCV,
)
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


# --------------------------------------------------
# Paths where trained artifacts will be saved
# --------------------------------------------------
MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.pkl"     # trained ML model
SCALER_PATH = MODELS_DIR / "scaler.pkl"   # fitted feature scaler


def train_and_save_model(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_type: str = "minmax",          # "minmax" or "standard"
    select_by: str = "roc_auc",           # "roc_auc", "f1", "accuracy"
    cv_splits: int = 5,
    save_paths: Tuple[Path, Path] = (MODEL_PATH, SCALER_PATH),
) -> Dict[str, Any]:
    """
    Train multiple classification models with hyperparameter tuning,
    select the best model using cross-validation, evaluate it on a
    hold-out test set, and save the model and scaler.

    Parameters
    ----------
    df : pd.DataFrame
        Cleaned dataset containing features and target column 'Outcome'
    test_size : float
        Proportion of data reserved for final testing
    random_state : int
        Seed for reproducibility
    scaler_type : str
        Feature scaling method ("minmax" or "standard")
    select_by : str
        Metric used to select best model ("roc_auc", "accuracy", "f1")
    cv_splits : int
        Number of cross-validation folds

    Returns
    -------
    Dict[str, Any]
        Summary of training results and saved artifact paths
    """

    # --------------------------------------------------
    # Input validation
    # --------------------------------------------------
    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    if "Outcome" not in df.columns:
        raise ValueError("Expected target column 'Outcome' in df.")

    if select_by not in ["roc_auc", "accuracy", "f1"]:
        raise ValueError("select_by must be 'roc_auc', 'accuracy', or 'f1'")

    # --------------------------------------------------
    # Separate features (X) and target (y)
    # --------------------------------------------------
    y = df["Outcome"].astype(int)

    # Remove target column and optional identifier column
    X = df.drop(columns=["Outcome", "subject_id"], errors="ignore")

    # Ensure no missing values remain
    if X.isna().any().any():
        raise ValueError("X contains NaNs. Handle missing values before training.")

    # --------------------------------------------------
    # Train / Test split (test set untouched by CV)
    # --------------------------------------------------
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=test_size,
        stratify=y,             # preserve class balance
        random_state=random_state,
    )

    # --------------------------------------------------
    # Feature scaling
    # --------------------------------------------------
    # Scaler is fitted ONLY on training data
    scaler = (
        MinMaxScaler()
        if scaler_type.lower() == "minmax"
        else StandardScaler()
    )

    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    # --------------------------------------------------
    # Cross-validation strategy
    # --------------------------------------------------
    cv = StratifiedKFold(
        n_splits=cv_splits,
        shuffle=True,
        random_state=random_state,
    )

    # --------------------------------------------------
    # Candidate models (base estimators)
    # --------------------------------------------------
    candidates = {
        "logreg": LogisticRegression(
            max_iter=3000,
            random_state=random_state,
        ),
        "svm_rbf": SVC(
            kernel="rbf",
            probability=True,    # required for ROC-AUC
            random_state=random_state,
        ),
        "random_forest": RandomForestClassifier(
            random_state=random_state,
        ),
    }

    # --------------------------------------------------
    # Hyperparameter search spaces
    # --------------------------------------------------
    param_grids = {
        "logreg": {
            "C": [0.01, 0.1, 1, 10, 100],
            "penalty": ["l1", "l2"],
            "solver": ["liblinear", "saga"],
            "class_weight": [None, "balanced"],
        },
        "svm_rbf": {
            "C": [0.1, 1, 10, 100],
            "gamma": ["scale", 0.01, 0.001, 0.0001],
            "class_weight": [None, "balanced"],
        },
        "random_forest": {
            "n_estimators": [200, 400, 800],
            "max_depth": [None, 5, 10, 20],
            "min_samples_leaf": [1, 2, 4, 8],
            "max_features": ["sqrt", "log2"],
            "class_weight": [None, "balanced", "balanced_subsample"],
        },
    }

    # --------------------------------------------------
    # Metric used during cross-validation
    # --------------------------------------------------
    scoring_map = {
        "roc_auc": "roc_auc",
        "accuracy": "accuracy",
        "f1": "f1",
    }
    scoring = scoring_map[select_by]

    tuned_models = {}
    cv_results = {}

    # --------------------------------------------------
    # Hyperparameter tuning for each model
    # --------------------------------------------------
    for name, model in candidates.items():

        search = GridSearchCV(
            estimator=model,
            param_grid=param_grids[name],
            scoring=scoring,
            cv=cv,
            n_jobs=-1,      # use all available cores
            refit=True,     # refit best model on full training set
        )

        search.fit(X_train_s, y_train)

        tuned_models[name] = search.best_estimator_
        cv_results[name] = {
            "best_cv_score": float(search.best_score_),
            "best_params": search.best_params_,
        }

    # --------------------------------------------------
    # Select the best model based on CV performance
    # --------------------------------------------------
    best_model_name = max(
        cv_results,
        key=lambda k: cv_results[k]["best_cv_score"],
    )

    best_model = tuned_models[best_model_name]
    best_cv_score = cv_results[best_model_name]["best_cv_score"]

    # --------------------------------------------------
    # Final evaluation on the unseen test set
    # --------------------------------------------------
    proba = best_model.predict_proba(X_test_s)[:, 1]
    pred = best_model.predict(X_test_s)

    test_scores = {
        "roc_auc": float(roc_auc_score(y_test, proba)),
        "accuracy": float(accuracy_score(y_test, pred)),
        "f1": float(f1_score(y_test, pred)),
    }

    # --------------------------------------------------
    # Save trained artifacts
    # --------------------------------------------------
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path, scaler_path = save_paths

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    # --------------------------------------------------
    # Return summary information
    # --------------------------------------------------
    return {
        "best_model_name": best_model_name,
        "selected_metric": select_by,
        "best_cv_score": best_cv_score,
        "cv_details": cv_results,
        "test_scores": test_scores,
        "feature_columns": list(X.columns),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
        "scaler_type": scaler_type,
        "cv_splits": cv_splits,
    }


# --------------------------------------------------
# Script entry point
# --------------------------------------------------
if __name__ == "__main__":
    clean_data_path = Path("data/diabetes_cleaned.csv")

    try:
        print("Loading data from", clean_data_path)
        df = pd.read_csv(clean_data_path)

        print("Training model with hyperparameter tuning...")
        results = train_and_save_model(df)

        print("\nTraining complete")
        print("Best model:", results["best_model_name"])
        print(
            "Best CV score (",
            results["selected_metric"],
            "):",
            results["best_cv_score"],
        )
        print("Test scores:", results["test_scores"])
        print("Saved model to:", results["model_path"])
        print("Saved scaler to:", results["scaler_path"])

    except FileNotFoundError:
        print(f"File {clean_data_path} does not exist.")
