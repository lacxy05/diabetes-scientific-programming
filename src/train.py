from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score

from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier


MODELS_DIR = Path("models")
MODEL_PATH = MODELS_DIR / "model.pkl"
SCALER_PATH = MODELS_DIR / "scaler.pkl"


def train_and_save_model(
    df: pd.DataFrame,
    *,
    test_size: float = 0.2,
    random_state: int = 42,
    scaler_type: str = "minmax",          # "minmax" or "standard"
    select_by: str = "roc_auc",           # "roc_auc", "f1", "accuracy"
    save_paths: Tuple[Path, Path] = (MODEL_PATH, SCALER_PATH),
) -> Dict[str, Any]:
    """
    Train multiple models and save the best model + scaler using joblib.
    Expects cleaned df with target column 'Outcome'.
    """

    if df is None or not isinstance(df, pd.DataFrame):
        raise ValueError("df must be a pandas DataFrame.")

    if "Outcome" not in df.columns:
        raise ValueError("Expected target column 'Outcome' in df.")

    y = df["Outcome"].astype(int)
    X = df.drop(columns=["Outcome", "subject_id"], errors="ignore")

    if X.isna().any().any():
        raise ValueError("X contains NaNs. Ensure preprocessing removed/handled missing values.")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    scaler = MinMaxScaler() if scaler_type.lower() == "minmax" else StandardScaler()
    X_train_s = scaler.fit_transform(X_train)
    X_test_s = scaler.transform(X_test)

    candidates = {
        "logreg": LogisticRegression(max_iter=2000, random_state=random_state),
        "svm_rbf": SVC(kernel="rbf", probability=True, random_state=random_state),
        "random_forest": RandomForestClassifier(
            n_estimators=400,
            random_state=random_state,
            class_weight="balanced",
        ),
    }

    def compute_scores(model) -> Dict[str, float]:
        proba = model.predict_proba(X_test_s)[:, 1]
        pred = model.predict(X_test_s)
        return {
            "roc_auc": float(roc_auc_score(y_test, proba)),
            "accuracy": float(accuracy_score(y_test, pred)),
            "f1": float(f1_score(y_test, pred)),
        }

    scores = {}
    trained_models = {}

    for name, model in candidates.items():
        model.fit(X_train_s, y_train)
        trained_models[name] = model
        scores[name] = compute_scores(model)

    if select_by not in ["roc_auc", "accuracy", "f1"]:
        raise ValueError("select_by must be one of: 'roc_auc', 'accuracy', 'f1'")

    best_model_name = max(scores, key=lambda k: scores[k][select_by])
    best_model = trained_models[best_model_name]
    best_score = scores[best_model_name][select_by]

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    model_path, scaler_path = save_paths

    joblib.dump(best_model, model_path)
    joblib.dump(scaler, scaler_path)

    return {
        "best_model_name": best_model_name,
        "best_score": best_score,
        "selected_metric": select_by,
        "all_scores": scores,
        "feature_columns": list(X.columns),
        "model_path": str(model_path),
        "scaler_path": str(scaler_path),
    }


if __name__ == "__main__":
    clean_data_path = Path("data/diabetes_cleaned.csv")

    try:
        print("Loading data from ", clean_data_path)
        df = pd.read_csv(clean_data_path)

        print("Training model...")
        results = train_and_save_model(df)

        print("Training complete")
        print("Best model:", results["best_model_name"])
        print("Best score (", results["selected_metric"], "): ", results["best_score"])
        print("Saved model to:", results["model_path"])

    except FileNotFoundError:
        print(f"File {clean_data_path} does not exist.")



