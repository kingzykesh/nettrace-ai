import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier

PROCESSED_DIR = Path("datasets/processed")
ARTIFACTS_DIR = Path("packages/ml/artifacts")
REPORTS_DIR = Path("packages/ml/reports")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_data():
    X_train = pd.read_parquet(PROCESSED_DIR / "X_train_binary.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test_binary.parquet")
    y_train = pd.read_parquet(PROCESSED_DIR / "y_train_binary.parquet")["Label"]
    y_test = pd.read_parquet(PROCESSED_DIR / "y_test_binary.parquet")["Label"]

    return X_train, X_test, y_train, y_test


def scale_data(X_train, X_test):
    scaler = StandardScaler()

    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, ARTIFACTS_DIR / "binary_scaler.joblib")
    joblib.dump(list(X_train.columns), ARTIFACTS_DIR / "feature_columns.joblib")

    return X_train_scaled, X_test_scaled


def evaluate_model(name, model, X_test, y_test):
    y_pred = model.predict(X_test)

    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
    else:
        roc_auc = None

    metrics = {
        "model": name,
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred, zero_division=0),
        "recall": recall_score(y_test, y_pred, zero_division=0),
        "f1_score": f1_score(y_test, y_pred, zero_division=0),
        "roc_auc": roc_auc,
        "confusion_matrix": confusion_matrix(y_test, y_pred).tolist(),
        "classification_report": classification_report(
            y_test,
            y_pred,
            target_names=["Benign", "Attack"],
            zero_division=0,
            output_dict=True,
        ),
    }

    with open(REPORTS_DIR / f"{name.lower()}_binary_metrics.json", "w") as file:
        json.dump(metrics, file, indent=4)

    print(f"\n===== {name} Binary Metrics =====")
    print(json.dumps(metrics, indent=4))

    return metrics


def train_random_forest(X_train, y_train):
    model = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1,
        max_depth=20,
    )

    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train):
    attack_count = int((y_train == 1).sum())
    benign_count = int((y_train == 0).sum())
    scale_pos_weight = benign_count / attack_count

    model = XGBClassifier(
        n_estimators=250,
        max_depth=8,
        learning_rate=0.08,
        subsample=0.8,
        colsample_bytree=0.8,
        objective="binary:logistic",
        eval_metric="logloss",
        scale_pos_weight=scale_pos_weight,
        tree_method="hist",
        random_state=42,
        n_jobs=-1,
    )

    model.fit(X_train, y_train)
    return model


def main():
    print("Loading binary dataset...")
    X_train, X_test, y_train, y_test = load_data()

    print("Scaling dataset...")
    X_train_scaled, X_test_scaled = scale_data(X_train, X_test)

    print("Training Random Forest binary model...")
    rf_model = train_random_forest(X_train_scaled, y_train)
    rf_metrics = evaluate_model("RandomForest", rf_model, X_test_scaled, y_test)

    print("Training XGBoost binary model...")
    xgb_model = train_xgboost(X_train_scaled, y_train)
    xgb_metrics = evaluate_model("XGBoost", xgb_model, X_test_scaled, y_test)

    if xgb_metrics["f1_score"] >= rf_metrics["f1_score"]:
        best_model = xgb_model
        best_name = "XGBoost"
    else:
        best_model = rf_model
        best_name = "RandomForest"

    joblib.dump(best_model, ARTIFACTS_DIR / "binary_model.joblib")

    metadata = {
        "task": "binary_intrusion_detection",
        "selected_model": best_name,
        "feature_count": X_train.shape[1],
        "classes": {
            "0": "Benign",
            "1": "Attack",
        },
    }

    with open(ARTIFACTS_DIR / "binary_training_metadata.json", "w") as file:
        json.dump(metadata, file, indent=4)

    print(f"\nBest binary model saved: {best_name}")
    print("Saved to: packages/ml/artifacts/binary_model.joblib")


if __name__ == "__main__":
    main()