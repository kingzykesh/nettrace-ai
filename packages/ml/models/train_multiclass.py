import json
from pathlib import Path

import joblib
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.preprocessing import LabelEncoder

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)

from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier

PROCESSED_DIR = Path("datasets/processed")
ARTIFACTS_DIR = Path("packages/ml/artifacts")
REPORTS_DIR = Path("packages/ml/reports")

ARTIFACTS_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)


def load_dataset():

    X_train = pd.read_parquet(PROCESSED_DIR / "X_train_multiclass.parquet")
    X_test = pd.read_parquet(PROCESSED_DIR / "X_test_multiclass.parquet")

    y_train = pd.read_parquet(
        PROCESSED_DIR / "y_train_multiclass.parquet"
    )["Attack"]

    y_test = pd.read_parquet(
        PROCESSED_DIR / "y_test_multiclass.parquet"
    )["Attack"]

    return X_train, X_test, y_train, y_test


def preprocess(X_train, X_test, y_train, y_test):

    scaler = StandardScaler()

    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)

    encoder = LabelEncoder()

    y_train = encoder.fit_transform(y_train)
    y_test = encoder.transform(y_test)

    joblib.dump(
        scaler,
        ARTIFACTS_DIR / "multiclass_scaler.joblib",
    )

    joblib.dump(
        encoder,
        ARTIFACTS_DIR / "attack_label_encoder.joblib",
    )

    return X_train, X_test, y_train, y_test, encoder


def evaluate(model_name, model, X_test, y_test, encoder):

    predictions = model.predict(X_test)

    metrics = {

        "model": model_name,

        "accuracy":
        accuracy_score(
            y_test,
            predictions,
        ),

        "weighted_f1":
        f1_score(
            y_test,
            predictions,
            average="weighted",
        ),

        "confusion_matrix":
        confusion_matrix(
            y_test,
            predictions,
        ).tolist(),

        "classification_report":
        classification_report(
            y_test,
            predictions,
            target_names=encoder.classes_,
            output_dict=True,
            zero_division=0,
        ),

    }

    with open(
        REPORTS_DIR / f"{model_name.lower()}_multiclass_metrics.json",
        "w",
    ) as file:

        json.dump(metrics, file, indent=4)

    print(json.dumps(metrics, indent=4))

    return metrics


def train_random_forest(X_train, y_train):

    model = RandomForestClassifier(

        n_estimators=200,

        random_state=42,

        class_weight="balanced",

        n_jobs=-1,

    )

    model.fit(X_train, y_train)

    return model


def train_xgboost(X_train, y_train):

    model = XGBClassifier(

        objective="multi:softprob",

        num_class=len(set(y_train)),

        eval_metric="mlogloss",

        n_estimators=300,

        learning_rate=0.08,

        max_depth=8,

        subsample=0.8,

        colsample_bytree=0.8,

        tree_method="hist",

        random_state=42,

        n_jobs=-1,

    )

    model.fit(X_train, y_train)

    return model


def main():

    print("Loading dataset...")

    X_train, X_test, y_train, y_test = load_dataset()

    print("Preprocessing...")

    X_train, X_test, y_train, y_test, encoder = preprocess(
        X_train,
        X_test,
        y_train,
        y_test,
    )

    print("\nTraining Random Forest...")

    rf = train_random_forest(
        X_train,
        y_train,
    )

    rf_metrics = evaluate(
        "RandomForest",
        rf,
        X_test,
        y_test,
        encoder,
    )

    print("\nTraining XGBoost...")

    xgb = train_xgboost(
        X_train,
        y_train,
    )

    xgb_metrics = evaluate(
        "XGBoost",
        xgb,
        X_test,
        y_test,
        encoder,
    )

    if (
        xgb_metrics["weighted_f1"]
        >=
        rf_metrics["weighted_f1"]
    ):

        best_model = xgb

        best_name = "XGBoost"

    else:

        best_model = rf

        best_name = "RandomForest"

    joblib.dump(

        best_model,

        ARTIFACTS_DIR / "multiclass_model.joblib",

    )

    metadata = {

        "selected_model": best_name,

        "classes": encoder.classes_.tolist(),

        "number_of_classes": len(encoder.classes_),

    }

    with open(

        ARTIFACTS_DIR / "multiclass_training_metadata.json",

        "w",

    ) as file:

        json.dump(metadata, file, indent=4)

    print("\n========================")

    print(f"Best Model : {best_name}")

    print("Model saved successfully.")

    print("========================")


if __name__ == "__main__":
    main()