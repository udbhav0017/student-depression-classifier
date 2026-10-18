"""Train and save a depression-classification model from the supplied CSV."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


TARGET = "Depression"


def load_data(path: Path) -> pd.DataFrame:
    data = pd.read_csv(path)
    if TARGET not in data.columns:
        raise ValueError(f"Expected target column '{TARGET}' was not found. Columns: {list(data.columns)}")
    data = data.dropna(subset=[TARGET]).copy()
    data[TARGET] = pd.to_numeric(data[TARGET], errors="raise").astype(int)
    if set(data[TARGET].unique()) - {0, 1}:
        raise ValueError(f"'{TARGET}' must contain only 0 and 1 for this classifier.")
    return data


def build_pipeline(numeric_columns: list[str], categorical_columns: list[str]) -> Pipeline:
    numeric = Pipeline([("imputer", SimpleImputer(strategy="median"))])
    categorical = Pipeline(
        [
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )
    preprocess = ColumnTransformer(
        [("numeric", numeric, numeric_columns), ("categorical", categorical, categorical_columns)]
    )
    return Pipeline(
        [("preprocess", preprocess), ("model", LogisticRegression(max_iter=2000, class_weight="balanced"))]
    )


def make_feature_schema(features: pd.DataFrame, numeric_columns: list[str], categorical_columns: list[str]) -> dict:
    schema: dict[str, dict] = {}
    for column in numeric_columns:
        values = pd.to_numeric(features[column], errors="coerce")
        schema[column] = {
            "kind": "numeric",
            "median": float(values.median()),
            "min": float(values.min()),
            "max": float(values.max()),
        }
    for column in categorical_columns:
        values = features[column].dropna().astype(str)
        schema[column] = {"kind": "categorical", "options": sorted(values.unique().tolist())}
    return schema


def main() -> None:
    parser = argparse.ArgumentParser(description="Train the student-depression classifier.")
    parser.add_argument("--data", required=True, help="Path to the source CSV file.")
    parser.add_argument("--output-dir", default="artifacts", help="Directory for saved model outputs.")
    parser.add_argument("--test-size", type=float, default=0.20, help="Held-out test proportion (default: 0.20).")
    args = parser.parse_args()

    if not 0 < args.test_size < 1:
        raise ValueError("--test-size must be between 0 and 1.")
    data_path = Path(args.data)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    data = load_data(data_path)
    # IDs identify rows but do not describe a student's condition, so exclude them from prediction.
    features = data.drop(columns=[TARGET, "id"], errors="ignore")
    target = data[TARGET]
    numeric_columns = features.select_dtypes(include=[np.number]).columns.tolist()
    categorical_columns = [column for column in features.columns if column not in numeric_columns]

    X_train, X_test, y_train, y_test = train_test_split(
        features, target, test_size=args.test_size, random_state=42, stratify=target
    )
    pipeline = build_pipeline(numeric_columns, categorical_columns)
    pipeline.fit(X_train, y_train)
    predictions = pipeline.predict(X_test)
    probabilities = pipeline.predict_proba(X_test)[:, 1]
    metrics = {
        "accuracy": float(accuracy_score(y_test, predictions)),
        "precision": float(precision_score(y_test, predictions, zero_division=0)),
        "recall": float(recall_score(y_test, predictions, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_test, probabilities)),
        "train_rows": int(len(X_train)),
        "test_rows": int(len(X_test)),
        "positive_rate_test": float(y_test.mean()),
    }
    metadata = {
        "target": TARGET,
        "features": features.columns.tolist(),
        "numeric_columns": numeric_columns,
        "categorical_columns": categorical_columns,
        "source_file": str(data_path),
    }

    joblib.dump(pipeline, output_dir / "depression_model.joblib")
    (output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    (output_dir / "feature_schema.json").write_text(
        json.dumps(make_feature_schema(features, numeric_columns, categorical_columns), indent=2), encoding="utf-8"
    )
    (output_dir / "metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    print(json.dumps(metrics, indent=2))
    print("\nClassification report:\n")
    print(classification_report(y_test, predictions, target_names=["Not depressed", "Depressed"], zero_division=0))
    print(f"\nSaved model and metadata to: {output_dir.resolve()}")


if __name__ == "__main__":
    main()

