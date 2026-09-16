"""
Training entry point.

Reproduces the leakage-safe pipeline from the research notebook (temporal
split, leakage-safe features, class-weighted LightGBM, F1-maximising
threshold selection on validation, final held-out test evaluation) and then
saves versioned, reloadable artifacts:

  artifacts/model_<timestamp>.joblib   the fitted sklearn Pipeline
  artifacts/model_latest.joblib        a copy of the most recent model
  artifacts/metadata.json              threshold, metrics, feature list,
                                        training data hash, timestamp

Usage:
    python -m src.train --data-path /path/to/fraud.csv
    # or
    FRAUD_DATA_PATH=/path/to/fraud.csv python -m src.train
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    average_precision_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder
import lightgbm as lgb

from src.features import (
    CATEGORICAL_FEATURES,
    FEATURE_COLUMNS,
    NUMERIC_FEATURES,
    engineer_features_batch,
)

RANDOM_STATE = 42
ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"


def temporal_split(data: pd.DataFrame, train_frac=0.70, val_frac=0.85):
    """Split by unique `step` values, not row position, so the split
    boundary always falls on a whole time step."""
    data = data.sort_values("step").reset_index(drop=True)
    unique_steps = np.sort(data["step"].unique())
    n_steps = len(unique_steps)

    train_end = unique_steps[int(n_steps * train_frac) - 1]
    val_end = unique_steps[int(n_steps * val_frac) - 1]

    train_mask = data["step"] <= train_end
    val_mask = (data["step"] > train_end) & (data["step"] <= val_end)
    test_mask = data["step"] > val_end
    return train_mask, val_mask, test_mask


def build_pipeline() -> Pipeline:
    preprocessor = ColumnTransformer(
        transformers=[
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    return Pipeline(
        [
            ("preprocess", preprocessor),
            (
                "model",
                lgb.LGBMClassifier(
                    n_estimators=300,
                    learning_rate=0.05,
                    num_leaves=31,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                    n_jobs=-1,
                    verbosity=-1,
                ),
            ),
        ]
    )


def evaluate(y_true, prob, threshold: float) -> dict:
    pred = (prob >= threshold).astype(int)
    cm = confusion_matrix(y_true, pred)
    return {
        "threshold": threshold,
        "precision": precision_score(y_true, pred, zero_division=0),
        "recall": recall_score(y_true, pred, zero_division=0),
        "f1": f1_score(y_true, pred, zero_division=0),
        "roc_auc": roc_auc_score(y_true, prob),
        "pr_auc": average_precision_score(y_true, prob),
        "false_positives": int(cm[0, 1]),
        "false_negatives": int(cm[1, 0]),
    }


def select_threshold(y_true, prob) -> float:
    """F1-maximising threshold on the validation set (a neutral default in
    the absence of a real operating cost matrix - see README)."""
    thresholds = np.arange(0.05, 1.00, 0.01)
    best_t, best_f1 = 0.5, -1.0
    for t in thresholds:
        pred = (prob >= t).astype(int)
        f1 = f1_score(y_true, pred, zero_division=0)
        if f1 > best_f1:
            best_f1, best_t = f1, t
    return float(best_t)


def file_hash(path: Path, chunk_size=8_388_608) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()[:16]


def main():
    parser = argparse.ArgumentParser(description="Train the fraud-detection model.")
    parser.add_argument(
        "--data-path",
        default=os.environ.get("FRAUD_DATA_PATH"),
        help="Path to the PaySim CSV. Falls back to $FRAUD_DATA_PATH.",
    )
    args = parser.parse_args()

    if not args.data_path:
        sys.exit("No data path given. Use --data-path or set FRAUD_DATA_PATH.")

    data_path = Path(args.data_path)
    if not data_path.exists():
        sys.exit(f"Data file not found: {data_path}")

    print(f"Loading data from {data_path} ...")
    data = pd.read_csv(data_path)
    print(f"Loaded {len(data):,} rows.")

    train_mask, val_mask, test_mask = temporal_split(data)

    print("Engineering leakage-safe features ...")
    engineered = engineer_features_batch(data)

    X = engineered[FEATURE_COLUMNS]
    y = engineered["isFraud"]

    X_train, y_train = X.loc[train_mask], y.loc[train_mask]
    X_val, y_val = X.loc[val_mask], y.loc[val_mask]
    X_test, y_test = X.loc[test_mask], y.loc[test_mask]

    print(f"Train: {len(X_train):,}  Val: {len(X_val):,}  Test: {len(X_test):,}")

    print("Training LightGBM ...")
    pipeline = build_pipeline()
    pipeline.fit(X_train, y_train)

    val_prob = pipeline.predict_proba(X_val)[:, 1]
    threshold = select_threshold(y_val, val_prob)
    val_metrics = evaluate(y_val, val_prob, threshold)
    print("Validation metrics:", json.dumps(val_metrics, indent=2))

    test_prob = pipeline.predict_proba(X_test)[:, 1]
    test_metrics = evaluate(y_test, test_prob, threshold)
    print("Held-out test metrics:", json.dumps(test_metrics, indent=2))

    # ---- Save versioned artifacts ------------------------------------
    ARTIFACT_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    model_path = ARTIFACT_DIR / f"model_{timestamp}.joblib"
    latest_path = ARTIFACT_DIR / "model_latest.joblib"

    joblib.dump(pipeline, model_path)
    joblib.dump(pipeline, latest_path)

    metadata = {
        "model_version": timestamp,
        "trained_at_utc": timestamp,
        "random_state": RANDOM_STATE,
        "feature_columns": FEATURE_COLUMNS,
        "decision_threshold": threshold,
        "validation_metrics": val_metrics,
        "test_metrics": test_metrics,
        "training_data_file": data_path.name,
        "training_data_hash_sha256_16": file_hash(data_path),
        "training_data_rows": int(len(data)),
    }
    with open(ARTIFACT_DIR / "metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\nSaved model to {model_path}")
    print(f"Saved metadata to {ARTIFACT_DIR / 'metadata.json'}")
    print(f"Selected decision threshold: {threshold}")


if __name__ == "__main__":
    main()
