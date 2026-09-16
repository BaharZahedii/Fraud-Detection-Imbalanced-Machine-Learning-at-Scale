import json

import joblib
import numpy as np
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from src.features import CATEGORICAL_FEATURES, FEATURE_COLUMNS, NUMERIC_FEATURES


@pytest.fixture(scope="module")
def dummy_artifacts(tmp_path_factory):
    """Train a tiny, fast dummy pipeline so tests don't need the real
    multi-million-row PaySim CSV. This only checks the API contract, not
    model quality."""
    rng = np.random.default_rng(0)
    n = 200
    df = pd.DataFrame(
        {
            "step": rng.integers(1, 50, n),
            "type": rng.choice(["PAYMENT", "TRANSFER", "CASH_OUT"], n),
            "amount": rng.uniform(1, 1000, n),
            "oldbalanceOrg": rng.uniform(0, 2000, n),
            "newbalanceOrig": rng.uniform(0, 2000, n),
            "oldbalanceDest": rng.uniform(0, 2000, n),
            "newbalanceDest": rng.uniform(0, 2000, n),
            "hist_per_sender": rng.integers(0, 10, n),
            "time_since_last": rng.integers(0, 20, n),
            "AVG_txn_amount": rng.uniform(0, 500, n),
            "total_past_amount": rng.uniform(0, 5000, n),
            "balance_change": rng.uniform(-500, 500, n),
            "insufficient_funds": rng.integers(0, 2, n),
            "is_first_txn": rng.integers(0, 2, n),
        }
    )
    y = rng.integers(0, 2, n)

    preprocessor = ColumnTransformer(
        [
            ("num", SimpleImputer(strategy="median"), NUMERIC_FEATURES),
            ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
        ]
    )
    pipeline = Pipeline([("preprocess", preprocessor), ("model", LogisticRegression())])
    pipeline.fit(df[FEATURE_COLUMNS], y)

    artifact_dir = tmp_path_factory.mktemp("artifacts")
    model_path = artifact_dir / "model_latest.joblib"
    metadata_path = artifact_dir / "metadata.json"

    joblib.dump(pipeline, model_path)
    metadata = {
        "model_version": "test-dummy",
        "decision_threshold": 0.5,
        "feature_columns": FEATURE_COLUMNS,
    }
    with open(metadata_path, "w") as f:
        json.dump(metadata, f)

    return artifact_dir


@pytest.fixture()
def client(dummy_artifacts, monkeypatch):
    import src.serve as serve_module

    monkeypatch.setattr(serve_module, "MODEL_PATH", dummy_artifacts / "model_latest.joblib")
    monkeypatch.setattr(serve_module, "METADATA_PATH", dummy_artifacts / "metadata.json")

    with TestClient(serve_module.app) as test_client:
        yield test_client


def test_health_reports_loaded_model(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert body["model_version"] == "test-dummy"


def test_predict_returns_expected_shape(client):
    payload = {
        "step": 10,
        "type": "TRANSFER",
        "amount": 5000.0,
        "nameOrig": "C_TEST_1",
        "oldbalanceOrg": 5000.0,
        "newbalanceOrig": 0.0,
        "nameDest": "C_TEST_2",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 5000.0,
    }
    response = client.post("/predict", json=payload)
    assert response.status_code == 200

    body = response.json()
    assert 0.0 <= body["fraud_probability"] <= 1.0
    assert isinstance(body["is_fraud_prediction"], bool)
    assert body["decision_threshold"] == 0.5
    assert body["sender_transaction_history_count"] == 0


def test_predict_updates_sender_history_across_calls(client):
    payload = {
        "step": 1,
        "type": "PAYMENT",
        "amount": 10.0,
        "nameOrig": "C_REPEAT",
        "oldbalanceOrg": 100.0,
        "newbalanceOrig": 90.0,
        "nameDest": "C_DEST",
        "oldbalanceDest": 0.0,
        "newbalanceDest": 10.0,
    }
    first = client.post("/predict", json=payload)
    assert first.json()["sender_transaction_history_count"] == 0

    payload["step"] = 2
    second = client.post("/predict", json=payload)
    assert second.json()["sender_transaction_history_count"] == 1
