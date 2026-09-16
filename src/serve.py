"""
FastAPI service exposing the fraud-detection model.

Run locally:
    uvicorn src.serve:app --reload

Run in Docker: see Dockerfile / README.

Endpoints:
    GET  /health    liveness + model version check
    POST /predict   score a single transaction
"""

from __future__ import annotations

import json
import logging
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, HTTPException

from src.features import SenderHistoryStore, build_feature_row
from src.schemas import PredictionResponse, Transaction

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("fraud-api")

ARTIFACT_DIR = Path(__file__).resolve().parent.parent / "artifacts"
MODEL_PATH = ARTIFACT_DIR / "model_latest.joblib"
METADATA_PATH = ARTIFACT_DIR / "metadata.json"

# Module-level state, populated at startup.
_model = None
_metadata: dict = {}
_history_store = SenderHistoryStore()


def load_model() -> None:
    global _model, _metadata

    if not MODEL_PATH.exists() or not METADATA_PATH.exists():
        logger.warning(
            "No trained model found at %s. Run `python -m src.train` first. "
            "The API will start but /predict will return 503 until a model exists.",
            MODEL_PATH,
        )
        return

    _model = joblib.load(MODEL_PATH)
    with open(METADATA_PATH) as f:
        _metadata = json.load(f)
    logger.info("Loaded model version %s", _metadata.get("model_version"))


@asynccontextmanager
async def lifespan(app: FastAPI):
    load_model()
    yield


app = FastAPI(
    title="Fraud Detection API",
    description="Serves the leakage-safe LightGBM fraud model.",
    version="1.0.0",
    lifespan=lifespan,
)


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok" if _model is not None else "no_model_loaded",
        "model_version": _metadata.get("model_version"),
        "decision_threshold": _metadata.get("decision_threshold"),
        "known_senders_in_memory": len(_history_store),
    }


@app.post("/predict", response_model=PredictionResponse)
def predict(transaction: Transaction) -> PredictionResponse:
    if _model is None:
        raise HTTPException(
            status_code=503,
            detail="No trained model is loaded. Run `python -m src.train` first.",
        )

    raw = transaction.model_dump()

    engineered = _history_store.get_features(
        name_orig=raw["nameOrig"],
        step=raw["step"],
        amount=raw["amount"],
        old_balance_org=raw["oldbalanceOrg"],
    )

    row = build_feature_row(raw, engineered)
    probability = float(_model.predict_proba(row)[:, 1][0])
    threshold = float(_metadata["decision_threshold"])

    # Update the sender's history AFTER scoring, so this transaction is
    # visible to the *next* one from the same sender — not to itself.
    _history_store.update(raw["nameOrig"], raw["step"], raw["amount"])

    return PredictionResponse(
        fraud_probability=probability,
        is_fraud_prediction=probability >= threshold,
        decision_threshold=threshold,
        model_version=_metadata.get("model_version", "unknown"),
        sender_transaction_history_count=engineered["hist_per_sender"],
    )
