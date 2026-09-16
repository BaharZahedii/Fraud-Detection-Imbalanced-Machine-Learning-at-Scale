# Fraud Detection — Deployable Service

This turns the research notebook (`Fraud_Detection.ipynb`) into a
small, deployable project: a training script that saves versioned model
artifacts, and a FastAPI service that scores one transaction at a time.

The modelling logic (temporal split, leakage-safe features, class-weighted
LightGBM, F1-maximising threshold) is unchanged from the notebook — this
project only adds the layer needed to actually run it as a service.

## Project layout

```
fraud_deploy/
  src/
    features.py   feature engineering — batch (training) and online (serving)
    train.py       CLI training script, saves artifacts/
    schemas.py     API request/response shapes
    serve.py       FastAPI app
  tests/
    test_features.py   unit tests for the feature logic
    test_api.py         API contract tests, using a small dummy model
  artifacts/       trained model + metadata land here (not committed)
  requirements.txt
  Dockerfile
  .github/workflows/ci.yml   runs tests + a Docker build on every push
```

## 1. Set up

```bash
python -m venv .venv
source .venv/bin/activate        # Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## 2. Train the model

You need the PaySim CSV locally (not included in this project — download it
from Kaggle: "Synthetic Financial Datasets For Fraud Detection").

```bash
python -m src.train --data-path /path/to/fraud.csv
```

This will:
- load and temporally split the data (70% train / 15% validation / 15% test)
- engineer the leakage-safe features
- train the LightGBM pipeline
- pick a decision threshold on the validation set
- evaluate once on the held-out test set
- save `artifacts/model_latest.joblib` and `artifacts/metadata.json`

`metadata.json` records the threshold, validation/test metrics, the exact
feature list, a hash of the training data file, and a timestamp — so any
saved model is traceable back to the data and settings that produced it.

## 3. Run the API locally

```bash
uvicorn src.serve:app --reload
```

Then, in another terminal:

```bash
curl http://localhost:8000/health

curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{
    "step": 743,
    "type": "TRANSFER",
    "amount": 181000.00,
    "nameOrig": "C1231006815",
    "oldbalanceOrg": 181000.00,
    "newbalanceOrig": 0.00,
    "nameDest": "C1900366749",
    "oldbalanceDest": 0.00,
    "newbalanceDest": 0.00
  }'
```

Interactive API docs (Swagger UI) are auto-generated at
`http://localhost:8000/docs`.

## 4. Run it in Docker

```bash
docker build -t fraud-api .
docker run -p 8000:8000 fraud-api
```

The trained model is baked into the image at build time. To ship a new
model version: retrain, then rebuild the image. (A real production setup
would instead pull the model from a model registry at container startup —
see "What's simplified" below.)

## 5. Run the tests

```bash
pytest -q
```

`test_features.py` checks the feature logic in isolation, including that
the batch (training-time) and online (serving-time) feature computations
agree with each other for the same sequence of transactions — if they ever
drifted apart, the model would see different features in production than
it saw in training. `test_api.py` checks the API's request/response
contract using a small dummy model, so tests run in seconds without
needing the full multi-million-row dataset. `.github/workflows/ci.yml`
runs both automatically on every push.

## Why an in-memory feature store, and what to change for real production

Several features (`hist_per_sender`, `time_since_last`, `total_past_amount`,
`AVG_txn_amount`) depend on a sender's transaction history. At training
time that's easy — the whole dataset is available and the notebook computes
them with a `groupby`. At serving time, a single transaction arrives with no
access to the rest of the dataset, so `SenderHistoryStore` in `features.py`
keeps a small running total per sender in memory instead.

That's fine for a demo on one process, but it has two real limitations
worth naming honestly:

- **Not persistent** — restart the API and all sender history is lost,
  so every sender looks like a first-time sender again.
- **Not shared** — running two API replicas behind a load balancer, each
  would keep its own separate history for the same sender.

For a real deployment, `SenderHistoryStore` would be backed by a shared,
persistent store (Redis is a natural fit for fast per-key counters) or a
proper feature store product, keyed by `nameOrig`. The rest of the code
would be unchanged — only `features.py`'s storage backend would need to
change.

## What's simplified here, on purpose

This is a portfolio-scale reference implementation, not a finished
production system. Deliberately out of scope:

- **Authentication / authorization** on the API
- **A model registry** (the model is baked into the Docker image rather
  than pulled from a registry at startup)
- **A real feature store** (see above)
- **Structured logging / metrics export** (e.g. to Prometheus) for the
  monitoring described in the notebook's productionisation section
- **A real operating cost matrix** — the decision threshold is still the
  F1-maximising point from the notebook, which is a reasonable neutral
  default but not a business decision about the true cost of a false
  positive versus a false negative
