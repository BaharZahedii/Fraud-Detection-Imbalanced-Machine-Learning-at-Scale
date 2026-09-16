# --- Fraud Detection API -----------------------------------------------
# Build:  docker build -t fraud-api .
# Run:    docker run -p 8000:8000 fraud-api
#
# The trained model (artifacts/model_latest.joblib + metadata.json) is
# copied into the image at build time. Retrain and rebuild the image to
# ship a new model version — see README for a note on doing this via a
# model registry instead, for a real production setup.

FROM python:3.11-slim

WORKDIR /app

# System deps needed by LightGBM's compiled component.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY src/ src/
COPY artifacts/ artifacts/

ENV PYTHONPATH=/app
EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=3s CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/health')" || exit 1

CMD ["uvicorn", "src.serve:app", "--host", "0.0.0.0", "--port", "8000"]
