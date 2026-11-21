# Serving image: FastAPI + ONNX Runtime only (no torch, no training deps).
# Models are trained on MVTec AD (CC BY-NC-SA 4.0) and are NOT baked into the
# image; mount them at runtime:
#   docker run -p 8000:8000 -v ./models:/app/models anomaly-mlops
FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    MODELS_DIR=/app/models \
    SAMPLES_DIR=/app/data/mvtec_ad \
    PORT=8000

WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install . && useradd --create-home appuser && mkdir -p /app/models /app/data \
    && chown -R appuser /app
USER appuser

EXPOSE 8000
CMD ["sh", "-c", "uvicorn anomaly.serve.api:app --host 0.0.0.0 --port ${PORT}"]
