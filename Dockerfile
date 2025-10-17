# syntax=docker/dockerfile:1.4

FROM python:3.11-slim AS base

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    PORT=8000 \
    TRAINING_DATA_BUCKET=raw-data-stocks \
    TRAINING_DATA_PREFIX=stock_data \
    TRAINING_MODEL_BUCKET=raw-data-stocks \
    TRAINING_MODEL_PREFIX=models

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        libgomp1 \
        libgl1 \
        libglib2.0-0 \
        curl \
        ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt ./

RUN pip install --upgrade pip setuptools wheel \
    && pip install --no-cache-dir -r requirements.txt

COPY . .

# Ensure both project root and src/ are on the Python path
ENV PYTHONPATH=/app:/app/src

RUN useradd --create-home --shell /bin/bash appuser \
    && chown -R appuser:appuser /app

RUN chmod +x scripts/entrypoint.sh scripts/train_once.py || true

USER appuser

EXPOSE ${PORT}

ENTRYPOINT ["/bin/sh", "/app/scripts/entrypoint.sh"]
