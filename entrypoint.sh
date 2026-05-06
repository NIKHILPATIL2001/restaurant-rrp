#!/usr/bin/env bash
set -euo pipefail

MODE="${1:-api}"

if [ "$MODE" = "dashboard" ]; then
    exec streamlit run dashboard/app.py \
        --server.port 8501 \
        --server.address 0.0.0.0 \
        --server.headless true
fi

# API mode: migrate -> seed -> train -> warmup -> serve
echo "[entrypoint] Running Alembic migrations..."
alembic upgrade head

echo "[entrypoint] Seeding synthetic data (idempotent)..."
python -m rrp.scripts.seed --if-empty

echo "[entrypoint] Training baseline model (idempotent)..."
python -m rrp.scripts.train_baseline --if-missing

WARMUP_DAYS="${RRP_WARMUP_DAYS:-30}"
echo "[entrypoint] Starting background warmup simulation (${WARMUP_DAYS} days)..."
python -m rrp.scripts.warmup_simulate --days "$WARMUP_DAYS" &

echo "[entrypoint] Starting API server..."
exec uvicorn rrp.api.main:app --host 0.0.0.0 --port 8000
