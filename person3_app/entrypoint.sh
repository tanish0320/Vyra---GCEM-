#!/bin/sh
# AnemiaVision AI — Docker entrypoint
# 1. Run DB migrations (creates tables in Postgres if they don't exist)
# 2. Launch gunicorn

set -e

echo "[entrypoint] Running database migrations..."
python migrate.py

echo "[entrypoint] Starting gunicorn..."
exec gunicorn \
  --bind "0.0.0.0:${PORT:-5000}" \
  --workers "${ANEMIA_GUNICORN_WORKERS:-1}" \
  --timeout "${ANEMIA_GUNICORN_TIMEOUT:-180}" \
  --log-level info \
  app:app
