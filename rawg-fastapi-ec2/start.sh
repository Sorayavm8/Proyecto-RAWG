#!/usr/bin/env bash
set -e
if [ -f ".env" ]; then
  set -a
  source .env
  set +a
fi
exec gunicorn -k uvicorn.workers.UvicornWorker app.main:app --bind 0.0.0.0:8000 --workers ${WORKERS:-2} --timeout ${TIMEOUT:-60}
