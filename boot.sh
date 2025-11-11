#!/usr/bin/env bash
set -e
if [ -n "$SA_JSON_B64" ]; then
  echo "$SA_JSON_B64" | base64 -d > service_account.json
  export SERVICE_ACCOUNT_JSON=service_account.json
fi
python index_drive.py || true
exec gunicorn -w 2 -b 0.0.0.0:${PORT:-10000} app:app
