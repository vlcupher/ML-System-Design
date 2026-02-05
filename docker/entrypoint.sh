#!/bin/sh
set -e
exec python -m uvicorn diagram_service.main:app --host 0.0.0.0 --port "${PORT:-8000}"
