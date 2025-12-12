#!/bin/sh
# Startup script for Railway/Cloud deployment
# Reads PORT from environment (Railway sets this automatically)

PORT=${PORT:-8000}
exec uvicorn api:app --host 0.0.0.0 --port $PORT

