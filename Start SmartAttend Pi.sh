#!/bin/bash
set -e

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON_BIN="$BASE_DIR/venv/bin/python"
APP_FILE="$BASE_DIR/app.py"

if [ ! -x "$PYTHON_BIN" ]; then
  echo "SmartAttend could not find $PYTHON_BIN"
  echo "Create the virtual environment and install dependencies first."
  exit 1
fi

if [ ! -f "$APP_FILE" ]; then
  echo "SmartAttend could not find $APP_FILE"
  exit 1
fi

export AUTO_OPEN_BROWSER="${AUTO_OPEN_BROWSER:-true}"
export AUTO_OPEN_BROWSER_URL="${AUTO_OPEN_BROWSER_URL:-http://127.0.0.1:5000}"
export AUTO_OPEN_BROWSER_DELAY_SEC="${AUTO_OPEN_BROWSER_DELAY_SEC:-2.5}"

cd "$BASE_DIR"
exec "$PYTHON_BIN" "$APP_FILE"
