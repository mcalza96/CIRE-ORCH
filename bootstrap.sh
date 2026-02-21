#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$BASE_DIR"

if command -v python3 >/dev/null 2>&1; then
  PYTHON_CMD=python3
elif command -v python >/dev/null 2>&1; then
  PYTHON_CMD=python
else
  echo "❌ Python 3 not found"
  exit 1
fi

echo "📦 Bootstrapping orchestrator..."

if [ ! -d "venv" ]; then
  $PYTHON_CMD -m venv venv
fi

source venv/bin/activate
./venv/bin/python -m pip install -r requirements-core.txt

echo "✅ orchestrator ready"
