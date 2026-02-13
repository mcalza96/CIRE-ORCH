#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

cleanup() {
  echo ""
  echo "🛑 Deteniendo orchestrator..."
  "$BASE_DIR/stack.sh" down
}

trap cleanup INT TERM

echo "🚀 Levantando stack de desarrollo (orchestrator)..."
"$BASE_DIR/stack.sh" up

echo ""
echo "📜 Logs en vivo (Ctrl+C para detener todo):"
echo "   - $BASE_DIR/.logs/orchestrator-api.log"

touch "$BASE_DIR/.logs/orchestrator-api.log"
tail -f "$BASE_DIR/.logs/orchestrator-api.log"
