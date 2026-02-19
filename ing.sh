#!/bin/bash

set -euo pipefail

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="$BASE_DIR/tools/ingestion-client/ing.sh"

if [[ ! -x "$TARGET_SCRIPT" ]]; then
  echo "❌ No se encontró el cliente de ingesta en: $TARGET_SCRIPT"
  exit 1
fi

echo "ℹ️  Usando cliente unificado de ingesta (tools/ingestion-client/ing.sh)"
exec "$TARGET_SCRIPT" "$@"
