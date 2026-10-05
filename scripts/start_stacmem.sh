#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON="${STACMEM_PYTHON:-$PROJECT_ROOT/.venv/bin/python}"

if [[ -f "$PROJECT_ROOT/.env" ]]; then
  set -a
  source "$PROJECT_ROOT/.env"
  set +a
fi

PORT="${STACMEM_PORT:-8020}"
DATABASE="${STACMEM_DATABASE:-$PROJECT_ROOT/runtime/stacmem.sqlite3}"

if [[ ! -x "$PYTHON" ]]; then
  echo "Missing $PYTHON. Run: bash scripts/quickstart.sh" >&2
  exit 1
fi
if [[ -z "${DASHSCOPE_API_KEY:-}" ]]; then
  echo "DASHSCOPE_API_KEY is required for natural-language ingestion." >&2
  exit 1
fi

cd "$PROJECT_ROOT"
exec "$PYTHON" -m stacmem.standalone_cli \
  --config configs/standalone_qwen.toml \
  --database "$DATABASE" \
  serve --port "$PORT"
