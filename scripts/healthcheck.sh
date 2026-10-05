#!/usr/bin/env bash
set -euo pipefail
PORT="${STACMEM_PORT:-8020}"
curl --noproxy '*' -fsS "http://127.0.0.1:${PORT}/health"
echo
