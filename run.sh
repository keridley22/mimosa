#!/bin/bash
set -euo pipefail
cd "$(dirname "$0")"
PYTHON_BIN="${PLAQUE_PYTHON:-/Users/katherineridley/anaconda3/bin/python}"
export MPLBACKEND=Agg
export MPLCONFIGDIR="${TMPDIR:-/tmp}/young-plaque-matplotlib"
exec "$PYTHON_BIN" main.py "$@"
