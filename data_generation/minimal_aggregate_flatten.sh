#!/usr/bin/env bash
# minimal_aggregate_flatten.sh
# Run minimal_aggregate_flatten.py for each immediate subdirectory ending with "_T0_F2.ds"
# Usage: minimal_aggregate_flatten.sh [ROOT_DIR]
set -euo pipefail

ROOT_DIR="${1:-.}"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PY_SCRIPT="${SCRIPT_DIR}/minimal_aggregate_flatten.py"

if [[ ! -f "$PY_SCRIPT" ]]; then
    echo "Error: $PY_SCRIPT not found." >&2
    exit 2
fi

shopt -s nullglob
for dir in "$ROOT_DIR"/*/*_T2_F0.ds; do
    if [[ -d "$dir" ]]; then
        echo "Processing: $dir"
        python3 "$PY_SCRIPT" --input-ds-path "$dir"
    fi
done
shopt -u nullglob
