#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_DIR="${SCRIPT_DIR}/../../results/optuna"
PORT="${PORT:-8080}"

if [ ! -d "${RESULTS_DIR}" ]; then
    echo "ERROR: Results directory not found: ${RESULTS_DIR}"
    echo "Run an experiment first."
    exit 1
fi

echo "============================================"
echo "Merging all studies..."
echo "============================================"

uv run python "${SCRIPT_DIR}/merge_studies.py" --optuna-dir "${RESULTS_DIR}"

MERGED_DB="${RESULTS_DIR}/all_studies.db"

if [ ! -f "${MERGED_DB}" ]; then
    echo "ERROR: No study.db files found in ${RESULTS_DIR}"
    echo "Run an experiment first."
    exit 1
fi

echo ""
echo "============================================"
echo "Optuna Dashboard"
echo "============================================"
echo "Database:   ${MERGED_DB}"
echo "URL:        http://localhost:${PORT}"
echo "============================================"
echo ""

uv run optuna-dashboard --port "${PORT}" "sqlite:///${MERGED_DB}"
