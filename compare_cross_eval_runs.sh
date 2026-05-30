#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cross-Eval Run Comparison Script
# Discovers all CrossEval run directories and generates comparison plots
# =============================================================================

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/results"
OUTPUT_DIR="${RESULTS_BASE}/cross_eval_comparison"

# Activate virtual environment
VENV_DIR="${SCRIPT_DIR}/.venv"
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

echo "============================================"
echo "Cross-Eval Run Comparison"
echo "============================================"
echo "Results base:     ${RESULTS_BASE}"
echo "Output dir:       ${OUTPUT_DIR}"
echo "============================================"

# Auto-discover CrossEval directories
RUN_DIRS=()
for d in "${RESULTS_BASE}"/CrossEval_*/; do
    if [ -d "$d" ] && [ -f "$d/results.csv" ]; then
        RUN_DIRS+=("$(realpath "$d")")
    fi
done

if [ ${#RUN_DIRS[@]} -eq 0 ]; then
    echo "ERROR: No CrossEval run directories with results.csv found in ${RESULTS_BASE}"
    exit 1
fi

echo ""
echo "Found ${#RUN_DIRS[@]} cross-eval runs:"
for d in "${RUN_DIRS[@]}"; do
    echo "  - $(basename "$d")"
done
echo ""

mkdir -p "${OUTPUT_DIR}"

# =============================================================================
# Comparison
# =============================================================================
echo "Generating comparison plots..."
echo ""

python "${SCRIPT_DIR}/compare_cross_eval_runs.py" \
    --runs "${RUN_DIRS[@]}" \
    --output-dir "${OUTPUT_DIR}"

echo ""
echo "============================================"
echo "Done! Comparison plots saved to: ${OUTPUT_DIR}"
echo "============================================"
