#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
N_TRIALS="${N_TRIALS:-100}"
GPU_ID="${GPU_ID:-0}"
TIMESTAMP=$(date +%Y%m%d)

BACKBONES=("dinov3_vits" "dinov3_vitb" "dinov3_vithplus")

echo "============================================"
echo "PatchCore Optuna Backbone Search"
echo "============================================"
echo "Timestamp:  ${TIMESTAMP}"
echo "Trials:     ${N_TRIALS} per backbone"
echo "GPU:        ${GPU_ID}"
echo "Backbones:  ${BACKBONES[*]}"
echo "============================================"
echo ""

for backbone in "${BACKBONES[@]}"; do
    STUDY_NAME="${backbone}_layer_search_${TIMESTAMP}"

    echo "============================================"
    echo "Starting: ${backbone} (${N_TRIALS} trials)"
    echo "Study:    ${STUDY_NAME}"
    echo "============================================"

    uv run python "${SCRIPT_DIR}/optimize_patchcore.py" \
        --backbone "${backbone}" \
        --n-trials "${N_TRIALS}" \
        --gpu "${GPU_ID}" \
        --timestamp "${TIMESTAMP}"

    echo ""
    echo "Done: ${backbone}"
    echo ""
done

echo "============================================"
echo "All experiments complete!"
echo "Results in: results/optuna/"
echo "============================================"
