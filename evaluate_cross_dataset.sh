#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Cross-Dataset Evaluation Script
# Evaluates trained PatchCore models on a different (real) dataset
# =============================================================================

# Configurable paths
MODEL_ROOT="${MODEL_ROOT:-results/IM224_MVTEC_BENCH_dinov3_vitl_L7-8_P0.1_D1024-1024_PS-3_AN-1_S0_0}"
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/mvtec_small}"

# Evaluation configuration
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-224}"
RESIZE="${RESIZE:-256}"
EVAL_LOG_GROUP="${EVAL_LOG_GROUP:-CrossEval_$(date +%Y%m%d_%H%M%S)}"
EVAL_LOG_PROJECT="${EVAL_LOG_PROJECT:-CrossEval_Results}"

# Derive model name from MODEL_ROOT for folder naming
MODEL_NAME=$(basename "$(realpath "${MODEL_ROOT}")")
EVAL_LOG_GROUP="${EVAL_LOG_GROUP}_${MODEL_NAME}"

# =============================================================================
# Script setup
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESULTS_BASE="${SCRIPT_DIR}/results"

# Activate virtual environment
VENV_DIR="${SCRIPT_DIR}/.venv"
if [ -d "${VENV_DIR}" ]; then
    source "${VENV_DIR}/bin/activate"
else
    echo "ERROR: Virtual environment not found at ${VENV_DIR}"
    exit 1
fi

echo "============================================"
echo "Cross-Dataset Evaluation"
echo "============================================"
echo "Model root:       ${MODEL_ROOT}"
echo "Dataset path:     ${EVAL_DATASET_PATH}"
echo "Results base:     ${RESULTS_BASE}"
echo "GPU:              ${GPU_ID}"
echo "Seed:             ${SEED}"
echo "============================================"

if [ ! -d "${MODEL_ROOT}" ]; then
    echo "ERROR: Model root does not exist: ${MODEL_ROOT}"
    exit 1
fi

if [ ! -d "${EVAL_DATASET_PATH}" ]; then
    echo "ERROR: Dataset path does not exist: ${EVAL_DATASET_PATH}"
    exit 1
fi

mkdir -p "${RESULTS_BASE}"

# Auto-detect categories from models directory
echo ""
echo "Detecting available models..."
CATEGORIES=()
if [ -d "${MODEL_ROOT}/models" ]; then
    for model_dir in "${MODEL_ROOT}/models"/mvtec_*; do
        if [ -d "${model_dir}" ]; then
            cat_name=$(basename "${model_dir}" | sed 's/mvtec_//')
            CATEGORIES+=("${cat_name}")
        fi
    done
fi

if [ ${#CATEGORIES[@]} -eq 0 ]; then
    echo "ERROR: No mvtec_* model directories found in ${MODEL_ROOT}/models/"
    exit 1
fi

echo "Found ${#CATEGORIES[@]} model categories: ${CATEGORIES[*]}"
echo ""

# Build dataset flags
dataset_flags=()
for category in "${CATEGORIES[@]}"; do
    dataset_flags+=('-d' "${category}")
done

# =============================================================================
# Evaluation
# =============================================================================
echo "Starting cross-dataset evaluation..."
echo ""

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/scripts/patchcore/evaluate_cross_dataset.py" \
    --gpu "${GPU_ID}" \
    --seed "${SEED}" \
    --save_segmentation_images \
    --log_group "${EVAL_LOG_GROUP}" \
    --log_project "${EVAL_LOG_PROJECT}" \
    "${RESULTS_BASE}" \
    cross_eval \
        --model_root "${MODEL_ROOT}" \
    dataset \
        --resize "${RESIZE}" \
        --imagesize "${IMAGE_SIZE}" \
        "${dataset_flags[@]}" \
        mvtec "${EVAL_DATASET_PATH}"

# Find the actual run folder created by create_storage_folder
RESULTS_DIR=$(find "${RESULTS_BASE}" -maxdepth 1 -name "${EVAL_LOG_GROUP}*" -type d | sort | tail -1)
if [ -z "${RESULTS_DIR}" ]; then
    echo "ERROR: Could not find results directory"
    exit 1
fi

echo ""
echo "Evaluation complete. Results saved to: ${RESULTS_DIR}"
echo ""

# =============================================================================
# Plotting
# =============================================================================
echo "Generating cross-evaluation plots..."
echo ""

python "${SCRIPT_DIR}/plot_cross_eval_results.py" \
    --results-dir "${RESULTS_DIR}" \
    --output-dir "${RESULTS_DIR}/plots"

echo ""
echo "============================================"
echo "Done! All results and plots saved to: ${RESULTS_DIR}"
echo "============================================"
