#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# PatchCore Training & Evaluation Script
# =============================================================================

# Configurable dataset path (MVTec AD format)
DATASET_PATH="${DATASET_PATH:-/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/BigD_synthetic_anomalies_v7}"

# Training configuration
GPU_ID="${GPU_ID:-0}"
SEED="${SEED:-0}"
IMAGE_SIZE="${IMAGE_SIZE:-384}"
RESIZE="${RESIZE:-384}"
BACKBONE="${BACKBONE:-dinov3_vitb}"
LAYERS="${LAYERS:-4 12}"
CORESET_PERCENTAGE="${CORESET_PERCENTAGE:-0.1}"
PRETRAIN_DIM="${PRETRAIN_DIM:-768}"
TARGET_DIM="${TARGET_DIM:-768}"
PATCH_SIZE="${PATCH_SIZE:-1}"
NUM_NN="${NUM_NN:-9}"

if [[ "${BACKBONE}" == wideresnet50 ]]; then
    PRETRAIN_DIM=1024
    TARGET_DIM=1024
fi

LOG_GROUP="${LOG_GROUP:-IM${IMAGE_SIZE}_${BACKBONE}_L${LAYERS// /-}_P${CORESET_PERCENTAGE}_D${PRETRAIN_DIM}-${TARGET_DIM}_PS-${PATCH_SIZE}_AN-${NUM_NN}_S${SEED}}"
LOG_PROJECT="${LOG_PROJECT:-PatchCore_Results}"

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
    echo "Run: cd ${SCRIPT_DIR} && uv venv && uv pip install -r requirements.txt"
    exit 1
fi

echo "============================================"
echo "PatchCore Training & Evaluation"
echo "============================================"
echo "Dataset path:     ${DATASET_PATH}"
echo "Results base:     ${RESULTS_BASE}"
echo "GPU:              ${GPU_ID}"
echo "Seed:             ${SEED}"
echo "Image size:       ${IMAGE_SIZE}"
echo "Resize:           ${RESIZE}"
echo "Backbone:         ${BACKBONE}"
echo "Layers:           ${LAYERS}"
echo "Coreset %:        ${CORESET_PERCENTAGE}"
echo "Pretrain dim:     ${PRETRAIN_DIM}"
echo "Target dim:       ${TARGET_DIM}"
echo "Patch size:       ${PATCH_SIZE}"
echo "Num NN:           ${NUM_NN}"
echo "Log group:        ${LOG_GROUP}"
echo "Log project:      ${LOG_PROJECT}"
echo "============================================"

if [ ! -d "${DATASET_PATH}" ]; then
    echo "ERROR: Dataset path does not exist: ${DATASET_PATH}"
    exit 1
fi

mkdir -p "${RESULTS_BASE}"

#DATASETS=('bottle' 'cable' 'capsule' 'carpet' 'grid' 'hazelnut' 'leather' 'metal_nut' 'pill' 'screw' 'tile' 'toothbrush' 'transistor' 'wood' 'zipper')
DATASETS=('candle' 'capsules' 'cashew' 'chewinggum' 'fryum' 'macaroni1' 'macaroni2' 'pcb1' 'pcb2' 'pcb3' 'pcb4' 'pipe_fryum')

dataset_flags=()
for dataset in "${DATASETS[@]}"; do
    dataset_flags+=('-d' "${dataset}")
done

layer_flags=()
for layer in ${LAYERS}; do
    layer_flags+=('-le' "${layer}")
done

# =============================================================================
# Training
# =============================================================================
echo ""
echo "Starting PatchCore training..."
echo ""

export PYTHONPATH="${SCRIPT_DIR}/src:${PYTHONPATH:-}"

python "${SCRIPT_DIR}/scripts/patchcore/run_patchcore.py" \
    --gpu "${GPU_ID}" \
    --seed "${SEED}" \
    --save_patchcore_model \
    --save_segmentation_images \
    --log_group "${LOG_GROUP}" \
    --log_project "${LOG_PROJECT}" \
    "${RESULTS_BASE}" \
    patch_core \
        -b "${BACKBONE}" \
        "${layer_flags[@]}" \
        --pretrain_embed_dimension "${PRETRAIN_DIM}" \
        --target_embed_dimension "${TARGET_DIM}" \
        --anomaly_scorer_num_nn "${NUM_NN}" \
        --patchsize "${PATCH_SIZE}" \
    sampler \
        -p "${CORESET_PERCENTAGE}" \
        approx_greedy_coreset \
    dataset \
        --resize "${RESIZE}" \
        --imagesize "${IMAGE_SIZE}" \
        "${dataset_flags[@]}" \
        mvtec "${DATASET_PATH}"

# Find the actual run folder created by create_storage_folder
RESULTS_DIR=$(find "${RESULTS_BASE}" -maxdepth 1 -name "${LOG_GROUP}*" -type d | sort | tail -1)
if [ -z "${RESULTS_DIR}" ]; then
    echo "ERROR: Could not find results directory"
    exit 1
fi

echo ""
echo "Training complete. Results saved to: ${RESULTS_DIR}"
echo ""

# =============================================================================
# Plotting
# =============================================================================
echo "Generating evaluation plots..."
echo ""

mkdir -p "${RESULTS_DIR}/plots"

python "${SCRIPT_DIR}/plot_results.py" \
    --curves-dir "${RESULTS_DIR}/curves" \
    --output-dir "${RESULTS_DIR}/plots"

echo ""
echo "============================================"
echo "Done! All results and plots saved to: ${RESULTS_DIR}"
echo "============================================"