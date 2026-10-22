#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Tiled PatchCore — train + evaluate on very large images (one command)
#
# Thin wrapper around tiled_patchcore/run_tiled_patchcore.py: native-resolution
# tiling, ONE global coreset over pooled tile features, full-resolution
# stitched anomaly maps, and the full analytics pack (plots/, per_image_scores,
# memory_bank.json, ...).
#
# Quickstart: edit DATASET_PATH below, then  ./run_tiled_patchcore.sh
# Everything is env-overridable:  BACKBONE=dinov3_vitl ./run_tiled_patchcore.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Config — the usual one to change: DATASET_PATH.
# Accepts EITHER a category folder (runs just that category) ...
DATASET_PATH="${DATASET_PATH:-/home/daniel/jetbrains-projects/dino-core-v1/dataset/40mp/mosque}"
# ... OR a dataset root containing category folders (runs all of them).
# (A folder counts as a category when it contains a train/ subdir.)
# -----------------------------------------------------------------------------

BACKBONE="${BACKBONE:-dinov3_vitb}"           # DINOv3 layers/dims auto-derived
TILE_SIZE="${TILE_SIZE:-384}"                 # = backbone input; multiple of 16
TILE_OVERLAP="${TILE_OVERLAP:-}"              # empty -> tile_size/2 (50% overlap)
TILE_BATCH_SIZE="${TILE_BATCH_SIZE:-32}"       # bounds GPU peak (tiles per forward)
CORESET_PERCENTAGE="${CORESET_PERCENTAGE:-0.01}"
NUM_NN="${NUM_NN:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"                 # images per dataloader batch
NUM_WORKERS="${NUM_WORKERS:-0}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-auto}"                      # auto (CUDA > Apple MPS > CPU) | cpu | <N> cuda id
RESULTS_PATH="${RESULTS_PATH:-}"              # empty -> <repo>/results_tiled
SAVE_SEGMENTATION_IMAGES="${SAVE_SEGMENTATION_IMAGES:-0}"  # 1 -> per-image renders
DRY_RUN="${DRY_RUN:-0}"                       # 1 -> print the command, don't run

# Advanced (leave empty for sensible defaults / auto-detection):
LAYERS="${LAYERS:-}"                          # e.g. "8,12" (comma-separated)
PRETRAIN_DIM="${PRETRAIN_DIM:-}"              # required only for exotic backbones
TARGET_DIM="${TARGET_DIM:-}"
FAISS_NUM_WORKERS="${FAISS_NUM_WORKERS:-8}"
PYTHON="${PYTHON:-}"                          # e.g. /path/to/python; empty -> uv run python

# =============================================================================
# Script setup (normally nothing to edit below this line)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"                            # weights/ paths are repo-root-relative

RESULTS_PATH="${RESULTS_PATH:-${SCRIPT_DIR}/results_tiled}"
LOG_GROUP="${LOG_GROUP:-tiled_${BACKBONE}_T${TILE_SIZE}_p${CORESET_PERCENTAGE}}"

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS ships faiss-cpu + torch with duplicate libomp copies; heavy FAISS
    # search threads can segfault.
    FAISS_NUM_WORKERS=4
    # Same known macOS packaging issue: torch+faiss both embed libomp and the
    # interpreter aborts at import (OMP Error #15) unless these are set.
    # Only applied when the user has not set them; harmless on Linux.
    export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi

if [[ ! -d "${DATASET_PATH}" ]]; then
    echo "ERROR: DATASET_PATH does not exist: ${DATASET_PATH}" >&2
    exit 1
fi

# Category folder or dataset root? (category dirs contain train/ per MVTec format)
dataset_flags=()
if [[ -d "${DATASET_PATH%/}/train" ]]; then
    DATA_ROOT="$(dirname "${DATASET_PATH%/}")"
    CATEGORY="$(basename "${DATASET_PATH%/}")"
    dataset_flags=(--subdatasets "${CATEGORY}")
else
    DATA_ROOT="${DATASET_PATH}"               # root -> auto-discover all categories
fi

cmd=()
if [[ -n "${PYTHON}" ]]; then
    cmd+=("${PYTHON}")
else
    cmd+=(uv run python)
fi
cmd+=("${SCRIPT_DIR}/tiled_patchcore/run_tiled_patchcore.py"
      --data_path "${DATA_ROOT}"
      --backbone "${BACKBONE}"
      --tile_size "${TILE_SIZE}"
      --tile_batch_size "${TILE_BATCH_SIZE}"
      --coreset_percentage "${CORESET_PERCENTAGE}"
      --anomaly_scorer_num_nn "${NUM_NN}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --seed "${SEED}"
      --faiss_num_workers "${FAISS_NUM_WORKERS}"
      --results_path "${RESULTS_PATH}"
      --log_group "${LOG_GROUP}")
cmd+=("${dataset_flags[@]+"${dataset_flags[@]}"}")

[[ -n "${TILE_OVERLAP}" ]]   && cmd+=(--tile_overlap "${TILE_OVERLAP}")
[[ -n "${LAYERS}" ]]         && cmd+=(--layers "${LAYERS}")
[[ -n "${PRETRAIN_DIM}" ]]   && cmd+=(--pretrain_embed_dimension "${PRETRAIN_DIM}")
[[ -n "${TARGET_DIM}" ]]     && cmd+=(--target_embed_dimension "${TARGET_DIM}")
[[ "${SAVE_SEGMENTATION_IMAGES}" == "1" ]] && cmd+=(--save_segmentation_images)
if [[ "${GPU_ID}" == "cpu" ]]; then
    cmd+=(--cpu)
elif [[ "${GPU_ID}" != "auto" ]]; then
    cmd+=(--gpu "${GPU_ID}")
fi

echo "============================================"
echo "Tiled PatchCore"
echo "============================================"
echo "Dataset root:     ${DATA_ROOT}"
echo "Category:         ${CATEGORY:-<all, auto-discovered>}"
echo "Backbone:         ${BACKBONE}"
echo "Tile size:        ${TILE_SIZE}  overlap: ${TILE_OVERLAP:-auto (tile/2)}"
echo "Tile batch:       ${TILE_BATCH_SIZE}"
echo "Coreset keep:     ${CORESET_PERCENTAGE}"
echo "Num NN / Seed:    ${NUM_NN} / ${SEED}"
echo "GPU:              ${GPU_ID}"
echo "Results:          ${RESULTS_PATH}/${LOG_GROUP}*"
echo "============================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run — would execute:"
    printf '  %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"

echo ""
echo "Done. Artifacts (results.csv, plots/, per_image_scores.csv, memory_bank.json, models/):"
find "${RESULTS_PATH}" -maxdepth 1 -name "${LOG_GROUP}*" -type d | sort | tail -1
