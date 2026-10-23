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
# Extra CLI flags are appended to every invocation:
#   ./run_tiled_patchcore.sh --save_segmentation_images
#
# Layout:
#   dataset root + PER_CATEGORY_RUNS=0 -> ONE run folder per arm containing
#     models/mvtec_<cat>/ for every category  (what cross_evaluate_tiled.sh wants)
#   dataset root + PER_CATEGORY_RUNS=1 -> one run folder PER category
#     (<prefix>_<cat>), crash of one category does not stop the others
#   category folder -> one run folder named <prefix>_<cat>
# =============================================================================

# -----------------------------------------------------------------------------
# Config — the usual one to change: DATASET_PATH.
# Accepts EITHER a category folder (runs just that category) ...
DATASET_PATH="${DATASET_PATH:-/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/BigD}"
#DATASET_PATH="${DATASET_PATH:-/home/daniel/jetbrains-projects/dino-core-v1/dataset/40mp/mosque}"
# ... OR a dataset root containing category folders (runs all of them).
# (A folder counts as a category when it contains a train/ subdir.)
# -----------------------------------------------------------------------------
CATEGORIES="${CATEGORIES:-}"                  # root mode subset, e.g. "pcb1 pcb2"; empty = all
PER_CATEGORY_RUNS="${PER_CATEGORY_RUNS:-0}"   # 1 -> one run folder + analytics pack per category

# Current best combo for BigD (vits / T192 / layers 3,7,12):
BACKBONE="${BACKBONE:-dinov3_vitb}"           # DINOv3 layers/dims auto-derived
LAYERS="${LAYERS:-3,7,12}"                    # empty -> family default ('7,8' for DINOv3)
TILE_SIZE="${TILE_SIZE:-768}"                 # = backbone input; multiple of 16
TILE_OVERLAP="${TILE_OVERLAP:-}"              # empty -> tile_size/2 (50% overlap)
TILE_BATCH_SIZE="${TILE_BATCH_SIZE:-32}"       # bounds GPU peak (tiles per forward)
PAD_VALUE="${PAD_VALUE:-0.0}"                 # padding fill for border tiles
PATCHSIZE="${PATCHSIZE:-1}"                   # 1 = per-patch features (DINOv3)
CORESET_PERCENTAGE="${CORESET_PERCENTAGE:-0.01}"
NUM_NN="${NUM_NN:-3}"
TRAIN_VAL_SPLIT="${TRAIN_VAL_SPLIT:-1.0}"     # < 1.0 -> hold out val samples (trainer has no VAL loader)
BATCH_SIZE="${BATCH_SIZE:-1}"                 # images per dataloader batch
NUM_WORKERS="${NUM_WORKERS:-8}"                # native-res dataloading is IO-heavy
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-auto}"                      # auto (CUDA > Apple MPS > CPU) | cpu | <N> cuda id
FAISS_NUM_WORKERS="${FAISS_NUM_WORKERS:-8}"
FAISS_TEMP_MEMORY_MB="${FAISS_TEMP_MEMORY_MB:-256}"

RESULTS_PATH="${RESULTS_PATH:-}"              # empty -> <repo>/results_tiled. Keep ONE
                                              # folder per training arm (real vs synthetic) —
                                              # same LOG_GROUP in both = indistinguishable _N runs
LOG_PROJECT="${LOG_PROJECT:-tiled_patchcore}"
LOG_GROUP_PREFIX="${LOG_GROUP_PREFIX:-tiled_${BACKBONE}_T${TILE_SIZE}_p${CORESET_PERCENTAGE}}"
SAVE_SEGMENTATION_IMAGES="${SAVE_SEGMENTATION_IMAGES:-1}"  # 1 -> per-image renders (heavy at native res)
TOP_ANOMALOUS="${TOP_ANOMALOUS:-4}"           # panels with tile-grid overlay per category (0 disables)
NO_SAVE_MODEL="${NO_SAVE_MODEL:-0}"           # 1 -> skip saving the memory bank
DRY_RUN="${DRY_RUN:-0}"                       # 1 -> print the commands, don't run

# Advanced (leave empty for sensible defaults / auto-detection):
PRETRAIN_DIM="${PRETRAIN_DIM:-}"              # required only for exotic backbones
TARGET_DIM="${TARGET_DIM:-}"
PYTHON="${PYTHON:-}"                          # e.g. /path/to/python; empty -> uv run python

# =============================================================================
# Script setup (normally nothing to edit below this line)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"                            # weights/ paths are repo-root-relative

RESULTS_PATH="${RESULTS_PATH:-${SCRIPT_DIR}/results_tiled}"
extra_args=("$@")                             # pass-through flags, applied to every run

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

# ---------------------------------------------------------------------------
# Command construction
# ---------------------------------------------------------------------------
python_cmd=()
if [[ -n "${PYTHON}" ]]; then
    python_cmd+=("${PYTHON}")
else
    python_cmd+=(uv run python)
fi
python_cmd+=("${SCRIPT_DIR}/tiled_patchcore/run_tiled_patchcore.py")

execute() {
    # execute <log_group> [category flags...]
    local log_group="$1"; shift
    local -a cmd=("${python_cmd[@]}"
        --data_path "${DATA_ROOT}"
        --backbone "${BACKBONE}"
        --tile_size "${TILE_SIZE}"
        --tile_batch_size "${TILE_BATCH_SIZE}"
        --pad_value "${PAD_VALUE}"
        --patchsize "${PATCHSIZE}"
        --coreset_percentage "${CORESET_PERCENTAGE}"
        --anomaly_scorer_num_nn "${NUM_NN}"
        --train_val_split "${TRAIN_VAL_SPLIT}"
        --batch_size "${BATCH_SIZE}"
        --num_workers "${NUM_WORKERS}"
        --seed "${SEED}"
        --faiss_num_workers "${FAISS_NUM_WORKERS}"
        --faiss_temp_memory_mb "${FAISS_TEMP_MEMORY_MB}"
        --results_path "${RESULTS_PATH}"
        --log_project "${LOG_PROJECT}"
        --log_group "${log_group}"
        --top_anomalous "${TOP_ANOMALOUS}"
    )
    cmd+=("$@")

    [[ -n "${TILE_OVERLAP}" ]]   && cmd+=(--tile_overlap "${TILE_OVERLAP}")
    [[ -n "${LAYERS}" ]]         && cmd+=(--layers "${LAYERS}")
    [[ -n "${PRETRAIN_DIM}" ]]   && cmd+=(--pretrain_embed_dimension "${PRETRAIN_DIM}")
    [[ -n "${TARGET_DIM}" ]]     && cmd+=(--target_embed_dimension "${TARGET_DIM}")
    [[ "${SAVE_SEGMENTATION_IMAGES}" == "1" ]] && cmd+=(--save_segmentation_images)
    [[ "${NO_SAVE_MODEL}" == "1" ]] && cmd+=(--no_save_model)
    if [[ "${GPU_ID}" == "cpu" ]]; then
        cmd+=(--cpu)
    elif [[ "${GPU_ID}" != "auto" ]]; then
        cmd+=(--gpu "${GPU_ID}")
    fi
    cmd+=("${extra_args[@]+"${extra_args[@]}"}")

    if [[ "${DRY_RUN}" == "1" ]]; then
        echo "### ${log_group}"
        printf '  %q' "${cmd[@]}"
        printf '\n'
        return 0
    fi
    "${cmd[@]}"
}

discover_categories() {
    local dir
    for dir in "${DATA_ROOT}"/*/; do
        [[ -d "${dir%/}/train" ]] && basename "${dir%/}"
    done
}

cats=()
load_categories() {
    # Fills the global `cats` array: CATEGORIES (space-separated) or discovery.
    # Validation runs in the main shell so a bad name exits the script.
    cats=()
    if [[ -n "${CATEGORIES}" ]]; then
        read -r -a cats <<< "${CATEGORIES}"
        local cat
        for cat in "${cats[@]}"; do
            if [[ ! -d "${DATA_ROOT}/${cat}/train" ]]; then
                echo "ERROR: '${cat}' is not a category under ${DATA_ROOT} (no train/)." >&2
                exit 1
            fi
        done
    else
        mapfile -t cats < <(discover_categories)
    fi
}

failed=()

# ---------------------------------------------------------------------------
# Resolve mode and run
# ---------------------------------------------------------------------------
if [[ -d "${DATASET_PATH%/}/train" ]]; then
    # Single category folder -> its own run folder (never a bare shared name).
    DATA_ROOT="$(dirname "${DATASET_PATH%/}")"
    CATEGORY="$(basename "${DATASET_PATH%/}")"
    MODE="single category"
    LOG_GROUP_NAME="${LOG_GROUP_PREFIX}_${CATEGORY}"
    execute "${LOG_GROUP_NAME}" -d "${CATEGORY}"
elif [[ "${PER_CATEGORY_RUNS}" == "1" ]]; then
    DATA_ROOT="${DATASET_PATH}"
    MODE="one run per category"
    load_categories
    for cat in "${cats[@]}"; do
        if ! execute "${LOG_GROUP_PREFIX}_${cat}" -d "${cat}"; then
            failed+=("${cat}")
        fi
    done
else
    DATA_ROOT="${DATASET_PATH}"
    MODE="one run, all selected categories"
    sub_flags=()
    if [[ -n "${CATEGORIES}" ]]; then
        load_categories
        for cat in "${cats[@]}"; do
            sub_flags+=(-d "${cat}")
        done
    fi
    execute "${LOG_GROUP_PREFIX}" "${sub_flags[@]+"${sub_flags[@]}"}"
fi

echo ""
echo "============================================"
echo "Tiled PatchCore"
echo "============================================"
echo "Mode:             ${MODE}"
echo "Dataset root:     ${DATA_ROOT}"
echo "Category:         ${CATEGORY:-<all, auto-discovered>}"
echo "Backbone/layers:  ${BACKBONE} / ${LAYERS:-family default}"
echo "Tile size:        ${TILE_SIZE}  overlap: ${TILE_OVERLAP:-auto (tile/2)}"
echo "Tile batch:       ${TILE_BATCH_SIZE}"
echo "Coreset keep:     ${CORESET_PERCENTAGE}"
echo "Num NN / Seed:    ${NUM_NN} / ${SEED}"
echo "GPU:              ${GPU_ID}"
echo "Results:          ${RESULTS_PATH}/${LOG_GROUP_PREFIX}*"
echo "============================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run — nothing executed."
    exit 0
fi

if (( ${#failed[@]} )); then
    echo "FAILED categories: ${failed[*]}" >&2
    exit 1
fi

echo ""
echo "Done. Artifacts (results.csv, plots/, per_image_scores.csv, memory_bank.json, models/):"
find "${RESULTS_PATH}" -maxdepth 1 -name "${LOG_GROUP_PREFIX}*" -type d | sort
