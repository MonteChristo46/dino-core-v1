#!/usr/bin/env bash
set -euo pipefail

# =============================================================================
# Tiled/Plain PatchCore — cross-evaluate saved runs on ONE real test set
#
# Thin wrapper around scripts/tiled_patchcore/cross_evaluate.py: takes any number of
# training-run folders (tiled and plain mix freely), evaluates each on the
# same REAL dataset root at each checkpoint's own geometry, and writes the
# legacy-compatible CrossEval_<ts>_<model>/ folders + the identical
# comparison chart pack (compare_cross_eval_runs.py) + paired scatters.
#
# Quickstart: edit MODEL_RUNS and EVAL_DATASET_PATH below, then
#   ./cross_evaluate_tiled.sh
# Everything is env-overridable:
#   MODEL_RUNS="runA runB" EVAL_DATASET_PATH=/data/real ./cross_evaluate_tiled.sh
# =============================================================================

# -----------------------------------------------------------------------------
# Config — the two you always set: MODEL_RUNS + EVAL_DATASET_PATH.
# Space-separated list of training-run folders (each needs models/ + curves/;
# tiled checkpoints are detected automatically via tiling_params.pkl).
# NOTE: output folders are named after the run basename, in the order listed
# here — same-basename runs get a _1 suffix (so below: plain = real, _1 = synth):
MODEL_RUNS="${MODEL_RUNS:-/home/daniel/jetbrains-projects/dino-core-v1/results_tiled_bigd/bigd_macaroni2 /home/daniel/jetbrains-projects/dino-core-v1/results_tiled_bigd_synthetic/bigd_macaroni2}"
# Optional display names, parallel to MODEL_RUNS (same order) — used for the
# output folders and chart legends. MODEL_NAMES=auto -> name from run folders
# (basename; same-basename collisions get their parent folder prepended).
MODEL_NAMES="${MODEL_NAMES:-bigd_real bigd_synth}"
# The REAL dataset root (MVTec format) every run is evaluated on.
# Accepts EITHER a category folder (evaluates just that category) ...
EVAL_DATASET_PATH="${EVAL_DATASET_PATH:-/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/BigD/macaroni2}"
# ... OR a dataset root containing category folders (evaluates all of them).
# -----------------------------------------------------------------------------

OUTPUT_ROOT="${OUTPUT_ROOT:-}"                # empty -> <repo>/results_cross_eval
SUBDATASETS="${SUBDATASETS:-}"                # space-separated; empty -> all matching
BATCH_SIZE="${BATCH_SIZE:-1}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SEED="${SEED:-0}"
GPU_ID="${GPU_ID:-auto}"                      # auto (CUDA > Apple MPS > CPU) | cpu | <N> cuda id
FAISS_NUM_WORKERS="${FAISS_NUM_WORKERS:-8}"
SKIP_COMPARISON="${SKIP_COMPARISON:-0}"       # 1 -> no compare_cross_eval_runs.py charts
PAIRED_SCATTER="${PAIRED_SCATTER:-1}"         # 1 -> per-image score scatter for run pairs
DRY_RUN="${DRY_RUN:-0}"                       # 1 -> print the command, don't run
PYTHON="${PYTHON:-}"                          # e.g. /path/to/python; empty -> uv run python

# =============================================================================
# Script setup (normally nothing to edit below this line)
# =============================================================================
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "${SCRIPT_DIR}"

OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}/results_cross_eval}"

if [[ "$(uname)" == "Darwin" ]]; then
    # macOS ships faiss-cpu + torch with duplicate libomp copies; heavy FAISS
    # search threads can segfault.
    FAISS_NUM_WORKERS=4
    # Same known macOS packaging issue: torch+faiss both embed libomp and the
    # interpreter aborts at import (OMP Error #15) unless these are set.
    export KMP_DUPLICATE_LIB_OK="${KMP_DUPLICATE_LIB_OK:-TRUE}"
    export OMP_NUM_THREADS="${OMP_NUM_THREADS:-1}"
fi

if [[ -z "${MODEL_RUNS}" ]]; then
    echo "ERROR: MODEL_RUNS is empty — set it to one or more training-run folders" >&2
    echo "       (each with models/ + curves/, e.g. results_tiled/tiled_..._0)" >&2
    exit 1
fi

[[ "${MODEL_NAMES}" == "auto" ]] && MODEL_NAMES=""
read -ra run_array <<< "${MODEL_RUNS}"
read -ra name_array <<< "${MODEL_NAMES}"

for run_dir in "${run_array[@]}"; do
    if [[ ! -d "${run_dir}" ]]; then
        echo "ERROR: model run does not exist: ${run_dir}" >&2
        exit 1
    fi
done

if [[ -n "${MODEL_NAMES}" && ${#name_array[@]} -ne ${#run_array[@]} ]]; then
    echo "ERROR: MODEL_NAMES count (${#name_array[@]}) != MODEL_RUNS count (${#run_array[@]})" >&2
    exit 1
fi

if [[ ! -d "${EVAL_DATASET_PATH}" ]]; then
    echo "ERROR: EVAL_DATASET_PATH does not exist: ${EVAL_DATASET_PATH}" >&2
    exit 1
fi

# Category folder or dataset root? (category dirs contain train/ per MVTec format)
dataset_flags=()
if [[ -d "${EVAL_DATASET_PATH%/}/train" ]]; then
    DATA_ROOT="$(dirname "${EVAL_DATASET_PATH%/}")"
    CATEGORY="$(basename "${EVAL_DATASET_PATH%/}")"
    dataset_flags=(--subdatasets "${CATEGORY}")
else
    DATA_ROOT="${EVAL_DATASET_PATH}"          # root -> all matching categories
fi

cmd=()
if [[ -n "${PYTHON}" ]]; then
    cmd+=("${PYTHON}")
else
    cmd+=(uv run python)
fi
cmd+=("${SCRIPT_DIR}/scripts/tiled_patchcore/cross_evaluate.py"
      --data_path "${DATA_ROOT}"
      --output_root "${OUTPUT_ROOT}"
      --batch_size "${BATCH_SIZE}"
      --num_workers "${NUM_WORKERS}"
      --seed "${SEED}"
      --faiss_num_workers "${FAISS_NUM_WORKERS}")
for i in "${!run_array[@]}"; do
    cmd+=(--model_run "${run_array[$i]}")
    [[ -z "${MODEL_NAMES}" ]] || cmd+=(--model_name "${name_array[$i]}")
done
cmd+=("${dataset_flags[@]+"${dataset_flags[@]}"}")

if [[ "${PAIRED_SCATTER}" == "1" ]]; then
    cmd+=(--paired_scatter)
else
    cmd+=(--no_paired_scatter)
fi
[[ "${SKIP_COMPARISON}" == "1" ]] && cmd+=(--skip_comparison)
if [[ "${GPU_ID}" == "cpu" ]]; then
    cmd+=(--cpu)
elif [[ "${GPU_ID}" != "auto" ]]; then
    cmd+=(--gpu "${GPU_ID}")
fi

echo "============================================"
echo "PatchCore Cross-Evaluation (tiled + plain)"
echo "============================================"
echo "Model runs:"
for i in "${!run_array[@]}"; do
    label=""
    [[ -z "${MODEL_NAMES}" ]] || label="  [${name_array[$i]}]"
    echo "  - ${run_array[$i]}${label}"
done
echo "Dataset root:     ${DATA_ROOT}"
echo "Category:         ${CATEGORY:-<all, matching>}"
echo "GPU:              ${GPU_ID}"
echo "Seed:             ${SEED}"
echo "Output:           ${OUTPUT_ROOT}"
echo "============================================"

if [[ "${DRY_RUN}" == "1" ]]; then
    echo "Dry run — would execute:"
    printf '  %q' "${cmd[@]}"
    printf '\n'
    exit 0
fi

"${cmd[@]}"

echo ""
echo "Done. Artifacts:"
echo "  - per-run CrossEval folders + comparison charts: ${OUTPUT_ROOT}"
find "${OUTPUT_ROOT}" -maxdepth 1 -name "CrossEval_*" -type d 2>/dev/null | sort | tail -5
[[ ! -d "${OUTPUT_ROOT}/cross_eval_comparison" ]] || \
    echo "  - comparison charts: ${OUTPUT_ROOT}/cross_eval_comparison"
[[ ! -d "${OUTPUT_ROOT}/paired_scatter" ]] || \
    echo "  - paired scatters:   ${OUTPUT_ROOT}/paired_scatter"
exit 0
