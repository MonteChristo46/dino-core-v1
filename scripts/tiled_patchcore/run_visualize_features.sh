#!/usr/bin/env bash
# Visualize per-patch DINOv3 features (UMAP/PCA -> RGB) of tiled PatchCore,
# stitched into one full-resolution map with the tile grid painted in.
# Adjust the CONFIG block, then: ./scripts/tiled_patchcore/run_visualize_features.sh
# Extra CLI flags are passed through, e.g.:
#   ./scripts/tiled_patchcore/run_visualize_features.sh --save_npz --umap_neighbors 30
set -euo pipefail

# ======================= CONFIG =======================
# Input (dataset mode)
#DATA_PATH="dataset/40mp"
#DATA_PATH="/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/BigD"
DATA_PATH="/home/daniel/jetbrains-projects/dino-core-v1/dataset/_ad2_view"
CATEGORY="vial"
SPLIT="test"                  # train | test | val
N_IMAGES=2                    # first N images of the split
# ...or explicit images instead (space-separated; empty = dataset mode):
IMAGE_PATHS=()                # e.g. ("dataset/40mp/mosque/test/good/test_40mp.jpg")

# Model — checkpoint or fresh backbone
MODEL_DIR=""                  # e.g. results_tiled/tiled/models/mvtec_mosque
                              # (empty = fresh backbone; tiling flags below apply)
BACKBONE="dinov3_vitb"        # dinov3_vits|vitb|vitl|vithplus, wideresnet50, ...
LAYERS="3,7,12"                     # empty = family default ('7,8' for DINOv3);
                              # each layer gets a panel + the patchcore average
TILE_SIZE=768
TILE_OVERLAP=""               # empty = TILE_SIZE/2 (50%)
TILE_BATCH_SIZE=8
PAD_VALUE=0.0
PATCHSIZE=1                   # 1 = per-patch features (DINOv3)

# Reduction to RGB
REDUCER="pca"                 # umap | pca | cuml_umap (RAPIDS, needs cuML)
FIT_SCOPE="image"             # image | category (comparable colors across images)
MAX_FIT_ROWS=50000            # subsample cap for reducer fitting
UMAP_NEIGHBORS=15
UMAP_MIN_DIST=0.1
FULL_IMAGE=1                  # 1 = second panel row: whole image in one global
                              #     pass (non-tiled reference, no tile grid)
FULL_IMAGE_SIZE=768           # squash resolution of that pass — independent of
                              #     TILE_SIZE; row 2 keeps its own-fit colors

# Compute / output
GPU=""                        # empty = auto (CUDA > MPS > CPU), else CUDA id
CPU_ONLY=0                    # 1 = force CPU
SEED=0
OUT_DIR="feature_maps"
SAVE_NPZ=0                    # 1 = also dump 3D coords + layout per image
# ======================================================

cd "$( cd "$( dirname "$0" )" && pwd )/../.."   # repo root (uv run from there)

args=(--split "$SPLIT" --n_images "$N_IMAGES" --backbone "$BACKBONE"
      --tile_size "$TILE_SIZE" --tile_batch_size "$TILE_BATCH_SIZE"
      --pad_value "$PAD_VALUE" --patchsize "$PATCHSIZE"
      --reducer "$REDUCER" --fit_scope "$FIT_SCOPE"
      --max_fit_rows "$MAX_FIT_ROWS" --umap_neighbors "$UMAP_NEIGHBORS"
      --umap_min_dist "$UMAP_MIN_DIST" --seed "$SEED" --out_dir "$OUT_DIR")

if [[ ${#IMAGE_PATHS[@]} -gt 0 ]]; then
  for path in "${IMAGE_PATHS[@]}"; do args+=(--image_path "$path"); done
else
  args+=(--data_path "$DATA_PATH" --category "$CATEGORY")
fi
[[ -n "$MODEL_DIR" ]]    && args+=(--model_dir "$MODEL_DIR")
[[ -n "$LAYERS" ]]       && args+=(--layers "$LAYERS")
[[ -n "$TILE_OVERLAP" ]] && args+=(--tile_overlap "$TILE_OVERLAP")
[[ -n "$GPU" ]]          && args+=(--gpu "$GPU")
[[ "$CPU_ONLY" == "1" ]] && args+=(--cpu)
[[ "$SAVE_NPZ" == "1" ]] && args+=(--save_npz)
[[ "$FULL_IMAGE" == "0" ]] && args+=(--no_full_image)
[[ "$FULL_IMAGE" != "0" ]] && args+=(--full_image_size "$FULL_IMAGE_SIZE")

echo "uv run python scripts/tiled_patchcore/visualize_features.py ${args[*]} $*"
exec uv run python scripts/tiled_patchcore/visualize_features.py "${args[@]}" "$@"
