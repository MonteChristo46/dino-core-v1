# Tiled PatchCore — Goal, Vision & Implementation Plan

Status: **planned, not yet implemented**
Scope: `src/patchcore/` (vendored PatchCore fork with DINOv3 support)

---

## 1. Goal

Add **tiled processing** to PatchCore so that very large images can be
handled without downscaling them to the backbone's nominal input size
(224/384 px). An image is decomposed into a grid of overlapping,
backbone-sized tiles; each tile is treated exactly like an independent
image through the feature pipeline.

Hard requirements:

1. **Global coreset, never per tile.** Coreset subsampling happens
   **once**, at training time, over the pooled patch features of
   *all tiles from all training images*. Per-tile subsampling is
   forbidden — it would shrink the memory bank per tile and destroy
   the coverage guarantee.
2. **No coreset at inference.** Inference is: extract tile features →
   kNN-search the (fixed) memory bank → score.
3. **Overlapping tiles.** A rigid non-overlapping grid risks cutting an
   anomaly in half and creating edge artifacts. Tiles overlap (default
   50%), and scores in overlap regions are **averaged** during
   post-processing to yield a smooth, full-resolution anomaly map.
4. **Encapsulation.** Tiling lives in new classes (`TileMaker` and a
   `TiledPatchCore` wrapper); the existing `PatchCore` core algorithm
   (`patchcore.py`, `common.py`, `sampler.py`) is not modified.

## 2. Vision

- **Why:** GlitchHunt images (and future industrial data) are much
  larger than 384 px. Downsizing to 384 destroys exactly the small
  defects an anomaly detector must find. Tiling keeps native
  resolution while keeping the backbone input fixed and batchable.
- **The key insight:** a tile *is* an image. The existing batch
  dimension simply becomes the tile dimension:
  `B images → B·T tiles → [B·T·s², D]` features. Feature extraction,
  preprocessing, aggregation, and kNN are already batched — they need
  no changes. Only **grid construction**, **padding**, **map
  stitching**, and the **memory-bank pooling discipline** are new, and
  all of that lives in the wrapper.
- **Outcome:** run `train_and_evaluate.sh` / `run_patchcore.py` with
  `--tile_size 384` and get native-resolution training and inference
  with full-resolution anomaly maps; flip the flag off and behavior is
  byte-identical to today.
- **Non-goals (for now):** learned/optimal tile placement, multi-scale
  tile pyramids, tile-aware GT mask synthesis, disk-spilled feature
  accumulation (memmap) — revisit only if memory numbers demand it.

## 3. Current architecture (baseline being extended)

| Stage | Code | Shape flow |
|---|---|---|
| Feature extraction | `NetworkFeatureAggregator` (`common.py`) | `[B,3,S,S]` → dict `layer → [B,C,s,s]` (DINOv3: hidden states, register tokens stripped, square grid) |
| Patchify + align + project | `PatchCore._embed` (`patchcore.py`) | `[B·s², C·p·p]` → `Preprocessing` → `Aggregator` → `[B·s², D]` |
| Memory bank | `PatchCore._fill_memory_bank` | concat all training features → `featuresampler.run()` (coreset) → `NearestNeighbourScorer.fit()` (FAISS) |
| Scoring | `PatchCore._predict` | kNN distances → max-pool = image score; `RescaleSegmentor` upsamples `[B,s,s]` grid + Gaussian σ=4 |

Relevant constraints discovered while exploring:

- `DinoV3Backbone.format_to_spatial_grid` requires a **perfect-square
  patch token grid** → tiles must be square and a multiple of 16
  (patch size). 384 → 24×24 = 576 tokens. ✔
- `MVTecDataset` currently always applies `Resize + CenterCrop`, which
  would destroy a large image before tiling sees it.
- All scripts already default to `ApproximateGreedyCoresetSampler`
  (avoids the O(N²) distance matrix of the exact greedy variant) —
  mandatory for tiling, where pooled N grows ~×T.
- `input_shape` is baked into `PatchCore.load` (feature-dim probe and
  `RescaleSegmentor.target_size`) — the wrapper fixes this to the tile
  size, so it becomes a non-issue.

## 4. Design principles

1. **Composition, not modification.** `TiledPatchCore` *owns* a
   `PatchCore` configured with `input_shape=(3, tile, tile)`; the inner
   model never knows tiling exists.
2. **Coreset global-by-construction.** The wrapper reimplements the
   memory-bank fill: pool **all** tiles' features from **all** images,
   call `featuresampler.run()` exactly once, then `fit()` the scorer.
   Inference code paths must never touch the sampler. Enforced and
   covered by a test.
3. **Padding to a fixed grid.** If the image doesn't fit an exact tile
   grid, pad right/bottom so no content is ever cropped away and
   stitching is lossless.
4. **Overlapping tiles + averaged stitch.** Default overlap = 50%
   (tile 384, stride 192). Anomalies smaller than the stride are
   guaranteed to be fully contained in at least one tile.
5. **Smooth once.** Stitch tile score grids at pixel level, *then*
   apply a single Gaussian smoothing over the whole stitched map
   (σ=4, same as `RescaleSegmentor`) — never per-tile smoothing
   (seams).

## 5. New components

### 5.1 `src/patchcore/tiling.py`

```python
@dataclass
class TileLayout:
    image_hw: tuple[int, int]
    padded_hw: tuple[int, int]
    tile_size: int
    stride: int
    origins: list[tuple[int, int]]   # row-major (y, x) tile corners
    # num_tiles = len(origins)

class TileMaker:
    def __init__(self, tile_size=384, overlap=None, stride=None,
                 pad_value=0.0):
        """Exactly one of overlap/stride; default overlap = tile_size // 2.
        Validates tile_size % 16 == 0 (DINOv3 square-grid constraint)."""

    def compute_layout(self, height, width) -> TileLayout:
        """padded = tile + ceil((H - tile)/stride) * stride  (per axis);
        origins = product(range(0, padded-tile+1, stride), repeat=2)."""

    def extract(self, images: Tensor[B,3,H,W]) -> tuple[Tensor[B*T,3,t,t], TileLayout]:
        """F.pad (right/bottom, constant) -> unfold(2).unfold(3) -> reshape.
        Pure tensor ops, device-agnostic."""

    def stitch(self, tile_maps: Tensor[B*T, m, m], layout, upsample=True) -> Tensor[B,H,W]:
        """Optional per-tile bilinear upsample m×m -> t×t, then overlap-add
        onto padded canvas (F.fold sums overlaps natively), divide by
        fold-of-ones weight map (= simple average in overlaps), crop to H,W."""
```

**Padding semantics (honest note):** dataset images are ImageNet-
normalized, so `pad_value=0.0` pads with the *dataset mean* (neutral
grey), not pixel-black. True black would be `(0-mean)/std ≈ -2.1` per
channel. Either choice creates artificial border tokens in edge tiles;
what matters is that training and inference pad identically so those
features are in-distribution. Default `0.0`, configurable; if border
false positives show up, reflect-padding is a one-line switch.

**Weighting:** default is plain averaging in overlaps (per the brief).
A Hann/triangular window is a trivial extension of the weight map but
is explicitly out of scope for v1.

### 5.2 `src/patchcore/tiled_patchcore.py`

Same public surface as `PatchCore` (`fit`, `predict`, `save_to_path`,
`load_from_path`) so scripts can swap it in transparently.

```python
class TiledPatchCore:
    def __init__(self, patchcore: PatchCore, tile_maker: TileMaker,
                 tile_batch_size: int = 8): ...

    @classmethod
    def load(cls, backbone, layers_to_extract_from, device, *,
             tile_size=384, overlap=None, tile_batch_size=8,
             **patchcore_kwargs) -> "TiledPatchCore":
        """Builds inner PatchCore with input_shape=(3, tile_size, tile_size)."""

    def _embed_tiled(self, images):
        """extract tiles -> chunked inner._embed(provide_patch_shapes=True)
        in tile_batch_size groups -> flat [B*T*s*s, D], layout, (s, s)."""

    def fit(self, training_data):
        """for batch in loader: _embed_tiled -> accumulate.
        features = np.concatenate(all)
        features = featuresampler.run(features)   # ONE global coreset call
        anomaly_scorer.fit([features])"""

    def _predict(self, images):
        """features -> anomaly_scorer.predict -> [B*T*s*s] patch scores
        -> reshape [B*T, s, s]
        image score = max over all T*s^2 patches   (== PatchMaker.score semantics)
        map         = tile_maker.stitch(...) -> Gaussian σ=4 once -> crop."""

    def save_to_path(self, save_path, prepend=""):
        """inner.save_to_path + pickle {tile_size, stride, tile_batch_size}
        as <prepend>tiling_params.pkl (self-describing checkpoints)."""

    def load_from_path(self, load_path, device, nn_method, prepend=""): ...
```

GPU-memory control: `tile_batch_size` bounds the number of tiles per
backbone forward, independent of the dataloader batch of images.

## 6. Changes to existing files

| File | Change | Rationale |
|---|---|---|
| `src/patchcore/datasets/mvtec.py` | `resize=None, imagesize=None` → transform is `ToTensor + Normalize` only (native resolution); same passthrough for GT masks | otherwise the dataset downscales/crops the large images before tiling ever sees them |
| `scripts/patchcore/run_patchcore.py` | new options on the `patch_core` group: `--tile_size` (None = current behavior), `--tile_overlap` (default tile_size//2), `--tile_batch_size` (8); when set, build `TiledPatchCore`; `input_shape=(3, tile, tile)` | CLI + library scope; downstream metrics/segmentation-save/results.csv unchanged (per-image scores, full-res maps) |
| `config/patchcore_config.py` | add `TilingConfig` dataclass: `enabled=False, tile_size=384, overlap=None, tile_batch_size=8, pad_value=0.0` | repo convention; CLI is the live wiring |
| `scripts/patchcore/load_and_evaluate_patchcore.py`, `evaluate_cross_dataset.py` | detect `tiling_params.pkl` beside `patchcore_params.pkl` → load wrapped model | saved tiled models stay usable by the eval tooling |
| `AGENTS.md` | add `src/patchcore/tiling.py` and `tiled_patchcore.py` rows to the module table | keep the agent doc current |

**Not touched:** `patchcore.py`, `common.py`, `sampler.py`, `backbones.py`,
`dinov3.py`, `metrics.py`. The coreset and FAISS machinery is reused
as-is; correctness of "global once, never at inference" is a property
of the wrapper's control flow, not of the samplers.

## 7. Testing plan — `test/test_tiling.py` (CPU-friendly)

1. **Grid math:** exact-fit image (e.g. 768² @384/192 → no padding,
   expected origins); partial fit (500² @384/192 → padded 576², 2×2
   tiles); non-square images; batch > 1.
2. **extract/stitch roundtrip:** constant tile maps survive identically;
   hand-constructed tile values verify overlap pixels equal the **mean
   of contributing tiles**; weight map ≥ 1 everywhere (no div-by-zero);
   stitched map cropped back to original H×W.
3. **Validation:** non-multiple-of-16 tile size raises; both overlap
   and stride given raises.
4. **Coreset invariant (the important one):** inject a recording mock
   sampler; call `fit()` on N images with T tiles each; assert
   `run()` called **exactly once** and received `(N·T·s², D)` rows;
   assert inference path never calls it.
5. **Predict contract:** one image score per input image; mask shape ==
   input H×W.
6. **Save/load roundtrip:** tiling params survive pickle roundtrip.
7. **Mini end-to-end:** WR50 backbone, CPU, small synthetic loader
   (style of existing `test/test_patchcore.py`).

## 8. Known risks / honest caveats

- **Tiled ≠ full-image features.** DINOv3 self-attention is global
  *within* an input; each tile sees only itself (plus CLS/registers).
  A large anomaly spanning multiple tiles, or one that needs global
  context, may score differently than at full resolution. This is an
  acceptable and expected trade-off — but validate AUROC on one
  category with known large anomalies before scaling to the suite.
- **Memory growth.** Pooled training features grow ~×T per image (a
  1536² image at 384/192 → 49 tiles ≈ 28k patches/image). Keep
  `ApproximateGreedyCoresetSampler`; the exact greedy variant's N²
  distance matrix will explode. `tile_batch_size` bounds GPU peak.
- **`patchsize > 1` border effect.** With WR50-style `patchsize=3`,
  tile-border patches lose cross-tile context (`Unfold` zero-pads).
  Irrelevant for the DINOv3 default (`patchsize=1`); mitigated
  otherwise by 50%-overlap averaging.
- **Padding artifacts.** Border tiles contain padded regions → slightly
  out-of-distribution features in a thin band. Same band is padded at
  train time, so it's consistent; watch for edge false positives.

## 9. Execution order

1. `src/patchcore/tiling.py` + grid/stitch unit tests
2. `src/patchcore/tiled_patchcore.py` + coreset-invariant & shape tests
3. `mvtec.py` native-resolution passthrough
4. `run_patchcore.py` CLI wiring (`--tile_size`, `--tile_overlap`, `--tile_batch_size`)
5. `config/patchcore_config.py` `TilingConfig` + eval-script loading support
6. `AGENTS.md` table update
7. `pytest test/` green + smoke run: one MVTec category
   `... patch_core --tile_size 384 ...` vs. untiled baseline

## 10. Definition of done

- `--tile_size 384` trains a memory bank where the coreset is taken
  once over pooled tile features and behaves identically (interface-wise)
  to untiled `PatchCore`.
- Inference on an image larger than the tile size returns an image score
  and a full-resolution, seam-smoothed anomaly map with averaged
  overlaps.
- Toggling tiling off reproduces current behavior; all existing tests
  pass unmodified.
