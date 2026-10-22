# dino-core-v1

Training-data pipeline and anomaly-detection experiments for **Glitch Hunt**.
The repo contains two cooperating subsystems:

1. **Synthetic anomaly generation** — turns *good* images into plausible *bad*
   ones (SAM2 masking → DINOv3 feature clustering → Perlin-noise islands →
   blending) and exports the result as an **MVTec-format dataset**.
2. **PatchCore anomaly detection** — a vendored, DINOv3-enabled PatchCore
   implementation that trains a one-class memory bank on good images and scores
   test images via k-NN search.

```
 raw dataset (good images)
        │
        ▼
 scripts/semantic_patching/generate_dataset.py     ── SAM2 + DINOv3 + HDBSCAN + Perlin
        │
        ▼
 MVTec-format dataset on disk
   <category>/train/good, test/good, test/bad, ground_truth/bad
        │
        ▼
 scripts/patchcore/run_patchcore.py                ── PatchCore fit (coreset → memory bank)
        │                                               predict (k-NN → scores + anomaly maps)
        ▼
 results/ (AUROC, curves, thresholds, saved models, segmentation images)
```

PatchCore itself is **one-class**: the memory bank is built exclusively from
`train/good` features. Synthetic *bad* images are used for threshold
calibration and evaluation (see `test_anomaly_scaling.py`), never for the
memory bank.

---

## Setup

Requirements: **Python 3.12**, **UV**, Linux + CUDA (see Gotchas).

```bash
uv sync
source .venv/bin/activate          # or prefix commands with `uv run`
```

Weights (gitignored) live in `weights/`:

```
weights/
  dinov3_vits/  dinov3_vitb/  dinov3_vitl/  dinov3_vithplus/   # HF AutoModel dirs
  sam2/  sam3/  sam3.1/  fast-sam/  rf-detr/  u2net.onnx
```

There are **two sys.path conventions** in this repo — scripts import either
`patchcore.*` or `src.*`/`config`:

| Subsystem | Import style | How paths get set |
|---|---|---|
| PatchCore (training/eval) | `import patchcore.…` | shell wrappers `export PYTHONPATH=src`; some scripts `sys.path.insert(0, …/src)` |
| Synthesis / generation | `from src.… import`, `from config import` | scripts insert the repo root themselves; run from repo root |

Both resolve the *same* canonical modules. The single canonical `DinoV3Backbone`
lives in `src/backbones/dinov3.py`; `src/patchcore/dinov3.py` is only a
compatibility shim re-exporting it.

---

## Repo map

| Path | Purpose |
|---|---|
| `src/patchcore/` | Vendored PatchCore (modified: DINOv3 support). Model, feature aggregator, coreset samplers, FAISS scorer, MVTec dataset. |
| `src/backbones/` | Canonical `DinoV3Backbone` + HF backbone factory. |
| `src/masking/` | `SAM2Masker` (+ `BaseMasker`) — object masks for synthesis. |
| `src/clustering/` | UMAP→HDBSCAN clustering of DINOv3 patch features. |
| `src/synthesis/` | Perlin island generator, patch composers, texture fillers. |
| `src/augmentation/` | Augmentation strategies (`SemanticPatchStrategy`, `RandomPatchStrategy`) + blenders + `AugmentationChain`. |
| `src/patchcore_optuna/` | Optuna evaluator/objective + search spaces (backbone/layer presets). |
| `config/` | Stdlib dataclass configs (`PatchCoreConfig`, `PipelineConfig`, `SAM2Config`, …). Documentation of defaults more than live wiring. |
| `scripts/patchcore/` | Train+eval CLI, load-and-evaluate, cross-dataset eval. |
| `tiled_patchcore/` | User-friendly **tiled** PatchCore for very large images: `run_tiled_patchcore.py`, `load_and_evaluate.py`, `shared.py`. |
| `scripts/semantic_patching/` | Synthetic dataset generation + visualization QA. |
| `scripts/datasets/` | Experiments: learning curves, synthetic-anomaly threshold scaling. |
| `scripts/optuna/` | HPO driver + dashboard. |
| `test/` | pytest suite (GPU-gated cases skipped automatically on CPU). |
| `*.sh` (repo root) | Convenience wrappers: `run_tiled_patchcore.sh`, `train_and_evaluate.sh`, `evaluate_cross_dataset.sh`, `sample_evaluation.sh`, `local_run_test.sh`. |

---

## Train PatchCore

Single-category or multi-category training goes through
`scripts/patchcore/run_patchcore.py`, a *chained* click CLI. Three subcommands
must all appear, in any order after the results path:

```bash
PYTHONPATH=src uv run python scripts/patchcore/run_patchcore.py \
    --gpu 0 --seed 0 --save_patchcore_model --save_segmentation_images \
    --log_project PatchCore_Results --log_group my_run \
    results/ \
    patch_core -b dinov3_vitl -le 8 -le 12 -le 16 \
        --pretrain_embed_dimension 1024 --target_embed_dimension 1024 \
        --anomaly_scorer_num_nn 9 --patchsize 1 \
    sampler approx_greedy_coreset -p 0.1 \
    dataset -d bottle --resize 384 --imagesize 384 --batch_size 2 \
        mvtec /path/to/mvtec_ad
```

- `patch_core` — backbone, layers, embedding dims, scorer/patch options.
  Multiple `-b` flags create an ensemble.
- `sampler` — memory-bank subsampling: `identity`, `greedy_coreset`,
  `approx_greedy_coreset` (recommended), `-p` fraction of patches kept.
- `dataset` — dataset name (`mvtec`), data path, and `-d` one or more
  categories (each is trained/evaluated independently). `--resize` /
  `--imagesize` control the preprocessing; the dataloader's `imagesize` becomes
  PatchCore's `input_shape`.

Outputs land in `results/<log_group>*/`: `results.csv` (image/pixel AUROC),
`curves/*.npz` (PR curves + F1-optimal thresholds), saved models with
`--save_patchcore_model`, segmentation renders with `--save_segmentation_images`.

The easiest path is the wrapper script (all knobs are env vars):

```bash
DATASET_PATH=/path/to/BigD BACKBONE=dinov3_vitl LAYERS="8 12 16" \
IMAGE_SIZE=384 RESIZE=384 NUM_NN=9 ./train_and_evaluate.sh
```

(Default wrapper run is the WR50 baseline: `wideresnet50`, `layer2 layer3`, 224px.)

---

## Tiled PatchCore (large images)

Images much larger than the backbone input (GlitchHunt crops, high-res
industrial cameras) would lose small defects when downscaled to 224/384 px.
**Tiled PatchCore** cuts each image into a grid of overlapping backbone-sized
tiles, treats every tile as an independent image through the *unmodified*
PatchCore pipeline, takes the coreset **once globally** over all pooled tile
features, and reassembles per-tile kNN scores into a **full-resolution**
anomaly map (overlaps averaged, one smoothing pass — no seams).

The friendly CLI lives in `tiled_patchcore/` — one command, sane defaults,
no chained subcommands, no `PYTHONPATH`:

```bash
# One-liner wrapper (edit DATASET_PATH at the top of the script):
./run_tiled_patchcore.sh

# Train + evaluate every category under the dataset root:
uv run python tiled_patchcore/run_tiled_patchcore.py \
    --data_path /path/to/synthetic_v1 \
    --backbone dinov3_vitb --tile_size 384 \
    --coreset_percentage 0.1 --results_path results_tiled

# Re-evaluate saved checkpoints later (geometry auto-read from checkpoint):
uv run python tiled_patchcore/load_and_evaluate.py \
    --model_root results_tiled/tiled/models --data_path /path/to/synthetic_v1
```

What it wires up for you: native-resolution dataloading, `TiledPatchCore`
(inner `PatchCore` pinned to the tile size), approx-greedy coreset (global,
once), DINOv3 layer/embed defaults per backbone family (`--pretrain/
target_embed_dimension` overrides exist for non-DINOv3 nets), and it trains
on **good images only** — `train/bad` samples of the synthetic-pipeline
format are dropped so they never leak into the memory bank.

Outputs are compatible with the existing results layout, plus an analytics
pack: `results.csv`, `confusion_matrix.csv` (always populated),
`per_image_scores.csv` (per-image score + TP/FP/TN/FN triage),
`memory_bank.json`, `timing.json`, `curves/*.npz`, and `plots/`:
tile-grid overlay on a real image, score distribution with threshold, image
ROC, and the top-N anomalous images as original | heatmap | heatmap+tile-grid
(quick seam/border-artifact check). Saved model dirs carry a
`tiling_params.pkl` sidecar, so the older
`scripts/patchcore/load_and_evaluate_patchcore.py` also reloads tiled models
transparently (use its `--native_resolution` flag).

Key knobs: `--tile_size` (multiple of 16; = backbone input), `--tile_overlap`
(default half the tile — anomalies smaller than the stride are guaranteed to
be fully contained in some tile), `--tile_batch_size` (bounds GPU peak during
backbone forwards). DINOv3 attention is *within* a tile: an anomaly spanning
many tiles or relying on global context can score differently than at full
resolution — validate AUROC per category before trusting the whole suite.

---

## Switching backbones

### DINOv3 (recommended)

| Name (`-b`) | Weights dir | Width | Transformer blocks | Suggested layers |
|---|---|---|---|---|
| `dinov3_vits` | `weights/dinov3_vits` | 384 | 12 | `"7" "8"` |
| `dinov3_vitb` | `weights/dinov3_vitb` | 768 | 12 | `"7" "8"` |
| `dinov3_vitl` | `weights/dinov3_vitl` | 1024 | 24 | `"8" "12" "16"` |
| `dinov3_vithplus` | `weights/dinov3_vithplus` | 1280 | 32 | `"8" "12" "16"` |

Notes:

- For DINOv3, `-le` values are **indices into HF `hidden_states`** (index 0 is
  the patch embedding, `"8"` is the output after block 8). If you omit `-le`
  with a DINOv3 backbone, the CLI defaults to layers `7` and `8`.
- DINOv3 is patch-16: `--imagesize` must be a multiple of 16 (224 → 14×14
  patch grid, 384 → 24×24). Register tokens (4 by default) are stripped before
  grid reshaping (`src/backbones/dinov3.py`).
- Set `--pretrain_embed_dimension`/`--target_embed_dimension` to the backbone
  width (or 1024 as a common default); mismatched values work but project
  needlessly.
- Vetted layer combinations per backbone live in
  `src/patchcore_optuna/search_spaces.py`.

### CNNs (torchvision / timm)

Any name from `_BACKBONES` in `src/patchcore/backbones.py`, e.g.
`wideresnet50`, `resnet50`, `densenet161`, `efficientnet_b7` — with
module-named layers (`-le layer2 -le layer3`).
`wideresnet50` is auto-loaded from `weights/wideresnet50/wide_resnet50_2.pth`
if present.

---

## Create synthetic anomaly data

`scripts/semantic_patching/generate_dataset.py` reads an MVTec-format source
dataset and writes an MVTec-format output dataset with **synthetic anomalies in
`test/bad`** (+ masks in `ground_truth/bad`, real good images copied to
`train/good` and `test/good`):

```bash
uv run python scripts/semantic_patching/generate_dataset.py \
    --categories bottle cable \
    --dataset-root /path/to/BigD \
    --output-dir /path/to/synthetic_v1 \
    --num-test 40 \
    --device cuda \
    --dinov3-size vitl \
    --blender feather --fill-strategy source
```

Pipeline per image: SAM2 masks the object → DINOv3 patch features are
UMAP/HDBSCAN-clustered → a Perlin-noise island is placed in a semantically
*different* region (falling back to random placement if clustering is too
poor) → feather/seamless-clone blending. Images where the object coverage or
cluster diversity is insufficient are skipped (see `PipelineConfig`).

Key flags: `--num-test` (images per category; split 50/50 good/bad),
`--fill-strategy source|dtd` (texture that fills the anomaly) with
`--dtd-mix-prob`, `--blender feather|seamless_clone`, `--sam2-*` masking
thresholds, `--reader-workers/--writer-workers` for throughput.
Defaults come from dataclasses in `config/` — change them there, not in the
script.

Visual QA before training:

```bash
uv run python scripts/semantic_patching/visualize.py   # see --help for args
```

---

## Evaluate, calibrate, experiment

| Script | Use |
|---|---|
| `scripts/patchcore/load_and_evaluate_patchcore.py` | Load saved model dirs (`patch_core_loader -p <dir>`), evaluate on a `dataset` spec. |
| `scripts/patchcore/evaluate_cross_dataset.sh` / `.py` | Evaluate a model trained on dataset A against dataset B using stored thresholds; chunked prediction for large sets. |
| `plot_results.py`, `plot_cross_eval_results.py`, `compare_cross_eval_runs.py` | Plot curves / compare runs. |
| `scripts/datasets/learning_curve.py` | AUROC vs. number of training images. |
| `scripts/datasets/test_anomaly_scaling.py` | How many synthetic bad images are needed to calibrate a stable detection threshold. Trains a memory bank on `train/good`, sweeps N synthetic test/bad + all good test images, fits the F1-optimal threshold per N. |

```bash
uv run python scripts/datasets/test_anomaly_scaling.py \
    --mvtec-dataset /path/to/BigD --synthetic-dataset /path/to/synthetic_v1 \
    --categories cable capsule
```

## Hyperparameter search (Optuna)

```bash
uv run python scripts/optuna/optimize_patchcore.py --storage sqlite/optuna.db …
bash scripts/optuna/dashboard.sh          # web UI on the study
```

Search spaces (backbones, layer combos, dims) are predefined in
`src/patchcore_optuna/search_spaces.py`.

---

## Datasets

Everything trains/evaluates through **MVTec format**
(`<root>/<category>/{train/good, test/<anomaly-type>/, ground_truth/}`),
consumed by `src/patchcore/datasets/mvtec.py`:

- real MVTec AD,
- the **BigD** glitch-hunt dataset (12 categories: candle, capsules, cashew,
  chewinggum, fryum, macaroni1/2, pcb1–4, pipe_fryum — plus standard MVTec
  categories in the same root),
- synthetic datasets produced by `generate_dataset.py`.

`resize`/`imagesize` do a square resize + center crop, so **keep images roughly
square or accept the crop**. GT masks are only read for `test` splits of
non-good anomaly types.

> `src/datasets/glitch_hunt_dataset.py` (binary good/bad folder walk) is
> **legacy and unused** — it also ignores its `split` argument when collecting
> files. Consume data via MVTec format instead.

---

## Tests

```bash
uv run pytest test/            # or: ./local_run_test.sh (CPU, PYTHONPATH=src)
```

GPU-dependent tests self-skip when CUDA is unavailable.

---

## Gotchas & known issues

- **Linux + CUDA target.** `pyproject.toml` pins the uv environment to
  `sys_platform == 'linux'` (`cuml-cu12` etc. don't exist elsewhere); on
  macOS you cannot create the venv — run experiments on the GPU box.
- **Memory bank is host-RAM-bound.** All patch features of the training set
  are concatenated before coreset sampling. Use `approx_greedy_coreset`; the
  exact `greedy_coreset` computes an N×N distance matrix and will explode on
  large sets.
- **FAISS is CPU by default** (`faiss-cpu`). `--faiss_on_gpu` requires
  faiss-gpu, which is *not* declared as a dependency.
- `train_and_evaluate.sh`'s venv hint references a nonexistent
  `requirements.txt`; the real setup is `uv sync`.
- Several scripts hardcode dataset paths from one machine
  (`/home/daniel/jetbrains-projects/…`); always pass explicit paths.
- A `TRAIN` split of `MVTecDataset` includes **every** subfolder under
  `train/` — keep synthetic `bad` images out of `train/` or PatchCore will bank
  anomalous features (the generator only writes `test/bad`, so this is
  currently consistent).
- `greedy/approx` coreset JL-projection uses an unseeded `nn.Linear` — seed
  reproducibility of memory banks depends on the global torch seed.

## Planned work

- **Tiled PatchCore** for very large images: `TileMaker` grid extraction with
  overlapping tiles (e.g. 384px, 50% overlap) + a `TiledPatchCore` wrapper —
  tile features pooled globally, coreset sampled *once* per training run,
  anomaly maps stitched back to full resolution.
