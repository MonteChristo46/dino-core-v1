# Agent.md — dino-core-v1 (Synthetic Anomaly Pipeline)

## Personality & Behavior

Act as a **Data Scientist** — not a code monkey. Be **honest** and **critical**. If an approach is over-engineered, unlikely to work, or makes bad assumptions, say so. Push back on complexity that doesn't serve the problem. Prioritize simple, testable experiments over elaborate architectures. Always ask: "what would we learn from this?" before coding.

## Project Goal

Build an anomaly detection training dataset for Glitch Hunt using **synthetic anomaly generation**:

1. **Extract good images** from glitch-hunt-dataset
2. **Mask objects** using SAM3 (or SAM2) with concept prompts:
   - **Required**: image exemplar (sample another image from the dataset)
   - **Optional**: text prompts, points
3. **Synthesize anomalies** on good images:
   - Sample **Perlin noise masks** constrained to the object mask
   - Use **DINOv3 feature clustering** to decide *where* to place the Perlin island (only in different semantic regions — avoid placing a stain on top of semantically identical area)
   - Blend the Perlin patch into the input image → synthetic anomaly
4. **Output dataset** structure:
   ```
   dataset/<category>/
     train/good/     (original good images)
     train/bad/      (synthetic anomalies)
     test/good/      (real good images)
     test/bad/       (real anomalies, if they exist)
   ```

## Current State

> **See `README.md` first** for end-to-end usage (training, backbone switching,
> synthetic data generation, evaluation, experiments).

### Existing Modules

| Path | Purpose |
|------|---------|
| `src/backbones/dinov3.py` | **Canonical** DinoV3Backbone — wraps HuggingFace DINOv3, parameterized register-token stripping, `is_dinov3`/`name` flags. Shared by both subsystems. |
| `src/patchcore/` | Vendored PatchCore (modified for DINOv3): model, feature aggregator, coreset samplers, FAISS scorer, `datasets/mvtec.py`. `src/patchcore/dinov3.py` is only a compatibility shim re-exporting the canonical backbone. `src/patchcore/tiling.py` = TileMaker/TileLayout (overlapping tile grid + overlap-averaged map stitching). `src/patchcore/tiled_patchcore.py` = TiledPatchCore wrapper (global coreset once at training, tile kNN + stitched full-res maps at inference, `tiling_params.pkl` checkpoints). |
| `src/clustering/base.py` | BaseClusterer + ClusteringResult — abstract clustering with UMAP preprocessing |
| `src/clustering/hdbscan_clusterer.py` | HDBSCANClusterer — UMAP→HDBSCAN on DINOv3 patch features |
| `src/masking/base.py` | BaseMasker — abstract interface for object segmentation maskers |
| `src/masking/sam2_masker.py` | SAM2Masker — object masks for the synthesis pipeline |
| `src/synthesis/` | Perlin island generator, patch composers, texture fillers |
| `src/augmentation/` | SemanticPatch/RandomPatch strategies, blenders, AugmentationChain |
| `src/datasets/glitch_hunt_dataset.py` | GlitchHuntDataset — **legacy/unused**; ignores its `split` arg. Use MVTec format instead. |
| `scripts/semantic_patching/generate_dataset.py` | Synthetic dataset generation pipeline (writes MVTec format: train/good, test/good, test/bad, ground_truth/bad) |
| `tiled_patchcore/` | User-friendly tiled PatchCore CLIs over `src/patchcore/tiled_patchcore.py`: `run_tiled_patchcore.py` (train+eval+full analytics pack, good-only memory bank, DINOv3 dims auto-detected), `load_and_evaluate.py` (reload checkpoints; dataset geometry auto-resolved from the checkpoint), `shared.py` (helpers). |

### Weights Available
`weights/` contains: dinov3_vits, dinov3_vitb, dinov3_vitl, dinov3_vithplus, dinov3_extractor, dinov3_raw, fast-sam, rf-detr, sam2, sam3, sam3.1, u2net.onnx

## What Needs Building

- [x] **Fix stale imports** — `clustering/`/`masking/` now use local imports; DINOv3 wrapper duplication resolved (canonical `src/backbones/dinov3.py` + shim)
- [x] **SAM masker implementation** (SAM2) with concept prompt support
- [x] **Perlin noise mask generator** + anomaly compositor + generation pipeline/CLI (`generate_dataset.py`)
- [x] **Tiled PatchCore** for very large images: TileMaker (overlapping tiles, mean-value padding, ~50% overlap, tile size = backbone input size e.g. 384) + TiledPatchCore wrapper. Coreset sampling stays **global** over pooled tile features, once at training time; inference = per-tile kNN + stitched, overlap-averaged anomaly maps. Power-user CLI: `patch_core --tile_size 384` + `dataset --native_resolution`; friendly one-command CLIs in `tiled_patchcore/` (run + load_and_evaluate, full analytics pack).
- [ ] **Config** via `.env` + pydantic-settings (matching workspace convention)

## Conventions

- **Python 3.12**. No linter/formatter/typechecker configured in this repo.
- **UV** as package manager (`uv sync`; Linux-only dependency set).
- **Two import layouts**: synthesis-side code uses `src.*`/`config` with the repo
  root on sys.path; PatchCore scripts use `patchcore.*` with `src/` on sys.path
  (`export PYTHONPATH=src` or `sys.path.insert` inside the script). The
  `patchcore.dinov3` shim works under both.
- **Weights are gitignored** (in `weights/`). Dataset output in `dataset/` also gitignored.
- **New capabilities go in `src/`** as new subpackages (e.g., `src/synthesis/`, `src/pipeline/`).
- **Scripts** go in `scripts/` and are run via `uv run python scripts/<name>.py`.

## Commands

```bash
uv sync && source .venv/bin/activate     # Setup (Linux/GPU box)
uv run python scripts/<name>.py          # Run a script
uv run pytest test/                      # Run tests
./local_run_test.sh                      # pytest with PYTHONPATH=src (CPU-safe subset)
```

## Key Design Notes (for the agent)

- Not every image needs a synthetic anomaly — only generate where clusters are diverse enough.
- The "image exemplar" prompt for SAM means: take a good image, compute its DINOv3/SAM embedding, and use that as the visual prompt to segment the same object class in another image.
- Clustering at 14x14 DINOv2 grid resolution means Perlin placement is coarse but semantically meaningful — that's fine, don't over-engineer it.
- Synthetic anomalies should look plausible but not require photo-realism. The anomaly detector needs to learn "something is different here," not "this is a photorealistic scratch."
