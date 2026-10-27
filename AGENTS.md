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
| `scripts/tiled_patchcore/` | User-friendly tiled PatchCore CLIs over `src/patchcore/tiled_patchcore.py`: `run_tiled_patchcore.py` (train+eval+full analytics pack, good-only memory bank, DINOv3 dims auto-detected; per-category failures are contained — traceback logged, run continues, results.csv/timing.json/memory_bank.json flush after EVERY category, exit 1 at the end if any category failed; every run folder carries `run_config.json` with the exact CLI + resolved options), `load_and_evaluate.py` (reload checkpoints; dataset geometry auto-resolved from the checkpoint), `cross_evaluate.py` (cross-evaluate any number of training runs — tiled and plain mixed — on ONE real dataset root; per-run `CrossEval_<ts>_<model>/` in the legacy results.csv schema + saved-threshold confusion + paired per-image score scatters, then runs `compare_cross_eval_runs.py` for the identical chart pack; `cross_evaluate_tiled.sh` wrapper with config block on top), `visualize_features.py` (per-patch backbone features → UMAP/cuML-UMAP/PCA→RGB → stitched full-res maps + tile grid; 2-row panel: tiled row per selected DINO layer + PatchCore average above, non-tiled whole-image→384² counterparts below in the column's color space; checkpoint or fresh backbone; `run_visualize_features.sh` wrapper with config block on top), `shared.py` (helpers). Single-class test splits (BigD has 17/27 categories without test anomalies): training still saves the model but records NaN metrics and skips curves; eval scripts skip such categories instead of crashing on `roc_curve`. The root `run_tiled_patchcore.sh` wrapper prints its config summary BEFORE running, tees every invocation to `<results>/logs/<log_group>_<ts>.log`, translates exit codes (137=OOM-kill → check dmesg, 139/134=native crash, 1=Python traceback), and refuses empty category discovery (exit 1, with a warning for broken train/ symlinks). |

### Weights Available
`weights/` contains: dinov3_vits, dinov3_vitb, dinov3_vitl, dinov3_vithplus, dinov3_extractor, dinov3_raw, fast-sam, rf-detr, sam2, sam3, sam3.1, u2net.onnx

## What Needs Building

- [x] **Fix stale imports** — `clustering/`/`masking/` now use local imports; DINOv3 wrapper duplication resolved (canonical `src/backbones/dinov3.py` + shim)
- [x] **SAM masker implementation** (SAM2) with concept prompt support
- [x] **Perlin noise mask generator** + anomaly compositor + generation pipeline/CLI (`generate_dataset.py`)
- [x] **Tiled PatchCore** for very large images: TileMaker (overlapping tiles, mean-value padding, ~50% overlap, tile size = backbone input size e.g. 384) + TiledPatchCore wrapper. Coreset sampling stays **global** over pooled tile features, once at training time; inference = per-tile kNN + stitched, overlap-averaged anomaly maps. Power-user CLI: `patch_core --tile_size 384` + `dataset --native_resolution`; friendly one-command CLIs in `scripts/tiled_patchcore/` (run + load_and_evaluate, full analytics pack).
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

Docker (tiled-PatchCore training image, GPU): `Dockerfile.patchcore` — code baked
in, weights/dataset/results mounted at runtime (`/app/weights`, `/data`,
`/results`); the venv is pruned via `--no-install-package` (no sam3/RAPIDS/gradio,
see the header comment for build/run commands and the exclusion list).

## Key Design Notes (for the agent)

- Not every image needs a synthetic anomaly — only generate where clusters are diverse enough.
- The "image exemplar" prompt for SAM means: take a good image, compute its DINOv3/SAM embedding, and use that as the visual prompt to segment the same object class in another image.
- Clustering at 14x14 DINOv2 grid resolution means Perlin placement is coarse but semantically meaningful — that's fine, don't over-engineer it.
- Synthetic anomalies should look plausible but not require photo-realism. The anomaly detector needs to learn "something is different here," not "this is a photorealistic scratch."

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

When the user types `/graphify`, use the installed graphify skill or instructions before doing anything else.

Rules:
- When graphify-out/graph.json exists, run `graphify query "<question>"` before your first grep, search or source-file read for a codebase question, and put this rule in every subagent prompt that explores code. Read raw files only for lines the graph or an earlier read already located. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- Dirty graphify-out/ files are expected after hooks or incremental updates; dirty graph files are not a reason to skip graphify. Only skip graphify if the task is about stale or incorrect graph output, or the user explicitly says not to use it.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
- A committed graphify-out/graph.json is rewritten whole on every build, so a large diff to it in a pull request is expected. Do not flag it as churn or suggest skipping the graph rebuild or its hooks.
