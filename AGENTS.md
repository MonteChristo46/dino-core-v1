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

## Current State (Early)

### Existing Modules

| Path | Purpose |
|------|---------|
| `src/backbones/dinov3.py` | DinoV3Backbone — wraps HuggingFace DINOv3, strips register tokens |
| `src/clustering/base.py` | BaseClusterer + ClusteringResult — abstract clustering with UMAP preprocessing |
| `src/clustering/hdbscan_clusterer.py` | HDBSCANClusterer — UMAP→HDBSCAN on DINOv3 patch features |
| `src/masking/base.py` | BaseMasker — abstract interface for object segmentation maskers |
| `src/datasets/glitch_hunt_dataset.py` | GlitchHuntDataset — reads good/bad folder structure |
| `src/augmentation/base.py` | BaseAugmentor — abstract interface for synthetic anomaly augmentors |
| `src/augmentation/augmentation_chain.py` | AugmentationChain — chain-of-responsibility that tries strategies in priority order |
| `src/augmentation/blenders/` | Blender implementations: FeatherBlender, PyramidBlender, SeamlessCloneBlender |

**Note**: `__init__.py` files in `clustering/` and `masking/` still import from `patchcore.*` — these are stale and need updating to local imports.

### Weights Available
`weights/` contains: dinov3_vitb, dinov3_vithplus, dinov3_vitl, dinov3, fast-sam, rf-detr, sam2, sam3.1, u2net.onnx

## What Needs Building

- [ ] **Fix stale imports**: `clustering/__init__.py` and `masking/__init__.py` reference `patchcore.*` — update to local relative imports
- [ ] **SAM masker implementation** (SAM2/SAM3) with concept prompt support (image exemplar, text, points)
- [ ] **Perlin noise mask generator** (constrained to object mask)
- [ ] **Anomaly compositor** — uses DINOv3 clusters + Perlin mask to blend synthetic defects
- [ ] **Pipeline orchestrator** — ties masking → clustering → synthesis → dataset export
- [ ] **CLI / script** to run the full pipeline per category
- [ ] **Config** via `.env` + pydantic-settings (matching workspace convention)

## Conventions

- **Python 3.12**. No linter/formatter/typechecker configured in this repo.
- **UV** as package manager.
- **Imports are local** — no `patchcore.*` package exists here. Use relative imports within `src/` (e.g., `from ..clustering.base import BaseClusterer`).
- **Weights are gitignored** (in `weights/`). Dataset output in `dataset/` also gitignored.
- **New capabilities go in `src/`** as new subpackages (e.g., `src/synthesis/`, `src/pipeline/`).
- **Scripts** go in `scripts/` and are run via `uv run python scripts/<name>.py`.

## Commands

```bash
uv sync && source .venv/bin/activate     # Setup
uv run python scripts/<name>.py          # Run a script
uv run pytest test/                      # Run tests (when they exist)
```

## Key Design Notes (for the agent)

- Not every image needs a synthetic anomaly — only generate where clusters are diverse enough.
- The "image exemplar" prompt for SAM means: take a good image, compute its DINOv3/SAM embedding, and use that as the visual prompt to segment the same object class in another image.
- Clustering at 14x14 DINOv2 grid resolution means Perlin placement is coarse but semantically meaningful — that's fine, don't over-engineer it.
- Synthetic anomalies should look plausible but not require photo-realism. The anomaly detector needs to learn "something is different here," not "this is a photorealistic scratch."
