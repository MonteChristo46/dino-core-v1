"""Export trained PatchCore models to Triton Inference Server format.

Creates a model_repository with:
  - Shared DINOv3 ONNX extractor
  - Per-category Python backend scorer (with FAISS index)
  - Per-category ensemble config

Usage:
    # Single category
    uv run python scripts/export_for_triton.py \
        --model-dir results/models/bottle \
        --output model_repository \
        --image-size 384 \
        --category bottle

    # Multiple categories (shared extractor)
    uv run python scripts/export_for_triton.py \
        --model-dir results/models/bottle results/models/cable \
        --output model_repository \
        --image-size 384 \
        --categories bottle cable
"""

import argparse
import logging
import os
import pickle
import shutil
from pathlib import Path

import torch

# Ensure src is on the path
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from export.onnx_extractor import export_to_onnx
from export.triton_configs import (
    generate_ensemble_config,
    generate_extractor_config,
    generate_scorer_config,
)

TRITON_SCORER_SRC = Path(__file__).parent.parent / "src" / "export" / "triton_scorer.py"

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def load_patchcore_model(model_dir: str, device: torch.device):
    """Load a trained PatchCore model from its saved directory."""
    import patchcore.backbones
    import patchcore.common
    import patchcore.patchcore

    params_files = [
        f for f in os.listdir(model_dir) if f.endswith("patchcore_params.pkl")
    ]
    if not params_files:
        raise FileNotFoundError(
            f"No patchcore_params.pkl found in {model_dir}"
        )

    params_file = params_files[0]
    prepend = params_file.replace("patchcore_params.pkl", "")

    with open(os.path.join(model_dir, params_file), "rb") as f:
        params = pickle.load(f)

    logger.info("Loading PatchCore model from %s", model_dir)
    logger.info("Parameters: %s", {k: v for k, v in params.items()})

    backbone = patchcore.backbones.load(params["backbone.name"])
    backbone.name = params["backbone.name"]

    nn_method = patchcore.common.FaissNN(
        params.get("faiss_on_gpu", False),
        params.get("faiss_num_workers", 4),
    )

    model = patchcore.patchcore.PatchCore(device)
    model.load(
        backbone=backbone,
        layers_to_extract_from=params["layers_to_extract_from"],
        device=device,
        input_shape=params["input_shape"],
        pretrain_embed_dimension=params["pretrain_embed_dimension"],
        target_embed_dimension=params["target_embed_dimension"],
        patchsize=params.get("patchsize", 1),
        patchstride=params.get("patchstride", 1),
        anomaly_score_num_nn=params.get("anomaly_scorer_num_nn", 9),
        nn_method=nn_method,
    )
    model.anomaly_scorer.load(model_dir, prepend)

    return model, params


def export_extractor(
    patchcore_model,
    output_dir: Path,
    image_size: int,
    device: torch.device,
) -> Path:
    """Export the shared DINOv3 ONNX extractor."""
    extractor_dir = output_dir / "dinov3_extractor" / "1"
    extractor_dir.mkdir(parents=True, exist_ok=True)

    onnx_path = extractor_dir / "model.onnx"
    export_to_onnx(
        patchcore_model,
        onnx_path,
        image_size=image_size,
        device=device,
    )
    return onnx_path


def export_scorer(
    output_dir: Path,
    category: str,
    model_dir: str,
    params: dict,
):
    """Export per-category scorer with FAISS index and model.py."""
    scorer_dir = output_dir / f"patchcore_scorer_{category}" / "1"
    scorer_dir.mkdir(parents=True, exist_ok=True)

    # Copy FAISS index
    index_files = [
        f for f in os.listdir(model_dir) if f.endswith(".faiss")
    ]
    if not index_files:
        raise FileNotFoundError(
            f"No .faiss index found in {model_dir}"
        )

    src_index = os.path.join(model_dir, index_files[0])
    dst_index = scorer_dir / "nn_scorer_search_index.faiss"
    shutil.copy2(src_index, dst_index)
    logger.info("Copied FAISS index: %s → %s", src_index, dst_index)

    # Copy params pickle (for scorer initialization metadata)
    params_files = [
        f for f in os.listdir(model_dir) if f.endswith("patchcore_params.pkl")
    ]
    if params_files:
        src_params = os.path.join(model_dir, params_files[0])
        dst_params = scorer_dir / "patchcore_params.pkl"
        shutil.copy2(src_params, dst_params)
        logger.info("Copied params: %s → %s", src_params, dst_params)

    # Write self-contained model.py
    model_py_path = scorer_dir / "model.py"
    shutil.copy2(TRITON_SCORER_SRC, model_py_path)
    logger.info("Wrote scorer model.py to %s", model_py_path)


def write_configs(
    output_dir: Path,
    category: str,
):
    """Write config.pbtxt files for extractor, scorer, and ensemble."""
    extractor_name = "dinov3_extractor"
    scorer_name = f"patchcore_scorer_{category}"
    ensemble_name = f"patchcore_ensemble_{category}"

    # Extractor config
    extractor_config_path = output_dir / extractor_name / "config.pbtxt"
    extractor_config_path.write_text(generate_extractor_config(extractor_name))
    logger.info("Wrote extractor config: %s", extractor_config_path)

    # Scorer config
    scorer_config_path = output_dir / scorer_name / "config.pbtxt"
    scorer_config_path.write_text(generate_scorer_config(scorer_name))
    logger.info("Wrote scorer config: %s", scorer_config_path)

    # Ensemble config
    ensemble_dir = output_dir / ensemble_name
    ensemble_dir.mkdir(parents=True, exist_ok=True)
    ensemble_config_path = ensemble_dir / "config.pbtxt"
    ensemble_config_path.write_text(
        generate_ensemble_config(ensemble_name, extractor_name, scorer_name)
    )
    logger.info("Wrote ensemble config: %s", ensemble_config_path)


def validate_repository(output_dir: Path):
    """Basic validation of the model repository structure."""
    errors = []

    # Check extractor
    extractor_onnx = output_dir / "dinov3_extractor" / "1" / "model.onnx"
    if not extractor_onnx.exists():
        errors.append(f"Missing ONNX extractor: {extractor_onnx}")
    else:
        size_mb = extractor_onnx.stat().st_size / (1024 * 1024)
        logger.info("Extractor ONNX: %.1f MB", size_mb)

    # Check each scorer
    for scorer_dir in output_dir.glob("patchcore_scorer_*"):
        model_py = scorer_dir / "1" / "model.py"
        faiss_index = scorer_dir / "1" / "nn_scorer_search_index.faiss"
        if not model_py.exists():
            errors.append(f"Missing model.py: {model_py}")
        if not faiss_index.exists():
            errors.append(f"Missing FAISS index: {faiss_index}")
        else:
            size_mb = faiss_index.stat().st_size / (1024 * 1024)
            logger.info(
                "Scorer %s FAISS: %.1f MB", scorer_dir.name, size_mb
            )

    # Check each ensemble
    for ensemble_dir in output_dir.glob("patchcore_ensemble_*"):
        config = ensemble_dir / "config.pbtxt"
        if not config.exists():
            errors.append(f"Missing ensemble config: {config}")

    if errors:
        logger.error("Validation failed:")
        for err in errors:
            logger.error("  - %s", err)
        raise RuntimeError("Model repository validation failed")

    logger.info("Model repository validation passed")


def main():
    parser = argparse.ArgumentParser(
        description="Export PatchCore models to Triton format"
    )
    parser.add_argument(
        "--model-dir",
        nargs="+",
        required=True,
        help="Directory(ies) containing trained PatchCore model(s)",
    )
    parser.add_argument(
        "--output",
        required=True,
        help="Output model_repository directory",
    )
    parser.add_argument(
        "--categories",
        nargs="+",
        default=None,
        help="Category name(s). Defaults to model-dir basename(s).",
    )
    parser.add_argument(
        "--image-size",
        type=int,
        default=384,
        help="Input image size (default: 384)",
    )
    parser.add_argument(
        "--device",
        default=None,
        help="Torch device (default: auto)",
    )

    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    if args.device:
        device = torch.device(args.device)
    else:
        device = torch.device(
            "cuda" if torch.cuda.is_available() else "cpu"
        )
    logger.info("Using device: %s", device)

    categories = args.categories or [
        os.path.basename(d.rstrip("/")) for d in args.model_dir
    ]

    if len(args.model_dir) != len(categories):
        raise ValueError(
            f"Number of model dirs ({len(args.model_dir)}) must match "
            f"number of categories ({len(categories)})"
        )

    # Export shared ONNX extractor from first model
    logger.info("=" * 60)
    logger.info("Exporting shared DINOv3 ONNX extractor")
    logger.info("=" * 60)

    first_model, first_params = load_patchcore_model(
        args.model_dir[0], device
    )
    export_extractor(first_model, output_dir, args.image_size, device)

    # Export per-category scorers
    for model_dir, category in zip(args.model_dir, categories):
        logger.info("=" * 60)
        logger.info("Exporting scorer for category: %s", category)
        logger.info("=" * 60)

        if model_dir == args.model_dir[0]:
            model, params = first_model, first_params
        else:
            model, params = load_patchcore_model(model_dir, device)

        export_scorer(output_dir, category, model_dir, params)
        write_configs(output_dir, category)

    # Validate
    logger.info("=" * 60)
    logger.info("Validating model repository")
    logger.info("=" * 60)
    validate_repository(output_dir)

    logger.info("=" * 60)
    logger.info("Export complete: %s", output_dir.resolve())
    logger.info("=" * 60)
    logger.info(
        "Start Triton: docker compose up --build"
    )
    logger.info(
        "Test: uv run python scripts/test_triton.py"
    )


if __name__ == "__main__":
    main()
