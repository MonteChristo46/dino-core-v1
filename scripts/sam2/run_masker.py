"""SAM2 masking script — creates object masks from images.

Usage:
    # Single image
    uv run python scripts/sam2/run_masker.py path/to/image.jpg

    # Directory of images
    uv run python scripts/sam2/run_masker.py path/to/images/

    # Dataset category
    uv run python scripts/sam2/run_masker.py --category bottle --dataset-root ../dataset/glitch_hunt_mvtec

    # With custom params
    uv run python scripts/sam2/run_masker.py image.jpg --model-size tiny --min-area-ratio 0.005 --device cuda

Output:
    output/sam2/<image_name>_mask.png   — merged mask overlay on original
    output/sam2/<image_name>_stats.json — per-stage stats + coverage
"""

import argparse
import json
import logging
import os
import sys
import time
from pathlib import Path

import matplotlib
import numpy as np
from PIL import Image

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.masking.sam2_masker import SAM2Masker

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg", ".bmp", ".tiff", ".tif"}


def collect_images(args):
    """Collect image paths from args (single file, directory, or dataset category)."""
    if args.category:
        dataset_root = args.dataset_root or os.path.join(PROJECT_ROOT, "..", "..", "dataset", "glitch_hunt_mvtec")
        dataset_root = os.path.abspath(dataset_root)
        good_dir = os.path.join(dataset_root, args.category, "good")
        if not os.path.isdir(good_dir):
            LOGGER.error(f"Category directory not found: {good_dir}")
            sys.exit(1)
        return sorted(
            os.path.join(good_dir, f)
            for f in os.listdir(good_dir)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

    if args.input is None:
        LOGGER.error("Provide --input (file/dir) or --category.")
        sys.exit(1)

    input_path = os.path.abspath(args.input)
    if os.path.isfile(input_path):
        return [input_path]

    if os.path.isdir(input_path):
        return sorted(
            os.path.join(input_path, f)
            for f in os.listdir(input_path)
            if os.path.splitext(f)[1].lower() in IMAGE_EXTENSIONS
        )

    LOGGER.error(f"Input not found: {input_path}")
    sys.exit(1)


def build_masker(args) -> SAM2Masker:
    weights_dir = args.weights_dir or os.path.join(PROJECT_ROOT, "weights", "sam2")
    return SAM2Masker(
        model_size=args.model_size,
        weights_dir=weights_dir,
        device=args.device,
        points_per_batch=args.points_per_batch,
        pred_iou_thresh=args.pred_iou_thresh,
        stability_score_thresh=args.stability_score_thresh,
        crop_n_layers=args.crop_n_layers,
        min_area_ratio=args.min_area_ratio,
        max_area_ratio=args.max_area_ratio,
        dedup_iou=args.dedup_iou,
        nms_iou_thresh=args.nms_iou_thresh,
        min_mask_region_area=args.min_mask_region_area,
        min_mask_ratio=args.min_mask_ratio,
        morph_cleanup=not args.no_morph,
        morph_kernel_size=args.morph_kernel_size,
        remove_background_like=not args.no_bg_filter,
    )


def save_results(image_path, img_np, merged_mask, masks, stats, output_dir, min_mask_ratio=0.05):
    """Save mask overlay image and stats JSON."""
    base_name = Path(image_path).stem
    os.makedirs(output_dir, exist_ok=True)

    mask_path = os.path.join(output_dir, f"{base_name}_mask.png")
    stats_path = os.path.join(output_dir, f"{base_name}_stats.json")

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.imshow(img_np)

    if np.count_nonzero(merged_mask) > 0:
        overlay = np.zeros_like(img_np)
        overlay[merged_mask > 0] = [255, 0, 0]
        ax.imshow(overlay, alpha=0.4)
        coverage = stats.get("coverage", 0)
        status = "VALID" if coverage >= min_mask_ratio else "NO_VALID_MASK"
        ax.set_title(f"{status} — coverage: {coverage:.1%}", fontsize=10, fontweight="bold")
    else:
        ax.set_title("NO_VALID_MASK — coverage: 0.0%", fontsize=10, fontweight="bold", color="red")

    ax.axis("off")
    plt.tight_layout()
    plt.savefig(mask_path, dpi=150, bbox_inches="tight", pad_inches=0)
    plt.close(fig)

    serializable_stats = {
        k: (float(v) if isinstance(v, (np.floating, np.integer)) else v)
        for k, v in stats.items()
        if k not in ("raw_masks", "final_masks")
    }
    serializable_stats["num_raw_masks"] = len(stats.get("raw_masks", []))
    serializable_stats["num_final_masks"] = len(stats.get("final_masks", []))
    serializable_stats["image"] = os.path.basename(image_path)
    serializable_stats["image_shape"] = list(img_np.shape)

    with open(stats_path, "w") as f:
        json.dump(serializable_stats, f, indent=2)

    return mask_path, stats_path


def main():
    parser = argparse.ArgumentParser(description="SAM2 masking script")
    parser.add_argument("input", nargs="?", help="Image file or directory")
    parser.add_argument("--category", help="Dataset category (e.g. bottle)")
    parser.add_argument("--dataset-root", help="Path to dataset root (used with --category)")
    parser.add_argument("--output-dir", default=os.path.join(PROJECT_ROOT, "output", "sam2"))
    parser.add_argument("--weights-dir", help="Path to SAM2 weights directory")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--model-size", default="small", choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--points-per-batch", type=int, default=64)
    parser.add_argument("--pred-iou-thresh", type=float, default=0.90)
    parser.add_argument("--stability-score-thresh", type=float, default=0.97)
    parser.add_argument("--crop-n-layers", type=int, default=0)
    parser.add_argument("--min-area-ratio", type=float, default=0.01, help="Min mask area as ratio of image (default: 0.01 = 1%%)")
    parser.add_argument("--max-area-ratio", type=float, default=0.70)
    parser.add_argument("--dedup-iou", type=float, default=0.95)
    parser.add_argument("--nms-iou-thresh", type=float, default=0.5,
        help="NMS IoU threshold for merging overlapping masks (default: 0.5)")
    parser.add_argument("--min-mask-region-area", type=int, default=100,
        help="Minimum mask region area in pixels to keep from SAM output (default: 100)")
    parser.add_argument("--min-mask-ratio", type=float, default=0.05, help="Min coverage to be considered VALID (default: 0.05 = 5%%)")
    parser.add_argument("--no-morph", action="store_true", help="Disable morphological cleanup")
    parser.add_argument("--morph-kernel-size", type=int, default=3)
    parser.add_argument("--no-bg-filter", action="store_true", help="Disable background-like filter")
    args = parser.parse_args()

    images = collect_images(args)
    if not images:
        LOGGER.error("No images found.")
        sys.exit(1)

    LOGGER.info(f"Found {len(images)} image(s). Loading SAM2 model...")
    masker = build_masker(args)
    LOGGER.info(f"Model loaded. Processing {len(images)} image(s)...")

    valid_count = 0
    no_mask_count = 0
    start_time = time.time()

    for idx, img_path in enumerate(images, 1):
        img_name = os.path.basename(img_path)
        LOGGER.info(f"[{idx}/{len(images)}] Processing {img_name}...")

        img_np = np.array(Image.open(img_path).convert("RGB"))
        merged_mask, masks, stats = masker.get_mask_with_stats(img_np)

        coverage = stats.get("coverage", 0)
        is_valid = coverage >= args.min_mask_ratio

        mask_path, stats_path = save_results(img_path, img_np, merged_mask, masks, stats, args.output_dir, args.min_mask_ratio)

        if is_valid:
            valid_count += 1
            LOGGER.info(f"  VALID — coverage: {coverage:.1%} — {len(masks)} mask(s) — {mask_path}")
        else:
            no_mask_count += 1
            LOGGER.info(f"  NO_VALID_MASK — coverage: {coverage:.1%} — {mask_path}")

    elapsed = time.time() - start_time
    LOGGER.info(f"\n{'='*50}")
    LOGGER.info(f"Done: {len(images)} images in {elapsed:.1f}s")
    LOGGER.info(f"  VALID:      {valid_count}")
    LOGGER.info(f"  NO_VALID:   {no_mask_count}")
    LOGGER.info(f"  Output dir: {args.output_dir}")


if __name__ == "__main__":
    main()
