"""Generate MVTec-format dataset with synthetic anomalies.

Produces a dataset in MVTec directory structure:
  dataset/semantic_synthetic_anomalies/<category>/
    ground_truth/bad/      # <idx>_mask.png (single-channel, 0/255)
    test/good/             # Real good images (from source test/good)
    test/bad/              # Synthetic anomaly images
    train/good/            # All original train/good images (copied)

Usage:
    uv run python scripts/semantic_patching/generate_dataset.py
    uv run python scripts/semantic_patching/generate_dataset.py --categories bottle cable
    uv run python scripts/semantic_patching/generate_dataset.py --device cuda --dinov3-size vithplus --batch-size 128
"""

import argparse
import logging
import os
import shutil
import time
import sys

import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.backbones.dinov3 import DinoV3Backbone
from src.clustering.hdbscan_clusterer import HDBSCANClusterer
from src.masking.sam2_masker import SAM2Masker
from src.synthesis.perlin_island import GammaSizeParams, PerlinIslandGenerator
from src.synthesis.fillers import DTDTextureFiller, SourceTextureFiller
from src.synthesis.random_composer import DTD_ROOT, RandomPatchComposer
from src.synthesis.patch_composer import PatchComposer
from src.augmentation import AugmentationChain
from src.augmentation.strategies.semantic_patch import SemanticPatchStrategy
from src.augmentation.strategies.random_patch import RandomPatchStrategy

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "mvtec_small")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "semantic_synthetic_anomalies_v3")

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

def collect_images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def main():
    overall_start = time.time()
    parser = argparse.ArgumentParser(description="Generate MVTec-format synthetic anomaly dataset")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Specific categories to process (default: all)")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--num-test", type=int, default=100,
                        help="Total test images per category (split 50/50 good/bad)")
    parser.add_argument("--device", default="cuda", choices=["cpu", "cuda", "mps"])
    parser.add_argument("--sam2-model-size", default="small",
                        choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--sam2-min-area-ratio", type=float, default=0.02)
    parser.add_argument("--sam2-max-area-ratio", type=float, default=0.50)
    parser.add_argument("--sam2-pred-iou", type=float, default=0.88)
    parser.add_argument("--sam2-stability", type=float, default=0.95)
    parser.add_argument("--dinov3-size", default="vitl",
                        choices=["vits", "vitsplus", "vitb", "vitl", "vithplus"])
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for DINOv3 feature extraction (default: 64)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fill-strategy", choices=["source", "dtd"], default="dtd",
                        help="Fill strategy for Perlin island (default: dtd)")
    parser.add_argument("--dtd-root", default=DTD_ROOT,
                        help="Path to DTD images directory (used when --fill-strategy=dtd)")
    args = parser.parse_args()

    categories = args.categories if args.categories else ALL_CATEGORIES
    num_good = args.num_test // 2
    num_bad = args.num_test - num_good

    LOGGER.info("Loading models...")
    sam2_weights = os.path.join(PROJECT_ROOT, "weights", "sam2")
    masker = SAM2Masker(
        model_size=args.sam2_model_size,
        weights_dir=sam2_weights,
        device=args.device,
        pred_iou_thresh=args.sam2_pred_iou,
        stability_score_thresh=args.sam2_stability,
        min_area_ratio=args.sam2_min_area_ratio,
        max_area_ratio=args.sam2_max_area_ratio,
        fallback_stability_thresholds=[0.90, 0.85, 0.80],
    )

    dinov3_weights = os.path.join(PROJECT_ROOT, "weights", f"dinov3_{args.dinov3_size}")
    if not os.path.exists(dinov3_weights):
        dinov3_weights = os.path.join(PROJECT_ROOT, "weights", "dinov3")
    backbone = DinoV3Backbone(weights_dir=dinov3_weights).to(args.device)
    backbone.eval()

    clusterer = HDBSCANClusterer(
        min_cluster_size=3,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        umap_dims=20,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
    )

    island_gen = PerlinIslandGenerator(
        size_params=GammaSizeParams(shape=1.5, scale=0.07, min_ratio=0.03, max_ratio=0.15),
        noise_octaves=4,
        noise_persistence=0.5,
        noise_lacunarity=2.0,
        noise_base_res=(8, 8),
        sea_level_range=(0.05, 0.25),
        seed=args.seed,
    )

    filler = DTDTextureFiller(args.dtd_root, seed=args.seed) if args.fill_strategy == "dtd" else SourceTextureFiller()

    patch_composer = PatchComposer(
        island_generator=island_gen,
        filler=filler,
        seed=args.seed,
    )
    random_composer = RandomPatchComposer(
        island_generator=island_gen,
        dtd_root=DTD_ROOT,
        seed=args.seed,
    )

    semantic_strategy = SemanticPatchStrategy(
        backbone=backbone,
        clusterer=clusterer,
        composer=patch_composer,
        random_composer=random_composer,
        device=args.device,
        min_coverage=0.05,
        min_clusters=2,
    )
    random_strategy = RandomPatchStrategy(composer=random_composer)

    chain = AugmentationChain([semantic_strategy, random_strategy])

    LOGGER.info("=" * 60)
    LOGGER.info("Configuration:")
    LOGGER.info(f"  categories: {categories}")
    LOGGER.info(f"  device: {args.device}")
    LOGGER.info(f"  DINOv3 size: {args.dinov3_size}")
    LOGGER.info(f"  DINOv3 weights: {dinov3_weights}")
    LOGGER.info(f"  DINOv3 batch size: {args.batch_size}")
    LOGGER.info(f"  SAM2 model size: {args.sam2_model_size}")
    LOGGER.info(f"  SAM2 pred_iou_thresh: {args.sam2_pred_iou}")
    LOGGER.info(f"  SAM2 stability: {args.sam2_stability}")
    LOGGER.info(f"  SAM2 area ratio: [{args.sam2_min_area_ratio}, {args.sam2_max_area_ratio}]")
    LOGGER.info(f"  num_test per category: {args.num_test} (good={num_good}, bad={num_bad})")
    LOGGER.info(f"  fill_strategy: {args.fill_strategy}")
    LOGGER.info(f"  Perlin: shape=1.5, scale=0.10, min_ratio=0.03, max_ratio=0.25")
    LOGGER.info(f"  Perlin: octaves=4, persistence=0.5, lacunarity=2.0, base_res=(8,8)")
    LOGGER.info(f"  Perlin: sea_level_range=(0.05, 0.25)")
    LOGGER.info(f"  seed: {args.seed}")
    LOGGER.info(f"  Augmentation chain: {chain}")
    LOGGER.info("=" * 60)
    LOGGER.info("")

    for category in categories:
        cat_start = time.time()
        LOGGER.info(f"Processing category: {category}")

        cat_src = os.path.join(args.dataset_root, category)
        cat_out = os.path.join(args.output_dir, category)

        train_good_dir = os.path.join(cat_out, "train", "good")
        test_good_dir = os.path.join(cat_out, "test", "good")
        test_bad_dir = os.path.join(cat_out, "test", "bad")
        gt_bad_dir = os.path.join(cat_out, "ground_truth", "bad")

        if os.path.exists(cat_out):
            shutil.rmtree(cat_out)

        os.makedirs(train_good_dir, exist_ok=True)
        os.makedirs(test_good_dir, exist_ok=True)
        os.makedirs(test_bad_dir, exist_ok=True)
        os.makedirs(gt_bad_dir, exist_ok=True)

        train_images = collect_images(os.path.join(cat_src, "train", "good"))
        test_images = collect_images(os.path.join(cat_src, "test", "good"))

        for img_path in train_images:
            shutil.copy2(img_path, os.path.join(train_good_dir, os.path.basename(img_path)))
        LOGGER.info(f"  Copied {len(train_images)} train/good images")

        test_good_src = test_images[:num_good]
        for img_path in test_good_src:
            shutil.copy2(img_path, os.path.join(test_good_dir, os.path.basename(img_path)))
        LOGGER.info(f"  Copied {len(test_good_src)} test/good images")

        source_images = test_images if test_images else train_images
        if not source_images:
            LOGGER.warning(f"  No source images for {category}, skipping bad generation")
            continue

        bad_idx = 0
        src_idx = 0
        valid_count = 0
        no_valid_count = 0
        fail_count = 0

        LOGGER.info(f"  Source images: {len(source_images)}, target bad: {num_bad}")
        LOGGER.info(f"  SAM2 masking + accumulating batches...")

        sam2_count = 0
        batch_buffer = []

        def _process_batch(buffer: list[dict]) -> None:
            nonlocal bad_idx, valid_count, no_valid_count, fail_count

            batch_start = time.time()
            images_np = [item["image_np"] for item in buffer]
            per_image_kwargs = [
                {"sam2_mask": item["merged_mask"], "coverage": item["coverage"]}
                for item in buffer
            ]

            results = chain.process_batch(images_np, per_image_kwargs, return_debug=True)

            batch_saved = 0
            for item, (augmented_img, anchor_crop, anomaly_mask, debug_info) in zip(buffer, results):
                strategy_name = debug_info.get("strategy", "unknown") if debug_info else "unknown"
                succeeded = debug_info.get("success", False) if debug_info else False

                if succeeded:
                    if strategy_name == "semantic_patch":
                        valid_count += 1
                    else:
                        no_valid_count += 1
                else:
                    fail_count += 1
                    continue

                if anomaly_mask is None or np.sum(anomaly_mask) == 0:
                    fail_count += 1
                    continue

                idx_str = f"{bad_idx:03d}"
                img_out = os.path.join(test_bad_dir, f"{idx_str}.png")
                mask_out = os.path.join(gt_bad_dir, f"{idx_str}_mask.png")

                Image.fromarray(augmented_img).save(img_out)

                mask_uint8 = (anomaly_mask > 0).astype(np.uint8) * 255
                Image.fromarray(mask_uint8, mode="L").save(mask_out)

                bad_idx += 1
                batch_saved += 1

            LOGGER.info(
                f"    Batch done: {len(buffer)} images processed, "
                f"{batch_saved} saved, "
                f"time={time.time() - batch_start:.3f}s"
            )

        while bad_idx < num_bad and src_idx < len(source_images) * 3:
            img_path = source_images[src_idx % len(source_images)]
            src_idx += 1

            image_pil = Image.open(img_path).convert("RGB")
            image_np = np.array(image_pil)

            merged_mask, _, stats = masker.get_mask_with_stats(image_np)
            coverage = stats.get("coverage", 0)
            sam2_count += 1

            batch_buffer.append({
                "image_pil": image_pil,
                "image_np": image_np,
                "merged_mask": merged_mask,
                "coverage": coverage,
            })

            if len(batch_buffer) >= args.batch_size:
                LOGGER.info(f"  Batch full ({len(batch_buffer)} images), running pipeline...")
                _process_batch(batch_buffer)
                batch_buffer = []

        if batch_buffer:
            LOGGER.info(f"  Final batch ({len(batch_buffer)} images), running pipeline...")
            _process_batch(batch_buffer)
            batch_buffer = []

        LOGGER.info(f"  SAM2 calls: {sam2_count}")
        LOGGER.info(f"  Generated {bad_idx} test/bad images (VALID={valid_count}, NO_VALID={no_valid_count}, failed={fail_count})")
        LOGGER.info(f"  Category {category} completed in {time.time() - cat_start:.1f}s")
        LOGGER.info("")

    LOGGER.info(f"\nDone: processed {len(categories)} categories in {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
