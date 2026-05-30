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
from contextlib import nullcontext
import logging
import os
import shutil
import time
import sys

import cuml.accel
import numpy as np
import torch
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from src.backbones.dinov3 import DinoV3Backbone
from src.clustering.hdbscan_clusterer import HDBSCANClusterer
from src.masking.sam2_masker import SAM2Masker
from src.synthesis.patch_composer import PatchComposer
from src.synthesis.perlin_island import PerlinIslandGenerator
from src.synthesis.random_composer import DTD_ROOT, RandomPatchComposer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "mvtec_small")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "semantic_synthetic_anomalies_visa")

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

#ALL_CATEGORIES = [
#    "screw",
#    "tile", "toothbrush", "transistor", "wood", "zipper",
#]

def collect_images(directory: str) -> list[str]:
    if not os.path.isdir(directory):
        return []
    return sorted(
        os.path.join(directory, f)
        for f in os.listdir(directory)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )


def extract_dino_features_batch(
    backbone: DinoV3Backbone,
    images_pil: list[Image.Image],
    device: str,
) -> list[torch.Tensor]:
    from torchvision import transforms

    t0 = time.time()
    preprocess = transforms.Compose([
        transforms.Resize(256),
        transforms.CenterCrop(224),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    batch_tensor = torch.stack([preprocess(img) for img in images_pil]).to(device)

    with torch.no_grad():
        outputs = backbone(batch_tensor)
        last_hidden = outputs.last_hidden_state

    spatial = DinoV3Backbone.format_to_spatial_grid(last_hidden)
    elapsed = time.time() - t0
    LOGGER.info(
        f"    DINOv3 batch: {len(images_pil)} images, "
        f"input_shape={list(batch_tensor.shape)}, "
        f"output_shape={list(spatial.shape)}, "
        f"time={elapsed:.3f}s"
    )
    return [spatial[i] for i in range(len(images_pil))]


def resize_mask_to_grid(mask: np.ndarray) -> np.ndarray:
    import cv2
    resized = cv2.resize(
        mask.astype(np.float32),
        (14, 14),
        interpolation=cv2.INTER_AREA,
    )
    return (resized > 0.5).astype(np.uint8)


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
    parser.add_argument("--sam2-max-area-ratio", type=float, default=0.35)
    parser.add_argument("--sam2-pred-iou", type=float, default=0.88)
    parser.add_argument("--sam2-stability", type=float, default=0.95)
    parser.add_argument("--dinov3-size", default="vithplus",
                        choices=["vits", "vitsplus", "vitb", "vitl", "vithplus"])
    parser.add_argument("--batch-size", type=int, default=64,
                        help="Batch size for DINOv3 feature extraction (default: 64)")
    parser.add_argument("--seed", type=int, default=42)
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
    LOGGER.info(f"  seed: {args.seed}")
    LOGGER.info("=" * 60)
    LOGGER.info("")

    clusterer = HDBSCANClusterer(
        min_cluster_size=3,
        min_samples=None,
        cluster_selection_epsilon=0.0,
        metric="euclidean",
        umap_dims=20,
        umap_n_neighbors=15,
        umap_min_dist=0.0,
    )

    island_gen = PerlinIslandGenerator(seed=args.seed)
    composer = PatchComposer(
        island_generator=island_gen,
        seed=args.seed,
    )
    random_composer = RandomPatchComposer(
        island_generator=island_gen,
        dtd_root=DTD_ROOT,
        seed=args.seed,
    )

    for category in categories:
        cat_start = time.time()
        LOGGER.info(f"Processing category: {category}")

        cat_src = os.path.join(args.dataset_root, category)
        cat_out = os.path.join(args.output_dir, category)

        train_good_dir = os.path.join(cat_out, "train", "good")
        test_good_dir = os.path.join(cat_out, "test", "good")
        test_bad_dir = os.path.join(cat_out, "test", "bad")
        gt_bad_dir = os.path.join(cat_out, "ground_truth", "bad")

        # Clean output directory for this category to avoid permission issues with stale files
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
            images_pil = [item["image_pil"] for item in buffer]
            features_list = extract_dino_features_batch(backbone, images_pil, args.device)

            batch_saved = 0
            profile_fn = getattr(cuml.accel, "profile", None)
            profile_cm = profile_fn(quiet=True) if callable(profile_fn) else nullcontext(None)
            with profile_cm as profile_result:
                for item, features_2d in zip(buffer, features_list):
                    image_np = item["image_np"]
                    merged_mask = item["merged_mask"]
                    coverage = item["coverage"]

                    mask_14 = resize_mask_to_grid(merged_mask)
                    feat_for_cluster = features_2d.permute(1, 2, 0).reshape(-1, features_2d.shape[0])
                    clustering = clusterer.cluster(feat_for_cluster, mask_14)

                    is_valid = coverage >= 0.05 and len(clustering.unique_clusters) >= 2

                    LOGGER.debug(
                        f"    img: coverage={coverage:.3f}, "
                        f"clusters={clustering.unique_clusters}, "
                        f"valid={is_valid}"
                    )

                    result = None
                    if is_valid:
                        result = composer.compose(
                            image=image_np,
                            sam2_mask=merged_mask,
                            clustering=clustering,
                            features_2d=features_2d,
                        )
                        if result is not None:
                            valid_count += 1
                    else:
                        result = random_composer.compose(image=image_np)
                        if result is not None:
                            no_valid_count += 1

                    if result is None:
                        fail_count += 1
                        continue

                    anomaly_img = result.augmented_image
                    anomaly_mask = result.anomaly_mask

                    if anomaly_mask is None or np.sum(anomaly_mask) == 0:
                        fail_count += 1
                        continue

                    idx_str = f"{bad_idx:03d}"
                    img_out = os.path.join(test_bad_dir, f"{idx_str}.png")
                    mask_out = os.path.join(gt_bad_dir, f"{idx_str}_mask.png")

                    Image.fromarray(anomaly_img).save(img_out)

                    mask_uint8 = (anomaly_mask > 0).astype(np.uint8) * 255
                    Image.fromarray(mask_uint8, mode="L").save(mask_out)

                    bad_idx += 1
                    batch_saved += 1

            LOGGER.info(
                f"    Batch done: {len(buffer)} images processed, "
                f"{batch_saved} saved, "
                f"time={time.time() - batch_start:.3f}s"
            )
            if profile_result is not None:
                print(profile_result)

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
                LOGGER.info(f"  Batch full ({len(batch_buffer)} images), running DINOv3...")
                _process_batch(batch_buffer)
                batch_buffer = []

        if batch_buffer:
            LOGGER.info(f"  Final batch ({len(batch_buffer)} images), running DINOv3...")
            _process_batch(batch_buffer)
            batch_buffer = []

        LOGGER.info(f"  SAM2 calls: {sam2_count}")

        LOGGER.info(f"  Generated {bad_idx} test/bad images (VALID={valid_count}, NO_VALID={no_valid_count}, failed={fail_count})")
        LOGGER.info(f"  Category {category} completed in {time.time() - cat_start:.1f}s")
        LOGGER.info("")

    LOGGER.info(f"\nDone: processed {len(categories)} categories in {time.time() - overall_start:.1f}s")


if __name__ == "__main__":
    main()
