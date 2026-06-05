"""Semantic patching visualization script.

Generates multi-panel visualizations per category:
  VALID mode (7 panels):
    1. Original image + SAM2 mask overlay (red)
    2. Perlin Island (magma heatmap)
    3. Source → Target clusters with pin markers
    4. PCA-reduced DINOv3 features (C → 3 channels → RGB)
    5. HDBSCAN clustering result (color-coded 14x14 grid)
    6. Synthetic Anomaly (clean)
    7. Synthetic Anomaly + mask overlay (green)

  NO_VALID mode (6 panels):
    1. Original image
    2. Random Perlin Island (magma heatmap)
    3. DTD texture sample
    4. Random placement visualization
    5. Synthetic Anomaly (seamless clone, clean)
    6. Synthetic Anomaly + mask overlay (green)

Usage:
    uv run python scripts/semantic_patching/visualize.py
    uv run python scripts/semantic_patching/visualize.py --categories bottle cable
    uv run python scripts/semantic_patching/visualize.py --device cuda
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import cv2
import matplotlib
import numpy as np
import torch
from PIL import Image
from sklearn.decomposition import PCA
from src.augmentation.blenders import FeatherBlender

matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, PROJECT_ROOT)

from config import ClusteringConfig, FillerConfig, PerlinConfig, SAM2Config
from src.backbones.dinov3 import DinoV3Backbone
from src.clustering.hdbscan_clusterer import HDBSCANClusterer
from src.masking.sam2_masker import SAM2Masker
from src.augmentation.blenders.seamless_clone import SeamlessCloneBlender
from src.synthesis.patch_composer import PatchComposer
from src.synthesis.perlin_island import PerlinIslandGenerator
from src.synthesis.fillers import DTDTextureFiller, SourceTextureFiller
from src.synthesis.random_composer import DTD_ROOT, RandomPatchComposer

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
LOGGER = logging.getLogger(__name__)

GRID_SIZE = 14

DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "BigD")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "semantic_patching")

#ALL_CATEGORIES = [
#    "bottle", "cable", "capsule", "carpet", "grid",
#    "hazelnut", "leather", "metal_nut", "pill", "screw",
#    "tile", "toothbrush", "transistor", "wood", "zipper",
#]

#ALL_CATEGORIES = [
#    "candle", "capsules", "cashew", "chewinggum", "fryum",
#    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
#]

ALL_CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
    "candle", "capsules", "cashew", "chewinggum", "fryum",
    "macaroni1", "macaroni2", "pcb1", "pcb2", "pcb3", "pcb4", "pipe_fryum",
]


def collect_image_path(category: str, dataset_root: str) -> str | None:
    good_dir = os.path.join(dataset_root, category, "train", "good")
    if not os.path.isdir(good_dir):
        good_dir = os.path.join(dataset_root, category, "good")
    if not os.path.isdir(good_dir):
        return None
    files = sorted(
        f for f in os.listdir(good_dir)
        if f.lower().endswith((".png", ".jpg", ".jpeg"))
    )
    if not files:
        return None
    return os.path.join(good_dir, files[0])


def extract_dino_features(
    backbone: DinoV3Backbone,
    image_pil: Image.Image,
    device: str,
) -> torch.Tensor:
    from torchvision import transforms

    preprocess = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    tensor = preprocess(image_pil).unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = backbone(tensor)
        last_hidden = outputs.last_hidden_state

    spatial = DinoV3Backbone.format_to_spatial_grid(last_hidden)
    return spatial.squeeze(0)


def resize_mask_to_grid(mask: np.ndarray) -> np.ndarray:
    resized = cv2.resize(
        mask.astype(np.float32),
        (GRID_SIZE, GRID_SIZE),
        interpolation=cv2.INTER_AREA,
    )
    return (resized > 0.5).astype(np.uint8)


def upscale_cluster_map(cluster_map_14: np.ndarray, h: int, w: int) -> np.ndarray:
    return cv2.resize(
        cluster_map_14.astype(np.int32),
        (w, h),
        interpolation=cv2.INTER_NEAREST,
    )


def visualize_perlin_island(island_intensity: np.ndarray, island_mask: np.ndarray) -> np.ndarray:
    h, w = island_intensity.shape
    rgb = np.zeros((h, w, 3), dtype=np.uint8)

    cmap = plt.get_cmap("magma")
    colored = (cmap(island_intensity)[:, :, :3] * 255).astype(np.uint8)
    rgb[island_mask > 0] = colored[island_mask > 0]

    return rgb


def visualize_source_target_pins(
    image_np: np.ndarray,
    cluster_full: np.ndarray,
    source_id: int,
    target_id: int,
    sam2_mask: np.ndarray,
) -> np.ndarray:
    result = image_np.copy()

    source_pixels = cluster_full == source_id
    target_pixels = cluster_full == target_id

    faint_blue = np.zeros_like(result)
    faint_blue[source_pixels] = [100, 100, 255]
    result = cv2.addWeighted(result, 0.85, faint_blue, 0.15, 0)

    faint_orange = np.zeros_like(result)
    faint_orange[target_pixels] = [255, 165, 0]
    result = cv2.addWeighted(result, 0.85, faint_orange, 0.15, 0)

    source_ys, source_xs = np.where(source_pixels & (sam2_mask > 0))
    target_ys, target_xs = np.where(target_pixels & (sam2_mask > 0))

    if len(source_ys) > 0:
        cy, cx = int(source_ys.mean()), int(source_xs.mean())
        cv2.circle(result, (cx, cy), 18, (255, 255, 255), 4)
        cv2.circle(result, (cx, cy), 14, (0, 0, 255), -1)
        cv2.putText(result, "SRC", (cx - 22, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    if len(target_ys) > 0:
        cy, cx = int(target_ys.mean()), int(target_xs.mean())
        cv2.circle(result, (cx, cy), 18, (255, 255, 255), 4)
        thickness = 4
        arm_len = 12
        cv2.line(result, (cx - arm_len, cy - arm_len), (cx + arm_len, cy + arm_len), (0, 140, 255), thickness)
        cv2.line(result, (cx + arm_len, cy - arm_len), (cx - arm_len, cy + arm_len), (0, 140, 255), thickness)
        cv2.putText(result, "TGT", (cx - 22, cy + 5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return result


def visualize_pca_features(features_2d: torch.Tensor, mask_14: np.ndarray) -> np.ndarray:
    c, h, w = features_2d.shape
    feat_flat = features_2d.permute(1, 2, 0).cpu().numpy().reshape(-1, c)

    pca = PCA(n_components=3)
    reduced = pca.fit_transform(feat_flat)

    reduced = (reduced - reduced.min(axis=0)) / (reduced.max(axis=0) - reduced.min(axis=0) + 1e-8)
    reduced = (reduced * 255).astype(np.uint8)

    rgb = reduced.reshape(h, w, 3)

    if mask_14 is not None:
        rgb[mask_14 == 0] = 0

    return rgb


def visualize_clustering(clustering, mask_14: np.ndarray, scale: int = 16) -> np.ndarray:
    cluster_map = clustering.cluster_map_masked.copy()

    unique_labels = sorted(set(cluster_map.flatten()) - {-1})
    n_clusters = len(unique_labels)

    if n_clusters == 0:
        img = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
        return img

    cmap = plt.get_cmap("tab20" if n_clusters <= 20 else "nipy_spectral")
    label_to_color = {}
    for i, label in enumerate(unique_labels):
        color = cmap(i / max(n_clusters - 1, 1))
        label_to_color[label] = (int(color[0] * 255), int(color[1] * 255), int(color[2] * 255))

    rgb = np.zeros((GRID_SIZE, GRID_SIZE, 3), dtype=np.uint8)
    for y in range(GRID_SIZE):
        for x in range(GRID_SIZE):
            label = cluster_map[y, x]
            if label in label_to_color:
                rgb[y, x] = label_to_color[label]

    if scale > 1:
        rgb = rgb.repeat(scale, axis=0).repeat(scale, axis=1)

    return rgb


def visualize_random_placement(
    image_np: np.ndarray,
    center: tuple[int, int],
    island_mask: np.ndarray,
) -> np.ndarray:
    result = image_np.copy()
    cx, cy = center

    cv2.circle(result, (cx, cy), 20, (255, 255, 255), 4)
    cv2.circle(result, (cx, cy), 16, (0, 255, 0), -1)
    cv2.putText(result, "RAND", (cx - 28, cy + 5),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    return result


def create_valid_panel(
    image_np: np.ndarray,
    sam2_mask: np.ndarray,
    island_rgb: np.ndarray,
    source_target_rgb: np.ndarray,
    pca_rgb: np.ndarray,
    cluster_rgb: np.ndarray,
    anomaly_clean: np.ndarray,
    anomaly_mask: np.ndarray,
    stats: dict,
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 7, figsize=(35, 5))

    axes[0].imshow(image_np)
    overlay = np.zeros_like(image_np)
    overlay[sam2_mask > 0] = [255, 0, 0]
    axes[0].imshow(overlay, alpha=0.4)
    axes[0].set_title("Original + SAM2 Mask", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(island_rgb)
    axes[1].set_title("Perlin Island", fontsize=9, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(source_target_rgb)
    axes[2].set_title(
        f"Source → Target [{stats.get('source_cluster', '?')}→{stats.get('target_cluster', '?')}]",
        fontsize=9, fontweight="bold",
    )
    axes[2].axis("off")

    axes[3].imshow(pca_rgb)
    axes[3].set_title("DINOv3 Features (PCA→RGB)", fontsize=9, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(cluster_rgb)
    axes[4].set_title(
        f"HDBSCAN ({stats.get('num_clusters', '?')} clusters, noise={stats.get('noise_ratio', '?'):.0%})",
        fontsize=9, fontweight="bold",
    )
    axes[4].axis("off")

    axes[5].imshow(anomaly_clean)
    axes[5].set_title("Synthetic Anomaly", fontsize=9, fontweight="bold")
    axes[5].axis("off")

    axes[6].imshow(anomaly_clean)
    if anomaly_mask is not None and np.sum(anomaly_mask) > 0:
        anomaly_overlay = np.zeros_like(anomaly_clean)
        anomaly_overlay[anomaly_mask > 0] = [0, 255, 0]
        axes[6].imshow(anomaly_overlay, alpha=0.4)
    axes[6].set_title("Anomaly + Mask", fontsize=9, fontweight="bold")
    axes[6].axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def create_no_valid_panel(
    image_np: np.ndarray,
    island_rgb: np.ndarray,
    dtd_texture: np.ndarray,
    placement_rgb: np.ndarray,
    anomaly_clean: np.ndarray,
    anomaly_mask: np.ndarray,
    stats: dict,
    title: str,
) -> plt.Figure:
    fig, axes = plt.subplots(1, 6, figsize=(30, 5))

    axes[0].imshow(image_np)
    axes[0].set_title("Original (No Mask)", fontsize=9, fontweight="bold")
    axes[0].axis("off")

    axes[1].imshow(island_rgb)
    axes[1].set_title("Random Perlin Island", fontsize=9, fontweight="bold")
    axes[1].axis("off")

    axes[2].imshow(dtd_texture)
    dtd_path = stats.get("dtd_path", "N/A")
    dtd_name = Path(dtd_path).name if dtd_path else "N/A"
    axes[2].set_title(f"DTD Texture: {dtd_name}", fontsize=9, fontweight="bold")
    axes[2].axis("off")

    axes[3].imshow(placement_rgb)
    axes[3].set_title("Random Placement", fontsize=9, fontweight="bold")
    axes[3].axis("off")

    axes[4].imshow(anomaly_clean)
    axes[4].set_title("Synthetic Anomaly", fontsize=9, fontweight="bold")
    axes[4].axis("off")

    axes[5].imshow(anomaly_clean)
    if anomaly_mask is not None and np.sum(anomaly_mask) > 0:
        anomaly_overlay = np.zeros_like(anomaly_clean)
        anomaly_overlay[anomaly_mask > 0] = [0, 255, 0]
        axes[5].imshow(anomaly_overlay, alpha=0.4)
    axes[5].set_title("Anomaly + Mask", fontsize=9, fontweight="bold")
    axes[5].axis("off")

    fig.suptitle(title, fontsize=11, fontweight="bold", y=1.02)
    fig.tight_layout()
    return fig


def main():
    sam2_defaults = SAM2Config()
    sam2_defaults = SAM2Config()
    clustering_defaults = ClusteringConfig()
    perlin_defaults = PerlinConfig()
    filler_defaults = FillerConfig()

    parser = argparse.ArgumentParser(description="Semantic patching visualization")
    parser.add_argument("--categories", nargs="*", default=None,
                        help="Specific categories to process (default: all)")
    parser.add_argument("--dataset-root", default=DATASET_ROOT)
    parser.add_argument("--output-dir", default=OUTPUT_DIR)
    parser.add_argument("--device", default=sam2_defaults.device, choices=["cpu", "cuda", "mps"])
    parser.add_argument("--sam2-model-size", default=sam2_defaults.model_size,
                        choices=["tiny", "small", "base_plus", "large"])
    parser.add_argument("--sam2-min-area-ratio", type=float, default=sam2_defaults.min_area_ratio,
                        help="Min mask area as ratio of image")
    parser.add_argument("--sam2-max-area-ratio", type=float, default=sam2_defaults.max_area_ratio,
                        help="Max mask area as ratio of image")
    parser.add_argument("--sam2-pred-iou", type=float, default=sam2_defaults.pred_iou_thresh,
                        help="SAM2 pred_iou_thresh")
    parser.add_argument("--sam2-stability", type=float, default=sam2_defaults.stability_score_thresh,
                        help="SAM2 stability_score_thresh")
    parser.add_argument("--dinov3-size", default="vitl",
                        choices=["vits", "vitsplus", "vitb", "vitl", "vithplus"])
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fill-strategy", choices=["source", "dtd"], default="source",
                        help="Fill strategy for Perlin island")
    parser.add_argument("--dtd-root", default=DTD_ROOT,
                        help="Path to DTD images directory (used when --fill-strategy=dtd)")
    parser.add_argument(
        "--dtd-mix-prob",
        type=float,
        default=filler_defaults.dtd_mix_prob,
        help="Probability of using DTD fill instead of source fill in semantic mode",
    )
    args = parser.parse_args()

    categories = args.categories if args.categories else ALL_CATEGORIES
    os.makedirs(args.output_dir, exist_ok=True)

    LOGGER.info("=" * 64)
    LOGGER.info("Configuration")
    LOGGER.info("=" * 64)
    LOGGER.info("  categories: %s", categories)
    LOGGER.info("  dataset_root: %s", args.dataset_root)
    LOGGER.info("  output_dir: %s", args.output_dir)
    LOGGER.info("  device: %s", args.device)
    LOGGER.info("  seed: %d", args.seed)
    LOGGER.info("  dinov3_size: %s", args.dinov3_size)
    LOGGER.info("  fill_strategy: %s", args.fill_strategy)
    LOGGER.info("  dtd_mix_prob: %.3f", args.dtd_mix_prob)
    LOGGER.info("  blender: seamless_clone (mode=mixed)")
    LOGGER.info("")
    LOGGER.info("  SAM2:")
    LOGGER.info("    model_size: %s", args.sam2_model_size)
    LOGGER.info("    min_area_ratio: %.4f", args.sam2_min_area_ratio)
    LOGGER.info("    max_area_ratio: %.4f", args.sam2_max_area_ratio)
    LOGGER.info("    pred_iou_thresh: %.4f", args.sam2_pred_iou)
    LOGGER.info("    stability_score_thresh: %.4f", args.sam2_stability)
    LOGGER.info("    fallback_stability_thresholds: [0.70]")
    LOGGER.info("")
    LOGGER.info("  Clustering:")
    LOGGER.info("    min_cluster_size: %d", clustering_defaults.min_cluster_size)
    LOGGER.info("    min_samples: %s", clustering_defaults.min_samples)
    LOGGER.info("    cluster_selection_epsilon: %.4f", clustering_defaults.cluster_selection_epsilon)
    LOGGER.info("    metric: %s", clustering_defaults.metric)
    LOGGER.info("    umap_dims: %d", clustering_defaults.umap_dims)
    LOGGER.info("    umap_n_neighbors: %d", clustering_defaults.umap_n_neighbors)
    LOGGER.info("    umap_min_dist: %.4f", clustering_defaults.umap_min_dist)
    LOGGER.info("")
    LOGGER.info("  Perlin:")
    LOGGER.info("    size_params: shape=%.4f, scale=%.4f, min_ratio=%.4f, max_ratio=%.4f",
                perlin_defaults.size_params.shape,
                perlin_defaults.size_params.scale,
                perlin_defaults.size_params.min_ratio,
                perlin_defaults.size_params.max_ratio)
    LOGGER.info("    noise_octaves: %d", perlin_defaults.noise_octaves)
    LOGGER.info("    noise_persistence: %.4f", perlin_defaults.noise_persistence)
    LOGGER.info("    noise_lacunarity: %.4f", perlin_defaults.noise_lacunarity)
    LOGGER.info("    noise_base_res: %s", perlin_defaults.noise_base_res)
    LOGGER.info("    sea_level_range: %s", perlin_defaults.sea_level_range)
    LOGGER.info("    gradient_types: %s", perlin_defaults.gradient_types)
    LOGGER.info("=" * 64)

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
        fallback_stability_thresholds=[0.70],
    )

    dinov3_weights = os.path.join(PROJECT_ROOT, "weights", f"dinov3_{args.dinov3_size}")
    if not os.path.exists(dinov3_weights):
        dinov3_weights = os.path.join(PROJECT_ROOT, "weights", "dinov3")
    backbone = DinoV3Backbone(weights_dir=dinov3_weights).to(args.device)
    backbone.eval()

    clusterer = HDBSCANClusterer(
        min_cluster_size=clustering_defaults.min_cluster_size,
        min_samples=clustering_defaults.min_samples,
        cluster_selection_epsilon=clustering_defaults.cluster_selection_epsilon,
        metric=clustering_defaults.metric,
        umap_dims=clustering_defaults.umap_dims,
        umap_n_neighbors=clustering_defaults.umap_n_neighbors,
        umap_min_dist=clustering_defaults.umap_min_dist,
    )

    island_gen = PerlinIslandGenerator(
        size_params=perlin_defaults.size_params,
        noise_octaves=perlin_defaults.noise_octaves,
        noise_persistence=perlin_defaults.noise_persistence,
        noise_lacunarity=perlin_defaults.noise_lacunarity,
        noise_base_res=perlin_defaults.noise_base_res,
        sea_level_range=perlin_defaults.sea_level_range,
        seed=args.seed,
    )

    dtd_mix_prob = float(np.clip(args.dtd_mix_prob, 0.0, 1.0))
    dtd_filler = DTDTextureFiller(args.dtd_root, seed=args.seed)
    filler = dtd_filler if args.fill_strategy == "dtd" else SourceTextureFiller()
    blender = FeatherBlender(20, 0.5)

    composer = PatchComposer(
        island_generator=island_gen,
        blender=blender,
        filler=filler,
        dtd_filler=(dtd_filler if args.fill_strategy == "source" else None),
        dtd_mix_prob=(dtd_mix_prob if args.fill_strategy == "source" else 0.0),
        seed=args.seed,
    )
    random_composer = RandomPatchComposer(
        island_generator=island_gen,
        dtd_root=DTD_ROOT,
        seed=args.seed,
    )

    results = []
    valid_count = 0
    skip_count = 0
    no_valid_count = 0

    for category in categories:
        LOGGER.info(f"Processing category: {category}")
        cat_output = os.path.join(args.output_dir, category)
        os.makedirs(cat_output, exist_ok=True)

        image_path = collect_image_path(category, args.dataset_root)
        if image_path is None:
            LOGGER.info(f"  SKIP: no images found for {category}")
            skip_count += 1
            continue

        image_pil = Image.open(image_path).convert("RGB")
        image_np = np.array(image_pil)
        h, w = image_np.shape[:2]

        merged_mask, masks, stats = masker.get_mask_with_stats(image_np)
        coverage = stats.get("coverage", 0)

        features_2d = extract_dino_features(backbone, image_pil, args.device)
        mask_14 = resize_mask_to_grid(merged_mask)
        feat_for_cluster = features_2d.permute(1, 2, 0).reshape(-1, features_2d.shape[0])
        clustering = clusterer.cluster(feat_for_cluster, mask_14)

        pca_rgb = visualize_pca_features(features_2d, mask_14)
        cluster_rgb = visualize_clustering(clustering, mask_14)

        is_valid = coverage >= sam2_defaults.min_area_ratio and len(clustering.unique_clusters) >= 2

        if is_valid:
            valid_count += 1
            LOGGER.info(f"  VALID: coverage {coverage:.1%}, {len(masks)} mask(s)")

            patch_result = composer.compose(
                image=image_np,
                sam2_mask=merged_mask,
                clustering=clustering,
                features_2d=features_2d,
            )

            if patch_result is not None:
                anomaly_clean = patch_result.augmented_image
                anomaly_mask = patch_result.anomaly_mask
                island_rgb = visualize_perlin_island(
                    patch_result.island.intensity,
                    patch_result.island.mask,
                )
                cluster_full = upscale_cluster_map(clustering.cluster_map_masked, h, w)
                source_target_rgb = visualize_source_target_pins(
                    image_np, cluster_full,
                    patch_result.source_cluster_id,
                    patch_result.target_cluster_id,
                    merged_mask,
                )
                patch_stats = {
                    "num_clusters": clustering.info.get("num_clusters", 0),
                    "noise_ratio": clustering.info.get("noise_ratio", 0),
                    "source_cluster": patch_result.source_cluster_id,
                    "target_cluster": patch_result.target_cluster_id,
                }
            else:
                anomaly_clean = image_np
                anomaly_mask = None
                island_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                source_target_rgb = image_np.copy()
                patch_stats = {
                    "num_clusters": clustering.info.get("num_clusters", 0),
                    "noise_ratio": clustering.info.get("noise_ratio", 0),
                    "source_cluster": "N/A",
                    "target_cluster": "N/A",
                }
                LOGGER.info(f"  PATCH: composition not possible for {category}")

            fig = create_valid_panel(
                image_np=image_np,
                sam2_mask=merged_mask,
                island_rgb=island_rgb,
                source_target_rgb=source_target_rgb,
                pca_rgb=pca_rgb,
                cluster_rgb=cluster_rgb,
                anomaly_clean=anomaly_clean,
                anomaly_mask=anomaly_mask,
                stats=patch_stats,
                title=f"{category} (VALID)",
            )

            results.append({
                "category": category,
                "mode": "valid",
                "image_np": image_np,
                "sam2_mask": merged_mask,
                "island_rgb": island_rgb,
                "source_target_rgb": source_target_rgb,
                "pca_rgb": pca_rgb,
                "cluster_rgb": cluster_rgb,
                "anomaly_clean": anomaly_clean,
                "anomaly_mask": anomaly_mask,
                "stats": patch_stats,
            })

        else:
            no_valid_count += 1
            LOGGER.info(f"  NO_VALID: coverage {coverage:.1%}, {len(masks)} mask(s)")

            random_result = random_composer.compose(image=image_np)

            if random_result is not None:
                anomaly_clean = random_result.augmented_image
                anomaly_mask = random_result.anomaly_mask
                island_rgb = visualize_perlin_island(
                    random_result.island.intensity,
                    random_result.island.mask,
                )
                placement_rgb = visualize_random_placement(
                    image_np,
                    random_result.placement_center,
                    random_result.island.mask,
                )
                dtd_texture = random_result.dtd_texture
                patch_stats = {
                    "dtd_path": random_result.dtd_path,
                    "placement_center": random_result.placement_center,
                }
            else:
                anomaly_clean = image_np
                anomaly_mask = None
                island_rgb = np.zeros((h, w, 3), dtype=np.uint8)
                placement_rgb = image_np.copy()
                dtd_texture = np.zeros((h, w, 3), dtype=np.uint8)
                patch_stats = {
                    "dtd_path": "N/A",
                    "placement_center": (0, 0),
                }
                LOGGER.info(f"  RANDOM: composition not possible for {category}")

            fig = create_no_valid_panel(
                image_np=image_np,
                island_rgb=island_rgb,
                dtd_texture=dtd_texture,
                placement_rgb=placement_rgb,
                anomaly_clean=anomaly_clean,
                anomaly_mask=anomaly_mask,
                stats=patch_stats,
                title=f"{category} (NO_VALID)",
            )

            results.append({
                "category": category,
                "mode": "no_valid",
                "image_np": image_np,
                "island_rgb": island_rgb,
                "dtd_texture": dtd_texture,
                "placement_rgb": placement_rgb,
                "anomaly_clean": anomaly_clean,
                "anomaly_mask": anomaly_mask,
                "stats": patch_stats,
            })

        out_path = os.path.join(cat_output, f"{Path(image_path).stem}.jpg")
        fig.savefig(out_path, dpi=150, bbox_inches="tight", pad_inches=0.1, format="jpeg")
        plt.close(fig)
        LOGGER.info(f"  Saved: {out_path}")

    if results:
        valid_results = [r for r in results if r["mode"] == "valid"]
        no_valid_results = [r for r in results if r["mode"] == "no_valid"]

        all_figs = []

        for r in valid_results:
            fig = create_valid_panel(
                image_np=r["image_np"],
                sam2_mask=r["sam2_mask"],
                island_rgb=r["island_rgb"],
                source_target_rgb=r["source_target_rgb"],
                pca_rgb=r["pca_rgb"],
                cluster_rgb=r["cluster_rgb"],
                anomaly_clean=r["anomaly_clean"],
                anomaly_mask=r["anomaly_mask"],
                stats=r["stats"],
                title=f"{r['category']} (VALID)",
            )
            all_figs.append(fig)

        for r in no_valid_results:
            fig = create_no_valid_panel(
                image_np=r["image_np"],
                island_rgb=r["island_rgb"],
                dtd_texture=r["dtd_texture"],
                placement_rgb=r["placement_rgb"],
                anomaly_clean=r["anomaly_clean"],
                anomaly_mask=r["anomaly_mask"],
                stats=r["stats"],
                title=f"{r['category']} (NO_VALID)",
            )
            all_figs.append(fig)

        n_rows = len(all_figs)
        n_cols = 7
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(35, 5 * n_rows))
        if n_rows == 1:
            axes = axes.reshape(1, -1)

        for i, r in enumerate(valid_results):
            axes[i, 0].imshow(r["image_np"])
            overlay = np.zeros_like(r["image_np"])
            overlay[r["sam2_mask"] > 0] = [255, 0, 0]
            axes[i, 0].imshow(overlay, alpha=0.4)
            axes[i, 0].set_ylabel(r["category"], fontsize=10, fontweight="bold")
            axes[i, 0].axis("off")

            axes[i, 1].imshow(r["island_rgb"])
            axes[i, 1].axis("off")

            axes[i, 2].imshow(r["source_target_rgb"])
            axes[i, 2].axis("off")

            axes[i, 3].imshow(r["pca_rgb"])
            axes[i, 3].axis("off")

            axes[i, 4].imshow(r["cluster_rgb"])
            axes[i, 4].axis("off")

            axes[i, 5].imshow(r["anomaly_clean"])
            axes[i, 5].axis("off")

            axes[i, 6].imshow(r["anomaly_clean"])
            if r["anomaly_mask"] is not None and np.sum(r["anomaly_mask"]) > 0:
                ao = np.zeros_like(r["anomaly_clean"])
                ao[r["anomaly_mask"] > 0] = [0, 255, 0]
                axes[i, 6].imshow(ao, alpha=0.4)
            axes[i, 6].axis("off")

        valid_offset = len(valid_results)
        for i, r in enumerate(no_valid_results):
            row = valid_offset + i
            axes[row, 0].imshow(r["image_np"])
            axes[row, 0].set_ylabel(r["category"], fontsize=10, fontweight="bold")
            axes[row, 0].axis("off")

            axes[row, 1].imshow(r["island_rgb"])
            axes[row, 1].axis("off")

            axes[row, 2].imshow(r["dtd_texture"])
            axes[row, 2].axis("off")

            axes[row, 3].imshow(r["placement_rgb"])
            axes[row, 3].axis("off")

            axes[row, 4].imshow(r["anomaly_clean"])
            axes[row, 4].axis("off")

            axes[row, 5].imshow(r["anomaly_clean"])
            if r["anomaly_mask"] is not None and np.sum(r["anomaly_mask"]) > 0:
                ao = np.zeros_like(r["anomaly_clean"])
                ao[r["anomaly_mask"] > 0] = [0, 255, 0]
                axes[row, 5].imshow(ao, alpha=0.4)
            axes[row, 5].axis("off")

            axes[row, 6].axis("off")

        axes[0, 0].set_title("Original + SAM2 Mask", fontsize=9, fontweight="bold")
        axes[0, 1].set_title("Perlin Island", fontsize=9, fontweight="bold")
        axes[0, 2].set_title("Source → Target", fontsize=9, fontweight="bold")
        axes[0, 3].set_title("DINOv3 Features (PCA→RGB)", fontsize=9, fontweight="bold")
        axes[0, 4].set_title("HDBSCAN Clustering", fontsize=9, fontweight="bold")
        axes[0, 5].set_title("Synthetic Anomaly", fontsize=9, fontweight="bold")
        axes[0, 6].set_title("Anomaly + Mask", fontsize=9, fontweight="bold")

        fig.suptitle("Semantic Patching — Summary", fontsize=14, fontweight="bold", y=1.01)
        fig.tight_layout()

        summary_path = os.path.join(args.output_dir, "summary.jpg")
        fig.savefig(summary_path, dpi=150, bbox_inches="tight", pad_inches=0.1, format="jpeg")
        plt.close(fig)
        LOGGER.info(f"Summary saved: {summary_path}")

    LOGGER.info(f"\nDone: {valid_count} valid, {no_valid_count} no_valid, {skip_count} skipped out of {len(categories)} categories")


if __name__ == "__main__":
    main()
