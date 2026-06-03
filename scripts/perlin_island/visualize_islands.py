"""Visualize Perlin noise islands across all dataset categories.

For each category, generates multiple Perlin islands with different gradient
types and displays them overlaid on the original image. Shows both the binary
island mask and the intensity heatmap.

Usage:
    uv run python scripts/perlin_island/visualize_islands.py
    uv run python scripts/perlin_island/visualize_islands.py --fill-strategy dtd --dtd-root /path/to/dtd/images
"""

import argparse
import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_ROOT, "src"))

from synthesis.fillers import DTDTextureFiller
from synthesis.perlin_island import GammaSizeParams, PerlinIslandGenerator

DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "..",
    "..",
    "dataset",
    "glitch_hunt_mvtec",
)
DTD_ROOT_DEFAULT = os.path.join(
    PROJECT_ROOT,
    "..",
    "..",
    "dataset",
    "dtd",
    "images",
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "perlin_islands")

GRADIENT_TYPES = (
    "circular",
    "elliptical",
    "linear",
    "multi_center",
    "ring",
    "freeform",
)

N_SAMPLES_PER_CATEGORY = 1
N_ISLANDS_PER_SAMPLE = 6
MAX_CATEGORIES = 10


def discover_categories(dataset_root: str) -> list[str]:
    categories = []
    if not os.path.isdir(dataset_root):
        print(f"Dataset root not found: {dataset_root}")
        return categories

    for cat in sorted(os.listdir(dataset_root)):
        cat_path = os.path.join(dataset_root, cat, "good")
        if not os.path.isdir(cat_path):
            continue
        images = [
            f for f in os.listdir(cat_path)
            if f.lower().endswith((".png", ".jpg", ".jpeg"))
        ]
        if not images:
            continue
        categories.append(cat)
    return categories


def get_sample_images(dataset_root: str, category: str, n_samples: int) -> list[np.ndarray]:
    good_dir = os.path.join(dataset_root, category, "good")
    if not os.path.exists(good_dir):
        return []

    images = sorted(
        [f for f in os.listdir(good_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if not images:
        return []

    sample_files = random.sample(images, min(n_samples, len(images)))
    results = []
    for img_name in sample_files:
        img_path = os.path.join(good_dir, img_name)
        img = np.array(Image.open(img_path).convert("RGB"))
        results.append(img)
    return results


def crop_bounds_from_mask(mask: np.ndarray, pad_ratio: float = 0.2) -> tuple[int, int, int, int]:
    ys, xs = np.where(mask > 0)
    h, w = mask.shape

    if ys.size == 0 or xs.size == 0:
        crop_h = min(128, h)
        crop_w = min(128, w)
        y0 = (h - crop_h) // 2
        x0 = (w - crop_w) // 2
        return y0, y0 + crop_h, x0, x0 + crop_w

    y_min, y_max = int(ys.min()), int(ys.max())
    x_min, x_max = int(xs.min()), int(xs.max())
    box_h = y_max - y_min + 1
    box_w = x_max - x_min + 1
    pad = max(8, int(max(box_h, box_w) * pad_ratio))

    y0 = max(0, y_min - pad)
    y1 = min(h, y_max + pad + 1)
    x0 = max(0, x_min - pad)
    x1 = min(w, x_max + pad + 1)
    return y0, y1, x0, x1


def crop_region(arr: np.ndarray, bounds: tuple[int, int, int, int]) -> np.ndarray:
    y0, y1, x0, x1 = bounds
    return arr[y0:y1, x0:x1]


def source_image_pool(dataset_root: str, categories: list[str], per_category: int = 2) -> list[np.ndarray]:
    pool: list[np.ndarray] = []
    for category in categories:
        pool.extend(get_sample_images(dataset_root, category, per_category))
    return pool


def compose_island_with_fill(
    img: np.ndarray,
    result,
    filler,
) -> np.ndarray:
    overlay = img.copy().astype(np.float64)
    mask_bool = result.mask > 0

    if not mask_bool.any():
        return overlay.astype(np.uint8)

    bounds = crop_bounds_from_mask(result.mask)
    mask_crop = crop_region(result.mask, bounds)
    intensity_crop = crop_region(result.intensity, bounds)

    dummy_source = np.zeros(img.shape[:2], dtype=bool)
    fill_patch = filler.fill(img, mask_crop, dummy_source)

    ih, iw = fill_patch.shape[:2]
    intensity_3ch = np.stack([intensity_crop] * 3, axis=-1)

    y0, y1, x0, x1 = bounds
    roi_h = min(ih, y1 - y0)
    roi_w = min(iw, x1 - x0)

    if roi_h > 0 and roi_w > 0:
        overlay[y0:y0+roi_h, x0:x0+roi_w] = (
            overlay[y0:y0+roi_h, x0:x0+roi_w] * (1.0 - intensity_3ch[:roi_h, :roi_w])
            + fill_patch[:roi_h, :roi_w].astype(np.float64) * intensity_3ch[:roi_h, :roi_w]
        )

    return np.clip(overlay, 0, 255).astype(np.uint8)


def main():
    parser = argparse.ArgumentParser(description="Visualize Perlin noise islands")
    parser.add_argument(
        "--dataset-root",
        default=DATASET_ROOT,
        help="Path to MVTec dataset root",
    )
    parser.add_argument(
        "--dtd-root",
        default=DTD_ROOT_DEFAULT,
        help="Path to DTD images directory (used when --fill-strategy=dtd)",
    )
    parser.add_argument(
        "--fill-strategy",
        choices=["noise", "dtd"],
        default="dtd",
        help="Fill strategy: 'dtd' composes DTD textures (default), 'noise' uses intensity colormap overlay",
    )
    parser.add_argument(
        "--samples-per-category",
        type=int,
        default=N_SAMPLES_PER_CATEGORY,
        help="Number of images to sample per category",
    )
    parser.add_argument(
        "--islands-per-sample",
        type=int,
        default=N_ISLANDS_PER_SAMPLE,
        help="Number of islands to generate per image",
    )
    parser.add_argument(
        "--output",
        default=os.path.join(OUTPUT_DIR, "perlin_islands.jpg"),
        help="Save path",
    )
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--max-categories",
        type=int,
        default=MAX_CATEGORIES,
        help="Limit to first N categories",
    )
    args = parser.parse_args()

    random.seed(args.seed)
    np.random.seed(args.seed)

    categories = discover_categories(args.dataset_root)
    if not categories:
        print("No categories found.")
        sys.exit(1)

    categories = categories[:args.max_categories]
    print(f"Found {len(categories)} categories (limited to {args.max_categories})")

    generator = PerlinIslandGenerator(
        size_params=GammaSizeParams(shape=1.5, scale=0.07, min_ratio=0.03, max_ratio=0.15),
        noise_octaves=4,
        noise_persistence=0.5,
        noise_lacunarity=2.0,
        noise_base_res=(8, 8),
        sea_level_range=(0.05, 0.25),
        seed=args.seed,
    )

    filler = None
    if args.fill_strategy == "dtd":
        print(f"Loading DTD textures from {args.dtd_root}...")
        filler = DTDTextureFiller(args.dtd_root, seed=args.seed)
        print(f"  Loaded {len(filler._texture_paths)} textures")

    n_cols = args.islands_per_sample + 1
    n_rows = len(categories) * args.samples_per_category

    fig_height = n_rows * 2.5 + 1.0
    fig_width = n_cols * 3.0 + 1.5
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(fig_width, fig_height))

    if n_rows == 1:
        axes = axes.reshape(1, -1)

    row_idx = 0
    for category in categories:
        print(f"Processing {category}...")
        samples = get_sample_images(args.dataset_root, category, args.samples_per_category)

        for sample_idx, img in enumerate(samples):
            h, w = img.shape[:2]
            ax_orig = axes[row_idx, 0]
            ax_orig.imshow(img)
            ax_orig.set_title(f"{category}", fontsize=8, fontweight="bold", pad=4)
            ax_orig.axis("off")

            for island_idx in range(args.islands_per_sample):
                ax = axes[row_idx, island_idx + 1]

                result = generator(h, w, mask=None)

                if filler is not None:
                    overlay = compose_island_with_fill(img, result, filler)
                else:
                    overlay = img.copy().astype(np.float64)
                    mask_bool = result.mask > 0

                    if mask_bool.any():
                        cmap = plt.get_cmap("hot")
                        colored = cmap(result.intensity)[..., :3]
                        colored = (colored * 255).astype(np.float64)

                        alpha = 0.65 * result.intensity[..., np.newaxis]
                        overlay = overlay * (1 - alpha) + colored * alpha
                        overlay = np.clip(overlay, 0, 255).astype(np.uint8)
                    else:
                        overlay = overlay.astype(np.uint8)

                ax.imshow(overlay)

                area_pct = result.area_ratio * 100
                ax.set_title(
                    f"{result.gradient_type}\n{area_pct:.1f}%",
                    fontsize=6,
                    pad=2,
                )
                ax.axis("off")

            row_idx += 1

    fill_label = args.fill_strategy.upper() if args.fill_strategy == "dtd" else "Noise Intensity"
    fig.suptitle(
        f"Perlin Noise Island Generator — {fill_label} Fill",
        fontsize=13,
        fontweight="bold",
        y=0.995,
    )

    plt.tight_layout(rect=[0, 0, 1, 0.985])

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.savefig(args.output, dpi=150, bbox_inches="tight", format="jpeg")
    print(f"Saved visualization to {args.output}")
    plt.close(fig)

    # Second visualization: island distribution statistics
    print("\nGenerating island size distribution plot...")
    n_test = 12000
    areas = []
    grad_counts = {gt: 0 for gt in GRADIENT_TYPES}

    total_area = 224 * 224

    for _ in range(n_test):
        pixels = int(generator._sample_area(mask_area=total_area, total_area=total_area))
        areas.append((pixels / total_area) * 100.0)
        grad_counts[random.choice(GRADIENT_TYPES)] += 1

    fig2, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    ax1.hist(areas, bins=50, edgecolor="black", alpha=0.7, color="steelblue")
    ax1.set_xlabel("Island area (% of image)")
    ax1.set_ylabel("Count")
    ax1.set_title(f"Island Size Distribution (n={n_test})")
    ax1.set_yscale("log")

    types = list(grad_counts.keys())
    counts = list(grad_counts.values())
    colors = plt.cm.tab10(np.linspace(0, 1, len(types)))
    ax2.bar(types, counts, color=colors, edgecolor="black", alpha=0.7)
    ax2.set_xlabel("Gradient Type")
    ax2.set_ylabel("Count")
    ax2.set_title("Gradient Type Distribution")
    ax2.tick_params(axis="x", rotation=30)

    plt.tight_layout()
    dist_path = args.output.replace(".jpg", "_distribution.jpg").replace(".jpeg", "_distribution.jpeg")
    plt.savefig(dist_path, dpi=150, bbox_inches="tight", format="jpeg")
    print(f"Saved distribution plot to {dist_path}")
    plt.close(fig2)

    # Third visualization: gradient type showcase
    print("\nGenerating gradient type showcase...")

    n_showcase = len(GRADIENT_TYPES) * 4
    fig3, axes3 = plt.subplots(4, len(GRADIENT_TYPES), figsize=(len(GRADIENT_TYPES) * 3, 10))

    for col_idx, grad_type in enumerate(GRADIENT_TYPES):
        type_gen = PerlinIslandGenerator(
            size_params=GammaSizeParams(shape=1.5, scale=0.12, min_ratio=0.03, max_ratio=0.30),
            noise_octaves=4,
            noise_persistence=0.5,
            noise_lacunarity=2.0,
            noise_base_res=(6, 6),
            sea_level_range=(0.05, 0.25),
            gradient_types=(grad_type,),
            seed=args.seed + 300 + col_idx,
        )

        for row_idx in range(4):
            ax = axes3[row_idx, col_idx]
            result = type_gen(224, 224, mask=None)

            display = np.zeros((224, 224, 3), dtype=np.float32)
            if result.mask.sum() > 0:
                cmap = plt.get_cmap("viridis")
                colored = cmap(result.intensity)[..., :3]
                display[result.mask > 0] = colored[result.mask > 0]

            ax.imshow(display)
            if row_idx == 0:
                ax.set_title(f"{grad_type}\n{result.area_ratio*100:.1f}%", fontsize=9, fontweight="bold")
            ax.axis("off")

    fig3.suptitle("Gradient Type Showcase — Intensity Heatmaps", fontsize=12, fontweight="bold")
    plt.tight_layout()
    showcase_path = args.output.replace(".jpg", "_showcase.jpg").replace(".jpeg", "_showcase.jpeg")
    plt.savefig(showcase_path, dpi=150, bbox_inches="tight", format="jpeg")
    print(f"Saved showcase to {showcase_path}")
    plt.close(fig3)

    # Fourth visualization: island fill showcase (crop only)
    print("\nGenerating fill-only island showcase...")
    source_pool = source_image_pool(args.dataset_root, categories, per_category=2)
    if not source_pool:
        print("No source images available for fill showcase.")
    else:
        fig4, axes4 = plt.subplots(4, len(GRADIENT_TYPES), figsize=(len(GRADIENT_TYPES) * 3.0, 10))

        for col_idx, grad_type in enumerate(GRADIENT_TYPES):
            type_gen = PerlinIslandGenerator(
                size_params=GammaSizeParams(shape=1.5, scale=0.10, min_ratio=0.03, max_ratio=0.25),
                noise_octaves=4,
                noise_persistence=0.5,
                noise_lacunarity=2.0,
                noise_base_res=(6, 6),
                sea_level_range=(0.05, 0.25),
                gradient_types=(grad_type,),
                seed=args.seed + 700 + col_idx,
            )

            src = random.choice(source_pool)
            h, w = src.shape[:2]
            for row_idx in range(4):
                result = type_gen(h, w, mask=None)
                bounds = crop_bounds_from_mask(result.mask)
                src_crop = crop_region(src, bounds)
                mask_crop = crop_region(result.mask, bounds)

                if filler is not None:
                    dummy_source = np.zeros(src.shape[:2], dtype=bool)
                    island_fill = filler.fill(src, mask_crop, dummy_source)
                    island_fill[mask_crop == 0] = 0
                else:
                    island_fill = np.zeros_like(src_crop, dtype=np.uint8)
                    island_fill[mask_crop > 0] = src_crop[mask_crop > 0]

                ax = axes4[row_idx, col_idx]
                ax.imshow(island_fill)
                if row_idx == 0:
                    ax.set_title(f"{grad_type}\n{result.area_ratio*100:.1f}%", fontsize=9, fontweight="bold")
                ax.axis("off")

        fill_label = "DTD Texture" if filler is not None else "Source Crop"
        fig4.suptitle(f"Island Fill Showcase — {fill_label} Inside Mask", fontsize=12, fontweight="bold")
        plt.tight_layout()
        fill_path = args.output.replace(".jpg", "_showcase_fill.jpg").replace(
            ".jpeg", "_showcase_fill.jpeg"
        )
        plt.savefig(fill_path, dpi=150, bbox_inches="tight", format="jpeg")
        print(f"Saved fill showcase to {fill_path}")
        plt.close(fig4)

    print("Done.")


if __name__ == "__main__":
    main()
