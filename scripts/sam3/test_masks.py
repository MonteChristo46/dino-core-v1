"""Quick test script to visualize SAM3 masks on MVTec images for all categories."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

# Enable bfloat16 for Ampere GPUs
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

# Paths
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights", "sam3")
DATASET_ROOT = os.path.join(
    PROJECT_ROOT,
    "..",
    "..",
    "dataset",
    "glitch_hunt_mvtec",
)
CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam3.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "sam3")

# Category-specific text prompts
CATEGORY_PROMPTS = {
    "bottle": "bottle",
    "cable": "cable",
    "capsule": "capsule",
    "carpet": "carpet",
    "grid": "grid",
    "hazelnut": "hazelnut",
    "leather": "leather",
    "metal_nut": "metal nut",
    "pill": "pill",
    "screw": "screw",
    "tile": "tile",
    "toothbrush": "toothbrush",
    "transistor": "transistor",
    "wood": "wood",
    "zipper": "zipper",
}

N_IMAGES_PER_CATEGORY = 4


def get_sample_images(category, prompt, processor, n_samples):
    good_dir = os.path.join(DATASET_ROOT, category, "good")
    if not os.path.exists(good_dir):
        return []

    images = sorted(
        [f for f in os.listdir(good_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    if not images:
        return []

    sample_images = random.sample(images, min(n_samples, len(images)))
    results = []

    for img_name in sample_images:
        img_path = os.path.join(good_dir, img_name)
        image = Image.open(img_path).convert("RGB")

        state = processor.set_image(image)
        state = processor.set_text_prompt(prompt=prompt, state=state)

        masks = state["masks"]
        boxes = state["boxes"]
        scores = state["scores"]

        img_np = np.array(image)

        mask_np = None
        box = None
        score = 0
        mask_area = 0

        if len(masks) > 0 and masks[0].numel() > 0:
            mask_np = masks[0, 0].cpu().float().numpy() > 0.5
            if mask_np.sum() > 0:
                mask_area = mask_np.sum()
                if len(boxes) > 0:
                    box = boxes[0].cpu().numpy()
                if len(scores) > 0:
                    score = scores[0].cpu().float().item()

        results.append({
            "img_np": img_np,
            "img_name": img_name,
            "mask_np": mask_np,
            "box": box,
            "score": score,
            "mask_area": mask_area,
        })

    return results


def main():
    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.5)

    categories = list(CATEGORY_PROMPTS.keys())
    n_categories = len(categories)

    fig, axes = plt.subplots(n_categories, N_IMAGES_PER_CATEGORY, figsize=(16, 60))

    for row_idx, category in enumerate(categories):
        prompt = CATEGORY_PROMPTS[category]
        print(f"Processing {category}...")

        results = get_sample_images(category, prompt, processor, N_IMAGES_PER_CATEGORY)

        for col_idx in range(N_IMAGES_PER_CATEGORY):
            ax = axes[row_idx, col_idx]

            if col_idx < len(results):
                result = results[col_idx]
                img_np = result["img_np"]
                mask_np = result["mask_np"]
                box = result["box"]
                score = result["score"]
                mask_area = result["mask_area"]
                img_name = result["img_name"]

                ax.imshow(img_np)

                if mask_np is not None and mask_np.sum() > 0:
                    overlay = np.zeros_like(img_np)
                    overlay[mask_np] = [255, 0, 0]
                    ax.imshow(overlay, alpha=0.4)

                    if box is not None:
                        x0, y0, x1, y1 = box.astype(int)
                        ax.add_patch(
                            plt.Rectangle(
                                (x0, y0), x1 - x0, y1 - y0, fill=False, edgecolor="lime", linewidth=2
                            )
                        )

                ax.set_title(f"{img_name}\nScore: {score:.3f}", fontsize=6)
            else:
                ax.set_title("No data", fontsize=6)
            ax.axis("off")

        # Category label on the left
        axes[row_idx, 0].set_ylabel(category, rotation=90, size=12, labelpad=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sam3_mask_all_categories.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, format="jpeg")
    print(f"Saved visualization to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
