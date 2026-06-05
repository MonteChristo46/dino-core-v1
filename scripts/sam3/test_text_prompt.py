"""Validate SAM3 text prompt masking on MVTec categories."""

import os
import random
import sys

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights", "sam3")
DATASET_ROOT = os.path.join(PROJECT_ROOT, "..", "..", "dataset", "glitch_hunt_mvtec")
CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam3.pt")
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "sam3")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

N_TARGETS = 4


def get_image_paths(category, n):
    good_dir = os.path.join(DATASET_ROOT, category, "good")
    if not os.path.exists(good_dir):
        return []
    images = sorted(
        [f for f in os.listdir(good_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    return random.sample(images, min(n, len(images)))


def load_image(path):
    return Image.open(path).convert("RGB")


def run_text_prompt(model, processor, pil_img, text_prompt):
    with torch.inference_mode():
        state = processor.set_image(pil_img)
        state = processor.set_text_prompt(text_prompt, state)

        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

        if masks is None or masks.numel() == 0:
            return None, None, None

        return masks, boxes, scores


def main():
    if len(sys.argv) < 2:
        print(f"Usage: uv run {sys.argv[0]} \"<text prompt>\"")
        print(f"Example: uv run {sys.argv[0]} \"objects in foreground\"")
        sys.exit(1)

    text_prompt = sys.argv[1]

    print(f"Loading SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.5)

    n_categories = len(CATEGORIES)
    fig, axes = plt.subplots(n_categories, N_TARGETS, figsize=(16, 60))
    fig.suptitle(f'Text prompt: "{text_prompt}"', fontsize=16, y=0.995)

    total_detections = 0
    total_images = 0
    all_scores = []

    for row_idx, category in enumerate(CATEGORIES):
        print(f"Processing {category}...")

        target_names = get_image_paths(category, N_TARGETS)

        detections = 0
        scores_list = []

        for col_idx, target_name in enumerate(target_names):
            ax = axes[row_idx, col_idx]
            target_path = os.path.join(DATASET_ROOT, category, "good", target_name)
            target_pil = load_image(target_path)
            target_np = np.array(target_pil)

            masks, boxes, scores = run_text_prompt(model, processor, target_pil, text_prompt)

            ax.imshow(target_np)

            if masks is not None and masks.numel() > 0:
                detections += 1
                mask_np = masks[0, 0].cpu().float().numpy() > 0.5
                if mask_np.sum() > 0:
                    overlay = np.zeros_like(target_np)
                    overlay[mask_np] = [255, 0, 0]
                    ax.imshow(overlay, alpha=0.4)

                if boxes is not None and boxes.numel() > 0:
                    x0, y0, x1, y1 = boxes[0].cpu().numpy().astype(int)
                    ax.add_patch(
                        plt.Rectangle(
                            (x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor="lime", linewidth=2
                        )
                    )

                score_val = scores[0].cpu().float().item()
                scores_list.append(score_val)
                ax.set_title(f"{target_name}\nScore: {score_val:.3f}", fontsize=6)
            else:
                ax.set_title(f"{target_name}\nNo detection", fontsize=6)

            ax.axis("off")

        axes[row_idx, 0].set_ylabel(category, rotation=90, size=12, labelpad=10)
        det_rate = detections / N_TARGETS if N_TARGETS > 0 else 0
        avg_score = np.mean(scores_list) if scores_list else 0
        total_detections += detections
        total_images += len(target_names)
        all_scores.extend(scores_list)
        print(f"  {category}: {detections}/{len(target_names)} detected, avg score: {avg_score:.3f}")

    overall_rate = total_detections / total_images if total_images > 0 else 0
    overall_avg = np.mean(all_scores) if all_scores else 0
    print(f"\nOverall: {total_detections}/{total_images} detected ({overall_rate:.1%}), avg score: {overall_avg:.3f}")

    plt.tight_layout()
    safe_name = text_prompt.replace(" ", "_").replace("/", "_")[:50]
    out_path = os.path.join(OUTPUT_DIR, f"sam3_text_prompt_{safe_name}.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, format="jpeg")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
