"""Validate SAM3 box prompt masking on MVTec categories."""

import os
import random

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
PROMPT_BOX = [0.5, 0.5, 0.95, 0.95]


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


def run_box_prompt(model, processor, pil_img, box, confidence_threshold=0.5):
    with torch.inference_mode():
        state = processor.set_image(pil_img)

        text_outputs = model.backbone.forward_text(["visual"], device=model.device)
        state["backbone_out"].update(text_outputs)

        boxes = torch.tensor(box, device=model.device, dtype=torch.float32).view(1, 1, 4)
        labels = torch.tensor([True], device=model.device, dtype=torch.bool).view(1, 1)

        from sam3.model.data_misc import FindStage
        find_stage = FindStage(
            img_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        from sam3.model.geometry_encoders import Prompt
        geometric_prompt = Prompt(
            box_embeddings=boxes,
            box_mask=torch.zeros(1, 1, device=model.device, dtype=torch.bool),
            box_labels=labels,
        )
        state["geometric_prompt"] = geometric_prompt

        outputs = model.forward_grounding(
            backbone_out=state["backbone_out"],
            find_input=find_stage,
            geometric_prompt=geometric_prompt,
            find_target=None,
        )

        from sam3.model import box_ops
        from sam3.model.data_misc import interpolate

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep] if out_masks is not None else None
        out_bbox = out_bbox[keep]

        if out_probs.numel() == 0:
            return None, None, None

        width, height = pil_img.size
        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([width, height, width, height], device=model.device)
        boxes_xyxy = boxes_xyxy * scale_fct[None, :]

        if out_masks is not None:
            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (height, width),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()

        return out_masks, boxes_xyxy, out_probs


def main():
    print("Loading SAM3 model...")
    model = build_sam3_image_model(
        checkpoint_path=CHECKPOINT,
        load_from_HF=False,
    )
    processor = Sam3Processor(model, confidence_threshold=0.5)

    n_categories = len(CATEGORIES)
    fig, axes = plt.subplots(n_categories, N_TARGETS, figsize=(16, 60))

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

            masks, boxes, scores = run_box_prompt(
                model, processor, target_pil, PROMPT_BOX
            )

            ax.imshow(target_np)

            h, w = target_pil.height, target_pil.width
            px0, py0, px1, py1 = (
                PROMPT_BOX[0] * w - PROMPT_BOX[2] * w / 2,
                PROMPT_BOX[1] * h - PROMPT_BOX[3] * h / 2,
                PROMPT_BOX[0] * w + PROMPT_BOX[2] * w / 2,
                PROMPT_BOX[1] * h + PROMPT_BOX[3] * h / 2,
            )
            ax.add_patch(
                plt.Rectangle(
                    (px0, py0), px1 - px0, py1 - py0,
                    fill=False, edgecolor="cyan", linewidth=1, linestyle="--"
                )
            )

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
        print(f"  {category}: {detections}/{N_TARGETS} detected, avg score: {avg_score:.3f}")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sam3_box_prompt_all_categories.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, format="jpeg")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
