"""Run DETR detection → SAM3 segmentation across all MVTec categories.

Produces a single image grid: one row per category, 4 images per row.
DETR boxes shown in blue, SAM3 masks in green.

Usage:
    python scripts/test_detr_sam3_all_categories.py
"""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
SAM3_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam3", "sam3.pt")
DETR_WEIGHTS = os.path.join(WEIGHTS_DIR, "rf-detr", "rf-detr-seg-large.pt")
DATASET_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "..", "dataset", "glitch_hunt_mvtec")
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "detr_sam3")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]

N_TARGETS = 4
DETR_THRESHOLD = 0.5
SAM3_THRESHOLD = 0.5


def get_image_paths(category, n):
    good_dir = os.path.join(DATASET_ROOT, category, "good")
    if not os.path.exists(good_dir):
        return []
    images = sorted(
        [f for f in os.listdir(good_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
    )
    return random.sample(images, min(n, len(images)))


def load_detr_model():
    from rfdetr import RFDETRSegLarge
    return RFDETRSegLarge(pretrain_weights=DETR_WEIGHTS)


def load_sam3_model():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=SAM3_THRESHOLD)
    return model, processor


def run_detr(model, pil_img, threshold=DETR_THRESHOLD):
    """Run DETR detection, return list of (box, label, score)."""
    detections = model.predict(pil_img, threshold=threshold)
    results = []

    for det in detections:
        if isinstance(det, tuple):
            box, mask, score, class_id, tracker_id, meta = det
            x0, y0, x1, y1 = box.astype(float)
            label = meta.get("class_name", "") if isinstance(meta, dict) else ""
            if not label:
                label = f"class_{class_id}"
        elif isinstance(det, dict):
            box = det.get("box") or det.get("bbox")
            if box is None:
                continue
            if isinstance(box, dict):
                x0 = box.get("x0", box.get("xmin", 0))
                y0 = box.get("y0", box.get("ymin", 0))
                x1 = box.get("x1", box.get("xmax", 0))
                y1 = box.get("y1", box.get("ymax", 0))
            elif isinstance(box, (list, tuple)):
                x0, y0, x1, y1 = box
            else:
                continue
            label = det.get("label", det.get("class", "object"))
            score = det.get("score", det.get("confidence", 0))
        else:
            continue

        results.append({
            "box": (x0, y0, x1, y1),
            "label": label,
            "score": float(score),
        })

    return results


def run_sam3_box(model, processor, pil_img, box, threshold=SAM3_THRESHOLD):
    """Run SAM3 on a single box, return (mask_np, score) or (None, None)."""
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage, interpolate
    from sam3.model.geometry_encoders import Prompt

    w, h = pil_img.size

    with torch.inference_mode():
        state = processor.set_image(pil_img)

        text_outputs = model.backbone.forward_text(["visual"], device=model.device)
        state["backbone_out"].update(text_outputs)

        boxes_norm = torch.tensor(
            [
                (box[0] + box[2]) / 2 / w,
                (box[1] + box[3]) / 2 / h,
                (box[2] - box[0]) / w,
                (box[3] - box[1]) / h,
            ],
            device=model.device,
            dtype=torch.float32,
        ).view(1, 1, 4)

        labels = torch.tensor([True], device=model.device, dtype=torch.bool).view(1, 1)

        find_stage = FindStage(
            img_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        geometric_prompt = Prompt(
            box_embeddings=boxes_norm,
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

        out_bbox = outputs["pred_boxes"]
        out_logits = outputs["pred_logits"]
        out_masks = outputs["pred_masks"]
        out_probs = out_logits.sigmoid()
        presence_score = outputs["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep] if out_masks is not None else None
        out_bbox = out_bbox[keep]

        if out_probs.numel() == 0:
            return None, None

        boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([w, h, w, h], device=model.device)
        boxes_xyxy = boxes_xyxy * scale_fct[None, :]

        if out_masks is not None:
            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (h, w),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()

        mask_np = out_masks[0, 0].cpu().float().numpy() > 0.5
        score = out_probs[0].cpu().float().item()
        return mask_np, score


def main():
    print("Loading RF-DETR nano model...")
    detr_model = load_detr_model()

    print("Loading SAM3 model...")
    sam3_model, sam3_processor = load_sam3_model()

    n_categories = len(CATEGORIES)
    fig, axes = plt.subplots(n_categories, N_TARGETS, figsize=(16, 60))

    for row_idx, category in enumerate(CATEGORIES):
        print(f"\nProcessing {category}...")

        target_names = get_image_paths(category, N_TARGETS)

        stats = {"detr_detections": 0, "sam3_masks": 0, "detr_scores": [], "sam3_scores": []}

        for col_idx, target_name in enumerate(target_names):
            ax = axes[row_idx, col_idx]
            target_path = os.path.join(DATASET_ROOT, category, "good", target_name)
            target_pil = Image.open(target_path).convert("RGB")
            target_np = np.array(target_pil)

            ax.imshow(target_np)

            detr_results = run_detr(detr_model, target_pil, DETR_THRESHOLD)

            if detr_results:
                stats["detr_detections"] += 1
                stats["detr_scores"].extend([d["score"] for d in detr_results])

                for det in detr_results:
                    x0, y0, x1, y1 = [int(v) for v in det["box"]]
                    ax.add_patch(
                        plt.Rectangle(
                            (x0, y0), x1 - x0, y1 - y0,
                            fill=False, edgecolor="blue", linewidth=2,
                        )
                    )

                    mask, sam3_score = run_sam3_box(
                        sam3_model, sam3_processor, target_pil, det["box"], SAM3_THRESHOLD
                    )

                    if mask is not None:
                        stats["sam3_masks"] += 1
                        stats["sam3_scores"].append(sam3_score)

                        overlay = np.zeros_like(target_np)
                        overlay[mask] = [0, 255, 0]
                        ax.imshow(overlay, alpha=0.4)

            title_lines = [target_name]
            if detr_results:
                best_detr = max(d["score"] for d in detr_results)
                title_lines.append(f"DETR={best_detr:.2f}")
                if stats["sam3_scores"]:
                    best_sam3 = max(stats["sam3_scores"][-len(detr_results):])
                    title_lines.append(f"SAM3={best_sam3:.2f}")
            else:
                title_lines.append("No detection")

            ax.set_title("\n".join(title_lines), fontsize=6)
            ax.axis("off")

            print(f"  {target_name}: {len(detr_results)} objects, {stats['sam3_masks']} masked")

        det_rate = stats["detr_detections"] / N_TARGETS if N_TARGETS > 0 else 0
        avg_detr = np.mean(stats["detr_scores"]) if stats["detr_scores"] else 0
        avg_sam3 = np.mean(stats["sam3_scores"]) if stats["sam3_scores"] else 0
        print(f"  {category}: {stats['detr_detections']}/{N_TARGETS} detected, "
              f"{stats['sam3_masks']} masked, avg DETR={avg_detr:.3f}, avg SAM3={avg_sam3:.3f}")

        axes[row_idx, 0].set_ylabel(category, rotation=90, size=12, labelpad=10)

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "detr_sam3_all_categories.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, format="jpeg")
    print(f"\nSaved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
