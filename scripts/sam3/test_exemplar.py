"""Validate SAM3 exemplar+box masking on MVTec categories."""

import os
import random

import matplotlib.pyplot as plt
import numpy as np
import torch
from PIL import Image
from torchvision.transforms import v2

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

N_EXEMPLARS = 4
N_TARGETS = 4
EXEMPLAR_BOX = [0.5, 0.5, 0.6, 0.6]


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


def run_grounding_with_box(model, processor, pil_img, box=None, confidence_threshold=0.5):
    """Run SAM3 grounding on an image with optional box prompt."""
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage, interpolate

    with torch.inference_mode():
        width, height = pil_img.size
        state = processor.set_image(pil_img)

        text_outputs = model.backbone.forward_text(["visual"], device=model.device)
        state["backbone_out"].update(text_outputs)

        find_stage = FindStage(
            img_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        geometric_prompt = model._get_dummy_prompt()
        if box is not None:
            boxes = torch.tensor(box, device=model.device, dtype=torch.float32).view(1, 1, 4)
            labels = torch.tensor([True], device=model.device, dtype=torch.bool).view(1, 1)
            geometric_prompt.append_boxes(boxes, labels)

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

        keep = out_probs > confidence_threshold
        out_probs = out_probs[keep]
        out_masks = out_masks[keep] if out_masks is not None else None
        out_bbox = out_bbox[keep]

        if out_probs.numel() == 0:
            return None, None, None

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([width, height, width, height], device=model.device)
        boxes = boxes * scale_fct[None, :]

        if out_masks is not None:
            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (height, width),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()

        return out_masks, boxes, out_probs


def extract_exemplar_features(exemplar_pils, model, processor):
    """Run exemplar images with center box to get grounded visual features."""
    transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.uint8, scale=True),
        v2.Resize(size=(processor.resolution, processor.resolution)),
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    features_list = []
    with torch.inference_mode():
        for pil_img in exemplar_pils:
            masks, boxes, scores = run_grounding_with_box(
                model, processor, pil_img, box=EXEMPLAR_BOX
            )

            img_tensor = transform(pil_img).unsqueeze(0).to(model.device)
            backbone_out = model.backbone.forward_image(img_tensor)
            vis_feats_raw = backbone_out["backbone_fpn"][-1]
            feat_h, feat_w = vis_feats_raw.shape[-2:]
            vis_feats = vis_feats_raw.flatten(-2).permute(2, 0, 1)

            if masks is not None and masks.numel() > 0:
                mask_np = masks[0, 0].cpu().float().numpy() > 0.5
                if mask_np.sum() > 0:
                    mask_resized = torch.nn.functional.interpolate(
                        torch.from_numpy(mask_np).float().unsqueeze(0).unsqueeze(0).to(model.device),
                        size=(feat_h, feat_w),
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze()
                    vis_feats_masked = vis_feats * mask_resized.view(-1, 1, 1)
                    features_list.append(vis_feats_masked)
                    continue

            features_list.append(vis_feats)

    combined = torch.cat(features_list, dim=0)
    mask = torch.zeros(1, combined.shape[0], dtype=torch.bool, device=combined.device)
    return combined, mask


def run_target_with_exemplar(target_pil, model, processor, visual_embed, visual_mask, confidence_threshold=0.5):
    """Run target image using exemplar visual features as visual prompt."""
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage, interpolate

    with torch.inference_mode():
        width, height = target_pil.size
        transform = v2.Compose([
            v2.ToImage(),
            v2.ToDtype(torch.uint8, scale=True),
            v2.Resize(size=(processor.resolution, processor.resolution)),
            v2.ToDtype(torch.float32, scale=True),
            v2.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
        ])

        img_tensor = transform(target_pil).unsqueeze(0).to(model.device)
        backbone_out = model.backbone.forward_image(img_tensor)

        inst_interactivity_en = model.inst_interactive_predictor is not None
        if inst_interactivity_en and "sam2_backbone_out" in backbone_out:
            sam2_backbone_out = backbone_out["sam2_backbone_out"]
            sam2_backbone_out["backbone_fpn"][0] = (
                model.inst_interactive_predictor.model.sam_mask_decoder.conv_s0(
                    sam2_backbone_out["backbone_fpn"][0]
                )
            )
            sam2_backbone_out["backbone_fpn"][1] = (
                model.inst_interactive_predictor.model.sam_mask_decoder.conv_s1(
                    sam2_backbone_out["backbone_fpn"][1]
                )
            )

        find_stage = FindStage(
            img_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=None,
        )

        geometric_prompt = model._get_dummy_prompt()

        text_outputs = model.backbone.forward_text(["visual"], device=model.device)
        backbone_out.update(text_outputs)

        prompt, prompt_mask, backbone_out = model._encode_prompt(
            backbone_out,
            find_stage,
            geometric_prompt,
            visual_prompt_embed=visual_embed,
            visual_prompt_mask=visual_mask,
            encode_text=True,
        )

        backbone_out, encoder_out, _ = model._run_encoder(
            backbone_out, find_stage, prompt, prompt_mask
        )

        out = {
            "encoder_hidden_states": encoder_out["encoder_hidden_states"],
            "prev_encoder_out": {
                "encoder_out": encoder_out,
                "backbone_out": backbone_out,
            },
        }

        out, hs = model._run_decoder(
            memory=out["encoder_hidden_states"],
            pos_embed=encoder_out["pos_embed"],
            src_mask=encoder_out["padding_mask"],
            out=out,
            prompt=prompt,
            prompt_mask=prompt_mask,
            encoder_out=encoder_out,
        )

        seg_img_ids = find_stage.img_ids
        if "id_mapping" in backbone_out and backbone_out["id_mapping"] is not None:
            seg_img_ids = backbone_out["id_mapping"][seg_img_ids]
        model._run_segmentation_heads(
            out=out,
            backbone_out=backbone_out,
            img_ids=seg_img_ids,
            vis_feat_sizes=encoder_out["vis_feat_sizes"],
            encoder_hidden_states=out["encoder_hidden_states"],
            prompt=prompt,
            prompt_mask=prompt_mask,
            hs=hs,
        )

        out_bbox = out["pred_boxes"]
        out_logits = out["pred_logits"]
        out_masks = out.get("pred_masks")

        if out_logits.numel() == 0:
            return None, None, None

        out_probs = out_logits.sigmoid()
        presence_score = out["presence_logit_dec"].sigmoid().unsqueeze(1)
        out_probs = (out_probs * presence_score).squeeze(-1)

        keep = out_probs > confidence_threshold
        if keep.sum() == 0:
            return None, None, None

        out_probs = out_probs[keep]
        out_masks = out_masks[keep] if out_masks is not None else None
        out_bbox = out_bbox[keep]

        boxes = box_ops.box_cxcywh_to_xyxy(out_bbox)
        scale_fct = torch.tensor([width, height, width, height], device=model.device)
        boxes = boxes * scale_fct[None, :]

        if out_masks is not None:
            out_masks = interpolate(
                out_masks.unsqueeze(1),
                (height, width),
                mode="bilinear",
                align_corners=False,
            ).sigmoid()

        return out_masks, boxes, out_probs


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

        exemplar_names = get_image_paths(category, N_EXEMPLARS)
        exemplar_pils = [load_image(os.path.join(DATASET_ROOT, category, "good", n)) for n in exemplar_names]

        visual_embed, visual_mask = extract_exemplar_features(exemplar_pils, model, processor)

        target_names = get_image_paths(category, N_TARGETS)

        detections = 0
        total_targets = len(target_names)
        scores_list = []

        for col_idx, target_name in enumerate(target_names):
            ax = axes[row_idx, col_idx]
            target_path = os.path.join(DATASET_ROOT, category, "good", target_name)
            target_pil = load_image(target_path)
            target_np = np.array(target_pil)

            masks, boxes, scores = run_target_with_exemplar(
                target_pil, model, processor, visual_embed, visual_mask
            )

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
        det_rate = detections / total_targets if total_targets > 0 else 0
        avg_score = np.mean(scores_list) if scores_list else 0
        print(f"  {category}: {detections}/{total_targets} detected, avg score: {avg_score:.3f}")

    plt.tight_layout()
    out_path = os.path.join(OUTPUT_DIR, "sam3_exemplar_box_mask_all_categories.jpg")
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    plt.savefig(out_path, dpi=150, format="jpeg")
    print(f"Saved to {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
