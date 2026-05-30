"""Test DETR detection → SAM3 box prompt segmentation pipeline.

Flow: Image → RF-DETR detects objects → boxes → SAM3 segments each box.

Usage:
    python scripts/test_detr_sam3_pipeline.py [image_path] [--gui]

If no image path is given, picks a random image from the dataset.
Default mode is terminal-based (works on headless servers).
Add --gui for Tkinter window (requires a display).
"""

import os
import random
import sys

import numpy as np
import torch
from PIL import Image

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights")
SAM3_CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam3", "sam3.pt")
DETR_WEIGHTS = os.path.join(WEIGHTS_DIR, "rf-detr", "rf-detr-seg-nano.pt")
DATASET_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "..", "dataset", "glitch_hunt_mvtec")
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "detr_sam3")

CATEGORIES = [
    "bottle", "cable", "capsule", "carpet", "grid",
    "hazelnut", "leather", "metal_nut", "pill", "screw",
    "tile", "toothbrush", "transistor", "wood", "zipper",
]


def has_display():
    return os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY")


def pick_random_image():
    if not os.path.isdir(DATASET_ROOT):
        return None
    for root, dirs, files in os.walk(DATASET_ROOT):
        if os.path.basename(root) == "good":
            imgs = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
            if imgs:
                return os.path.join(root, random.choice(imgs))
    return None


def load_detr_model():
    from rfdetr import RFDETRSegNano
    return RFDETRSegNano(pretrain_weights=DETR_WEIGHTS)


def load_sam3_model():
    from sam3.model_builder import build_sam3_image_model
    from sam3.model.sam3_image_processor import Sam3Processor

    model = build_sam3_image_model(checkpoint_path=SAM3_CHECKPOINT, load_from_HF=False)
    processor = Sam3Processor(model, confidence_threshold=0.5)
    return model, processor


def run_sam3_box_inference(model, processor, pil_img, box, confidence_threshold=0.5):
    """Run SAM3 segmentation on a single box prompt.

    Args:
        box: (x0, y0, x1, y1) in pixel coordinates.

    Returns:
        (mask_np, score) or (None, None) if no detection.
    """
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

        keep = out_probs > confidence_threshold
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


def save_result_image(pil_img, detections, output_path):
    """Save image with DETR boxes and SAM3 mask overlays.

    Args:
        detections: list of dicts with 'box', 'mask', 'score', 'label'.
    """
    img = pil_img.copy().convert("RGBA")

    for det in detections:
        mask = det.get("mask")
        if mask is not None:
            mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
            overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
            overlay_data = overlay.load()
            mask_data = mask_img.load()
            w, h = img.size
            for y in range(h):
                for x in range(w):
                    if mask_data[x, y] > 0:
                        overlay_data[x, y] = (0, 255, 0, 100)
            img = Image.alpha_composite(img, overlay)

    img_rgb = img.convert("RGB")

    from PIL import ImageDraw
    draw = ImageDraw.Draw(img_rgb)

    for det in detections:
        x0, y0, x1, y1 = [int(v) for v in det["box"]]
        color = (0, 0, 255)
        draw.rectangle([x0, y0, x1, y1], outline=color, width=2)

        label = det.get("label", "object")
        score = det.get("score", 0)
        text = f"{label} {score:.2f}"
        bbox = draw.textbbox((x0, y0 - 20), text)
        draw.rectangle(bbox, fill=color)
        draw.text((x0, y0 - 20), text, fill=(255, 255, 255))

    img_rgb.save(output_path)
    return output_path


class TerminalPipelineTester:
    def __init__(self, image_path, detr_threshold=0.5, sam3_threshold=0.5):
        self.image_path = image_path
        self.detr_threshold = detr_threshold
        self.sam3_threshold = sam3_threshold
        self.pil_img = Image.open(image_path).convert("RGB")
        self.detections = []

        print(f"\nImage: {image_path}")
        print(f"Size: {self.pil_img.size[0]}x{self.pil_img.size[1]}")

        print("\nLoading RF-DETR nano model...")
        self.detr_model = load_detr_model()

        print("Loading SAM3 model...")
        self.sam3_model, self.sam3_processor = load_sam3_model()

        print("\nRunning DETR detection...")
        self._run_detr()

        if self.detections:
            print(f"\nFound {len(self.detections)} objects. Running SAM3 segmentation...")
            self._run_sam3()
        else:
            print("\nNo objects detected by DETR.")

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"detr_sam3_{base_name}.png")
        save_result_image(self.pil_img, self.detections, out_path)
        print(f"\nSaved result to {out_path}")

        self._print_summary()

    def _run_detr(self):
        detections = self.detr_model.predict(self.pil_img, threshold=self.detr_threshold)

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

            self.detections.append({
                "box": (x0, y0, x1, y1),
                "mask": None,
                "score": float(score),
                "label": label,
            })

            print(f"  DETR: {label} ({score:.3f}) box=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")

    def _run_sam3(self):
        for i, det in enumerate(self.detections):
            box = det["box"]
            print(f"  SAM3 [{i+1}/{len(self.detections)}] box=[{box[0]:.0f},{box[1]:.0f},{box[2]:.0f},{box[3]:.0f}]... ", end="")

            mask, score = run_sam3_box_inference(
                self.sam3_model, self.sam3_processor, self.pil_img, box, self.sam3_threshold
            )

            if mask is not None:
                det["mask"] = mask
                det["sam3_score"] = score
                print(f"score={score:.3f}")
            else:
                print("no segmentation")

    def _print_summary(self):
        print(f"\n{'='*50}")
        print(f"Summary: {len(self.detections)} objects detected")
        for i, det in enumerate(self.detections):
            has_mask = det.get("mask") is not None
            sam3_score = f"SAM3={det.get('sam3_score', 0):.3f}" if has_mask else "SAM3=N/A"
            mask_status = "masked" if has_mask else "no mask"
            print(f"  [{i+1}] {det['label']} DETR={det['score']:.3f} {sam3_score} {mask_status}")


class GUIPipelineTester:
    def __init__(self, root, image_path, detr_threshold=0.5, sam3_threshold=0.5):
        import tkinter as tk
        from tkinter import ttk
        from PIL import ImageTk

        self.root = root
        self.image_path = image_path
        self.detr_threshold = detr_threshold
        self.sam3_threshold = sam3_threshold
        self.pil_img = Image.open(image_path).convert("RGB")
        self.detections = []
        self.current_step = 0

        self.root.title("DETR → SAM3 Pipeline Tester")
        self.root.geometry("900x700")

        control_frame = ttk.Frame(root, padding=10)
        control_frame.pack(fill=tk.X)

        self.status_label = ttk.Label(control_frame, text="Ready", font=("TkDefaultFont", 10))
        self.status_label.pack(side=tk.LEFT)

        self.step_btn = ttk.Button(control_frame, text="Next Step", command=self._next_step)
        self.step_btn.pack(side=tk.RIGHT)

        self.auto_btn = ttk.Button(control_frame, text="Run All", command=self._run_all)
        self.auto_btn.pack(side=tk.RIGHT, padx=5)

        self.canvas = tk.Canvas(root, bg="gray")
        self.canvas.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

        self._show_image(self.pil_img)

        print(f"\nImage: {image_path}")
        print("Loading models...")
        self.detr_model = load_detr_model()
        self.sam3_model, self.sam3_processor = load_sam3_model()
        print("Models loaded.")

    def _show_image(self, img):
        display_w = self.canvas.winfo_width() or 800
        display_h = self.canvas.winfo_height() or 600

        iw, ih = img.size
        scale = min(display_w / iw, display_h / ih, 1.0)
        new_w, new_h = int(iw * scale), int(ih * scale)

        resized = img.resize((new_w, new_h), Image.LANCZOS)
        self.tk_img = ImageTk.PhotoImage(resized)
        self.canvas.create_image(display_w // 2, display_h // 2, anchor=tk.CENTER, image=self.tk_img)

    def _next_step(self):
        if self.current_step == 0:
            self._run_detr()
            self.current_step = 1
        elif self.current_step == 1:
            self._run_sam3()
            self.current_step = 2
        else:
            os.makedirs(OUTPUT_DIR, exist_ok=True)
            base_name = os.path.splitext(os.path.basename(self.image_path))[0]
            out_path = os.path.join(OUTPUT_DIR, f"detr_sam3_{base_name}.png")
            save_result_image(self.pil_img, self.detections, out_path)
            self.status_label.config(text=f"Saved to {out_path}")
            print(f"Saved to {out_path}")

    def _run_all(self):
        self._run_detr()
        self._run_sam3()
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base_name = os.path.splitext(os.path.basename(self.image_path))[0]
        out_path = os.path.join(OUTPUT_DIR, f"detr_sam3_{base_name}.png")
        save_result_image(self.pil_img, self.detections, out_path)
        self.status_label.config(text=f"Done — {len(self.detections)} objects — saved to {out_path}")
        print(f"Saved to {out_path}")

    def _run_detr(self):
        self.status_label.config(text="Running DETR detection...")
        self.root.update()

        detections = self.detr_model.predict(self.pil_img, threshold=self.detr_threshold)

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

            self.detections.append({
                "box": (x0, y0, x1, y1),
                "mask": None,
                "score": float(score),
                "label": label,
            })

            print(f"  DETR: {label} ({score:.3f}) box=[{x0:.0f},{y0:.0f},{x1:.0f},{y1:.0f}]")

        self.status_label.config(text=f"DETR found {len(self.detections)} objects")
        self._draw_boxes()

    def _draw_boxes(self):
        img = self.pil_img.copy()
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img)

        for det in self.detections:
            x0, y0, x1, y1 = [int(v) for v in det["box"]]
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=3)

        self._show_image(img)

    def _run_sam3(self):
        self.status_label.config(text="Running SAM3 segmentation...")
        self.root.update()

        for i, det in enumerate(self.detections):
            print(f"  SAM3 [{i+1}/{len(self.detections)}]... ", end="")
            mask, score = run_sam3_box_inference(
                self.sam3_model, self.sam3_processor, self.pil_img, det["box"], self.sam3_threshold
            )
            if mask is not None:
                det["mask"] = mask
                det["sam3_score"] = score
                print(f"score={score:.3f}")
            else:
                print("no segmentation")

        self._draw_masks()
        self.status_label.config(text=f"SAM3 done — {sum(1 for d in self.detections if d.get('mask'))} masked")

    def _draw_masks(self):
        img = self.pil_img.copy().convert("RGBA")

        for det in self.detections:
            if det.get("mask") is not None:
                mask = det["mask"]
                mask_img = Image.fromarray((mask * 255).astype(np.uint8)).convert("L")
                overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
                overlay_data = overlay.load()
                mask_data = mask_img.load()
                w, h = img.size
                for y in range(h):
                    for x in range(w):
                        if mask_data[x, y] > 0:
                            overlay_data[x, y] = (0, 255, 0, 100)
                img = Image.alpha_composite(img, overlay)

        img_rgb = img.convert("RGB")
        from PIL import ImageDraw
        draw = ImageDraw.Draw(img_rgb)
        for det in self.detections:
            x0, y0, x1, y1 = [int(v) for v in det["box"]]
            draw.rectangle([x0, y0, x1, y1], outline=(0, 0, 255), width=3)

        self._show_image(img_rgb)


def main():
    use_gui = "--gui" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--gui"]
    image_path = args[0] if args else pick_random_image()

    if not image_path:
        print("No dataset found. Please pass an image path as argument.")
        sys.exit(1)

    if not os.path.exists(image_path):
        print(f"Image not found: {image_path}")
        sys.exit(1)

    if use_gui and has_display():
        import tkinter as tk
        root = tk.Tk()
        GUIPipelineTester(root, image_path)
        root.mainloop()
    else:
        if use_gui and not has_display():
            print("Warning: --gui requested but no display available, falling back to terminal mode.")
        TerminalPipelineTester(image_path)


if __name__ == "__main__":
    main()
