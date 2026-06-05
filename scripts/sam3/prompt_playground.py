"""Tkinter playground for SAM3 prompt APIs (text, points, and box)."""

from __future__ import annotations

import argparse
import contextlib
import os
import random
import tkinter as tk
from tkinter import filedialog, ttk

import numpy as np
import torch
from PIL import Image, ImageTk

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
DATASET_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "..", "dataset", "glitch_hunt_mvtec")
)
CHECKPOINT_DEFAULT = os.path.join(PROJECT_ROOT, "weights", "sam3", "sam3.pt")

DISPLAY_MAX_SIZE = 1000
POINT_FG = 1
POINT_BG = 0


def pick_random_image() -> str | None:
    if not os.path.exists(DATASET_ROOT):
        return None
    candidates: list[str] = []
    for root, _, files in os.walk(DATASET_ROOT):
        if os.path.basename(root) != "good":
            continue
        for name in files:
            if name.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff")):
                candidates.append(os.path.join(root, name))
    if not candidates:
        return None
    return random.choice(candidates)


def _autocast_context() -> contextlib.AbstractContextManager:
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


def run_text_prompt(
    processor: Sam3Processor,
    pil_img: Image.Image,
    text_prompt: str,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    with torch.inference_mode(), _autocast_context():
        state = processor.set_image(pil_img)
        state = processor.set_text_prompt(text_prompt, state)
        masks = state.get("masks")
        boxes = state.get("boxes")
        scores = state.get("scores")

    if masks is None or masks.numel() == 0:
        return None, None, None

    mask_np = masks[0, 0].detach().cpu().float().numpy() > 0.5
    box_np = None
    if boxes is not None and boxes.numel() > 0:
        box_np = boxes[0].detach().cpu().numpy().astype(np.float32)
    score = None
    if scores is not None and scores.numel() > 0:
        score = float(scores[0].detach().cpu().item())
    return mask_np, box_np, score


def run_point_prompt(
    model,
    processor: Sam3Processor,
    pil_img: Image.Image,
    points_norm: list[tuple[float, float, int]],
    confidence_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage, interpolate

    with torch.inference_mode(), _autocast_context():
        width, height = pil_img.size
        state = processor.set_image(pil_img)
        text_outputs = model.backbone.forward_text(["visual"], device=model.device)
        state["backbone_out"].update(text_outputs)

        if points_norm:
            pts = torch.tensor(points_norm, device=model.device, dtype=torch.float32).view(1, -1, 3)
            pts_mask = torch.zeros(1, len(points_norm), device=model.device, dtype=torch.bool)
        else:
            pts = None
            pts_mask = None

        find_stage = FindStage(
            img_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            text_ids=torch.tensor([0], device=model.device, dtype=torch.long),
            input_boxes=None,
            input_boxes_mask=None,
            input_boxes_label=None,
            input_points=None,
            input_points_mask=pts_mask,
            input_points_before_embed=pts,
        )

        geometric_prompt = model._get_geo_prompt_from_find_input(find_stage)
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

    mask_np = out_masks[0, 0].detach().cpu().float().numpy() > 0.5 if out_masks is not None else None
    box_np = boxes_xyxy[0].detach().cpu().numpy().astype(np.float32) if boxes_xyxy.numel() > 0 else None
    score = float(out_probs[0].detach().cpu().item()) if out_probs.numel() > 0 else None
    return mask_np, box_np, score


def run_box_prompt(
    model,
    processor: Sam3Processor,
    pil_img: Image.Image,
    box_xyxy_norm: tuple[float, float, float, float],
    confidence_threshold: float,
) -> tuple[np.ndarray | None, np.ndarray | None, float | None]:
    from sam3.model import box_ops
    from sam3.model.data_misc import FindStage, interpolate
    from sam3.model.geometry_encoders import Prompt

    x0, y0, x1, y1 = box_xyxy_norm
    cx = (x0 + x1) * 0.5
    cy = (y0 + y1) * 0.5
    bw = max(1e-6, x1 - x0)
    bh = max(1e-6, y1 - y0)

    with torch.inference_mode(), _autocast_context():
        width, height = pil_img.size
        state = processor.set_image(pil_img)
        text_outputs = model.backbone.forward_text(["visual"], device=model.device)
        state["backbone_out"].update(text_outputs)

        boxes = torch.tensor([cx, cy, bw, bh], device=model.device, dtype=torch.float32).view(1, 1, 4)
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
            box_embeddings=boxes,
            box_mask=torch.zeros(1, 1, device=model.device, dtype=torch.bool),
            box_labels=labels,
        )

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

    mask_np = out_masks[0, 0].detach().cpu().float().numpy() > 0.5 if out_masks is not None else None
    box_np = boxes_xyxy[0].detach().cpu().numpy().astype(np.float32) if boxes_xyxy.numel() > 0 else None
    score = float(out_probs[0].detach().cpu().item()) if out_probs.numel() > 0 else None
    return mask_np, box_np, score


class Sam3PromptPlayground:
    def __init__(self, root: tk.Tk, image_path: str, checkpoint_path: str, confidence_threshold: float):
        self.root = root
        self.root.title("SAM3 Prompt Playground")

        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = confidence_threshold

        self.model = None
        self.processor = None

        self.image_path = image_path
        self.pil_img = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.pil_img.size
        self.scale = 1.0
        self.disp_w = 1
        self.disp_h = 1
        self.disp_img = self.pil_img

        self.points: list[tuple[float, float, int]] = []
        self.box_px: tuple[int, int, int, int] | None = None
        self.drag_start: tuple[int, int] | None = None
        self.mask_overlay: np.ndarray | None = None
        self.pred_box: np.ndarray | None = None
        self.last_score: float | None = None

        self.mode_var = tk.StringVar(value="points")
        self.point_label_var = tk.IntVar(value=POINT_FG)
        self.text_prompt_var = tk.StringVar(value="object in foreground")

        self._recompute_display_image()
        self._build_ui()
        self._load_model()
        self._redraw()

    def _build_ui(self):
        controls = tk.Frame(self.root)
        controls.pack(side=tk.TOP, fill=tk.X, padx=6, pady=6)

        tk.Button(controls, text="Open Image", command=self._open_image).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="Clear Overlay", command=self._clear_overlay).pack(side=tk.LEFT, padx=2)
        tk.Button(controls, text="Clear Prompts", command=self._clear_prompts).pack(side=tk.LEFT, padx=2)

        tk.Label(controls, text="Mode").pack(side=tk.LEFT, padx=(10, 2))
        mode_box = ttk.Combobox(
            controls,
            textvariable=self.mode_var,
            values=["text", "points", "box"],
            width=8,
            state="readonly",
        )
        mode_box.pack(side=tk.LEFT, padx=2)
        mode_box.bind("<<ComboboxSelected>>", lambda _: self._on_mode_changed())

        tk.Button(controls, text="Run Active", command=self._run_active).pack(side=tk.LEFT, padx=6)

        self.status = tk.Label(controls, text="Loading model...", anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        text_frame = tk.Frame(self.root)
        text_frame.pack(side=tk.TOP, fill=tk.X, padx=6)
        tk.Label(text_frame, text="Text prompt").pack(side=tk.LEFT)
        self.text_entry = tk.Entry(text_frame, textvariable=self.text_prompt_var)
        self.text_entry.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=6)
        tk.Button(text_frame, text="Run Text", command=self._run_text_prompt).pack(side=tk.LEFT)

        point_frame = tk.Frame(self.root)
        point_frame.pack(side=tk.TOP, fill=tk.X, padx=6, pady=(4, 0))
        tk.Label(point_frame, text="Point label").pack(side=tk.LEFT)
        tk.Radiobutton(point_frame, text="Foreground", variable=self.point_label_var, value=POINT_FG).pack(
            side=tk.LEFT, padx=4
        )
        tk.Radiobutton(point_frame, text="Background", variable=self.point_label_var, value=POINT_BG).pack(
            side=tk.LEFT, padx=4
        )
        tk.Label(point_frame, text="(Right-click always adds background point)").pack(side=tk.LEFT, padx=10)

        self.canvas = tk.Canvas(
            self.root,
            width=self.disp_w,
            height=self.disp_h,
            bg="#202020",
            cursor="crosshair",
        )
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True, padx=6, pady=6)
        self.canvas.bind("<Button-1>", self._on_left_down)
        self.canvas.bind("<B1-Motion>", self._on_left_drag)
        self.canvas.bind("<ButtonRelease-1>", self._on_left_up)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Button-2>", self._on_right_click)

    def _load_model(self):
        self._set_status("Loading SAM3 model...")
        self.root.update()
        self.model = build_sam3_image_model(checkpoint_path=self.checkpoint_path, load_from_HF=False)
        self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
        self._set_status("Model loaded. Use mode selector for text/points/box.")

    def _set_status(self, text: str):
        self.status.config(text=text)

    def _recompute_display_image(self):
        self.orig_w, self.orig_h = self.pil_img.size
        self.scale = min(DISPLAY_MAX_SIZE / self.orig_w, DISPLAY_MAX_SIZE / self.orig_h, 1.0)
        self.disp_w = max(1, int(round(self.orig_w * self.scale)))
        self.disp_h = max(1, int(round(self.orig_h * self.scale)))
        self.disp_img = self.pil_img.resize((self.disp_w, self.disp_h), Image.LANCZOS)

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif *.tiff")]
        )
        if not path:
            return
        self.image_path = path
        self.pil_img = Image.open(path).convert("RGB")
        self._recompute_display_image()
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self._clear_prompts(clear_overlay=True)
        self._set_status(f"Loaded {os.path.basename(path)}")
        self._redraw()

    def _clear_prompts(self, clear_overlay: bool = False):
        self.points = []
        self.box_px = None
        self.drag_start = None
        self.pred_box = None
        if clear_overlay:
            self.mask_overlay = None
            self.last_score = None
        self._redraw()

    def _clear_overlay(self):
        self.mask_overlay = None
        self.pred_box = None
        self.last_score = None
        self._set_status("Overlay cleared")
        self._redraw()

    def _on_mode_changed(self):
        mode = self.mode_var.get()
        if mode == "text":
            self._set_status("Text mode: edit prompt and click Run Text or Run Active")
        elif mode == "points":
            self._set_status("Points mode: left click adds selected label, right click adds background")
        else:
            self._set_status("Box mode: drag left mouse to draw a box")

    def _to_orig_xy(self, disp_x: int, disp_y: int) -> tuple[int, int]:
        x = int(np.clip(round(disp_x / self.scale), 0, self.orig_w - 1))
        y = int(np.clip(round(disp_y / self.scale), 0, self.orig_h - 1))
        return x, y

    def _to_disp_xy(self, orig_x: int, orig_y: int) -> tuple[int, int]:
        x = int(np.clip(round(orig_x * self.scale), 0, self.disp_w - 1))
        y = int(np.clip(round(orig_y * self.scale), 0, self.disp_h - 1))
        return x, y

    def _on_left_down(self, event):
        mode = self.mode_var.get()
        if mode == "points":
            self._add_point(event.x, event.y, self.point_label_var.get())
            return
        if mode == "box":
            self.drag_start = (event.x, event.y)

    def _on_left_drag(self, event):
        if self.mode_var.get() != "box" or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        x1, y1 = event.x, event.y
        ox0, oy0 = self._to_orig_xy(x0, y0)
        ox1, oy1 = self._to_orig_xy(x1, y1)
        self.box_px = (min(ox0, ox1), min(oy0, oy1), max(ox0, ox1), max(oy0, oy1))
        self._redraw()

    def _on_left_up(self, event):
        if self.mode_var.get() != "box" or self.drag_start is None:
            return
        x0, y0 = self.drag_start
        self.drag_start = None
        ox0, oy0 = self._to_orig_xy(x0, y0)
        ox1, oy1 = self._to_orig_xy(event.x, event.y)
        self.box_px = (min(ox0, ox1), min(oy0, oy1), max(ox0, ox1), max(oy0, oy1))
        self._run_box_prompt()

    def _on_right_click(self, event):
        if self.mode_var.get() != "points":
            return
        self._add_point(event.x, event.y, POINT_BG)

    def _add_point(self, disp_x: int, disp_y: int, label: int):
        ox, oy = self._to_orig_xy(disp_x, disp_y)
        x_norm = ox / max(1, self.orig_w)
        y_norm = oy / max(1, self.orig_h)
        self.points.append((x_norm, y_norm, int(label)))
        self._run_point_prompt()

    def _run_active(self):
        mode = self.mode_var.get()
        if mode == "text":
            self._run_text_prompt()
        elif mode == "points":
            self._run_point_prompt()
        else:
            self._run_box_prompt()

    def _run_text_prompt(self):
        prompt = self.text_prompt_var.get().strip()
        if not prompt:
            self._set_status("Text prompt is empty")
            return
        self._set_status("Running text prompt...")
        self.root.update()
        mask_np, box_np, score = run_text_prompt(self.processor, self.pil_img, prompt)
        self._apply_prediction(mask_np, box_np, score, f"text='{prompt}'")

    def _run_point_prompt(self):
        if not self.points:
            self._set_status("No points yet")
            self._redraw()
            return
        self._set_status(f"Running points prompt ({len(self.points)} points)...")
        self.root.update()
        mask_np, box_np, score = run_point_prompt(
            self.model,
            self.processor,
            self.pil_img,
            self.points,
            self.confidence_threshold,
        )
        self._apply_prediction(mask_np, box_np, score, f"points={len(self.points)}")

    def _run_box_prompt(self):
        if self.box_px is None:
            self._set_status("No box yet - drag on image in box mode")
            self._redraw()
            return
        x0, y0, x1, y1 = self.box_px
        if abs(x1 - x0) < 2 or abs(y1 - y0) < 2:
            self._set_status("Box too small")
            self._redraw()
            return
        box_norm = (
            x0 / max(1, self.orig_w),
            y0 / max(1, self.orig_h),
            x1 / max(1, self.orig_w),
            y1 / max(1, self.orig_h),
        )
        self._set_status("Running box prompt...")
        self.root.update()
        mask_np, box_np, score = run_box_prompt(
            self.model,
            self.processor,
            self.pil_img,
            box_norm,
            self.confidence_threshold,
        )
        self._apply_prediction(mask_np, box_np, score, f"box=({x0},{y0},{x1},{y1})")

    def _apply_prediction(
        self,
        mask_np: np.ndarray | None,
        box_np: np.ndarray | None,
        score: float | None,
        tag: str,
    ):
        self.mask_overlay = mask_np
        self.pred_box = box_np
        self.last_score = score
        if score is None:
            self._set_status(f"No detection ({tag})")
        else:
            self._set_status(f"Score={score:.3f} ({tag})")
        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")

        base_photo = ImageTk.PhotoImage(self.disp_img)
        self.canvas.create_image(0, 0, image=base_photo, anchor=tk.NW)
        self.canvas._base_photo = base_photo

        if self.mask_overlay is not None:
            mask_resized = Image.fromarray((self.mask_overlay.astype(np.uint8) * 255)).resize(
                (self.disp_w, self.disp_h), Image.NEAREST
            )
            rgba = Image.new("RGBA", (self.disp_w, self.disp_h), (0, 0, 0, 0))
            red = Image.new("RGBA", (self.disp_w, self.disp_h), (255, 0, 0, 100))
            rgba.paste(red, mask=mask_resized)
            mask_photo = ImageTk.PhotoImage(rgba)
            self.canvas.create_image(0, 0, image=mask_photo, anchor=tk.NW)
            self.canvas._mask_photo = mask_photo

        for x_norm, y_norm, label in self.points:
            ox = int(round(x_norm * self.orig_w))
            oy = int(round(y_norm * self.orig_h))
            dx, dy = self._to_disp_xy(ox, oy)
            color = "#00ff00" if label == POINT_FG else "#ff0000"
            self.canvas.create_oval(dx - 4, dy - 4, dx + 4, dy + 4, fill=color, outline="white", width=1)

        if self.box_px is not None:
            x0, y0, x1, y1 = self.box_px
            dx0, dy0 = self._to_disp_xy(x0, y0)
            dx1, dy1 = self._to_disp_xy(x1, y1)
            self.canvas.create_rectangle(dx0, dy0, dx1, dy1, outline="#00d5ff", width=2, dash=(4, 3))

        if self.pred_box is not None and len(self.pred_box) == 4:
            x0, y0, x1, y1 = [int(v) for v in self.pred_box]
            dx0, dy0 = self._to_disp_xy(x0, y0)
            dx1, dy1 = self._to_disp_xy(x1, y1)
            self.canvas.create_rectangle(dx0, dy0, dx1, dy1, outline="#00ff66", width=2)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 Tkinter prompt playground")
    parser.add_argument("image", nargs="?", help="Optional image path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT, help="Path to sam3.pt checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="Confidence threshold")
    return parser.parse_args()


def main():
    args = parse_args()
    image_path = args.image
    if image_path is None:
        image_path = pick_random_image()
    if image_path is None:
        raise SystemExit("No image found. Pass an image path explicitly.")
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")
    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    root = tk.Tk()
    Sam3PromptPlayground(
        root=root,
        image_path=image_path,
        checkpoint_path=args.checkpoint,
        confidence_threshold=float(args.confidence_threshold),
    )
    root.mainloop()


if __name__ == "__main__":
    main()
