"""Headless-friendly SAM3 prompt playground served in the browser via Gradio.

Supports three prompt modes:
- text prompt
- click points (FG/BG)
- box prompt (normalized xyxy)
"""

from __future__ import annotations

import argparse
import contextlib
import os
import random
from dataclasses import dataclass
from typing import Any

import gradio as gr
import numpy as np
import torch
from PIL import Image

from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

if torch.cuda.is_available():
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
DATASET_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "..", "dataset", "glitch_hunt_mvtec")
)
CHECKPOINT_DEFAULT = os.path.join(PROJECT_ROOT, "weights", "sam3", "sam3.pt")

FG_LABEL = 1
BG_LABEL = 0

APP_CSS = """
#click-image img {
    -webkit-user-drag: none !important;
    user-select: none !important;
    -webkit-user-select: none !important;
}
#click-image {
    touch-action: none;
}
"""


def _autocast_context() -> contextlib.AbstractContextManager[Any]:
    if torch.cuda.is_available():
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return contextlib.nullcontext()


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


def list_class_representatives() -> dict[str, str]:
    """Return one representative 'good' image per MVTec class."""
    reps: dict[str, str] = {}
    if not os.path.exists(DATASET_ROOT):
        return reps

    for class_name in sorted(os.listdir(DATASET_ROOT)):
        class_dir = os.path.join(DATASET_ROOT, class_name)
        if not os.path.isdir(class_dir):
            continue
        good_dir = os.path.join(class_dir, "good")
        if not os.path.isdir(good_dir):
            continue

        names = sorted(
            n
            for n in os.listdir(good_dir)
            if n.lower().endswith((".png", ".jpg", ".jpeg", ".bmp", ".tif", ".tiff"))
        )
        if not names:
            continue
        reps[class_name] = os.path.join(good_dir, names[0])

    return reps


@dataclass
class Sam3InferenceResult:
    mask: np.ndarray | None
    box_xyxy: np.ndarray | None
    score: float | None


class Sam3Runner:
    def __init__(self, checkpoint_path: str, confidence_threshold: float, device: str = "auto"):
        self.checkpoint_path = checkpoint_path
        self.confidence_threshold = float(confidence_threshold)
        self.device_preference = device
        self.device_in_use: str | None = None
        self._force_device: str | None = None
        self.model = None
        self.processor = None

    def _candidate_devices(self) -> list[str]:
        if self._force_device is not None:
            return [self._force_device]
        if self.device_preference == "cpu":
            return ["cpu"]
        if self.device_preference == "cuda":
            return ["cuda"]
        if torch.cuda.is_available():
            return ["cuda", "cpu"]
        return ["cpu"]

    def ensure_loaded(self):
        if self.model is not None and self.processor is not None:
            return
        last_error: Exception | None = None
        for device_name in self._candidate_devices():
            try:
                if device_name == "cpu" and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                self.model = build_sam3_image_model(
                    checkpoint_path=self.checkpoint_path,
                    load_from_HF=False,
                    device=device_name,
                )
                self.processor = Sam3Processor(self.model, confidence_threshold=self.confidence_threshold)
                self.device_in_use = device_name
                self._force_device = None
                return
            except torch.OutOfMemoryError as exc:
                last_error = exc
                self.model = None
                self.processor = None
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                continue
            except RuntimeError as exc:
                last_error = exc
                self.model = None
                self.processor = None
                if "out of memory" in str(exc).lower() and torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    continue
                raise

        msg = "Failed to initialize SAM3 model"
        if last_error is not None:
            msg = f"{msg}: {last_error}"
        raise RuntimeError(msg)

    def _switch_to_cpu(self):
        self.model = None
        self.processor = None
        self.device_in_use = None
        self._force_device = "cpu"
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _run_with_cpu_fallback(self, fn):
        try:
            return fn()
        except torch.OutOfMemoryError:
            if self.device_in_use == "cuda":
                self._switch_to_cpu()
                self.ensure_loaded()
                return fn()
            raise
        except RuntimeError as exc:
            if "out of memory" in str(exc).lower() and self.device_in_use == "cuda":
                self._switch_to_cpu()
                self.ensure_loaded()
                return fn()
            raise

    def run_text(self, pil_img: Image.Image, text_prompt: str) -> Sam3InferenceResult:
        def _run() -> Sam3InferenceResult:
            self.ensure_loaded()
            with torch.inference_mode(), _autocast_context():
                state = self.processor.set_image(pil_img)
                state = self.processor.set_text_prompt(text_prompt, state)
                masks = state.get("masks")
                boxes = state.get("boxes")
                scores = state.get("scores")

            if masks is None or masks.numel() == 0:
                return Sam3InferenceResult(None, None, None)

            mask = masks[0, 0].detach().cpu().float().numpy() > 0.5
            box = boxes[0].detach().cpu().numpy().astype(np.float32) if boxes is not None and boxes.numel() > 0 else None
            score = float(scores[0].detach().cpu().item()) if scores is not None and scores.numel() > 0 else None
            return Sam3InferenceResult(mask, box, score)

        return self._run_with_cpu_fallback(_run)

    def run_points(self, pil_img: Image.Image, points_norm: list[list[float]]) -> Sam3InferenceResult:
        def _run() -> Sam3InferenceResult:
            self.ensure_loaded()
            from sam3.model import box_ops
            from sam3.model.data_misc import FindStage, interpolate

            with torch.inference_mode(), _autocast_context():
                width, height = pil_img.size
                state = self.processor.set_image(pil_img)

                text_outputs = self.model.backbone.forward_text(["visual"], device=self.model.device)
                state["backbone_out"].update(text_outputs)

                if points_norm:
                    points_tensor = torch.tensor(points_norm, device=self.model.device, dtype=torch.float32).view(1, -1, 3)
                    points_mask = torch.zeros(1, len(points_norm), device=self.model.device, dtype=torch.bool)
                else:
                    points_tensor = None
                    points_mask = None

                find_stage = FindStage(
                    img_ids=torch.tensor([0], device=self.model.device, dtype=torch.long),
                    text_ids=torch.tensor([0], device=self.model.device, dtype=torch.long),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=points_mask,
                    input_points_before_embed=points_tensor,
                )

                geometric_prompt = self.model._get_geo_prompt_from_find_input(find_stage)
                out = self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=find_stage,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )

                out_bbox = out["pred_boxes"]
                out_logits = out["pred_logits"]
                out_masks = out["pred_masks"]

                out_probs = out_logits.sigmoid()
                presence = out["presence_logit_dec"].sigmoid().unsqueeze(1)
                out_probs = (out_probs * presence).squeeze(-1)

                keep = out_probs > self.confidence_threshold
                out_probs = out_probs[keep]
                out_bbox = out_bbox[keep]
                out_masks = out_masks[keep] if out_masks is not None else None

                if out_probs.numel() == 0:
                    return Sam3InferenceResult(None, None, None)

                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
                scale = torch.tensor([width, height, width, height], device=self.model.device)
                boxes_xyxy = boxes_xyxy * scale[None, :]

                if out_masks is not None:
                    out_masks = interpolate(
                        out_masks.unsqueeze(1),
                        (height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()

            mask = out_masks[0, 0].detach().cpu().float().numpy() > 0.5 if out_masks is not None else None
            box = boxes_xyxy[0].detach().cpu().numpy().astype(np.float32) if boxes_xyxy.numel() > 0 else None
            score = float(out_probs[0].detach().cpu().item()) if out_probs.numel() > 0 else None
            return Sam3InferenceResult(mask, box, score)

        return self._run_with_cpu_fallback(_run)

    def run_box(self, pil_img: Image.Image, box_xyxy_norm: tuple[float, float, float, float]) -> Sam3InferenceResult:
        def _run() -> Sam3InferenceResult:
            self.ensure_loaded()
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
                state = self.processor.set_image(pil_img)

                text_outputs = self.model.backbone.forward_text(["visual"], device=self.model.device)
                state["backbone_out"].update(text_outputs)

                box_tensor = torch.tensor([cx, cy, bw, bh], device=self.model.device, dtype=torch.float32).view(1, 1, 4)
                box_labels = torch.tensor([True], device=self.model.device, dtype=torch.bool).view(1, 1)

                find_stage = FindStage(
                    img_ids=torch.tensor([0], device=self.model.device, dtype=torch.long),
                    text_ids=torch.tensor([0], device=self.model.device, dtype=torch.long),
                    input_boxes=None,
                    input_boxes_mask=None,
                    input_boxes_label=None,
                    input_points=None,
                    input_points_mask=None,
                )

                geometric_prompt = Prompt(
                    box_embeddings=box_tensor,
                    box_mask=torch.zeros(1, 1, device=self.model.device, dtype=torch.bool),
                    box_labels=box_labels,
                )

                out = self.model.forward_grounding(
                    backbone_out=state["backbone_out"],
                    find_input=find_stage,
                    geometric_prompt=geometric_prompt,
                    find_target=None,
                )

                out_bbox = out["pred_boxes"]
                out_logits = out["pred_logits"]
                out_masks = out["pred_masks"]

                out_probs = out_logits.sigmoid()
                presence = out["presence_logit_dec"].sigmoid().unsqueeze(1)
                out_probs = (out_probs * presence).squeeze(-1)

                keep = out_probs > self.confidence_threshold
                out_probs = out_probs[keep]
                out_bbox = out_bbox[keep]
                out_masks = out_masks[keep] if out_masks is not None else None

                if out_probs.numel() == 0:
                    return Sam3InferenceResult(None, None, None)

                boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
                scale = torch.tensor([width, height, width, height], device=self.model.device)
                boxes_xyxy = boxes_xyxy * scale[None, :]

                if out_masks is not None:
                    out_masks = interpolate(
                        out_masks.unsqueeze(1),
                        (height, width),
                        mode="bilinear",
                        align_corners=False,
                    ).sigmoid()

            mask = out_masks[0, 0].detach().cpu().float().numpy() > 0.5 if out_masks is not None else None
            box = boxes_xyxy[0].detach().cpu().numpy().astype(np.float32) if boxes_xyxy.numel() > 0 else None
            score = float(out_probs[0].detach().cpu().item()) if out_probs.numel() > 0 else None
            return Sam3InferenceResult(mask, box, score)

        return self._run_with_cpu_fallback(_run)

    def _run_grounded(
        self,
        pil_img: Image.Image,
        text_prompt: str,
        points_norm: list[list[float]],
        box_xyxy_norm: tuple[float, float, float, float] | None,
    ) -> Sam3InferenceResult:
        self.ensure_loaded()
        from sam3.model import box_ops
        from sam3.model.data_misc import FindStage, interpolate

        with torch.inference_mode(), _autocast_context():
            width, height = pil_img.size
            state = self.processor.set_image(pil_img)

            text_query = text_prompt.strip() if text_prompt.strip() else "visual"
            text_outputs = self.model.backbone.forward_text([text_query], device=self.model.device)
            state["backbone_out"].update(text_outputs)

            if points_norm:
                points_tensor = torch.tensor(points_norm, device=self.model.device, dtype=torch.float32).view(1, -1, 3)
                points_mask = torch.zeros(1, len(points_norm), device=self.model.device, dtype=torch.bool)
            else:
                points_tensor = None
                points_mask = None

            find_stage = FindStage(
                img_ids=torch.tensor([0], device=self.model.device, dtype=torch.long),
                text_ids=torch.tensor([0], device=self.model.device, dtype=torch.long),
                input_boxes=None,
                input_boxes_mask=None,
                input_boxes_label=None,
                input_points=None,
                input_points_mask=points_mask,
                input_points_before_embed=points_tensor,
            )

            geometric_prompt = self.model._get_geo_prompt_from_find_input(find_stage)

            if box_xyxy_norm is not None:
                x0, y0, x1, y1 = box_xyxy_norm
                cx = (x0 + x1) * 0.5
                cy = (y0 + y1) * 0.5
                bw = max(1e-6, x1 - x0)
                bh = max(1e-6, y1 - y0)
                box_tensor = torch.tensor([cx, cy, bw, bh], device=self.model.device, dtype=torch.float32).view(1, 1, 4)
                box_labels = torch.tensor([True], device=self.model.device, dtype=torch.bool).view(1, 1)
                geometric_prompt.append_boxes(box_tensor, box_labels)

            out = self.model.forward_grounding(
                backbone_out=state["backbone_out"],
                find_input=find_stage,
                geometric_prompt=geometric_prompt,
                find_target=None,
            )

            out_bbox = out["pred_boxes"]
            out_logits = out["pred_logits"]
            out_masks = out["pred_masks"]

            out_probs = out_logits.sigmoid()
            presence = out["presence_logit_dec"].sigmoid().unsqueeze(1)
            out_probs = (out_probs * presence).squeeze(-1)

            keep = out_probs > self.confidence_threshold
            out_probs = out_probs[keep]
            out_bbox = out_bbox[keep]
            out_masks = out_masks[keep] if out_masks is not None else None

            if out_probs.numel() == 0:
                return Sam3InferenceResult(None, None, None)

            boxes_xyxy = box_ops.box_cxcywh_to_xyxy(out_bbox)
            scale = torch.tensor([width, height, width, height], device=self.model.device)
            boxes_xyxy = boxes_xyxy * scale[None, :]

            if out_masks is not None:
                out_masks = interpolate(
                    out_masks.unsqueeze(1),
                    (height, width),
                    mode="bilinear",
                    align_corners=False,
                ).sigmoid()

        mask = out_masks[0, 0].detach().cpu().float().numpy() > 0.5 if out_masks is not None else None
        box = boxes_xyxy[0].detach().cpu().numpy().astype(np.float32) if boxes_xyxy.numel() > 0 else None
        score = float(out_probs[0].detach().cpu().item()) if out_probs.numel() > 0 else None
        return Sam3InferenceResult(mask, box, score)

    def run_combined(
        self,
        pil_img: Image.Image,
        text_prompt: str,
        points_norm: list[list[float]],
        box_xyxy_norm: tuple[float, float, float, float] | None,
    ) -> Sam3InferenceResult:
        points_norm = _normalize_points(points_norm)
        has_points = len(points_norm) > 0
        has_box = box_xyxy_norm is not None
        has_text = bool(text_prompt.strip())

        if has_points or has_box:
            return self._run_with_cpu_fallback(
                lambda: self._run_grounded(
                    pil_img=pil_img,
                    text_prompt=text_prompt,
                    points_norm=points_norm,
                    box_xyxy_norm=box_xyxy_norm,
                )
            )
        if has_text:
            return self.run_text(pil_img, text_prompt)
        return Sam3InferenceResult(None, None, None)


def _overlay_preview(
    pil_img: Image.Image,
    mask: np.ndarray | None,
    pred_box: np.ndarray | None,
    points: list[list[float]] | None,
    input_box_norm: tuple[float, float, float, float] | None,
    box_anchor_xy: list[int] | None = None,
) -> np.ndarray:
    img = np.array(pil_img).copy()
    h, w = img.shape[:2]

    if mask is not None and mask.shape[:2] == (h, w):
        red = np.zeros_like(img)
        red[..., 0] = 255
        alpha = 0.35
        img[mask] = (img[mask] * (1.0 - alpha) + red[mask] * alpha).astype(np.uint8)

    if input_box_norm is not None:
        x0 = int(np.clip(round(input_box_norm[0] * w), 0, w - 1))
        y0 = int(np.clip(round(input_box_norm[1] * h), 0, h - 1))
        x1 = int(np.clip(round(input_box_norm[2] * w), 0, w - 1))
        y1 = int(np.clip(round(input_box_norm[3] * h), 0, h - 1))
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        img[y0 : y1 + 1, x0 : x0 + 2] = [0, 213, 255]
        img[y0 : y1 + 1, x1 - 1 : x1 + 1] = [0, 213, 255]
        img[y0 : y0 + 2, x0 : x1 + 1] = [0, 213, 255]
        img[y1 - 1 : y1 + 1, x0 : x1 + 1] = [0, 213, 255]

    if box_anchor_xy is not None:
        ax = int(np.clip(box_anchor_xy[0], 0, w - 1))
        ay = int(np.clip(box_anchor_xy[1], 0, h - 1))
        y0, y1 = max(0, ay - 8), min(h, ay + 9)
        x0, x1 = max(0, ax - 8), min(w, ax + 9)
        img[ay : ay + 1, x0:x1] = [255, 215, 0]
        img[y0:y1, ax : ax + 1] = [255, 215, 0]

    if pred_box is not None and len(pred_box) == 4:
        x0, y0, x1, y1 = [int(np.clip(round(v), 0, w - 1)) for v in pred_box]
        x0, x1 = min(x0, x1), max(x0, x1)
        y0, y1 = min(y0, y1), max(y0, y1)
        img[y0 : y1 + 1, x0 : x0 + 2] = [0, 255, 102]
        img[y0 : y1 + 1, x1 - 1 : x1 + 1] = [0, 255, 102]
        img[y0 : y0 + 2, x0 : x1 + 1] = [0, 255, 102]
        img[y1 - 1 : y1 + 1, x0 : x1 + 1] = [0, 255, 102]

    for point in points or []:
        x_norm, y_norm, label = point
        px = int(np.clip(round(float(x_norm) * w), 0, w - 1))
        py = int(np.clip(round(float(y_norm) * h), 0, h - 1))
        color = np.array([0, 255, 0], dtype=np.uint8) if int(label) == FG_LABEL else np.array([255, 0, 0], dtype=np.uint8)
        y0, y1 = max(0, py - 4), min(h, py + 5)
        x0, x1 = max(0, px - 4), min(w, px + 5)
        img[y0:y1, x0:x1] = color

    return img


def _extract_click_xy(evt: gr.SelectData) -> tuple[int, int] | None:
    idx = getattr(evt, "index", None)
    if isinstance(idx, (list, tuple)) and len(idx) >= 2:
        return int(idx[0]), int(idx[1])
    value = getattr(evt, "value", None)
    if isinstance(value, dict):
        if "x" in value and "y" in value:
            return int(value["x"]), int(value["y"])
    return None


def _normalize_box_anchor(box_anchor: Any) -> list[int] | None:
    if box_anchor is None:
        return None
    arr = np.asarray(box_anchor, dtype=np.float64).reshape(-1)
    if arr.size < 2:
        return None
    return [int(round(float(arr[0]))), int(round(float(arr[1])))]


def _normalize_box_norm(box_norm: Any) -> list[float] | None:
    if box_norm is None:
        return None
    arr = np.asarray(box_norm, dtype=np.float64).reshape(-1)
    if arr.size < 4:
        return None
    return [float(arr[0]), float(arr[1]), float(arr[2]), float(arr[3])]


def _normalize_points(points: Any) -> list[list[float]]:
    if points is None:
        return []
    normalized: list[list[float]] = []
    for point in points:
        arr = np.asarray(point, dtype=np.float64).reshape(-1)
        if arr.size < 3:
            continue
        normalized.append([float(arr[0]), float(arr[1]), int(round(float(arr[2])))])
    return normalized


def _prompt_recipe(text_prompt: str, points: list[list[float]], box_norm: list[float] | None) -> str:
    parts: list[str] = []
    if text_prompt.strip():
        parts.append("text")
    if points:
        parts.append(f"points:{len(points)}")
    if box_norm is not None:
        parts.append("box")
    return "+".join(parts) if parts else "none"


def _format_prompt_state(points: list[list[float]], box_norm: list[float] | None, box_anchor: list[int] | None) -> str:
    points = _normalize_points(points)
    box_norm = _normalize_box_norm(box_norm)
    box_anchor = _normalize_box_anchor(box_anchor)

    lines = ["### Prompt State"]
    lines.append(f"- Points: **{len(points)}**")
    if points:
        last = points[-1]
        label = "FG" if int(last[2]) == FG_LABEL else "BG"
        lines.append(f"- Last point: **{label}** at ({last[0]:.3f}, {last[1]:.3f})")
    else:
        lines.append("- Last point: _(none)_")

    if box_anchor is not None:
        lines.append(f"- Box anchor A: ({int(box_anchor[0])}, {int(box_anchor[1])}) px")
    else:
        lines.append("- Box anchor A: _(none)_")

    if box_norm is not None:
        lines.append(
            f"- Box: x0={box_norm[0]:.3f}, y0={box_norm[1]:.3f}, x1={box_norm[2]:.3f}, y1={box_norm[3]:.3f}"
        )
    else:
        lines.append("- Box: _(none)_")

    return "\n".join(lines)


def build_app(
    runner: Sam3Runner,
    initial_image: Image.Image,
    class_to_image: dict[str, str],
    initial_class: str | None,
):
    with gr.Blocks(title="SAM3 Web Prompt Playground") as demo:
        gr.Markdown(
            "# SAM3 Prompt Playground\n"
            "Use this on a headless server via SSH tunnel.\n\n"
            "**Quick start:**\n"
            "1) Choose class and load image\n"
            "2) Choose mode (`text`, `points`, `box`)\n"
            "3) Interact on image and read status under controls"
        )

        points_state = gr.State([])
        box_state = gr.State(None)
        box_anchor_state = gr.State(None)
        active_mode_state = gr.State("points")

        class_names = sorted(class_to_image.keys())
        selected_class = initial_class if initial_class in class_to_image else (class_names[0] if class_names else None)

        with gr.Row():
            class_dropdown = gr.Dropdown(
                choices=class_names,
                value=selected_class,
                label="MVTec class",
                allow_custom_value=False,
            )
            prev_class_button = gr.Button("Prev class")
            next_class_button = gr.Button("Next class")
            load_class_button = gr.Button("Load class image")

        with gr.Row():
            with gr.Column(scale=1):
                mode = gr.Radio(choices=["text", "points", "box"], value="points", label="Mode")
                mode_hint = gr.Markdown(
                    "**Points mode**: click image to add point and auto-run SAM3."
                )
                with gr.Row():
                    mode_text_button = gr.Button("Text mode")
                    mode_points_button = gr.Button("Points mode")
                    mode_box_button = gr.Button("Box mode")

                with gr.Group() as text_group:
                    text_prompt = gr.Textbox(
                        value="object in foreground",
                        label="Text prompt",
                        placeholder="Example: object in foreground",
                    )
                    run_text_button = gr.Button("Run Text Prompt", variant="primary")

                with gr.Group() as points_group:
                    point_type = gr.Radio(choices=["fg", "bg"], value="fg", label="Point type")
                    with gr.Row():
                        undo_point = gr.Button("Undo point")
                        clear_points = gr.Button("Clear points")

                with gr.Group() as box_group:
                    gr.Markdown("Click once for corner A, click again for corner B.")
                    with gr.Row():
                        clear_box = gr.Button("Clear box")
                        run_box_button = gr.Button("Run box again")

                with gr.Row():
                    run_button = gr.Button("Run Active Mode", variant="secondary")
                    clear_overlay = gr.Button("Clear overlay")
                    clear_all = gr.Button("Reset all")

                status = gr.Textbox(value="Ready", label="Status", interactive=False)
                prompt_state_md = gr.Markdown(_format_prompt_state([], None, None))

            with gr.Column(scale=2):
                gr.Markdown("**Interaction:** click on either image to add points or define box corners (left/original is usually most reliable).")
                with gr.Row():
                    image_input = gr.Image(type="pil", value=initial_image, label="Original image (click/upload/load)", elem_id="click-image")
                    image_preview = gr.Image(
                        type="numpy",
                        value=np.array(initial_image),
                        label="Overlay preview",
                        interactive=True,
                    )

        def _mode_hint(current_mode: str) -> str:
            if current_mode == "text":
                return "**Text mode**: type prompt and click `Run Text Prompt` (or press Enter)."
            elif current_mode == "points":
                return "**Points mode**: each click adds FG/BG point and runs SAM3 immediately."
            return "**Box mode**: click corner A, then click corner B to run SAM3."

        def on_mode_changed(current_mode: str):
            return current_mode, _mode_hint(current_mode), f"Mode switched to {current_mode}"

        def set_mode(mode_name: str):
            return mode_name, mode_name, _mode_hint(mode_name), f"Mode switched to {mode_name}"

        def on_image_changed(img: Image.Image | None):
            if img is None:
                return [], None, None, None, "Upload/select an image", _format_prompt_state([], None, None)
            return [], None, np.array(img), None, "Image loaded", _format_prompt_state([], None, None)

        def _load_class_image(class_name: str | None):
            if not class_name or class_name not in class_to_image:
                return None, None, [], None, None, "Class has no representative image", _format_prompt_state([], None, None)
            img = Image.open(class_to_image[class_name]).convert("RGB")
            preview = np.array(img)
            return img, preview, [], None, None, f"Loaded class '{class_name}'", _format_prompt_state([], None, None)

        def on_load_class(class_name: str | None):
            return _load_class_image(class_name)

        def _step_class(current_class: str | None, step: int):
            if not class_names:
                return None, None, None, [], None, None, "No class images found", _format_prompt_state([], None, None)
            if current_class not in class_names:
                new_class = class_names[0]
            else:
                idx = class_names.index(current_class)
                new_class = class_names[(idx + step) % len(class_names)]
            img, preview, points, box, anchor, msg, prompt_state = _load_class_image(new_class)
            return new_class, img, preview, points, box, anchor, msg, prompt_state

        def on_prev_class(current_class: str | None):
            return _step_class(current_class, -1)

        def on_next_class(current_class: str | None):
            return _step_class(current_class, 1)

        def on_click(
            img: Image.Image | None,
            active_mode: str,
            current_text_prompt: str,
            current_point_type: str,
            points: list[list[float]],
            box_norm: list[float] | None,
            box_anchor: list[int] | None,
            evt: gr.SelectData,
        ):
            points = _normalize_points(points)
            box_norm = _normalize_box_norm(box_norm)
            box_anchor = _normalize_box_anchor(box_anchor)

            if img is None:
                return points, box_norm, box_anchor, None, "No image", _format_prompt_state(points, box_norm, box_anchor)

            click_xy = _extract_click_xy(evt)
            if click_xy is None:
                preview = _overlay_preview(
                    img,
                    None,
                    None,
                    points,
                    tuple(box_norm) if box_norm is not None else None,
                    box_anchor_xy=box_anchor,
                )
                return points, box_norm, box_anchor, preview, "Could not read click coordinates", _format_prompt_state(points, box_norm, box_anchor)

            x, y = click_xy
            width, height = img.size
            x = int(np.clip(x, 0, width - 1))
            y = int(np.clip(y, 0, height - 1))

            if active_mode == "points":
                label = FG_LABEL if current_point_type == "fg" else BG_LABEL
                points = list(points) + [[x / max(1, width), y / max(1, height), label]]
                try:
                    result = runner.run_combined(
                        img,
                        text_prompt=current_text_prompt,
                        points_norm=points,
                        box_xyxy_norm=(tuple(box_norm) if box_norm is not None else None),
                    )
                except Exception as exc:
                    preview = _overlay_preview(
                        img,
                        None,
                        None,
                        points,
                        tuple(box_norm) if box_norm is not None else None,
                        box_anchor_xy=box_anchor,
                    )
                    return points, box_norm, box_anchor, preview, f"Inference error: {exc}", _format_prompt_state(points, box_norm, box_anchor)
                preview = _overlay_preview(
                    img,
                    result.mask,
                    result.box_xyxy,
                    points,
                    tuple(box_norm) if box_norm is not None else None,
                    box_anchor_xy=box_anchor,
                )
                if result.score is None:
                    return points, box_norm, box_anchor, preview, f"No detection with recipe={_prompt_recipe(current_text_prompt, points, box_norm)}", _format_prompt_state(points, box_norm, box_anchor)
                return points, box_norm, box_anchor, preview, f"Point added. score={result.score:.3f} | recipe={_prompt_recipe(current_text_prompt, points, box_norm)}", _format_prompt_state(points, box_norm, box_anchor)

            if active_mode == "box":
                x_norm = x / max(1, width)
                y_norm = y / max(1, height)
                if box_anchor is None:
                    box_anchor = [x, y]
                    preview = _overlay_preview(img, None, None, points, None, box_anchor_xy=box_anchor)
                    return points, box_norm, box_anchor, preview, "Box corner A set. Click second corner.", _format_prompt_state(points, box_norm, box_anchor)

                ax, ay = int(box_anchor[0]), int(box_anchor[1])
                box_anchor = None
                x0 = min(ax, x) / max(1, width)
                y0 = min(ay, y) / max(1, height)
                x1 = max(ax, x) / max(1, width)
                y1 = max(ay, y) / max(1, height)
                box_norm = [float(x0), float(y0), float(x1), float(y1)]

                if x1 - x0 < 1e-4 or y1 - y0 < 1e-4:
                    preview = _overlay_preview(img, None, None, points, tuple(box_norm), box_anchor_xy=box_anchor)
                    return points, box_norm, box_anchor, preview, "Invalid box, try again", _format_prompt_state(points, box_norm, box_anchor)

                try:
                    result = runner.run_combined(
                        img,
                        text_prompt=current_text_prompt,
                        points_norm=points,
                        box_xyxy_norm=(box_norm[0], box_norm[1], box_norm[2], box_norm[3]),
                    )
                except Exception as exc:
                    preview = _overlay_preview(img, None, None, points, tuple(box_norm), box_anchor_xy=box_anchor)
                    return points, box_norm, box_anchor, preview, f"Inference error: {exc}", _format_prompt_state(points, box_norm, box_anchor)
                preview = _overlay_preview(img, result.mask, result.box_xyxy, points, tuple(box_norm), box_anchor_xy=box_anchor)
                if result.score is None:
                    return points, box_norm, box_anchor, preview, f"No detection with recipe={_prompt_recipe(current_text_prompt, points, box_norm)}", _format_prompt_state(points, box_norm, box_anchor)
                return points, box_norm, box_anchor, preview, f"Box set. score={result.score:.3f} | recipe={_prompt_recipe(current_text_prompt, points, box_norm)}", _format_prompt_state(points, box_norm, box_anchor)

            preview = _overlay_preview(
                img,
                None,
                None,
                points,
                tuple(box_norm) if box_norm is not None else None,
                box_anchor_xy=box_anchor,
            )
            return points, box_norm, box_anchor, preview, "Text mode: click is ignored", _format_prompt_state(points, box_norm, box_anchor)

        def on_clear_points(img: Image.Image | None):
            if img is None:
                return [], None, "No image", _format_prompt_state([], None, None)
            return [], _overlay_preview(img, None, None, [], None), "Points cleared", _format_prompt_state([], None, None)

        def on_undo_point(img: Image.Image | None, points: list[list[float]], box_norm: list[float] | None):
            points = _normalize_points(points)
            box_norm = _normalize_box_norm(box_norm)
            if img is None:
                return points, None, "No image", _format_prompt_state(points, box_norm, None)
            if not points:
                preview = _overlay_preview(img, None, None, points, tuple(box_norm) if box_norm is not None else None)
                return points, preview, "No points to undo", _format_prompt_state(points, box_norm, None)
            points = list(points[:-1])
            preview = _overlay_preview(img, None, None, points, tuple(box_norm) if box_norm is not None else None)
            return points, preview, f"Point removed. Remaining={len(points)}", _format_prompt_state(points, box_norm, None)

        def on_clear_box(img: Image.Image | None, points: list[list[float]]):
            points = _normalize_points(points)
            if img is None:
                return None, None, None, "No image", _format_prompt_state(points, None, None)
            return None, None, _overlay_preview(img, None, None, points, None), "Box cleared", _format_prompt_state(points, None, None)

        def on_clear_overlay(img: Image.Image | None, points: list[list[float]], box_norm: list[float] | None):
            points = _normalize_points(points)
            box_norm = _normalize_box_norm(box_norm)
            if img is None:
                return None, "No image"
            preview = _overlay_preview(img, None, None, points, tuple(box_norm) if box_norm is not None else None)
            return preview, "Overlay cleared"

        def on_clear_all(img: Image.Image | None):
            if img is None:
                return [], None, None, None, "Reset complete", _format_prompt_state([], None, None)
            return [], None, None, np.array(img), "Reset complete", _format_prompt_state([], None, None)

        def on_run(
            img: Image.Image | None,
            current_mode: str,
            prompt: str,
            points: list[list[float]],
            box_norm: list[float] | None,
        ):
            points = _normalize_points(points)
            box_norm = _normalize_box_norm(box_norm)
            if img is None:
                return None, "No image"

            box_tuple = tuple(box_norm) if box_norm is not None else None
            if box_tuple is not None:
                bx0 = float(np.clip(min(box_tuple[0], box_tuple[2]), 0.0, 1.0))
                by0 = float(np.clip(min(box_tuple[1], box_tuple[3]), 0.0, 1.0))
                bx1 = float(np.clip(max(box_tuple[0], box_tuple[2]), 0.0, 1.0))
                by1 = float(np.clip(max(box_tuple[1], box_tuple[3]), 0.0, 1.0))
                box_tuple = (bx0, by0, bx1, by1)

            if not prompt.strip() and not points and box_tuple is None:
                return _overlay_preview(img, None, None, points, None), "No active prompt: add text, points, or box"

            try:
                result = runner.run_combined(
                    pil_img=img,
                    text_prompt=prompt,
                    points_norm=points,
                    box_xyxy_norm=box_tuple,
                )
            except Exception as exc:
                return _overlay_preview(img, None, None, points, box_tuple), f"Inference error: {exc}"

            preview = _overlay_preview(img, result.mask, result.box_xyxy, points, box_tuple)
            recipe = _prompt_recipe(prompt, points, box_norm)
            if result.score is None:
                return preview, f"No detection with recipe={recipe}"
            return preview, f"score={result.score:.3f} | recipe={recipe}"

        def on_run_text(img: Image.Image | None, prompt: str, points: list[list[float]], box_norm: list[float] | None):
            points = _normalize_points(points)
            box_norm = _normalize_box_norm(box_norm)
            if img is None:
                return None, "No image"
            box_tuple = tuple(box_norm) if box_norm is not None else None
            if not prompt.strip() and not points and box_tuple is None:
                return _overlay_preview(img, None, None, points, box_tuple), "No active prompt"
            try:
                result = runner.run_combined(
                    pil_img=img,
                    text_prompt=prompt,
                    points_norm=points,
                    box_xyxy_norm=box_tuple,
                )
            except Exception as exc:
                return _overlay_preview(img, None, None, points, box_tuple), f"Inference error: {exc}"
            preview = _overlay_preview(img, result.mask, result.box_xyxy, points, box_tuple)
            recipe = _prompt_recipe(prompt, points, box_norm)
            if result.score is None:
                return preview, f"No detection with recipe={recipe}"
            return preview, f"score={result.score:.3f} | recipe={recipe}"

        def on_run_box(img: Image.Image | None, prompt: str, points: list[list[float]], box_norm: list[float] | None):
            points = _normalize_points(points)
            box_norm = _normalize_box_norm(box_norm)
            if img is None:
                return None, "No image"
            if box_norm is None:
                return _overlay_preview(img, None, None, points, None), "No box yet (click two corners)"
            bx0 = float(np.clip(min(box_norm[0], box_norm[2]), 0.0, 1.0))
            by0 = float(np.clip(min(box_norm[1], box_norm[3]), 0.0, 1.0))
            bx1 = float(np.clip(max(box_norm[0], box_norm[2]), 0.0, 1.0))
            by1 = float(np.clip(max(box_norm[1], box_norm[3]), 0.0, 1.0))
            try:
                result = runner.run_combined(
                    pil_img=img,
                    text_prompt=prompt,
                    points_norm=points,
                    box_xyxy_norm=(bx0, by0, bx1, by1),
                )
            except Exception as exc:
                return _overlay_preview(img, None, None, points, (bx0, by0, bx1, by1)), f"Inference error: {exc}"
            preview = _overlay_preview(img, result.mask, result.box_xyxy, points, (bx0, by0, bx1, by1))
            recipe = _prompt_recipe(prompt, points, [bx0, by0, bx1, by1])
            if result.score is None:
                return preview, f"No detection with recipe={recipe}"
            return preview, f"score={result.score:.3f} | recipe={recipe}"

        mode.change(
            on_mode_changed,
            inputs=[mode],
            outputs=[active_mode_state, mode_hint, status],
        )
        mode_text_button.click(
            lambda: set_mode("text"),
            inputs=None,
            outputs=[active_mode_state, mode, mode_hint, status],
        )
        mode_points_button.click(
            lambda: set_mode("points"),
            inputs=None,
            outputs=[active_mode_state, mode, mode_hint, status],
        )
        mode_box_button.click(
            lambda: set_mode("box"),
            inputs=None,
            outputs=[active_mode_state, mode, mode_hint, status],
        )
        image_input.change(
            on_image_changed,
            inputs=[image_input],
            outputs=[points_state, box_state, box_anchor_state, image_preview, status, prompt_state_md],
        )
        prev_class_button.click(
            on_prev_class,
            inputs=[class_dropdown],
            outputs=[class_dropdown, image_input, image_preview, points_state, box_state, box_anchor_state, status, prompt_state_md],
        )
        next_class_button.click(
            on_next_class,
            inputs=[class_dropdown],
            outputs=[class_dropdown, image_input, image_preview, points_state, box_state, box_anchor_state, status, prompt_state_md],
        )
        load_class_button.click(
            on_load_class,
            inputs=[class_dropdown],
            outputs=[image_input, image_preview, points_state, box_state, box_anchor_state, status, prompt_state_md],
        )
        image_input.select(
            on_click,
            inputs=[image_input, active_mode_state, text_prompt, point_type, points_state, box_state, box_anchor_state],
            outputs=[points_state, box_state, box_anchor_state, image_preview, status, prompt_state_md],
        )
        image_preview.select(
            on_click,
            inputs=[image_input, active_mode_state, text_prompt, point_type, points_state, box_state, box_anchor_state],
            outputs=[points_state, box_state, box_anchor_state, image_preview, status, prompt_state_md],
        )
        clear_points.click(
            on_clear_points,
            inputs=[image_input],
            outputs=[points_state, image_preview, status, prompt_state_md],
        )
        undo_point.click(
            on_undo_point,
            inputs=[image_input, points_state, box_state],
            outputs=[points_state, image_preview, status, prompt_state_md],
        )
        clear_box.click(
            on_clear_box,
            inputs=[image_input, points_state],
            outputs=[box_state, box_anchor_state, image_preview, status, prompt_state_md],
        )
        clear_overlay.click(
            on_clear_overlay,
            inputs=[image_input, points_state, box_state],
            outputs=[image_preview, status],
        )
        clear_all.click(
            on_clear_all,
            inputs=[image_input],
            outputs=[points_state, box_state, box_anchor_state, image_preview, status, prompt_state_md],
        )
        run_button.click(
            on_run,
            inputs=[image_input, mode, text_prompt, points_state, box_state],
            outputs=[image_preview, status],
        )
        run_text_button.click(
            on_run_text,
            inputs=[image_input, text_prompt, points_state, box_state],
            outputs=[image_preview, status],
        )
        text_prompt.submit(
            on_run_text,
            inputs=[image_input, text_prompt, points_state, box_state],
            outputs=[image_preview, status],
        )
        run_box_button.click(
            on_run_box,
            inputs=[image_input, text_prompt, points_state, box_state],
            outputs=[image_preview, status],
        )

    return demo


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="SAM3 browser prompt playground")
    parser.add_argument("image", nargs="?", help="Optional initial image path")
    parser.add_argument("--checkpoint", default=CHECKPOINT_DEFAULT, help="Path to sam3 checkpoint")
    parser.add_argument("--confidence-threshold", type=float, default=0.5, help="SAM3 confidence threshold")
    parser.add_argument("--device", choices=["auto", "cuda", "cpu"], default="auto", help="Inference device")
    parser.add_argument("--host", default="127.0.0.1", help="Bind host")
    parser.add_argument("--port", type=int, default=7860, help="Bind port")
    parser.add_argument("--share", action="store_true", help="Enable Gradio share URL")
    return parser.parse_args()


def main():
    args = parse_args()

    if not os.path.exists(args.checkpoint):
        raise SystemExit(f"Checkpoint not found: {args.checkpoint}")

    class_to_image = list_class_representatives()

    image_path = args.image or pick_random_image()
    if image_path is None:
        raise SystemExit("No image found. Pass an image path explicitly.")
    if not os.path.exists(image_path):
        raise SystemExit(f"Image not found: {image_path}")

    initial_image = Image.open(image_path).convert("RGB")
    runner = Sam3Runner(
        checkpoint_path=args.checkpoint,
        confidence_threshold=float(args.confidence_threshold),
        device=str(args.device),
    )

    initial_class = None
    image_path_abs = os.path.abspath(image_path)
    for class_name, class_image_path in class_to_image.items():
        if os.path.abspath(class_image_path) == image_path_abs:
            initial_class = class_name
            break

    app = build_app(runner, initial_image, class_to_image=class_to_image, initial_class=initial_class)
    app.launch(server_name=args.host, server_port=args.port, share=args.share, css=APP_CSS)


if __name__ == "__main__":
    main()
