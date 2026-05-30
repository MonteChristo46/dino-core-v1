"""Interactive script to test SAM3 point prompts.

Left-click = foreground point (green)
Right-click = background point (red)
Mask updates on every click.

Usage:
    python scripts/test_sam3_point_prompt.py [image_path] [--gui]

If no image path is given, picks a random image from the dataset.
Default mode is terminal-based (works on headless servers).
Add --gui for Tkinter window (requires a display).
"""

import os
import random
import sys
import tkinter as tk
from tkinter import filedialog

import numpy as np
import torch
from PIL import Image, ImageTk

from sam3.model.data_misc import FindStage
from sam3.model.geometry_encoders import Prompt
from sam3.model_builder import build_sam3_image_model
from sam3.model.sam3_image_processor import Sam3Processor

torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True
torch.autocast("cuda", dtype=torch.bfloat16).__enter__()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
WEIGHTS_DIR = os.path.join(PROJECT_ROOT, "weights", "sam3")
DATASET_ROOT = os.path.abspath(
    os.path.join(PROJECT_ROOT, "..", "..", "dataset", "glitch_hunt_mvtec")
)
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "output", "sam3")
CHECKPOINT = os.path.join(WEIGHTS_DIR, "sam3.pt")

DISPLAY_MAX_SIZE = 800
LABEL_FG = 1
LABEL_BG = 0
COLOR_FG = "#00ff00"
COLOR_BG = "#ff0000"
POINT_RADIUS = 5


def has_display():
    if os.environ.get("DISPLAY") or os.environ.get("WAYLAND_DISPLAY"):
        return True
    try:
        import tkinter as tk
        root = tk.Tk()
        root.destroy()
        return True
    except Exception:
        return False


def pick_random_image():
    if not os.path.exists(DATASET_ROOT):
        return None
    for root, dirs, files in os.walk(DATASET_ROOT):
        if os.path.basename(root) != "good":
            continue
        good_files = [f for f in files if f.lower().endswith((".png", ".jpg", ".jpeg"))]
        if good_files:
            return os.path.join(root, random.choice(good_files))
    return None


def run_point_inference(model, processor, pil_img, points, confidence_threshold=0.5):
    """Run SAM3 inference with point prompts.

    points: list of (x_norm, y_norm, label) tuples, label 1=fg, 0=bg
    """
    from sam3.model import box_ops
    from sam3.model.data_misc import interpolate

    with torch.inference_mode():
        width, height = pil_img.size
        state = processor.set_image(pil_img)

        text_outputs = model.backbone.forward_text(["visual"], device=model.device)
        state["backbone_out"].update(text_outputs)

        if points:
            pts = torch.tensor(points, device=model.device, dtype=torch.float32).view(1, -1, 3)
            pts_mask = torch.zeros(1, len(points), device=model.device, dtype=torch.bool)
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


def save_result_image(pil_img, mask_overlay, points, output_path):
    """Save image with mask overlay and point markers using PIL only."""
    img = pil_img.copy()

    if mask_overlay is not None:
        mask_img = Image.fromarray((mask_overlay * 255).astype(np.uint8)).convert("L")
        red_overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
        red_data = red_overlay.load()
        mask_data = mask_img.load()
        w, h = img.size
        for y in range(h):
            for x in range(w):
                if mask_data[x, y] > 0:
                    red_data[x, y] = (255, 0, 0, 100)
        img = img.convert("RGBA")
        img = Image.alpha_composite(img, red_overlay).convert("RGB")

    w, h = img.size
    for x_norm, y_norm, label in points:
        cx = int(x_norm * w)
        cy = int(y_norm * h)
        r = 8
        color = (0, 255, 0) if label == LABEL_FG else (255, 0, 0)
        for dy in range(-r, r + 1):
            for dx in range(-r, r + 1):
                if dx * dx + dy * dy <= r * r:
                    px, py = cx + dx, cy + dy
                    if 0 <= px < w and 0 <= py < h:
                        img.putpixel((px, py), color)

    img.save(output_path)
    return output_path


class TerminalPointPromptTester:
    def __init__(self, image_path):
        self.pil_img = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.pil_img.size
        self.points = []
        self.mask_overlay = None
        self.model = None
        self.processor = None
        self.image_path = image_path

        print(f"\nImage: {image_path}")
        print(f"Size: {self.orig_w}x{self.orig_h}")
        print("\nLoading SAM3 model...")
        self.model = build_sam3_image_model(checkpoint_path=CHECKPOINT, load_from_HF=False)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)
        print("Model loaded.\n")
        self._print_help()
        self._run_loop()

    def _print_help(self):
        print("=" * 50)
        print("Commands:")
        print("  <x> <y> f   - Add foreground point")
        print("  <x> <y> b   - Add background point")
        print("  clear       - Clear all points")
        print("  save        - Save current result to output/")
        print("  quit        - Exit")
        print("=" * 50)

    def _run_loop(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        while True:
            try:
                pts_str = f" [{len(self.points)} pts]" if self.points else ""
                score_str = ""
                if self.mask_overlay is not None:
                    score_str = f" score={self._last_score:.3f}"
                prompt = f"\n{pts_str}{score_str}> "
                line = input(prompt).strip()
            except (EOFError, KeyboardInterrupt):
                print("\nExiting.")
                break

            if not line:
                continue

            parts = line.lower().split()

            if parts[0] == "quit" or parts[0] == "q" or parts[0] == "exit":
                print("Exiting.")
                break

            if parts[0] == "clear":
                self.points = []
                self.mask_overlay = None
                self._last_score = None
                print("Points cleared.")
                continue

            if parts[0] == "save":
                path = self._save()
                print(f"Saved to {path}")
                continue

            if len(parts) >= 3:
                try:
                    x = int(parts[0])
                    y = int(parts[1])
                    label = LABEL_FG if parts[2].startswith("f") else LABEL_BG
                except ValueError:
                    print("Invalid format. Use: <x> <y> <f|b>")
                    continue

                if x < 0 or x >= self.orig_w or y < 0 or y >= self.orig_h:
                    print(f"Coordinates out of range. Image is {self.orig_w}x{self.orig_h}")
                    continue

                x_norm = x / self.orig_w
                y_norm = y / self.orig_h
                self.points.append((x_norm, y_norm, label))

                print(f"Added {'FG' if label == LABEL_FG else 'BG'} point at ({x}, {y})... ", end="", flush=True)

                masks, boxes, scores = run_point_inference(
                    self.model, self.processor, self.pil_img, self.points
                )

                if masks is not None and masks.numel() > 0:
                    mask_np = masks[0, 0].cpu().float().numpy() > 0.5
                    self.mask_overlay = mask_np
                    self._last_score = scores[0].cpu().float().item()
                    print(f"score={self._last_score:.3f}")
                else:
                    self.mask_overlay = None
                    self._last_score = None
                    print("no detection")

    def _save(self):
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        base = os.path.splitext(os.path.basename(self.image_path))[0]
        n = len(self.points)
        path = os.path.join(OUTPUT_DIR, f"point_{base}_n{n}.png")
        save_result_image(self.pil_img, self.mask_overlay, self.points, path)
        return path


class PointPromptTester:
    def __init__(self, root, image_path):
        self.root = root
        self.root.title("SAM3 Point Prompt Tester — Left=FG, Right=BG")

        self.pil_img = Image.open(image_path).convert("RGB")
        self.orig_w, self.orig_h = self.pil_img.size
        self.scale = min(DISPLAY_MAX_SIZE / self.orig_w, DISPLAY_MAX_SIZE / self.orig_h, 1.0)
        self.disp_w = int(self.orig_w * self.scale)
        self.disp_h = int(self.orig_h * self.scale)
        self.disp_img = self.pil_img.resize((self.disp_w, self.disp_h), Image.LANCZOS)

        self.points = []
        self.mask_overlay = None
        self.model = None
        self.processor = None

        self._build_ui()
        self._load_model()
        self._draw_base()

    def _build_ui(self):
        toolbar = tk.Frame(self.root)
        toolbar.pack(side=tk.TOP, fill=tk.X, padx=4, pady=4)

        tk.Button(toolbar, text="Clear Points", command=self._clear_points).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Open Image", command=self._open_image).pack(side=tk.LEFT, padx=2)
        tk.Button(toolbar, text="Quit", command=self.root.quit).pack(side=tk.RIGHT, padx=2)

        self.status = tk.Label(toolbar, text="Loading model…", anchor=tk.W)
        self.status.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=8)

        self.canvas = tk.Canvas(
            self.root, width=self.disp_w, height=self.disp_h, bg="#222", cursor="crosshair"
        )
        self.canvas.pack(fill=tk.BOTH, expand=True)

        self.canvas.bind("<Button-1>", self._on_left_click)
        self.canvas.bind("<Button-3>", self._on_right_click)
        self.canvas.bind("<Button-2>", self._on_right_click)

    def _load_model(self):
        self.root.update()
        self.status.config(text="Loading SAM3 model…")
        self.root.update()

        self.model = build_sam3_image_model(checkpoint_path=CHECKPOINT, load_from_HF=False)
        self.processor = Sam3Processor(self.model, confidence_threshold=0.5)

        self.status.config(text="Model loaded. Click to add points.")

    def _draw_base(self):
        self.photo = ImageTk.PhotoImage(self.disp_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.image_ref = self.photo

    def _clear_points(self):
        self.points = []
        self.mask_overlay = None
        self._redraw()
        self.status.config(text="Points cleared.")

    def _open_image(self):
        path = filedialog.askopenfilename(
            filetypes=[("Images", "*.png *.jpg *.jpeg *.bmp *.tif")]
        )
        if not path:
            return
        self.pil_img = Image.open(path).convert("RGB")
        self.orig_w, self.orig_h = self.pil_img.size
        self.scale = min(DISPLAY_MAX_SIZE / self.orig_w, DISPLAY_MAX_SIZE / self.orig_h, 1.0)
        self.disp_w = int(self.orig_w * self.scale)
        self.disp_h = int(self.orig_h * self.scale)
        self.disp_img = self.pil_img.resize((self.disp_w, self.disp_h), Image.LANCZOS)
        self.points = []
        self.mask_overlay = None
        self.canvas.config(width=self.disp_w, height=self.disp_h)
        self._draw_base()
        self.status.config(text=f"Loaded: {os.path.basename(path)}")

    def _on_left_click(self, event):
        self._add_point(event.x, event.y, LABEL_FG)

    def _on_right_click(self, event):
        self._add_point(event.x, event.y, LABEL_BG)

    def _add_point(self, cx, cy, label):
        x_norm = cx / self.disp_w
        y_norm = cy / self.disp_h
        self.points.append((x_norm, y_norm, label))

        self.status.config(text=f"Points: {len(self.points)} | Running inference…")
        self.root.update()

        masks, boxes, scores = run_point_inference(
            self.model, self.processor, self.pil_img, self.points
        )

        if masks is not None and masks.numel() > 0:
            mask_np = masks[0, 0].cpu().float().numpy() > 0.5
            self.mask_overlay = mask_np
            score_val = scores[0].cpu().float().item()
            self.status.config(text=f"Points: {len(self.points)} | Score: {score_val:.3f}")
        else:
            self.mask_overlay = None
            self.status.config(text=f"Points: {len(self.points)} | No detection")

        self._redraw()

    def _redraw(self):
        self.canvas.delete("all")
        self.photo = ImageTk.PhotoImage(self.disp_img)
        self.canvas.create_image(0, 0, image=self.photo, anchor=tk.NW)
        self.canvas.image_ref = self.photo

        if self.mask_overlay is not None:
            mask_resized = Image.fromarray((self.mask_overlay * 255).astype(np.uint8)).resize(
                (self.disp_w, self.disp_h), Image.NEAREST
            )
            mask_rgba = Image.new("RGBA", (self.disp_w, self.disp_h), (0, 0, 0, 0))
            mask_rgba.paste(Image.new("RGBA", mask_resized.size, (255, 0, 0, 100)), mask=mask_resized)
            mask_photo = ImageTk.PhotoImage(mask_rgba)
            self.canvas.create_image(0, 0, image=mask_photo, anchor=tk.NW)
            self.canvas._mask_photo = mask_photo

        for x_norm, y_norm, label in self.points:
            cx = int(x_norm * self.disp_w)
            cy = int(y_norm * self.disp_h)
            color = COLOR_FG if label == LABEL_FG else COLOR_BG
            self.canvas.create_oval(
                cx - POINT_RADIUS,
                cy - POINT_RADIUS,
                cx + POINT_RADIUS,
                cy + POINT_RADIUS,
                fill=color,
                outline="white",
                width=1,
            )


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
        root = tk.Tk()
        PointPromptTester(root, image_path)
        root.mainloop()
    else:
        if use_gui and not has_display():
            print("Warning: --gui requested but no display available, falling back to terminal mode.")
        TerminalPointPromptTester(image_path)


if __name__ == "__main__":
    main()
