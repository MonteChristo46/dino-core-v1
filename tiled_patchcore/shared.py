"""Shared helpers for the ``tiled_patchcore`` user-facing scripts.

Usability rules encoded here:
- Tiled models are trained on **native-resolution** images (the tiler needs
  the original pixels) and the memory bank is built from **good images only**
  (the MVTec-style datasets of the synthetic pipeline also contain train/bad,
  which must never leak into the memory bank).
- DINOv3 embedding dimensions are derived from the backbone name; non-DINOv3
  backbones must pass them explicitly.
- A saved checkpoint fully determines how it must be evaluated (tiled ->
  native resolution, plain -> its saved input size), so the eval script needs
  no dataset geometry flags.
"""
import csv
import json
import logging
import os
import sys
import time

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_PATH = os.path.join(_REPO_ROOT, "src")
if _SRC_PATH not in sys.path:
    sys.path.insert(0, _SRC_PATH)

import matplotlib

matplotlib.use("Agg")  # headless-safe plotting for all analytics artifacts
import matplotlib.patches  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

import numpy as np
import PIL
import torch
import torch.utils.data
import tqdm

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.tiled_patchcore
import patchcore.utils
from patchcore.datasets import mvtec

LOGGER = logging.getLogger(__name__)

# Feature dimensions of the supported backbones (auto-detection table).
_EMBED_DIMS = {
    "dinov3_vits": 384,
    "dinov3_vitsplus": 384,
    "dinov3_vitb": 768,
    "dinov3_vitl": 1024,
    "dinov3_vithplus": 1280,
    "wideresnet50": 1024,
    "wideresnet101": 1024,
}

_DEFAULT_LAYERS = {
    "dinov3": ["7", "8"],
    "wideresnet": ["layer2", "layer3"],
}

# --- anomaly-map rendering -------------------------------------------------
# One place defines the heatmap look, so the top-N plot and the per-image
# segmentation pack show the very same thing (the pack used to show an
# auto-scaled pure map, which reads as noise: every panel got stretched to its
# own min/max, so a flat good image looked as loud as a real defect).
_HEAT_CMAP = "magma"          # standalone anomaly-map panels
_OVERLAY_CMAP = "autumn"      # hot-region tint painted over the image
_OVERLAY_ALPHA = 0.75         # alpha of the flagged (>= cutoff) layer
_CONTEXT_ALPHA = 0.45         # peak alpha of the faint below-cutoff layer
_CUTOFF_EDGE = "white"        # iso-line drawn exactly at the decision cutoff
_GT_EDGE = "lime"             # ground-truth mask outline
_TILE_EDGE = "cyan"           # tile borders (seam / padding-artifact check)
_SEG_PANEL_INCHES = 3.4       # per-panel render size of the segmentation pack
_SEG_DPI = 180                # ~610 px per panel: defects stay visible
_SEG_SAVE_DEPTH = 4           # path components used for the jpg filenames


def resolve_embed_dims(backbone_name, pretrain=None, target=None):
    """Return (pretrain_embed_dimension, target_embed_dimension).

    Auto-derives the DINOv3/WR dimensions from the backbone name; explicit
    arguments always win.
    """
    name = backbone_name.lower()
    resolved = None
    for key, dim in _EMBED_DIMS.items():
        if key in name:
            resolved = dim
            break
    pretrain = pretrain if pretrain is not None else resolved
    target = target if target is not None else resolved
    if pretrain is None or target is None:
        raise ValueError(
            f"Cannot auto-derive embedding dimensions for backbone "
            f"'{backbone_name}'. Pass --pretrain_embed_dimension and "
            f"--target_embed_dimension explicitly. Known: {sorted(_EMBED_DIMS)}."
        )
    return int(pretrain), int(target)


def default_layers(backbone_name):
    """Feature layers to extract, per backbone family (None if unknown)."""
    name = backbone_name.lower()
    for key, layers in _DEFAULT_LAYERS.items():
        if key in name:
            return list(layers)
    return None


def discover_categories(data_path):
    """Subdirectories of a dataset root that look like MVTec categories."""
    return sorted(
        entry
        for entry in os.listdir(data_path)
        if os.path.isdir(os.path.join(data_path, entry, "train"))
    )


def build_dataset(
    source,
    category,
    split,
    resize=None,
    imagesize=None,
    train_val_split=1.0,
    seed=0,
    augment=False,
    good_only=False,
):
    """MVTec-format dataset. ``resize=imagesize=None`` => native resolution.

    ``split`` accepts "TRAIN"/"TEST"/"VAL" (case-insensitive) or a
    ``mvtec.DatasetSplit``. ``good_only`` drops every non-good training sample
    (protects the memory bank from train/bad in the synthetic-pipeline dataset
    format). Returns (dataset, n_excluded).
    """
    if isinstance(split, str):
        split = mvtec.DatasetSplit[split.upper()]
    dataset = mvtec.MVTecDataset(
        source,
        classname=category,
        resize=resize,
        imagesize=imagesize,
        split=split,
        train_val_split=train_val_split,
        seed=seed,
        augment=augment,
    )
    n_excluded = 0
    if good_only and split == mvtec.DatasetSplit.TRAIN:
        kept = [t for t in dataset.data_to_iterate if t[1] == "good"]
        n_excluded = len(dataset.data_to_iterate) - len(kept)
        dataset.data_to_iterate = kept
        if n_excluded:
            LOGGER.info(
                "Training set [%s]: dropped %d non-good samples (good-only).",
                category,
                n_excluded,
            )
    return dataset, n_excluded


def resolve_device(gpu=None, cpu=False):
    """Pick the compute device: ``--cpu`` > ``--gpu`` (CUDA) > MPS > CPU.

    On Apple Silicon the default is now the Metal (MPS) backend instead of
    silently falling back to CPU — the tiled pipeline is MPS-safe because
    stitching/Gaussian smoothing already run on the CPU and FAISS has no MPS
    backend, so the kNN index stays on the CPU either way.

    Returns ``(device, faiss_on_gpu)``.
    """
    if cpu:
        LOGGER.info("Device: cpu (forced with --cpu).")
        return torch.device("cpu"), False
    if gpu is not None:
        device = torch.device(f"cuda:{gpu}")
        LOGGER.info("Device: %s (from --gpu).", device)
        return device, True
    if torch.cuda.is_available():
        LOGGER.info("Device: cuda:0 (CUDA available).")
        return torch.device("cuda:0"), True

    mps = getattr(torch.backends, "mps", None)
    if mps is not None and mps.is_available():
        LOGGER.info(
            "Device: mps (Apple Silicon GPU) — no CUDA, using Metal. "
            "FAISS kNN stays on CPU. Pass --cpu to force the CPU instead."
        )
        return torch.device("mps"), False

    LOGGER.info("Device: cpu (no CUDA or MPS backend available).")
    return torch.device("cpu"), False


def build_dataloader(dataset, batch_size, num_workers):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    if dataset.classnames_to_use:
        loader.name = "mvtec_" + dataset.classnames_to_use[0]
    return loader


def dataset_image_hw(dataset):
    """Native (H, W) of the first image of a dataset (category-uniform)."""
    image_path = dataset.data_to_iterate[0][2]
    with PIL.Image.open(image_path) as image:
        return image.size[1], image.size[0]


def has_both_classes(dataset):
    """True when a split holds at least one good AND one non-good sample.

    Single-class splits (e.g. BigD categories whose test/ has no anomalies)
    make instance AUROC/thresholds undefined — sklearn's roc_curve even
    raises — so callers warn and skip metric computation instead of crashing.
    """
    labels = [anomaly != "good" for _cls, anomaly, _img, _mask in dataset.data_to_iterate]
    return bool(labels) and any(labels) and not all(labels)


class RecordingSampler:
    """Wraps a featuresampler to record pooled/coreset row counts."""

    def __init__(self, inner):
        self.inner = inner
        self.pooled_rows = None
        self.coreset_rows = None

    def run(self, features):
        self.pooled_rows = int(features.shape[0])
        sampled = self.inner.run(features)
        self.coreset_rows = int(sampled.shape[0])
        return sampled


def load_models_from_dir(model_dir, device, nn_method, seed=0):
    """Load all models saved in ``model_dir`` (single + Ensemble-* prepends).

    Tiled checkpoints are detected via their ``tiling_params.pkl`` sidecar and
    loaded through the wrapper. The global torch seed is reset before each
    backbone construction so untrained test backbones (random init) are
    reproducible, matching how the training script seeds its backbone; real
    checkpoints carry their own weights and are unaffected.
    """
    params_files = [
        f
        for f in os.listdir(model_dir)
        if f.endswith("patchcore_params.pkl")
    ]
    models = []
    for params_file in sorted(params_files):
        prepend = params_file[: -len("patchcore_params.pkl")]
        torch.manual_seed(seed)
        if patchcore.tiled_patchcore.TiledPatchCore.has_tiling_params(
            model_dir, prepend
        ):
            model = patchcore.tiled_patchcore.tiled_from_path(
                model_dir, device, nn_method, prepend=prepend
            )
            LOGGER.info("Loaded TiledPatchCore from %s (prepend='%s').", model_dir, prepend)
            models.append((model, True))
        else:
            model = patchcore.patchcore.PatchCore(device)
            model.load_from_path(
                load_path=model_dir, device=device, nn_method=nn_method, prepend=prepend
            )
            models.append((model, False))
    if not models:
        raise FileNotFoundError(f"No patchcore models found in {model_dir}")
    return models


def dataset_geometry_for_model(model, is_tiled):
    """Dataset (resize, imagesize) that matches a loaded checkpoint.

    Tiled models require native resolution. Plain models are evaluated at
    their training crop size (center-crop only; pass --resize explicitly to
    reproduce the historical Resize+CenterCrop pipeline exactly).
    """
    if is_tiled:
        return None, None
    input_shape = tuple(model.input_shape)
    return None, int(input_shape[-1])


def tile_layout_for_model(model, is_tiled, height, width):
    if not is_tiled:
        return None
    return model.tile_maker.compute_layout(height, width)


def evaluate_models(models, test_dataloader):
    """Predict with each model, min-max normalize + average (ensemble
    semantics identical to scripts/patchcore/run_patchcore.py), compute
    image- and pixel-level metrics.
    """
    raw_scores, raw_segmentations = [], []
    labels_gt, masks_gt, timing = None, None, 0.0
    for model, _is_tiled in models:
        start = time.perf_counter()
        scores, segmentations, labels_gt, masks_gt = model.predict(test_dataloader)
        timing += time.perf_counter() - start
        raw_scores.append(scores)
        raw_segmentations.append(segmentations)

    scores = np.array(raw_scores)
    min_scores = scores.min(axis=-1).reshape(-1, 1)
    max_scores = scores.max(axis=-1).reshape(-1, 1)
    scores = np.mean((scores - min_scores) / np.maximum(max_scores - min_scores, 1e-12), axis=0)

    segmentations = np.array(raw_segmentations)
    seg_min = (
        segmentations.reshape(len(segmentations), -1).min(axis=-1).reshape(-1, 1, 1, 1)
    )
    seg_max = (
        segmentations.reshape(len(segmentations), -1).max(axis=-1).reshape(-1, 1, 1, 1)
    )
    segmentations = np.mean(
        (segmentations - seg_min) / np.maximum(seg_max - seg_min, 1e-12), axis=0
    )
    # Keep the four normalization constants. Scores are min-max normalized over
    # *image* scores, maps over *all pixels*: the two [0,1] scales have the same
    # top but different bottoms, so a threshold cannot be read off a map directly.
    # map_cutoff_for_threshold() does the conversion for the renderers.
    norm_stats = {
        "score_min": min_scores.reshape(-1),
        "score_max": max_scores.reshape(-1),
        "map_min": seg_min.reshape(-1),
        "map_max": seg_max.reshape(-1),
    }

    anomaly_labels = [
        anomaly != "good" for _cls, anomaly, _img, _mask in test_dataloader.dataset.data_to_iterate
    ]
    image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
        scores, anomaly_labels
    )
    # Ground-truth masks are optional (see mvtec.py). Path-level check —
    # cheap, unlike summing native-resolution mask tensors.
    has_gt_masks = any(
        _mask is not None
        for _cls, _anomaly, _img, _mask in test_dataloader.dataset.data_to_iterate
    )
    if has_gt_masks:
        pixel_metrics = patchcore.metrics.compute_pixelwise_retrieval_metrics(
            segmentations, masks_gt
        )
        sel_idxs = [i for i in range(len(masks_gt)) if np.sum(masks_gt[i]) > 0]
        anomaly_pixel_metrics = (
            patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs], [masks_gt[i] for i in sel_idxs]
            )
            if sel_idxs
            else None
        )
    else:
        pixel_metrics = None
        anomaly_pixel_metrics = None
        LOGGER.info(
            "No ground_truth masks in this split — pixel-level AUROCs "
            "unavailable (reported as NaN); instance metrics unaffected."
        )
    return {
        "scores": scores,
        "segmentations": segmentations,
        "norm_stats": norm_stats,
        "labels_gt": anomaly_labels,
        "masks_gt": masks_gt,
        "has_gt_masks": has_gt_masks,
        "image_metrics": image_metrics,
        "pixel_metrics": pixel_metrics,
        "anomaly_pixel_metrics": anomaly_pixel_metrics,
        "predict_seconds": timing,
    }


def classify_predictions(scores, labels, threshold):
    """TP/FP/TN/FN + predicted flags per image (run_patchcore semantics)."""
    records = []
    for score, label in zip(scores, labels):
        predicted = bool(score >= threshold)
        is_anomaly = bool(label)
        if is_anomaly and predicted:
            category = "TP"
        elif not is_anomaly and predicted:
            category = "FP"
        elif not is_anomaly and not predicted:
            category = "TN"
        else:
            category = "FN"
        records.append((category, predicted))
    return records


def write_per_image_scores_csv(path, category, dataset, scores, records):
    rows = [
        {
            "category": category,
            "image_name": "/".join(image_path.split("/")[-4:]),
            "image_path": image_path,
            "is_anomaly": int(anomaly != "good"),
            "score": float(score),
            "predicted_anomaly": int(predicted),
            "result": result,
        }
        for (_cls, anomaly, image_path, _mask), score, (result, predicted) in zip(
            dataset.data_to_iterate, scores, records
        )
    ]
    write = not os.path.exists(path)
    with open(path, "a", newline="") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=[
                "category",
                "image_name",
                "image_path",
                "is_anomaly",
                "score",
                "predicted_anomaly",
                "result",
            ],
        )
        if write:
            writer.writeheader()
        writer.writerows(rows)


def confusion_counts(records):
    counts = {"TP": 0, "FP": 0, "FN": 0, "TN": 0}
    for result, _predicted in records:
        counts[result] += 1
    tp, fp, fn = counts["TP"], counts["FP"], counts["FN"]
    counts["Precision"] = tp / (tp + fp) if (tp + fp) else 0.0
    counts["Recall"] = tp / (tp + fn) if (tp + fn) else 0.0
    return counts


def write_json(path, payload):
    with open(path, "w") as json_file:
        json.dump(payload, json_file, indent=2, sort_keys=True)


def plot_score_distribution(scores, labels, threshold, save_path, title):
    scores = np.asarray(scores)
    labels = np.asarray(labels)
    if len(set(labels.tolist())) < 2:
        return False  # ROC/histogram meaningless without both classes
    fig, axis = plt.subplots(figsize=(5.5, 3.5), dpi=120)
    bins = np.linspace(scores.min(), scores.max(), 21)
    axis.hist(scores[labels == 0], bins=bins, alpha=0.6, label="good", color="tab:blue")
    axis.hist(scores[labels == 1], bins=bins, alpha=0.6, label="anomaly", color="tab:red")
    axis.axvline(threshold, color="black", linestyle="--", linewidth=1.2,
                 label=f"threshold={threshold:.3f}")
    axis.set_xlabel("normalized anomaly score")
    axis.set_ylabel("#images")
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)
    return True


def plot_roc(image_metrics, save_path, title):
    fig, axis = plt.subplots(figsize=(4.2, 4.2), dpi=120)
    axis.plot(image_metrics["fpr"], image_metrics["tpr"], label=f"AUROC={image_metrics['auroc']:.3f}")
    axis.plot([0, 1], [0, 1], "k--", linewidth=0.8)
    axis.set_xlabel("FPR")
    axis.set_ylabel("TPR")
    axis.set_title(title)
    axis.legend()
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def plot_tile_grid(image_path, layout, save_path, title):
    """Draw the tile grid on an original image — sanity-check tiling geometry."""
    with PIL.Image.open(image_path) as image:
        rgb = np.asarray(image.convert("RGB"))
    height, width = rgb.shape[:2]
    fig, axis = plt.subplots(figsize=(6.5, 6.5 * height / width), dpi=120)
    axis.imshow(rgb)
    tile = layout.tile_size
    for idx, (y, x) in enumerate(layout.origins):
        color = "tab:red" if idx % 2 else "tab:blue"
        axis.add_patch(
            matplotlib.patches.Rectangle((x, y), tile, tile, fill=False,
                                         edgecolor=color, linewidth=1.2)
        )
    axis.set_xlim(-0.5, width - 0.5)
    axis.set_ylim(height - 0.5, -0.5)
    axis.set_title(
        f"{title} — {layout.num_tiles} tiles "
        f"(tile={tile}, stride={layout.stride}, padded={layout.padded_hw})"
    )
    axis.axis("off")
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def _resize_2d(array, target_hw):
    """Bilinear-resample a 2-D map/mask onto ``target_hw`` if its size differs.

    Tiled models emit maps already cropped to the original image size
    (``TileMaker.stitch``), so this is a no-op there. Plain checkpoints emit maps
    at the model input size, and overlaying those on a native-resolution image
    without resampling paints the anomaly at the wrong scale.
    """
    array = np.asarray(array, dtype=np.float32)
    height, width = int(target_hw[0]), int(target_hw[1])
    if array.shape[:2] == (height, width):
        return array
    resized = PIL.Image.fromarray(array, mode="F").resize((width, height), PIL.Image.BILINEAR)
    return np.asarray(resized, dtype=np.float32)


def _alpha_ramp_cmap(name, peak_alpha):
    """``name`` colormap whose own alpha grows with the value (transparent low).

    Keeps a below-threshold context layer from tinting the whole image: the
    colormap's dark low end is also its transparent end, so only warm regions
    visibly wash the picture.
    """
    stops = np.linspace(0.0, 1.0, 32)
    rgba = matplotlib.colormaps[name](stops)
    colors = [(rgb[0], rgb[1], rgb[2], peak_alpha * stop ** 2)
              for rgb, stop in zip(rgba, stops)]
    return matplotlib.colors.ListedColormap(colors)


def map_cutoff_for_threshold(evaluation, threshold):
    """Decision threshold in *anomaly-map* units, or None when undefined.

    The threshold lives on the normalized-score scale; map pixels live on the
    normalized-pixel scale (same maximum, different minimum). Reading the
    threshold straight off the map therefore flags far too much area. Returns
    None for single-class splits / missing stats, where no decision boundary
    exists and the caller should fall back to a fixed fraction of the global max.
    """
    if threshold is None or not np.isfinite(threshold):
        return None
    stats = (evaluation or {}).get("norm_stats")
    if stats is None:
        return float(threshold)
    raw = stats["score_min"] + threshold * (stats["score_max"] - stats["score_min"])
    cutoff = (raw - stats["map_min"]) / np.maximum(
        stats["map_max"] - stats["map_min"], 1e-12
    )
    return float(np.clip(cutoff.mean(), 0.0, 1.0))


def overlay_anomaly_map(axis, rgb, heatmap, *, vmax, cutoff, layout=None,
                        show_grid=False, gt_mask=None):
    """Paint an anomaly map *over* its image — the top-N plot's look, in one place.

    Three layers: a faint alpha-ramped heatmap so sub-threshold structure stays
    readable, the flagged regions (``>= cutoff``, fully opaque tint) that actually
    made the detector fire, and a thin iso-line exactly at ``cutoff``. Optional
    tile grid (seam / padding artifacts) and ground-truth outline, so
    prediction-vs-GT overlap is answerable from a single panel.

    ``cutoff`` is in map units — pass ``map_cutoff_for_threshold`` output, not a
    score threshold. ``vmax`` is the *shared* scale so brightness means the same
    thing across every rendered file.
    """
    height, width = rgb.shape[:2]
    heatmap = _resize_2d(heatmap, (height, width))
    axis.imshow(rgb)
    axis.imshow(heatmap, cmap=_alpha_ramp_cmap(_HEAT_CMAP, _CONTEXT_ALPHA),
                vmin=0.0, vmax=vmax)
    axis.imshow(np.ma.masked_where(heatmap < cutoff, heatmap),
                cmap=_OVERLAY_CMAP, alpha=_OVERLAY_ALPHA, vmin=0.0, vmax=vmax)
    axis.contour(heatmap, levels=[cutoff], colors=_CUTOFF_EDGE, linewidths=0.7)
    if gt_mask is not None:
        gt_mask = _resize_2d(gt_mask, (height, width))
        if gt_mask.max() > 0.5:
            axis.contour(gt_mask, levels=[0.5], colors=_GT_EDGE, linewidths=1.0)
    if show_grid and layout is not None:
        for y, x in layout.origins:
            axis.add_patch(
                matplotlib.patches.Rectangle((x, y), layout.tile_size, layout.tile_size,
                                             fill=False, edgecolor=_TILE_EDGE,
                                             linewidth=0.7)
            )


def plot_top_anomalous(image_paths, scores, segmentations, layout, save_path,
                       n=4, title="Top anomaly candidates", cutoff=None):
    """Original | heatmap | overlay(+tile grid) for the highest-scoring images.

    The overlay shows whether hotspots coincide with tile seams/borders (padding
    artifacts) or sit on real content. Pass ``cutoff`` (map units, see
    map_cutoff_for_threshold) to draw exactly what crossed the decision line;
    without it, half the global max is used.
    """
    order = np.argsort(scores)[::-1][:n]
    count = len(order)
    fig, axes = plt.subplots(count, 3, figsize=(9.5, 3.2 * count), dpi=110,
                             squeeze=False)
    vmax = max(float(np.max(segmentations)), 1e-12)
    if cutoff is None or not np.isfinite(cutoff):
        cutoff = 0.5 * vmax
    for row, idx in enumerate(order):
        with PIL.Image.open(image_paths[idx]) as image:
            rgb = np.asarray(image.convert("RGB"))
        heatmap = _resize_2d(segmentations[idx], rgb.shape[:2])
        axes[row][0].imshow(rgb)
        axes[row][0].set_title(f"{os.path.basename(image_paths[idx])}\nscore={scores[idx]:.3f}")
        axes[row][1].imshow(heatmap, cmap=_HEAT_CMAP, vmin=0.0, vmax=vmax)
        axes[row][1].set_title("anomaly map")
        overlay_anomaly_map(axes[row][2], rgb, heatmap, vmax=vmax, cutoff=cutoff,
                            layout=layout, show_grid=layout is not None)
        axes[row][2].set_title(
            "overlay" + (" + tile grid" if layout else "")
            + f" (>= {cutoff:.3f})"
        )
        for axis_row in axes[row]:
            axis_row.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_segmentation_images_for_category(
    savefolder, dataset, scores, segmentations, records, threshold,
    layout=None, cutoff=None
):
    """Render ``Original | Ground Truth | Heatmap | Overlay`` per test image.

    Owns its figure instead of reusing patchcore.utils.plot_segmentation_images
    (which stays as-is for the legacy scripts) because that renderer shows the
    map as a standalone panel with no vmin/vmax: matplotlib then stretches *each
    image's own* min/max to the full colormap, so a flat, low-scoring image
    looks exactly as alarming as a real defect and nothing is comparable across
    files. Here every map shares one scale (the category's global max) and the
    hot region is painted over the image, like the top-N plot.

    Output layout is unchanged: ``<savefolder>/<TP|FP|TN|FN>/<path-tail>.jpg``,
    one file per test image. The GT panel and the green outline on the overlay
    disappear when the split has no ground_truth masks.
    """
    image_paths = [t[2] for t in dataset.data_to_iterate]
    mask_paths = [t[3] for t in dataset.data_to_iterate]
    has_gt = any(mask is not None for mask in mask_paths)
    categories = [result for result, _ in records]
    for category in set(categories):
        os.makedirs(os.path.join(savefolder, category), exist_ok=True)

    vmax = max(float(np.max(segmentations)), 1e-12) if len(segmentations) else 1e-12
    if cutoff is None or not np.isfinite(cutoff):
        cutoff = 0.5 * vmax
    threshold_str = (
        f" | Threshold: {threshold:.3f}"
        if threshold is not None and np.isfinite(threshold) else ""
    )

    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)

    def denormalized(image):
        # Exactly what the model saw (same transform, ImageNet stats undone), so
        # the RGB and the map share one pixel grid.
        tensor = dataset.transform_img(image)
        return np.clip((tensor.numpy() * std + mean) * 255, 0, 255).astype(np.uint8)

    panels = 4 if has_gt else 3
    for image_path, mask_path, score, segmentation, category in tqdm.tqdm(
        zip(image_paths, mask_paths, scores, segmentations, categories),
        total=len(image_paths),
        desc="Rendering segmentation images...",
        leave=False,
    ):
        savename = os.path.join(
            savefolder, category,
            "_".join(image_path.split("/")[-_SEG_SAVE_DEPTH:]).rsplit(".", 1)[0] + ".jpg",
        )
        with PIL.Image.open(image_path) as image:
            rgb = denormalized(image.convert("RGB")).transpose(1, 2, 0)
        mask = None
        if has_gt:
            if mask_path is not None:
                with PIL.Image.open(mask_path) as mask_image:
                    mask = np.asarray(mask_image.convert("L"), dtype=np.float32) / 255.0
            mask = _resize_2d(mask, rgb.shape[:2]) if mask is not None else None

        figure, axes = plt.subplots(
            1, panels,
            figsize=(_SEG_PANEL_INCHES * panels, _SEG_PANEL_INCHES), dpi=_SEG_DPI,
        )
        col = 0
        is_correct = category in ("TP", "TN")
        axes[col].imshow(rgb)
        axes[col].set_title(
            f"Score: {score:.3f}{threshold_str}\n{category}",
            fontsize=9, color="green" if is_correct else "red", fontweight="bold",
        )
        col += 1
        if has_gt:
            axes[col].imshow(rgb)
            if mask is not None and mask.max() > 0.5:
                axes[col].contour(mask, levels=[0.5], colors=_GT_EDGE, linewidths=1.0)
            axes[col].set_title(
                f"ground truth ({int((mask > 0.5).sum())} px)"
                if mask is not None and mask.max() > 0.5 else "ground truth: none",
                fontsize=9,
            )
            col += 1
        heatmap = _resize_2d(segmentation, rgb.shape[:2])
        axes[col].imshow(heatmap, cmap=_HEAT_CMAP, vmin=0.0, vmax=vmax)
        axes[col].set_title(f"anomaly map (0-{vmax:.3f})", fontsize=9)
        col += 1
        overlay_anomaly_map(axes[col], rgb, heatmap, vmax=vmax, cutoff=cutoff,
                            layout=layout, show_grid=layout is not None,
                            gt_mask=mask)
        axes[col].set_title(
            ("overlay + tile grid" if layout is not None else "overlay")
            + f" (>= {cutoff:.3f})",
            fontsize=9,
        )
        for axis in axes:
            axis.axis("off")
        figure.tight_layout()
        figure.savefig(savename, format="jpeg", dpi=_SEG_DPI,
                       pil_kwargs={"quality": 85})
        plt.close(figure)


def write_results_and_confusion(results_dir, rows, model_name):
    """results.csv (existing-compatible) + always-populated confusion CSV."""
    if not rows:
        return
    column_names = [
        "instance_auroc",
        "full_pixel_auroc",
        "anomaly_pixel_auroc",
        "optimal_threshold",
    ]
    compute_and_store = patchcore.utils.compute_and_store_final_results
    result_scores = [[
        row["instance_auroc"],
        row["full_pixel_auroc"],
        row["anomaly_pixel_auroc"],
        row["optimal_threshold"],
    ] for row in rows]
    compute_and_store(
        results_dir,
        result_scores,
        row_names=[row["category"] for row in rows],
        column_names=column_names,
        model_names=[model_name] * len(rows),
    )
    # Categories that could not be evaluated at all (single-class test splits)
    # carry no confusion dict and are absent from the confusion CSV.
    conf_rows = [row for row in rows if row.get("confusion")]
    if not conf_rows:
        LOGGER.info("No evaluated categories with confusion data — no confusion_matrix.csv written.")
        return
    cm_path = os.path.join(results_dir, "confusion_matrix.csv")
    cm_columns = ["TP", "FP", "FN", "TN", "Precision", "Recall"]
    with open(cm_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Model", "Category"] + cm_columns)
        for row in conf_rows:
            counts = row["confusion"]
            writer.writerow([model_name, row["category"]] + [counts.get(c, 0) for c in cm_columns])
        mean_values = [
            float(np.mean([r["confusion"].get(c, 0) for r in conf_rows])) for c in cm_columns
        ]
        writer.writerow([model_name, "Mean"] + mean_values)
