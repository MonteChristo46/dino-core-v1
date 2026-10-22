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


def plot_top_anomalous(image_paths, scores, segmentations, layout, save_path,
                       n=4, title="Top anomaly candidates"):
    """Original | heatmap | heatmap+tile-grid for the highest-scoring images.

    The grid overlay shows whether hotspots coincide with tile seams/borders
    (padding artifacts) or sit on real content.
    """
    order = np.argsort(scores)[::-1][:n]
    count = len(order)
    fig, axes = plt.subplots(count, 3, figsize=(9.5, 3.2 * count), dpi=110,
                             squeeze=False)
    vmax = max(float(np.max(segmentations)), 1e-12)
    for row, idx in enumerate(order):
        with PIL.Image.open(image_paths[idx]) as image:
            rgb = np.asarray(image.convert("RGB"))
        heatmap = segmentations[idx]
        axes[row][0].imshow(rgb)
        axes[row][0].set_title(f"{os.path.basename(image_paths[idx])}\nscore={scores[idx]:.3f}")
        axes[row][1].imshow(heatmap, cmap="magma", vmin=0.0, vmax=vmax)
        axes[row][1].set_title("anomaly map")
        axes[row][2].imshow(rgb)
        axes[row][2].imshow(np.ma.masked_where(heatmap < 0.5 * vmax, heatmap),
                            cmap="autumn", alpha=0.75)
        if layout is not None:
            for y, x in layout.origins:
                axes[row][2].add_patch(
                    matplotlib.patches.Rectangle((x, y), layout.tile_size, layout.tile_size,
                                                 fill=False, edgecolor="cyan", linewidth=0.7)
                )
        axes[row][2].set_title("overlay" + (" + tile grid" if layout else ""))
        for axis_row in axes[row]:
            axis_row.axis("off")
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(save_path)
    plt.close(fig)


def save_segmentation_images_for_category(
    savefolder, dataset, scores, segmentations, records, threshold
):
    """Reuse patchcore.utils plotting with ImageNet denormalization."""
    image_paths = [t[2] for t in dataset.data_to_iterate]
    mask_paths = [t[3] for t in dataset.data_to_iterate]
    if not any(mask is not None for mask in mask_paths):
        mask_paths = None  # -> utils skips the (all-black) GT panel
    categories = [result for result, _ in records]
    std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
    mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)

    def image_transform(image):
        tensor = dataset.transform_img(image)
        return np.clip((tensor.numpy() * std + mean) * 255, 0, 255).astype(np.uint8)

    def mask_transform(mask):
        return dataset.transform_mask(mask).numpy()

    patchcore.utils.plot_segmentation_images(
        savefolder,
        image_paths,
        segmentations,
        scores,
        mask_paths,
        image_transform=image_transform,
        mask_transform=mask_transform,
        categories=categories,
        optimal_threshold=threshold,
    )


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
    cm_path = os.path.join(results_dir, "confusion_matrix.csv")
    cm_columns = ["TP", "FP", "FN", "TN", "Precision", "Recall"]
    with open(cm_path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Model", "Category"] + cm_columns)
        for row in rows:
            counts = row["confusion"]
            writer.writerow([model_name, row["category"]] + [counts.get(c, 0) for c in cm_columns])
        mean_values = [
            float(np.mean([r["confusion"].get(c, 0) for r in rows])) for c in cm_columns
        ]
        writer.writerow([model_name, "Mean"] + mean_values)
