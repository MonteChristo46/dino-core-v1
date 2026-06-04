"""
Learning curve experiment: F1 / AUROC / Precision / Recall vs training dataset size.

Runs PatchCore with varying numbers of training images across multiple seeds,
producing a CSV and a multi-panel plot with one line per category and error bars.

Usage:
    uv run python scripts/datasets/learning_curve.py \
        --dataset /path/to/mvtec_small \
        --categories cable \
        --sizes 1 5 10 20 50 100 200 \
        --n-repeats 3 \
        --output output/learning_curve
"""

import argparse
import logging
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from sklearn.metrics import f1_score

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

import patchcore.backbones
import patchcore.common
import patchcore.datasets.mvtec as mvtec
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
LOGGER = logging.getLogger(__name__)

SIZES_DEFAULT = [1, 3, 5, 10, 20, 50, 100, 200]
CATEGORIES_DEFAULT = ["cable", "capsule", "screw", "wood"]

# Default model config (matches IM384_MVTEC_BENCH_dinov3_vitl_L8-12-16_P0.1_D1024-1024_PS-3_AN-9_S0)
BACKBONE_NAME = "dinov3_vitl"
LAYERS = ("8", "12", "16")
CORESET_PCT = 0.1
PRETRAIN_DIM = 1024
TARGET_DIM = 1024
PATCH_SIZE = 3
NUM_NN = 9
IMAGE_SIZE = 384
RESIZE = 384
BATCH_SIZE = 2
NUM_WORKERS = 8


def _subsample_train_dataset(dataset, n: int, rng: random.Random) -> torch.utils.data.Dataset:
    """Return a wrapper that only exposes a random subset of the training data."""
    indices = rng.sample(range(len(dataset)), min(n, len(dataset)))
    indices.sort()

    class SubsetDataset:
        def __init__(self, base_ds, idxs):
            self.base = base_ds
            self.idxs = idxs
            self.imagesize = base_ds.imagesize

        def __len__(self):
            return len(self.idxs)

        def __getitem__(self, idx):
            return self.base[self.idxs[idx]]

    return SubsetDataset(dataset, indices)


def _create_patchcore(device: torch.device) -> patchcore.patchcore.PatchCore:
    nn_method = patchcore.common.FaissNN(False, NUM_WORKERS)
    pc = patchcore.patchcore.PatchCore(device)
    pc.load(
        backbone=_load_backbone(),
        layers_to_extract_from=list(LAYERS),
        device=device,
        input_shape=(3, IMAGE_SIZE, IMAGE_SIZE),
        pretrain_embed_dimension=PRETRAIN_DIM,
        target_embed_dimension=TARGET_DIM,
        patchsize=PATCH_SIZE,
        featuresampler=patchcore.sampler.ApproximateGreedyCoresetSampler(CORESET_PCT, device),
        anomaly_scorer_num_nn=NUM_NN,
        nn_method=nn_method,
    )
    return pc


_backbone_cache = None


def _load_backbone():
    global _backbone_cache
    if _backbone_cache is not None:
        return _backbone_cache
    backbone = patchcore.backbones.load(BACKBONE_NAME)
    backbone.name = BACKBONE_NAME
    _backbone_cache = backbone
    return backbone


def run_single(
    category: str,
    dataset_path: str,
    train_size: int,
    seed: int,
    device: torch.device,
) -> dict[str, float]:
    rng = random.Random(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)

    train_dataset = mvtec.MVTecDataset(
        source=dataset_path,
        classname=category,
        resize=RESIZE,
        imagesize=IMAGE_SIZE,
        split=mvtec.DatasetSplit.TRAIN,
        train_val_split=1.0,
    )
    test_dataset = mvtec.MVTecDataset(
        source=dataset_path,
        classname=category,
        resize=RESIZE,
        imagesize=IMAGE_SIZE,
        split=mvtec.DatasetSplit.TEST,
    )

    train_subset = _subsample_train_dataset(train_dataset, train_size, rng)

    train_loader = torch.utils.data.DataLoader(
        train_subset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    test_loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    if "cuda" in device.type.lower():
        with torch.cuda.device(device.index):
            torch.cuda.empty_cache()

    pc = _create_patchcore(device)
    pc.fit(train_loader)
    scores, segmentations, labels_gt, masks_gt = pc.predict(test_loader)

    if "cuda" in device.type.lower():
        with torch.cuda.device(device.index):
            torch.cuda.empty_cache()

    scores_np = np.array(scores)
    min_s = scores_np.min()
    max_s = scores_np.max()
    if max_s - min_s > 0:
        scores_np = (scores_np - min_s) / (max_s - min_s)

    image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
        scores_np, labels_gt
    )
    auroc = image_metrics["auroc"]
    optimal_threshold = image_metrics["optimal_threshold"]

    preds = [1 if s >= optimal_threshold else 0 for s in scores_np]
    f1 = f1_score(labels_gt, preds)
    tp = sum(1 for p, l in zip(preds, labels_gt) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, labels_gt) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, labels_gt) if p == 0 and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "f1": f1,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "optimal_threshold": optimal_threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Learning curve experiment")
    parser.add_argument(
        "--dataset",
        type=str,
        default="/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/mvtec_small",
        help="Path to MVTec-style dataset",
    )
    parser.add_argument("--categories", nargs="+", default=CATEGORIES_DEFAULT)
    parser.add_argument("--sizes", nargs="+", type=int, default=SIZES_DEFAULT)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default="output/learning_curve")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = patchcore.utils.set_torch_device([args.gpu])
    _load_backbone().to(device)

    sizes = sorted(args.sizes)
    categories = args.categories

    # results[category][size_idx][repeat_idx] = {f1, auroc, precision, recall}
    results = {}

    for category in categories:
        LOGGER.info(f"=== Category: {category} ===")
        results[category] = {}
        for size_idx, size in enumerate(sizes):
            results[category][size_idx] = {}
            for repeat in range(args.n_repeats):
                seed = repeat * 1000 + 42
                LOGGER.info(f"  size={size}, repeat={repeat}, seed={seed}")
                metrics = run_single(category, args.dataset, size, seed, device)
                results[category][size_idx][repeat] = metrics
                LOGGER.info(
                    f"    F1={metrics['f1']:.4f}, AUROC={metrics['auroc']:.4f}, "
                    f"P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
                )

    # Write CSV
    csv_path = output_dir / "learning_curve.csv"
    with open(csv_path, "w") as f:
        f.write("category,train_size,repeat,f1,auroc,precision,recall,optimal_threshold\n")
        for cat in categories:
            for size_idx, size in enumerate(sizes):
                for repeat in range(args.n_repeats):
                    m = results[cat][size_idx][repeat]
                    f.write(f"{cat},{size},{repeat},{m['f1']:.6f},{m['auroc']:.6f},"
                            f"{m['precision']:.6f},{m['recall']:.6f},{m['optimal_threshold']:.6f}\n")
    LOGGER.info(f"CSV saved to {csv_path}")

    # Plot
    metrics_keys = ["f1", "auroc", "precision", "recall"]
    metric_labels = {"f1": "F1 Score", "auroc": "AUROC", "precision": "Precision", "recall": "Recall"}
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric_key in zip(axes, metrics_keys):
        for ci, cat in enumerate(categories):
            means = []
            stds = []
            for size_idx in range(len(sizes)):
                vals = [results[cat][size_idx][r][metric_key] for r in range(args.n_repeats)]
                means.append(np.mean(vals))
                stds.append(np.std(vals))
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(
                sizes, means, yerr=stds,
                marker="o", capsize=4, capthick=1,
                label=cat, color=colors[ci], linewidth=1.5,
            )
        ax.set_xlabel("Training images")
        ax.set_ylabel(metric_labels[metric_key])
        ax.set_title(metric_labels[metric_key])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)
        ax.set_xscale("log")
        ax.set_xticks(sizes)
        ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    fig.suptitle(f"Learning Curve — {', '.join(categories)} — {BACKBONE_NAME} L{'-'.join(LAYERS)}", fontsize=14)
    fig.tight_layout()

    plot_path = output_dir / "learning_curve.jpg"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    LOGGER.info(f"Plot saved to {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
