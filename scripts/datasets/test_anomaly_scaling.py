"""
Synthetic anomaly scaling experiment: How many synthetic anomalies are needed
to calibrate a good decision threshold?

Flow:
  1. Train PatchCore on 100 fixed good images.
  2. Run inference once on the real MVTec test set (good + real bad).
  3. For each bad_size N, repeat R times:
     a. Sample N synthetic bad images + all synthetic good test images.
     b. Compute optimal threshold from PR curve on synthetic data.
     c. Apply that threshold to the real MVTec scores -> F1, Precision, Recall.
     d. AUROC is threshold-independent, computed directly on real scores.

Usage:
    uv run python scripts/datasets/test_anomaly_scaling.py \
        --synthetic-dataset /path/to/semantic_synthetic_anomalies_v4 \
        --mvtec-dataset /path/to/mvtec_small \
        --categories cable \
        --bad-sizes 1 5 10 20 30 40 50 \
        --n-repeats 3 \
        --output output/test_anomaly_scaling
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

BAD_SIZES_DEFAULT = [1, 5, 10, 20, 30, 40, 50,100]
CATEGORIES_DEFAULT = ["cable", "capsule", "screw", "wood"]

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
TRAIN_GOOD_FIXED = 100


def _subsample(dataset, n: int, rng: random.Random) -> torch.utils.data.Dataset:
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


def train_once(dataset_path: str, category: str, seed: int, device: torch.device):
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
    train_subset = _subsample(train_dataset, TRAIN_GOOD_FIXED, rng)
    train_loader = torch.utils.data.DataLoader(
        train_subset,
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
    return pc


def run_inference(pc, dataset, device: torch.device):
    loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )
    scores, _, labels_gt, _ = pc.predict(loader)
    scores_np = np.array(scores)
    min_s = scores_np.min()
    max_s = scores_np.max()
    if max_s - min_s > 0:
        scores_np = (scores_np - min_s) / (max_s - min_s)
    return scores_np, np.array(labels_gt)


def compute_threshold_from_synthetic(pc, synthetic_test_ds, good_indices, bad_indices, device: torch.device):
    class TestSubset(torch.utils.data.Dataset):
        def __init__(self, base_ds, good_idxs, bad_idxs):
            self.base = base_ds
            self.good_idxs = good_idxs
            self.bad_idxs = bad_idxs
            self.imagesize = base_ds.imagesize

        def __len__(self):
            return len(self.good_idxs) + len(self.bad_idxs)

        def __getitem__(self, idx):
            if idx < len(self.good_idxs):
                return self.base[self.good_idxs[idx]]
            else:
                return self.base[self.bad_idxs[idx - len(self.good_idxs)]]

    subset = TestSubset(synthetic_test_ds, good_indices, bad_indices)
    scores, labels = run_inference(pc, subset, device)
    image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(scores, labels)
    return image_metrics["optimal_threshold"]


def evaluate_on_real(real_scores, real_labels, threshold):
    auroc_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(real_scores, real_labels)
    auroc = auroc_metrics["auroc"]

    preds = [1 if s >= threshold else 0 for s in real_scores]
    f1 = f1_score(real_labels, preds)
    tp = sum(1 for p, l in zip(preds, real_labels) if p == 1 and l == 1)
    fp = sum(1 for p, l in zip(preds, real_labels) if p == 1 and l == 0)
    fn = sum(1 for p, l in zip(preds, real_labels) if p == 0 and l == 1)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

    return {
        "f1": f1,
        "auroc": auroc,
        "precision": precision,
        "recall": recall,
        "threshold_used": threshold,
    }


def main():
    parser = argparse.ArgumentParser(description="Synthetic anomaly scaling experiment")
    parser.add_argument(
        "--synthetic-dataset",
        type=str,
        default="/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/semantic_synthetic_anomalies_v4",
        help="Path to synthetic anomaly dataset (for threshold calibration)",
    )
    parser.add_argument(
        "--mvtec-dataset",
        type=str,
        default="/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/mvtec_small",
        help="Path to real MVTec dataset (for evaluation)",
    )
    parser.add_argument("--categories", nargs="+", default=CATEGORIES_DEFAULT)
    parser.add_argument("--bad-sizes", nargs="+", type=int, default=BAD_SIZES_DEFAULT)
    parser.add_argument("--n-repeats", type=int, default=3)
    parser.add_argument("--output", type=str, default="output/test_anomaly_scaling")
    parser.add_argument("--gpu", type=int, default=0)
    args = parser.parse_args()

    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)

    device = patchcore.utils.set_torch_device([args.gpu])
    _load_backbone().to(device)

    bad_sizes = sorted(args.bad_sizes)
    categories = args.categories

    results = {}
    valid_sizes_map = {}

    for category in categories:
        LOGGER.info(f"=== Category: {category} ===")

        synthetic_test_ds = mvtec.MVTecDataset(
            source=args.synthetic_dataset,
            classname=category,
            resize=RESIZE,
            imagesize=IMAGE_SIZE,
            split=mvtec.DatasetSplit.TEST,
        )
        real_test_ds = mvtec.MVTecDataset(
            source=args.mvtec_dataset,
            classname=category,
            resize=RESIZE,
            imagesize=IMAGE_SIZE,
            split=mvtec.DatasetSplit.TEST,
        )

        synth_indices = list(range(len(synthetic_test_ds)))
        synth_is_bad = [synthetic_test_ds.data_to_iterate[i][1] != "good" for i in synth_indices]
        synth_good_indices = [i for i, b in zip(synth_indices, synth_is_bad) if not b]
        synth_bad_indices = [i for i, b in zip(synth_indices, synth_is_bad) if b]

        max_available = len(synth_bad_indices)
        LOGGER.info(f"  Synthetic test/bad available: {max_available}")
        LOGGER.info(f"  Real test set size: {len(real_test_ds)}")

        valid_sizes = [s for s in bad_sizes if s <= max_available]
        if not valid_sizes:
            LOGGER.warning(f"  No valid bad sizes for {category} (max={max_available}), skipping")
            continue

        LOGGER.info(f"  Training PatchCore on {TRAIN_GOOD_FIXED} good images (seed=42)")
        pc = train_once(args.synthetic_dataset, category, seed=42, device=device)
        LOGGER.info(f"  Training done.")

        LOGGER.info(f"  Running inference on real MVTec test set (once)...")
        real_scores, real_labels = run_inference(pc, real_test_ds, device)
        LOGGER.info(f"  Real test: {len(real_scores)} images, {sum(real_labels)} bad, {len(real_labels) - sum(real_labels)} good")

        results[category] = {}
        valid_sizes_map[category] = valid_sizes
        for size_idx, size in enumerate(valid_sizes):
            results[category][size_idx] = {}
            for repeat in range(args.n_repeats):
                rng = random.Random(repeat * 1000 + 42)
                sampled_bad = rng.sample(synth_bad_indices, size)
                LOGGER.info(f"  bad_size={size}, repeat={repeat}")

                threshold = compute_threshold_from_synthetic(
                    pc, synthetic_test_ds, synth_good_indices, sampled_bad, device
                )
                metrics = evaluate_on_real(real_scores, real_labels, threshold)
                results[category][size_idx][repeat] = metrics
                LOGGER.info(
                    f"    threshold={threshold:.4f}, F1={metrics['f1']:.4f}, "
                    f"AUROC={metrics['auroc']:.4f}, P={metrics['precision']:.4f}, R={metrics['recall']:.4f}"
                )

    csv_path = output_dir / "test_anomaly_scaling.csv"
    with open(csv_path, "w") as f:
        f.write("category,synth_bad_size,repeat,threshold_used,f1,auroc,precision,recall\n")
        for cat in categories:
            if cat not in results:
                continue
            for size_idx, size in enumerate(valid_sizes_map[cat]):
                for repeat in range(args.n_repeats):
                    if repeat not in results[cat][size_idx]:
                        continue
                    m = results[cat][size_idx][repeat]
                    f.write(f"{cat},{size},{repeat},{m['threshold_used']:.6f},{m['f1']:.6f},"
                            f"{m['auroc']:.6f},{m['precision']:.6f},{m['recall']:.6f}\n")
    LOGGER.info(f"CSV saved to {csv_path}")

    metrics_keys = ["f1", "auroc", "precision", "recall"]
    metric_labels = {"f1": "F1 Score", "auroc": "AUROC", "precision": "Precision", "recall": "Recall"}
    colors = plt.cm.tab10(np.linspace(0, 1, max(len(categories), 1)))

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    for ax, metric_key in zip(axes, metrics_keys):
        for ci, cat in enumerate(categories):
            if cat not in results:
                continue
            means = []
            stds = []
            xs = []
            for size_idx, size in enumerate(valid_sizes_map[cat]):
                vals = [results[cat][size_idx][r][metric_key] for r in range(args.n_repeats) if r in results[cat][size_idx]]
                if vals:
                    means.append(np.mean(vals))
                    stds.append(np.std(vals))
                    xs.append(size)
            if not means:
                continue
            means = np.array(means)
            stds = np.array(stds)
            ax.errorbar(
                xs, means, yerr=stds,
                marker="o", capsize=4, capthick=1,
                label=cat, color=colors[ci], linewidth=1.5,
            )
        ax.set_xlabel("Synthetic anomalies used for threshold calibration")
        ax.set_ylabel(metric_labels[metric_key])
        ax.set_title(metric_labels[metric_key])
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8)

    fig.suptitle(f"Synthetic Anomaly Scaling (evaluated on real MVTec) — {', '.join(categories)} — {BACKBONE_NAME} L{'-'.join(LAYERS)}", fontsize=14)
    fig.tight_layout()

    plot_path = output_dir / "test_anomaly_scaling.jpg"
    fig.savefig(plot_path, dpi=150, bbox_inches="tight")
    LOGGER.info(f"Plot saved to {plot_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
