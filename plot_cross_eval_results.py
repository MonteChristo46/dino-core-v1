#!/usr/bin/env python3
"""
PatchCore Cross-Evaluation Plotting Script

Reads results.csv and confusion_matrix.csv from a cross-evaluation run
and generates visualizations at the fixed saved threshold.
No curve recomputation — plots only the already-computed metrics.
"""

import argparse
import csv
import logging
import textwrap
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore Cross-Evaluation Plotting")
    parser.add_argument("--results-dir", type=str, required=True,
                        help="Path to the cross-eval run folder containing results.csv and confusion_matrix.csv")
    parser.add_argument("--output-dir", type=str, default=None,
                        help="Where to save plots (defaults to <results-dir>/plots)")
    return parser.parse_args()


def load_csv(filepath):
    rows = []
    with open(filepath, "r") as f:
        reader = csv.DictReader(f)
        for row in reader:
            rows.append(row)
    return rows


def save_plot(fig, output_dir, filename):
    jpg_path = output_dir / f"{filename}.jpg"
    fig.savefig(jpg_path, format="jpeg", dpi=150, pil_kwargs={"quality": 85}, bbox_inches="tight")
    LOGGER.info(f"Saved plot to {jpg_path}")


def get_category(row):
    for key in ("Category", "category", "Row Names", "row_names", "row name"):
        if key in row:
            raw = row[key]
            if " | " in raw:
                return raw.split(" | ")[-1].strip()
            return raw.strip()
    return ""


def wrap_labels(labels, max_width=12):
    return ["\n".join(textwrap.wrap(l, width=max_width, break_long_words=False)) for l in labels]


def compute_figsize(n_categories, max_label_len, bar_groups=2):
    width = max(8, n_categories * max(1.5, max_label_len * 0.12))
    height = 6 + (max_label_len // 15) * 1.5
    return (width, height)


def plot_precision_recall_barchart(cm_rows, output_dir):
    categories = []
    precisions = []
    recalls = []
    for row in cm_rows:
        cat = get_category(row)
        if cat.lower() == "mean":
            continue
        categories.append(cat)
        precisions.append(float(row["Precision"]))
        recalls.append(float(row["Recall"]))

    if not categories:
        LOGGER.warning("No category rows in confusion_matrix.csv, skipping precision/recall bar chart.")
        return

    labels = wrap_labels(categories)
    max_len = max(len(c) for c in categories)
    fig, ax = plt.subplots(1, 1, figsize=compute_figsize(len(categories), max_len))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, precisions, width, label="Precision", color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x + width / 2, recalls, width, label="Recall", color="#4CAF50", alpha=0.85)

    mean_precision = np.mean(precisions)
    mean_recall = np.mean(recalls)
    ax.axhline(mean_precision, color="#1565C0", linestyle="--", linewidth=1.5,
               label=f"Mean Precision ({mean_precision:.3f})")
    ax.axhline(mean_recall, color="#2E7D32", linestyle="--", linewidth=1.5,
               label=f"Mean Recall ({mean_recall:.3f})")

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    ax.set_title("Precision & Recall per Category (Fixed Threshold)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.legend(fontsize=11, loc="lower left")
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.2f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    plt.tight_layout()
    save_plot(fig, output_dir, "precision_recall_barchart")
    plt.close()


def plot_confusion_matrix_heatmap(cm_rows, output_dir):
    categories = []
    tp_vals = []
    fp_vals = []
    fn_vals = []
    tn_vals = []
    for row in cm_rows:
        cat = get_category(row)
        if cat.lower() == "mean":
            continue
        categories.append(cat)
        tp_vals.append(float(row["TP"]))
        fp_vals.append(float(row["FP"]))
        fn_vals.append(float(row["FN"]))
        tn_vals.append(float(row["TN"]))

    if not categories:
        LOGGER.warning("No category rows in confusion_matrix.csv, skipping confusion matrix heatmap.")
        return

    data = np.array([tp_vals, fp_vals, fn_vals, tn_vals]).T
    col_labels = ["TP", "FP", "FN", "TN"]
    labels = wrap_labels(categories, max_width=14)

    max_len = max(len(c) for c in categories)
    fig_width = max(9, max_len * 0.55, len(categories) * 0.5)
    fig_height = max(5, len(categories) * 0.55)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    im = ax.imshow(data, cmap="YlOrRd", aspect="auto")

    for i in range(len(categories)):
        for j in range(len(col_labels)):
            val = data[i, j]
            text_color = "white" if val > data.max() * 0.6 else "black"
            ax.text(j, i, f"{val:.0f}", ha="center", va="center",
                    fontsize=11, fontweight="bold", color=text_color)

    ax.set_xticks(np.arange(len(col_labels)))
    ax.set_xticklabels(col_labels, fontsize=13, fontweight="bold")
    ax.set_yticks(np.arange(len(categories)))
    ax.set_yticklabels(labels, fontsize=9)

    ax.set_xlabel("Prediction Outcome", fontsize=14, fontweight="bold")
    ax.set_ylabel("Category", fontsize=14, fontweight="bold")
    ax.set_title("Confusion Matrix Counts per Category", fontsize=16, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label("Count", fontsize=12)

    plt.tight_layout()
    save_plot(fig, output_dir, "confusion_matrix_heatmap")
    plt.close()


def plot_threshold_comparison(results_rows, output_dir):
    categories = []
    saved_thresholds = []
    computed_thresholds = []
    for row in results_rows:
        cat = get_category(row)
        try:
            saved = float(row["saved_threshold"])
            computed = float(row["computed_threshold"])
        except (KeyError, ValueError):
            continue
        categories.append(cat)
        saved_thresholds.append(saved)
        computed_thresholds.append(computed)

    if not categories:
        LOGGER.warning("No threshold data in results.csv, skipping threshold comparison.")
        return

    saved_arr = np.array(saved_thresholds)
    computed_arr = np.array(computed_thresholds)
    deltas = computed_arr - saved_arr
    abs_deltas = np.abs(deltas)

    mean_abs_delta = np.mean(abs_deltas)
    mean_signed_delta = np.mean(deltas)
    max_abs_delta = np.max(abs_deltas)
    median_abs_delta = np.median(abs_deltas)

    labels = wrap_labels(categories)
    max_len = max(len(c) for c in categories)
    fig, ax = plt.subplots(1, 1, figsize=compute_figsize(len(categories), max_len))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, saved_thresholds, width,
                   label="Saved Threshold (source domain)", color="#FF9800", alpha=0.85)
    bars2 = ax.bar(x + width / 2, computed_thresholds, width,
                   label="Computed Threshold (target domain)", color="#9C27B0", alpha=0.85)

    ax.plot(x, deltas, "o-", color="#E53935", linewidth=2, markersize=6,
            label="Δ (computed − saved)", zorder=5)
    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Threshold", fontsize=14)
    ax.set_title("Saved vs Computed Threshold per Category", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.legend(fontsize=10, loc="upper left", ncol=2)
    ax.grid(True, axis="y", alpha=0.3)

    for bar in bars1:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)
    for bar in bars2:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}",
                    xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 3), textcoords="offset points",
                    ha="center", va="bottom", fontsize=8)

    delta_text = (
        f"Threshold Drift Summary\n"
        f"Mean |Δ|: {mean_abs_delta:.4f}\n"
        f"Mean Δ:   {mean_signed_delta:+.4f}\n"
        f"Median |Δ|: {median_abs_delta:.4f}\n"
        f"Max |Δ|:  {max_abs_delta:.4f}"
    )
    ax.text(0.98, 0.02, delta_text, transform=ax.transAxes,
            fontsize=10, verticalalignment="bottom", horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.5", facecolor="white", edgecolor="gray", alpha=0.9))

    plt.tight_layout()
    save_plot(fig, output_dir, "threshold_comparison")
    plt.close()


def plot_auroc_transfer_barchart(results_rows, output_dir):
    categories = []
    instance_aurocs = []
    full_pixel_aurocs = []
    anomaly_pixel_aurocs = []
    for row in results_rows:
        cat = get_category(row)
        try:
            ia = float(row["instance_auroc"])
            fpa = float(row["full_pixel_auroc"])
            apa = float(row["anomaly_pixel_auroc"])
        except (KeyError, ValueError):
            continue
        categories.append(cat)
        instance_aurocs.append(ia)
        full_pixel_aurocs.append(fpa)
        anomaly_pixel_aurocs.append(apa)

    if not categories:
        LOGGER.warning("No AUROC data in results.csv, skipping AUROC transfer chart.")
        return

    labels = wrap_labels(categories)
    max_len = max(len(c) for c in categories)
    fig, ax = plt.subplots(1, 1, figsize=compute_figsize(len(categories), max_len, bar_groups=3))

    x = np.arange(len(categories))
    width = 0.25

    bars1 = ax.bar(x - width, instance_aurocs, width, label="Instance AUROC",
                   color="#2196F3", alpha=0.85)
    bars2 = ax.bar(x, full_pixel_aurocs, width, label="Full Pixel AUROC",
                   color="#FF9800", alpha=0.85)
    bars3 = ax.bar(x + width, anomaly_pixel_aurocs, width, label="Anomaly Pixel AUROC",
                   color="#4CAF50", alpha=0.85)

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("AUROC", fontsize=14)
    ax.set_title("AUROC Transfer Performance per Category", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    plt.tight_layout()
    save_plot(fig, output_dir, "auroc_transfer_barchart")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()
    results_dir = Path(args.results_dir)
    output_dir = Path(args.output_dir) if args.output_dir else results_dir / "plots"
    output_dir.mkdir(parents=True, exist_ok=True)

    cm_path = results_dir / "confusion_matrix.csv"
    results_path = results_dir / "results.csv"

    if not cm_path.exists():
        LOGGER.error(f"confusion_matrix.csv not found at {cm_path}")
        return
    if not results_path.exists():
        LOGGER.error(f"results.csv not found at {results_path}")
        return

    LOGGER.info(f"Loading confusion_matrix.csv from {cm_path}")
    cm_rows = load_csv(cm_path)
    LOGGER.info(f"Loading results.csv from {results_path}")
    results_rows = load_csv(results_path)

    LOGGER.info(f"Found {len(cm_rows)} rows in confusion_matrix.csv")
    LOGGER.info(f"Found {len(results_rows)} rows in results.csv")

    LOGGER.info("Generating cross-evaluation plots...")
    plot_precision_recall_barchart(cm_rows, output_dir)
    plot_confusion_matrix_heatmap(cm_rows, output_dir)
    plot_threshold_comparison(results_rows, output_dir)
    plot_auroc_transfer_barchart(results_rows, output_dir)

    LOGGER.info(f"All cross-evaluation plots saved to {output_dir}")


if __name__ == "__main__":
    main()
