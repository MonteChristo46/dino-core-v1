#!/usr/bin/env python3
"""
PatchCore Plotting Script

Reads saved FPR/TPR curves from training and generates visualization plots.
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from sklearn import metrics

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore Plotting")
    parser.add_argument("--curves-dir", type=str, required=True)
    parser.add_argument("--output-dir", type=str, required=True)
    return parser.parse_args()


def load_curves(curves_dir):
    results = {}
    curves_path = Path(curves_dir)
    for npz_file in sorted(curves_path.glob("*_curves.npz")):
        cat_name = npz_file.stem.replace("_curves", "")
        data = np.load(npz_file)
        results[cat_name] = {
            "image_fpr": data["image_fpr"],
            "image_tpr": data["image_tpr"],
            "pixel_fpr": data["pixel_fpr"],
            "pixel_tpr": data["pixel_tpr"],
            "image_auroc": float(data["image_auroc"]),
            "pixel_auroc": float(data["pixel_auroc"]),
            "anomaly_pixel_auroc": float(data["anomaly_pixel_auroc"]),
            "optimal_threshold": float(data.get("optimal_threshold", 0.5)),
        }
    return results


def save_plot(fig, output_dir, filename):
    jpg_path = output_dir / f"{filename}.jpg"
    fig.savefig(jpg_path, format="jpeg", dpi=150, pil_kwargs={"quality": 85}, bbox_inches="tight")
    LOGGER.info(f"Saved plot to {jpg_path}")


def plot_roc_curves(results, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    all_fpr = []
    all_tpr = []
    mean_auroc = 0.0

    for cat_name, m in results.items():
        fpr = m["image_fpr"]
        tpr = m["image_tpr"]
        auroc = m["image_auroc"]
        mean_auroc += auroc

        ax.plot(fpr, tpr, label=f"{cat_name} (AUROC={auroc:.3f})", alpha=0.7, linewidth=1.5)
        all_fpr.append(fpr)
        all_tpr.append(tpr)

    if all_fpr:
        mean_fpr = np.linspace(0, 1, 100)
        mean_tpr = np.zeros_like(mean_fpr)
        for fpr, tpr in zip(all_fpr, all_tpr):
            mean_tpr += np.interp(mean_fpr, fpr, tpr)
        mean_tpr /= len(all_fpr)
        mean_tpr[0] = 0.0
        mean_auroc /= len(results)

        ax.plot(mean_fpr, mean_tpr, color="black", linewidth=3, linestyle="--",
                label=f"Mean (AUROC={mean_auroc:.3f})")

    ax.plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, alpha=0.5)
    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("False Positive Rate", fontsize=14)
    ax.set_ylabel("True Positive Rate", fontsize=14)
    ax.set_title("ROC Curves - Image-Level Anomaly Detection", fontsize=16)
    ax.legend(loc="lower right", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, output_dir, "roc_curves")
    plt.close()


def plot_pr_curves(results, output_dir):
    fig, ax = plt.subplots(1, 1, figsize=(10, 8))

    for cat_name, m in results.items():
        fpr = m["pixel_fpr"]
        tpr = m["pixel_tpr"]
        precision = tpr
        recall = 1 - fpr

        valid = recall > 0
        ax.plot(recall[valid], precision[valid], label=f"{cat_name}", alpha=0.7, linewidth=1.5)

    ax.set_xlim([0.0, 1.0])
    ax.set_ylim([0.0, 1.05])
    ax.set_xlabel("Recall", fontsize=14)
    ax.set_ylabel("Precision", fontsize=14)
    ax.set_title("Precision-Recall Curves - Pixel-Level", fontsize=16)
    ax.legend(loc="lower left", fontsize=8, ncol=2)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_plot(fig, output_dir, "pr_curves")
    plt.close()


def plot_auroc_comparison(results, output_dir):
    categories = list(results.keys())
    image_aurocs = [results[c]["image_auroc"] for c in categories]
    pixel_aurocs = [results[c]["pixel_auroc"] for c in categories]
    anomaly_pixel_aurocs = [results[c]["anomaly_pixel_auroc"] for c in categories]

    x = np.arange(len(categories))
    width = 0.25

    fig, ax = plt.subplots(1, 1, figsize=(14, 7))

    bars1 = ax.bar(x - width, image_aurocs, width, label="Image AUROC", alpha=0.8)
    bars2 = ax.bar(x, pixel_aurocs, width, label="Pixel AUROC (all)", alpha=0.8)
    bars3 = ax.bar(x + width, anomaly_pixel_aurocs, width, label="Pixel AUROC (anomaly only)", alpha=0.8)

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("AUROC", fontsize=14)
    ax.set_title("AUROC Comparison by Category", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(categories, rotation=45, ha="right", fontsize=9)
    ax.legend(fontsize=11)
    ax.set_ylim([0.0, 1.05])
    ax.grid(True, axis="y", alpha=0.3)

    for bars in [bars1, bars2, bars3]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.2f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=6, rotation=0)

    plt.tight_layout()
    save_plot(fig, output_dir, "auroc_comparison")
    plt.close()


def plot_per_category_roc(results, output_dir):
    per_cat_dir = output_dir / "per_category"
    per_cat_dir.mkdir(parents=True, exist_ok=True)

    for cat_name, m in results.items():
        fig, axes = plt.subplots(1, 2, figsize=(14, 5))

        axes[0].plot(m["image_fpr"], m["image_tpr"], color="blue", linewidth=2,
                     label=f"AUROC={m['image_auroc']:.3f}")
        axes[0].plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, alpha=0.5)
        axes[0].set_xlim([0.0, 1.0])
        axes[0].set_ylim([0.0, 1.05])
        axes[0].set_xlabel("False Positive Rate", fontsize=12)
        axes[0].set_ylabel("True Positive Rate", fontsize=12)
        axes[0].set_title(f"{cat_name} - Image-Level ROC", fontsize=14)
        axes[0].legend(fontsize=11)
        axes[0].grid(True, alpha=0.3)

        axes[1].plot(m["pixel_fpr"], m["pixel_tpr"], color="red", linewidth=2,
                     label=f"AUROC={m['pixel_auroc']:.3f}")
        axes[1].plot([0, 1], [0, 1], color="gray", linestyle=":", linewidth=1, alpha=0.5)
        axes[1].set_xlim([0.0, 1.0])
        axes[1].set_ylim([0.0, 1.05])
        axes[1].set_xlabel("False Positive Rate", fontsize=12)
        axes[1].set_ylabel("True Positive Rate", fontsize=12)
        axes[1].set_title(f"{cat_name} - Pixel-Level ROC", fontsize=14)
        axes[1].legend(fontsize=11)
        axes[1].grid(True, alpha=0.3)

        plt.tight_layout()
        save_plot(fig, per_cat_dir, f"{cat_name}_roc")
        plt.close()


def main():
    logging.basicConfig(level=logging.INFO)

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    LOGGER.info(f"Loading curves from {args.curves_dir}")
    results = load_curves(args.curves_dir)

    if not results:
        LOGGER.error("No curve data found. Make sure training completed successfully.")
        sys.exit(1)

    LOGGER.info(f"Found curves for {len(results)} categories: {', '.join(results.keys())}")

    LOGGER.info("Generating plots...")
    plot_roc_curves(results, output_dir)
    plot_pr_curves(results, output_dir)
    plot_auroc_comparison(results, output_dir)
    plot_per_category_roc(results, output_dir)

    LOGGER.info(f"All plots saved to {output_dir}")


if __name__ == "__main__":
    main()
