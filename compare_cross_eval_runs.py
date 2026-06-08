#!/usr/bin/env python3
"""
PatchCore Cross-Evaluation Run Comparison Script

Takes multiple cross-eval run directories and generates comparison plots
across all runs: threshold delta, AUROC transfer, threshold scatter,
precision heatmap, recall heatmap, F1 heatmap, and delta boxplot.
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

PALETTE = [
    "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#E53935",
    "#00BCD4", "#FF5722", "#8BC34A", "#673AB7", "#F44336",
]


def parse_args():
    parser = argparse.ArgumentParser(description="PatchCore Cross-Eval Run Comparison")
    parser.add_argument("--runs", nargs="+", required=True,
                        help="Paths to cross-eval run directories containing results.csv")
    parser.add_argument("--output-dir", type=str, required=True,
                        help="Where to save comparison plots")
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


def get_model_name(row):
    for key in ("Model", "model"):
        if key in row:
            return row[key].strip()
    return ""


def wrap_labels(labels, max_width=12):
    return ["\n".join(textwrap.wrap(l, width=max_width, break_long_words=False)) for l in labels]


def compute_figsize(n_categories, max_label_len, bar_groups=2):
    width = max(8, n_categories * max(1.5, max_label_len * 0.12))
    height = 6 + (max_label_len // 15) * 1.5
    return (width, height)


def model_label(name):
    return name.replace("__", " / ")



def load_run_data(run_dir):
    results_path = Path(run_dir) / "results.csv"
    if not results_path.exists():
        LOGGER.warning(f"results.csv not found in {run_dir}, skipping.")
        return None

    rows = load_csv(results_path)
    categories = []
    saved = []
    computed = []
    instance_auroc = []
    full_pixel_auroc = []
    anomaly_pixel_auroc = []
    precision = []
    recall = []

    model_name = ""
    for row in rows:
        cat = get_category(row)
        if cat == "Mean" or not cat:
            continue
        if not model_name:
            model_name = get_model_name(row)
        try:
            categories.append(cat)
            saved.append(float(row["saved_threshold"]))
            computed.append(float(row["computed_threshold"]))
            instance_auroc.append(float(row["instance_auroc"]))
            full_pixel_auroc.append(float(row["full_pixel_auroc"]))
            anomaly_pixel_auroc.append(float(row["anomaly_pixel_auroc"]))
            precision.append(float(row["Precision"]))
            recall.append(float(row["Recall"]))
        except (KeyError, ValueError):
            continue

    return {
        "model": model_name,
        "categories": categories,
        "saved_threshold": np.array(saved),
        "computed_threshold": np.array(computed),
        "delta": np.array(computed) - np.array(saved),
        "abs_delta": np.abs(np.array(computed) - np.array(saved)),
        "instance_auroc": np.array(instance_auroc),
        "full_pixel_auroc": np.array(full_pixel_auroc),
        "anomaly_pixel_auroc": np.array(anomaly_pixel_auroc),
        "precision": np.array(precision),
        "recall": np.array(recall),
        "f1": np.array([
            2 * p * r / (p + r) if (p + r) > 0 else 0.0
            for p, r in zip(precision, recall)
        ]),
    }


def plot_threshold_delta_grouped(run_data_list, categories, output_dir):
    n_models = len(run_data_list)
    n_cats = len(categories)
    labels = wrap_labels(categories)
    max_len = max(len(c) for c in categories)
    max_model_len = max(len(model_label(rd["model"])) for rd in run_data_list) if run_data_list else 20
    fig_width = max(12, n_cats * max(1.5, max_len * 0.12), max_model_len * 0.12 * 10)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    x = np.arange(n_cats)
    width = 0.8 / max(n_models, 1)

    for i, rd in enumerate(run_data_list):
        offset = (i - (n_models - 1) / 2) * width
        bars = ax.bar(x + offset, rd["abs_delta"], width,
                      label=model_label(rd["model"]), color=PALETTE[i % len(PALETTE)], alpha=0.85)
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.3f}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=7)

    for i, rd in enumerate(run_data_list):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, rd["delta"], "o-", color=color, linewidth=1.5, markersize=4,
                alpha=0.35, zorder=3, label=f"{model_label(rd['model'])} Δ (signed)")

    ax.axhline(0, color="gray", linewidth=1, linestyle="--", alpha=0.5)

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("|Δ| (|computed − saved|)", fontsize=14)
    ax.set_title("Threshold Drift per Category (Cross-Model)", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.legend(fontsize=8, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
    save_plot(fig, output_dir, "threshold_delta_grouped")
    plt.close()


def plot_delta_lines(run_data_list, categories, output_dir):
    labels = wrap_labels(categories)
    max_len = max(len(c) for c in categories)
    max_model_len = max(len(model_label(rd["model"])) for rd in run_data_list) if run_data_list else 20
    fig_width = max(12, len(categories) * max(1.5, max_len * 0.12), max_model_len * 0.12 * 10)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    x = np.arange(len(categories))

    for i, rd in enumerate(run_data_list):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, rd["delta"], "o-", color=color, linewidth=2.5, markersize=7,
                label=model_label(rd["model"]))
        for xi, yi in zip(x, rd["delta"]):
            ax.annotate(f"{yi:+.3f}", xy=(xi, yi), xytext=(0, 6),
                        textcoords="offset points", ha="center", fontsize=7, color=color)

    ax.axhline(0, color="gray", linewidth=1.5, linestyle="--", alpha=0.6)
    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Δ (computed − saved)", fontsize=14)
    ax.set_title("Signed Threshold Delta per Model", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
    save_plot(fig, output_dir, "delta_lines")
    plt.close()


def plot_auroc_lines(run_data_list, categories, output_dir):
    labels = wrap_labels(categories)
    max_len = max(len(c) for c in categories)
    max_model_len = max(len(model_label(rd["model"])) for rd in run_data_list) if run_data_list else 20
    fig_width = max(12, len(categories) * max(1.5, max_len * 0.12), max_model_len * 0.12 * 10)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 6))

    x = np.arange(len(categories))

    for i, rd in enumerate(run_data_list):
        color = PALETTE[i % len(PALETTE)]
        ax.plot(x, rd["instance_auroc"], "o-", color=color, linewidth=2, markersize=7,
                label=model_label(rd["model"]))
        for xi, yi in zip(x, rd["instance_auroc"]):
            ax.annotate(f"{yi:.2f}", xy=(xi, yi), xytext=(0, 5),
                        textcoords="offset points", ha="center", fontsize=7, color=color)

    ax.set_xlabel("Category", fontsize=14)
    ax.set_ylabel("Instance AUROC", fontsize=14)
    ax.set_title("Instance AUROC Transfer per Model", fontsize=16)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=0, ha="center", fontsize=9)
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.set_ylim([0.7, 1.02])
    ax.grid(True, axis="y", alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(right=0.7)
    save_plot(fig, output_dir, "auroc_lines")
    plt.close()


def plot_threshold_scatter(run_data_list, output_dir):
    max_model_len = max(len(model_label(rd["model"])) for rd in run_data_list) if run_data_list else 20
    fig_width = max(10, max_model_len * 0.12 * 10)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, 7))

    all_saved = []
    all_computed = []

    for i, rd in enumerate(run_data_list):
        color = PALETTE[i % len(PALETTE)]
        ax.scatter(rd["saved_threshold"], rd["computed_threshold"],
                   c=color, s=80, alpha=0.8, edgecolors="white", linewidth=1,
                   label=model_label(rd["model"]), zorder=5)
        all_saved.extend(rd["saved_threshold"].tolist())
        all_computed.extend(rd["computed_threshold"].tolist())

    all_saved = np.array(all_saved)
    all_computed = np.array(all_computed)
    lim_min = min(all_saved.min(), all_computed.min()) * 0.9
    lim_max = max(all_saved.max(), all_computed.max()) * 1.1
    ax.plot([lim_min, lim_max], [lim_min, lim_max], "k--", alpha=0.4, linewidth=1.5,
            label="Δ = 0 (no drift)")

    ax.set_xlabel("Saved Threshold (source domain)", fontsize=13)
    ax.set_ylabel("Computed Threshold (target domain)", fontsize=13)
    ax.set_title("Saved vs Computed Threshold (colored by model)", fontsize=15)
    ax.set_xlim([lim_min, lim_max])
    ax.set_ylim([lim_min, lim_max])
    ax.legend(fontsize=9, loc="upper left", bbox_to_anchor=(1.01, 1))
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    fig.subplots_adjust(right=0.75)
    save_plot(fig, output_dir, "threshold_scatter")
    plt.close()


def _plot_metric_heatmap(run_data_list, categories, metric_key, metric_label, output_dir, filename):
    model_names = [model_label(rd["model"]) for rd in run_data_list]
    matrix = np.array([rd[metric_key] for rd in run_data_list])

    labels = wrap_labels(categories, max_width=10)

    max_model_len = max(len(m) for m in model_names) if model_names else 20
    # Width: ensure long model names fit without truncation
    fig_width = max(14, len(categories) * 1.5, max_model_len * 0.12 * 10)
    fig_height = max(4, len(model_names) * 1.0)
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    im = ax.imshow(matrix, cmap="RdYlGn", aspect="auto", vmin=0, vmax=1)

    for i in range(len(model_names)):
        for j in range(len(categories)):
            val = matrix[i, j]
            text_color = "white" if val > 0.6 else "black"
            ax.text(j, i, f"{val:.2f}", ha="center", va="center",
                    fontsize=12, fontweight="bold", color=text_color)

    ax.set_xticks(np.arange(len(categories)))
    ax.set_xticklabels(labels, fontsize=10, rotation=45, ha="right")
    ax.set_yticks(np.arange(len(model_names)))
    ax.set_yticklabels(model_names, fontsize=10)
    plt.setp(ax.get_yticklabels(), ha="right", va="center")

    ax.set_xlabel("Category", fontsize=14, fontweight="bold")
    ax.set_ylabel("Model", fontsize=14, fontweight="bold")
    ax.set_title(f"{metric_label} at Computed Threshold", fontsize=16, fontweight="bold")

    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    cbar.set_label(metric_label, fontsize=12)

    # Reserve space for y-axis labels using tight_layout rect parameter
    # rect=[left, bottom, right, top] in figure coordinates
    left_reserve = min(0.5, max(0.25, max_model_len * 0.015))
    fig.tight_layout(rect=[left_reserve, 0, 1, 1])
    save_plot(fig, output_dir, filename)
    plt.close()


def plot_precision_heatmap(run_data_list, categories, output_dir):
    _plot_metric_heatmap(run_data_list, categories, "precision", "Precision", output_dir, "precision_heatmap")


def plot_recall_heatmap(run_data_list, categories, output_dir):
    _plot_metric_heatmap(run_data_list, categories, "recall", "Recall", output_dir, "recall_heatmap")


def plot_f1_heatmap(run_data_list, categories, output_dir):
    _plot_metric_heatmap(run_data_list, categories, "f1", "F1 Score", output_dir, "f1_heatmap")


def plot_delta_boxplot(run_data_list, output_dir):
    labels = [model_label(rd["model"]) for rd in run_data_list]
    max_label_len = max(len(l) for l in labels) if labels else 20
    fig_width = max(10, len(run_data_list) * 3, max_label_len * 0.12 * 10)
    fig_height = 6 + (max_label_len > 30) * 1.5
    fig, ax = plt.subplots(1, 1, figsize=(fig_width, fig_height))

    deltas = [rd["abs_delta"] for rd in run_data_list]

    bp = ax.boxplot(deltas, tick_labels=labels, patch_artist=True, widths=0.5,
                    medianprops=dict(color="black", linewidth=2),
                    boxprops=dict(linewidth=1.5),
                    whiskerprops=dict(linewidth=1.5),
                    capprops=dict(linewidth=1.5))

    for i, box in enumerate(bp["boxes"]):
        box.set_facecolor(PALETTE[i % len(PALETTE)])
        box.set_alpha(0.7)

    for i, rd in enumerate(run_data_list):
        mean_val = np.mean(rd["abs_delta"])
        ax.scatter(i + 1, mean_val, marker="D", color="red", s=60, zorder=5,
                   edgecolors="white", linewidth=1.5)
        ax.annotate(f"μ={mean_val:.3f}", xy=(i + 1, mean_val),
                    xytext=(8, 5), textcoords="offset points",
                    fontsize=9, fontweight="bold", color="red")

    ax.set_xlabel("Model", fontsize=13)
    ax.set_ylabel("|Δ| (|computed − saved|)", fontsize=13)
    ax.set_title("Threshold Drift Distribution per Model\n(red diamond = mean)", fontsize=15)
    ax.grid(True, axis="y", alpha=0.3)

    # Rotate x labels if they are long
    if max_label_len > 25:
        plt.setp(ax.get_xticklabels(), rotation=30, ha="right")

    plt.tight_layout()
    # Extra bottom margin for rotated labels
    if max_label_len > 25:
        fig.subplots_adjust(bottom=0.25)
    save_plot(fig, output_dir, "delta_boxplot")
    plt.close()


def main():
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")

    args = parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    run_data_list = []
    for run_dir in args.runs:
        rd = load_run_data(run_dir)
        if rd is not None:
            run_data_list.append(rd)
            LOGGER.info(f"Loaded {rd['model']}: {len(rd['categories'])} categories")

    if len(run_data_list) < 1:
        LOGGER.error("No valid runs found.")
        return

    # Compute intersection of categories across all runs
    cat_sets = [set(rd["categories"]) for rd in run_data_list]
    common_categories = cat_sets[0]
    for cs in cat_sets[1:]:
        common_categories &= cs

    # Preserve first run's order for common categories
    categories = [c for c in run_data_list[0]["categories"] if c in common_categories]

    # Report excluded categories
    for rd in run_data_list:
        excluded = set(rd["categories"]) - common_categories
        if excluded:
            LOGGER.warning(f"{rd['model']}: excluding {len(excluded)} category(ies) not present in all runs: {sorted(excluded)}")

    # Filter each run's data to only common categories (preserving order)
    for rd in run_data_list:
        mask = [c in common_categories for c in rd["categories"]]
        rd["categories"] = [c for c, m in zip(rd["categories"], mask) if m]
        rd["saved_threshold"] = rd["saved_threshold"][mask]
        rd["computed_threshold"] = rd["computed_threshold"][mask]
        rd["delta"] = rd["delta"][mask]
        rd["abs_delta"] = rd["abs_delta"][mask]
        rd["instance_auroc"] = rd["instance_auroc"][mask]
        rd["full_pixel_auroc"] = rd["full_pixel_auroc"][mask]
        rd["anomaly_pixel_auroc"] = rd["anomaly_pixel_auroc"][mask]
        rd["precision"] = rd["precision"][mask]
        rd["recall"] = rd["recall"][mask]
        rd["f1"] = rd["f1"][mask]

    LOGGER.info(f"Using {len(categories)} common categories across all runs: {categories}")
    LOGGER.info(f"Generating comparison plots for {len(run_data_list)} models...")
    plot_threshold_delta_grouped(run_data_list, categories, output_dir)
    plot_delta_lines(run_data_list, categories, output_dir)
    plot_auroc_lines(run_data_list, categories, output_dir)
    plot_threshold_scatter(run_data_list, output_dir)
    plot_precision_heatmap(run_data_list, categories, output_dir)
    plot_recall_heatmap(run_data_list, categories, output_dir)
    plot_f1_heatmap(run_data_list, categories, output_dir)
    plot_delta_boxplot(run_data_list, output_dir)

    LOGGER.info(f"All comparison plots saved to {output_dir}")


if __name__ == "__main__":
    main()
