"""Cross-evaluate saved PatchCore / TiledPatchCore runs on one real test set.

Takes any number of training-run folders (each with ``models/mvtec_<cat>/`` +
``curves/mvtec_<cat>_curves.npz`` — the contract written by both
run_tiled_patchcore.py and scripts/patchcore/run_patchcore.py) and evaluates
every run on the SAME real dataset root. Tiled and plain checkpoints can be
mixed freely; each model's required dataset geometry (native resolution for
tiled, its saved crop for plain) is resolved from the checkpoint itself.

Every run gets a ``CrossEval_<timestamp>_<model>/`` folder whose results.csv
uses the exact schema of the legacy scripts/patchcore/evaluate_cross_dataset.py
(saved + computed thresholds, TP/FP/FN/TN, Precision/Recall), so
compare_cross_eval_runs.py produces the identical chart pack on the outputs.
With >=2 runs, a paired per-image score scatter per category is added — the
honest comparison when test sets are small.

Examples
--------
uv run python scripts/tiled_patchcore/cross_evaluate.py \
    --model_run results_tiled/tiled_dinov3_vitb_T384_p0.01_0 \
    --model_run /path/to/plain/run \
    --data_path dataset/40mp

Outputs:
    <output_root>/CrossEval_<ts>_<model>/results.csv, confusion_matrix.csv,
        curves/, per_image_scores.csv
    <output_root>/cross_eval_comparison/   (identical legacy chart pack)
    <output_root>/paired_scatter/          (per-image score scatter, run pairs)
"""
import csv
import gc
import itertools
import logging
import os
import re
import subprocess
import sys
import textwrap
from datetime import datetime

import click
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shared  # noqa: F401,E402  — must come first: puts repo src/ on sys.path

import matplotlib.pyplot as plt  # noqa: E402  (shared already set Agg)

import patchcore.common  # noqa: E402
import patchcore.utils  # noqa: E402

LOGGER = logging.getLogger("cross_evaluate")

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

# Exact column schema of scripts/patchcore/evaluate_cross_dataset.py —
# compare_cross_eval_runs.py parses these names verbatim.
RESULT_COLUMNS = [
    "instance_auroc",
    "full_pixel_auroc",
    "anomaly_pixel_auroc",
    "saved_threshold",
    "computed_threshold",
    "TP",
    "FP",
    "FN",
    "TN",
    "Precision",
    "Recall",
]

_PALETTE = [
    "#2196F3", "#FF9800", "#4CAF50", "#9C27B0", "#E53935",
    "#00BCD4", "#FF5722", "#8BC34A", "#673AB7", "#F44336",
]


def _validate_run(run_dir):
    """Check the run-dir contract; return {category: model_dir} + curves dir.

    A valid run has models/mvtec_<category>/ containing a
    *patchcore_params.pkl. curves/ is required for the saved threshold —
    categories without one are skipped later with a warning (legacy
    cross-eval semantics).
    """
    models_root = os.path.join(run_dir, "models")
    if not os.path.isdir(models_root):
        raise click.ClickException(
            f"{run_dir}: no models/ directory (not a training-run folder)."
        )
    model_dirs = {}
    for entry in sorted(os.listdir(models_root)):
        entry_path = os.path.join(models_root, entry)
        if not os.path.isdir(entry_path) or not entry.startswith("mvtec_"):
            continue
        if any(f.endswith("patchcore_params.pkl") for f in os.listdir(entry_path)):
            model_dirs[entry[len("mvtec_"):]] = entry_path
    if not model_dirs:
        raise click.ClickException(
            f"{run_dir}: no models/mvtec_*/ folders with patchcore_params.pkl."
        )
    return model_dirs, os.path.join(run_dir, "curves")


def _saved_threshold(curves_dir, category):
    """Training-time optimal threshold for a category (None if unavailable)."""
    curves_file = os.path.join(curves_dir, f"mvtec_{category}_curves.npz")
    if not os.path.exists(curves_file):
        return None
    with np.load(curves_file) as curves_data:
        if "optimal_threshold" not in curves_data:
            return None
        return float(curves_data["optimal_threshold"])


def _sanitize_name(name):
    """Filesystem-safe model name (used in output dir names + chart legends)."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", name).strip("_") or "model"


def _resolve_names(run_paths, explicit_names=()):
    """Model name per run: explicit > basename > parent-prefix on collisions.

    Explicit names (positionally paired with runs) always win. Otherwise the
    run folder's basename is used — unless several runs share it, in which
    case each collided name is disambiguated as ``<parent_dir>__<basename>``
    (the dataset-variant axis usually lives one level up, e.g.
    results_tiled_bigd vs results_tiled_bigd_synthetic). Any remaining
    duplicate gets a ``_N`` suffix. All names are filesystem-sanitized.
    """
    if explicit_names:
        if len(explicit_names) != len(run_paths):
            raise click.UsageError(
                f"--model_name count ({len(explicit_names)}) must match "
                f"--model_run count ({len(run_paths)})."
            )
        names = list(explicit_names)
    else:
        names = [os.path.basename(os.path.normpath(p)) for p in run_paths]
        if len(set(names)) != len(names):
            names = [
                f"{os.path.basename(os.path.dirname(os.path.normpath(p)))}__{name}"
                if names.count(name) > 1 else name
                for name, p in zip(names, run_paths)
            ]
    names = [_sanitize_name(n) for n in names]
    counts, result = {}, []
    for name in names:
        index = counts.get(name, 0)
        result.append(name if index == 0 else f"{name}_{index}")
        counts[name] = index + 1
    return result


def _output_dir(output_root, timestamp, name):
    """CrossEval_<ts>_<name> folder — the naming compare_cross_eval_runs.py
    parses the model name back out of (CrossEval_\\d{8}_\\d+_(.+))."""
    candidate = os.path.join(output_root, f"CrossEval_{timestamp}_{name}")
    suffix = 1
    while os.path.exists(candidate):
        candidate = os.path.join(output_root, f"CrossEval_{timestamp}_{name}_{suffix}")
        suffix += 1
    os.makedirs(candidate)
    return candidate


def _write_confusion_csv(path, model_name, rows):
    """Legacy confusion_matrix.csv: one row per category + a Mean row."""
    cm_columns = ["TP", "FP", "FN", "TN", "Precision", "Recall"]
    with open(path, "w", newline="") as csv_file:
        writer = csv.writer(csv_file, delimiter=",")
        writer.writerow(["Model", "Category"] + cm_columns)
        for category_label, counts in rows:
            writer.writerow(
                [model_name, category_label] + [counts.get(c, 0) for c in cm_columns]
            )
        if rows:
            mean_values = [
                float(np.mean([counts.get(c, 0) for _label, counts in rows]))
                for c in cm_columns
            ]
            writer.writerow([model_name, "Mean"] + mean_values)


def _write_run_results(run_out_dir, model_name, rows):
    """results.csv in the legacy schema + confusion_matrix.csv.

    rows: list of dicts with 'category' (bare) + every key of RESULT_COLUMNS.
    """
    result_scores = [[row[col] for col in RESULT_COLUMNS] for row in rows]
    patchcore.utils.compute_and_store_final_results(
        run_out_dir,
        result_scores,
        column_names=RESULT_COLUMNS,
        row_names=[f"{model_name} | mvtec_{row['category']}" for row in rows],
        model_names=[model_name] * len(rows),
    )
    _write_confusion_csv(
        os.path.join(run_out_dir, "confusion_matrix.csv"),
        model_name,
        [(f"{model_name} | mvtec_{row['category']}", row["confusion"]) for row in rows],
    )


def _plot_paired_scatters(paired_store, output_dir):
    """Per-image score scatter for every run pair x common category.

    Same images on both axes, points colored by ground-truth label: instant
    view of whether two models agree per image (paired evidence) instead of
    two incomparable aggregate AUROCs.
    """
    run_names = list(paired_store.keys())
    if len(run_names) < 2:
        return 0
    os.makedirs(output_dir, exist_ok=True)
    written = 0
    for name_a, name_b in itertools.combinations(run_names, 2):
        common = sorted(
            set(paired_store[name_a]) & set(paired_store[name_b])
        )
        for category in common:
            data_a = paired_store[name_a][category]
            data_b = paired_store[name_b][category]
            scores_b_by_path = dict(zip(data_b["paths"], data_b["scores"]))
            shared_paths = [p for p in data_a["paths"] if p in scores_b_by_path]
            if not shared_paths:
                continue
            scores_a = np.array([data_a["scores"][data_a["paths"].index(p)]
                                 for p in shared_paths])
            scores_b = np.array([scores_b_by_path[p] for p in shared_paths])
            labels = np.array([data_a["labels"][data_a["paths"].index(p)]
                               for p in shared_paths])

            fig, axis = plt.subplots(figsize=(5.5, 5.5), dpi=120)
            for flag, color, label in ((0, "tab:blue", "good"), (1, "tab:red", "anomaly")):
                mask = labels == flag
                if mask.any():
                    axis.scatter(scores_a[mask], scores_b[mask], c=color, s=60,
                                 alpha=0.85, edgecolors="white", linewidth=1,
                                 label=label, zorder=5)
            lim_min = float(min(scores_a.min(), scores_b.min()))
            lim_max = float(max(scores_a.max(), scores_b.max()))
            pad = 0.05 * max(lim_max - lim_min, 1e-12)
            axis.plot([lim_min - pad, lim_max + pad], [lim_min - pad, lim_max + pad],
                      "k--", linewidth=1.2, alpha=0.5, label="agreement")
            axis.set_xlim(lim_min - pad, lim_max + pad)
            axis.set_ylim(lim_min - pad, lim_max + pad)
            wrap = lambda s: "\n".join(textwrap.wrap(s, width=28, break_long_words=False))
            axis.set_xlabel(wrap(name_a), fontsize=10)
            axis.set_ylabel(wrap(name_b), fontsize=10)
            axis.set_title(f"mvtec_{category}: per-image scores (paired)", fontsize=11)
            axis.legend(fontsize=8)
            axis.grid(True, alpha=0.3)
            fig.tight_layout()
            safe = lambda s: s.replace(os.sep, "_").replace(" ", "_")
            fig.savefig(os.path.join(
                output_dir, f"paired_scatter_{safe(name_a)}_vs_{safe(name_b)}__mvtec_{category}.png"))
            plt.close(fig)
            written += 1
    return written


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--model_run", multiple=True, required=True,
              type=click.Path(exists=True, file_okay=False),
              help="Training-run folder with models/ + curves/ (repeatable; "
              "tiled and plain runs mix freely).")
@click.option("--model_name", multiple=True,
              help="Display name per --model_run (same order; used for output "
              "folders and chart legends). Default: run folder basename; "
              "basename collisions get the parent folder prepended.")
@click.option("--data_path", required=True,
              type=click.Path(exists=True, file_okay=False),
              help="REAL dataset root (MVTec format) every run is evaluated on.")
@click.option("--subdatasets", "-d", multiple=True,
              help="Restrict evaluation to these categories.")
@click.option("--output_root", default="results_cross_eval", show_default=True,
              help="CrossEval_* folders + cross_eval_comparison/ go here.")
@click.option("--batch_size", type=int, default=1, show_default=True)
@click.option("--num_workers", type=int, default=8, show_default=True)
@click.option("--gpu", type=int, default=None,
              help="CUDA device id. Default: auto — CUDA, then Apple MPS, then CPU.")
@click.option("--cpu", is_flag=True, help="Force CPU (auto prefers CUDA, then Apple MPS).")
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--skip_comparison", is_flag=True,
              help="Do not run compare_cross_eval_runs.py on the outputs.")
@click.option("--paired_scatter/--no_paired_scatter", default=True, show_default=True,
              help="Paired per-image score scatter for run pairs (needs >=2 runs).")
def main(model_run, model_name, data_path, subdatasets, output_root, batch_size,
         num_workers, gpu, cpu, faiss_num_workers, seed, skip_comparison,
         paired_scatter):
    """Cross-evaluate saved PatchCore / TiledPatchCore runs on one real test set.

    \b
    Examples:
      uv run python scripts/tiled_patchcore/cross_evaluate.py \\
          --model_run results_tiled/tiled_dinov3_vitb_T384_p0.01_0 \\
          --model_run /path/to/plain/run --data_path dataset/40mp
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    device, faiss_on_gpu = shared.resolve_device(gpu=gpu, cpu=cpu)
    patchcore.utils.fix_seeds(seed, device)

    names = _resolve_names(model_run, model_name)
    runs = []
    for name, run_dir in zip(names, model_run):
        model_dirs, curves_dir = _validate_run(run_dir)
        runs.append({
            "name": name,
            "path": run_dir,
            "model_dirs": model_dirs,
            "curves_dir": curves_dir,
        })
        LOGGER.info("Run '%s': %d categor(ies): %s",
                    name, len(model_dirs), sorted(model_dirs))

    available = set(shared.discover_categories(data_path))
    selected = set(subdatasets)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    dataset_cache = {}  # (category, resize, imagesize) -> (dataset, loader)
    paired_store = {}   # run name -> category -> {paths, scores, labels}
    produced_dirs = []

    for run in runs:
        categories = sorted(set(run["model_dirs"]) & available)
        if selected:
            categories = [c for c in categories if c in selected]
        if not categories:
            LOGGER.warning("Run '%s': no categories match %s — skipped.",
                           run["name"], data_path)
            continue

        run_out_dir = _output_dir(output_root, timestamp, run["name"])
        os.makedirs(os.path.join(run_out_dir, "curves"), exist_ok=True)
        LOGGER.info("=== Run: %s -> %s ===", run["name"], run_out_dir)

        run_info = {
            "model_name": run["name"],
            "source_run_path": os.path.abspath(run["path"]),
            "eval_data_path": os.path.abspath(data_path),
            "timestamp": timestamp,
            "categories": [],
            "saved_thresholds": {},
        }
        result_rows = []
        for category in categories:
            saved = _saved_threshold(run["curves_dir"], category)
            if saved is None:
                LOGGER.warning(
                    "[%s] no saved threshold under %s — skipped.",
                    category, run["curves_dir"],
                )
                continue

            models = shared.load_models_from_dir(
                run["model_dirs"][category], device,
                patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers),
                seed=seed,
            )
            if len({tiled for _m, tiled in models}) > 1:
                raise click.ClickException(
                    f"{run['model_dirs'][category]} mixes tiled and plain "
                    "checkpoints; cross-evaluate them separately."
                )
            first_model, first_tiled = models[0]
            if "backbone" not in run_info:
                # Checkpoint metadata: provenance that doesn't belong in a
                # filename but should survive with the results.
                run_info.update({
                    "backbone": getattr(first_model.backbone, "name", None),
                    "layers": list(first_model.layers_to_extract_from),
                    "input_shape": list(first_model.input_shape),
                    "tiled": bool(first_tiled),
                })
                if first_tiled:
                    run_info.update({
                        "tile_size": int(first_model.tile_maker.tile_size),
                        "tile_stride": int(first_model.tile_maker.stride),
                        "tile_overlap": int(first_model.tile_maker.overlap),
                    })
            resize, imagesize = shared.dataset_geometry_for_model(first_model, first_tiled)
            cache_key = (category, resize, imagesize)
            if cache_key not in dataset_cache:
                test_set, _ = shared.build_dataset(
                    data_path, category, "TEST",
                    resize=resize, imagesize=imagesize, seed=seed,
                )
                test_loader = shared.build_dataloader(test_set, batch_size, num_workers)
                dataset_cache[cache_key] = (test_set, test_loader)
            test_set, test_loader = dataset_cache[cache_key]

            if not shared.has_both_classes(test_set):
                LOGGER.warning(
                    "[%s] test split under %s has only one class — skipped "
                    "(AUROC/threshold undefined; nothing to compare).",
                    category, data_path,
                )
                del models
                gc.collect()
                continue
            run_info["categories"].append(category)
            run_info["saved_thresholds"][category] = saved

            LOGGER.info(
                "[%s] %s | %d model(s) | geometry resize=%s imagesize=%s | saved threshold %.4f",
                category, "TILED" if first_tiled else "plain",
                len(models), resize, imagesize, saved,
            )
            evaluation = shared.evaluate_models(models, test_loader)
            scores = evaluation["scores"]
            labels = [int(label) for label in evaluation["labels_gt"]]
            image_metrics = evaluation["image_metrics"]
            pixel_metrics = evaluation["pixel_metrics"]
            anomaly_pixel_metrics = evaluation["anomaly_pixel_metrics"]
            computed = float(image_metrics["optimal_threshold"])
            records = shared.classify_predictions(scores, labels, saved)
            confusion = shared.confusion_counts(records)

            result_rows.append({
                "category": category,
                "instance_auroc": image_metrics["auroc"],
                "full_pixel_auroc": pixel_metrics["auroc"] if pixel_metrics else float("nan"),
                "anomaly_pixel_auroc": anomaly_pixel_metrics["auroc"]
                if anomaly_pixel_metrics else float("nan"),
                "saved_threshold": saved,
                "computed_threshold": computed,
                "TP": confusion["TP"],
                "FP": confusion["FP"],
                "FN": confusion["FN"],
                "TN": confusion["TN"],
                "Precision": confusion["Precision"],
                "Recall": confusion["Recall"],
                "confusion": confusion,
            })
            shared.write_per_image_scores_csv(
                os.path.join(run_out_dir, "per_image_scores.csv"),
                category, test_set, scores, records,
            )
            np.savez(
                os.path.join(run_out_dir, "curves", f"mvtec_{category}_curves.npz"),
                image_fpr=image_metrics["fpr"],
                image_tpr=image_metrics["tpr"],
                image_thresholds=image_metrics["threshold"],
                pixel_fpr=pixel_metrics["fpr"] if pixel_metrics else np.array([]),
                pixel_tpr=pixel_metrics["tpr"] if pixel_metrics else np.array([]),
                image_auroc=image_metrics["auroc"],
                pixel_auroc=pixel_metrics["auroc"] if pixel_metrics else float("nan"),
                anomaly_pixel_auroc=anomaly_pixel_metrics["auroc"]
                if anomaly_pixel_metrics else float("nan"),
                optimal_threshold=saved,
                computed_threshold=computed,
            )
            paired_store.setdefault(run["name"], {})[category] = {
                "paths": [t[2] for t in test_set.data_to_iterate],
                "scores": [float(s) for s in scores],
                "labels": labels,
            }
            LOGGER.info(
                "mvtec_%s: instance_auroc=%.3f saved=%.3f computed=%.3f | P=%.3f R=%.3f",
                category, image_metrics["auroc"], saved, computed,
                confusion["Precision"], confusion["Recall"],
            )

            del models, evaluation
            gc.collect()
            if device.type == "cuda":
                torch.cuda.empty_cache()

        shared.write_json(os.path.join(run_out_dir, "run_info.json"), run_info)
        if not result_rows:
            LOGGER.warning("Run '%s': nothing evaluated — no results written.", run["name"])
            continue
        _write_run_results(run_out_dir, run["name"], result_rows)
        produced_dirs.append(run_out_dir)

    if not produced_dirs:
        raise click.ClickException("Nothing evaluated — check --model_run/--data_path/--subdatasets.")

    if paired_scatter:
        scatter_dir = os.path.join(output_root, "paired_scatter")
        written = _plot_paired_scatters(paired_store, scatter_dir)
        if written:
            LOGGER.info("Paired scatters: %d plot(s) in %s", written, scatter_dir)

    if not skip_comparison:
        comparison_dir = os.path.join(output_root, "cross_eval_comparison")
        os.makedirs(comparison_dir, exist_ok=True)
        compare_script = os.path.join(_REPO_ROOT, "compare_cross_eval_runs.py")
        command = [sys.executable, compare_script,
                   "--runs", *produced_dirs, "--output-dir", comparison_dir]
        LOGGER.info("Comparison charts: %s", " ".join(command))
        subprocess.run(command, check=True)

    LOGGER.info("All artifacts in %s", output_root)


if __name__ == "__main__":
    main()
