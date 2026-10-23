"""Evaluate saved PatchCore / TiledPatchCore models — one command.

Takes a model folder (as written by run_tiled_patchcore.py or
scripts/patchcore/run_patchcore.py with --save_patchcore_model) and the
dataset root; each checkpoint's required image geometry (native resolution
for tiled models, its training crop size for plain ones) is resolved from
the checkpoint itself — no flags to get wrong. Ensemble directories
(Ensemble-*_ prepends) are averaged exactly like the training script.

Examples
--------
# All category models under a run's models/ dir, same categories:
uv run python tiled_patchcore/load_and_evaluate.py \
    --model_root results_tiled/tiled/models --data_path dataset

# One specific model:
uv run python tiled_patchcore/load_and_evaluate.py \
    --model_path results_tiled/tiled/models/mvtec_bottle --data_path dataset

Outputs (full analytics pack):
    <output_dir>/results.csv, confusion_matrix.csv, per_image_scores.csv,
    timing.json, curves/, plots/
"""
import logging
import os
import sys

import click
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shared  # noqa: F401,E402  — must come first: puts repo src/ on sys.path

import patchcore.common
import patchcore.utils

LOGGER = logging.getLogger("load_and_evaluate")


def _discover_model_dirs(model_root):
    return sorted(
        os.path.join(model_root, entry)
        for entry in os.listdir(model_root)
        if os.path.isdir(os.path.join(model_root, entry))
        and any(f.endswith("patchcore_params.pkl") for f in os.listdir(os.path.join(model_root, entry)))
    )


def _category_from_model_dir(model_dir):
    name = os.path.basename(os.path.normpath(model_dir))
    return name[len("mvtec_"):] if name.startswith("mvtec_") else name


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option("--model_root", type=click.Path(exists=True, file_okay=False),
              help="Directory of per-category model folders (e.g. <run>/models).")
@click.option("--model_path", "-p", multiple=True,
              help="Explicit model folder(s); overrides --model_root.")
@click.option("--data_path", required=True, type=click.Path(exists=True, file_okay=False),
              help="Dataset root with MVTec-format categories.")
@click.option("--subdatasets", "-d", multiple=True,
              help="Restrict evaluation to these categories.")
@click.option("--output_dir", default=None,
              help="Default: 'eval' next to the models folder / 'results_tiled_eval'.")
@click.option("--batch_size", type=int, default=1, show_default=True)
@click.option("--num_workers", type=int, default=8, show_default=True)
@click.option("--resize", type=int, default=None,
              help="Override auto dataset geometry (resize) — only for plain models.")
@click.option("--imagesize", type=int, default=None,
              help="Override auto dataset geometry (center crop).")
@click.option("--gpu", type=int, default=None,
              help="CUDA device id. Default: auto — CUDA, then Apple MPS, then CPU.")
@click.option("--cpu", is_flag=True,
              help="Force CPU (auto prefers CUDA, then Apple MPS).")
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--save_segmentation_images", is_flag=True,
              help="Per-image original/GT/heatmap/overlay panels (more disk).")
@click.option("--top_anomalous", type=int, default=4, show_default=True)
def main(model_root, model_path, data_path, subdatasets, output_dir, batch_size,
         num_workers, resize, imagesize, gpu, cpu, faiss_num_workers, seed,
         save_segmentation_images, top_anomalous):
    """Evaluate saved PatchCore / TiledPatchCore checkpoints.

    \b
    Examples:
      uv run python tiled_patchcore/load_and_evaluate.py \
          --model_root results_tiled/tiled/models --data_path dataset
      uv run python tiled_patchcore/load_and_evaluate.py \
          -p results_tiled/tiled/models/mvtec_bottle --data_path dataset

    Dataset geometry is resolved from each checkpoint automatically: tiled
    models demand native resolution; plain models use their saved crop size.
    """
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if not model_root and not model_path:
        raise click.UsageError("Provide --model_root or one or more --model_path.")

    model_dirs = list(model_path) or _discover_model_dirs(model_root)
    if output_dir is None:
        if model_root:
            output_dir = os.path.join(os.path.dirname(os.path.normpath(model_root)), "eval")
        else:
            output_dir = "results_tiled_eval"

    device, faiss_on_gpu = shared.resolve_device(gpu=gpu, cpu=cpu)
    patchcore.utils.fix_seeds(seed, device)

    for folder in ("curves", "plots"):
        os.makedirs(os.path.join(output_dir, folder), exist_ok=True)

    selected = set(subdatasets)
    result_rows, timings = [], {}
    for model_dir in model_dirs:
        category = _category_from_model_dir(model_dir)
        if selected and category not in selected:
            continue
        test_root = os.path.join(data_path, category, "test")
        if not os.path.isdir(test_root):
            LOGGER.warning("Skipping %s: no category '%s' under %s", model_dir, category, data_path)
            continue

        models = shared.load_models_from_dir(
            model_dir,
            device,
            patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers),
            seed=seed,
        )
        first_model, first_tiled = models[0]
        auto_resize, auto_imagesize = shared.dataset_geometry_for_model(first_model, first_tiled)
        eff_resize = resize if resize is not None else auto_resize
        eff_imagesize = imagesize if imagesize is not None else auto_imagesize
        LOGGER.info(
            "Evaluating mvtec_%s | %d model(s) | %s | dataset geometry resize=%s imagesize=%s",
            category, len(models), "TILED (native)" if first_tiled else "plain",
            eff_resize, eff_imagesize,
        )
        mixed = len({tiled for _m, tiled in models}) > 1
        if mixed:
            raise click.ClickException(
                f"{model_dir} mixes tiled and plain checkpoints; evaluate them separately."
            )

        test_set, _ = shared.build_dataset(
            data_path, category, "TEST", resize=eff_resize, imagesize=eff_imagesize, seed=seed
        )
        if not shared.has_both_classes(test_set):
            LOGGER.warning(
                "Skipping mvtec_%s: test split has only one class — instance "
                "metrics are undefined for this checkpoint on this dataset.", category,
            )
            continue
        test_loader = shared.build_dataloader(test_set, batch_size, num_workers)

        evaluation = shared.evaluate_models(models, test_loader)
        scores = evaluation["scores"]
        segmentations = evaluation["segmentations"]
        labels = [int(label) for label in evaluation["labels_gt"]]
        image_metrics = evaluation["image_metrics"]
        pixel_metrics = evaluation["pixel_metrics"]
        anomaly_pixel_metrics = evaluation["anomaly_pixel_metrics"]
        threshold = float(image_metrics["optimal_threshold"])
        # Decision threshold converted into anomaly-map units (see
        # map_cutoff_for_threshold), so overlays show what actually fired.
        cutoff = shared.map_cutoff_for_threshold(evaluation, threshold)
        records = shared.classify_predictions(scores, labels, threshold)

        result_rows.append({
            "category": f"mvtec_{category}",
            "instance_auroc": image_metrics["auroc"],
            "full_pixel_auroc": pixel_metrics["auroc"] if pixel_metrics else float("nan"),
            "anomaly_pixel_auroc": anomaly_pixel_metrics["auroc"]
            if anomaly_pixel_metrics else float("nan"),
            "optimal_threshold": threshold,
            "confusion": shared.confusion_counts(records),
        })
        shared.write_per_image_scores_csv(
            os.path.join(output_dir, "per_image_scores.csv"),
            category, test_set, scores, records,
        )
        timings[f"mvtec_{category}"] = {
            "predict_seconds": round(evaluation["predict_seconds"], 3),
            "n_test_images": len(test_set),
            "n_models": len(models),
            "tiled": first_tiled,
        }
        np.savez(
            os.path.join(output_dir, "curves", f"mvtec_{category}_curves.npz"),
            image_fpr=image_metrics["fpr"],
            image_tpr=image_metrics["tpr"],
            image_thresholds=image_metrics["threshold"],
            pixel_fpr=pixel_metrics["fpr"] if pixel_metrics else np.array([]),
            pixel_tpr=pixel_metrics["tpr"] if pixel_metrics else np.array([]),
            image_auroc=image_metrics["auroc"],
            pixel_auroc=pixel_metrics["auroc"] if pixel_metrics else float("nan"),
            anomaly_pixel_auroc=anomaly_pixel_metrics["auroc"]
            if anomaly_pixel_metrics else float("nan"),
            optimal_threshold=threshold,
        )

        image_h, image_w = shared.dataset_image_hw(test_set)
        if eff_resize is None and eff_imagesize is not None:
            # Center-cropped native: actual tensor size is the crop.
            image_h = image_w = eff_imagesize
        tile_layout = shared.tile_layout_for_model(first_model, first_tiled, image_h, image_w)
        if tile_layout is not None:
            shared.plot_tile_grid(
                test_set.data_to_iterate[0][2], tile_layout,
                os.path.join(output_dir, "plots", f"mvtec_{category}_tile_grid.png"),
                title=f"mvtec_{category} tile grid (checkpoint geometry)",
            )
        if shared.plot_score_distribution(
            scores, labels, threshold,
            os.path.join(output_dir, "plots", f"mvtec_{category}_score_dist.png"),
            title=f"mvtec_{category}: score distribution",
        ):
            shared.plot_roc(
                image_metrics,
                os.path.join(output_dir, "plots", f"mvtec_{category}_roc.png"),
                title=f"mvtec_{category}: image-level ROC",
            )
        if top_anomalous > 0:
            image_paths = [t[2] for t in test_set.data_to_iterate]
            shared.plot_top_anomalous(
                image_paths, scores, segmentations, tile_layout,
                os.path.join(output_dir, "plots", f"mvtec_{category}_top_anomalous.png"),
                n=top_anomalous,
                title=f"mvtec_{category}: top scores" + (" + tile grid" if tile_layout else ""),
                cutoff=cutoff,
            )
        if save_segmentation_images:
            shared.save_segmentation_images_for_category(
                os.path.join(output_dir, "segmentation_images", f"mvtec_{category}"),
                test_set, scores, segmentations, records, threshold,
                layout=tile_layout, cutoff=cutoff,
            )

        LOGGER.info(
            "mvtec_%s: instance_auroc=%.3f full_pixel_auroc=%s anomaly_pixel_auroc=%s",
            category, image_metrics["auroc"],
            f"{pixel_metrics['auroc']:.3f}" if pixel_metrics else "n/a",
            f"{anomaly_pixel_metrics['auroc']:.3f}" if anomaly_pixel_metrics else "n/a",
        )

    if not result_rows:
        raise click.ClickException("Nothing evaluated — check --model_root/--subdatasets/--data_path.")
    shared.write_results_and_confusion(output_dir, result_rows, model_name="loaded")
    shared.write_json(os.path.join(output_dir, "timing.json"), timings)
    LOGGER.info("All artifacts in %s", output_dir)


if __name__ == "__main__":
    main()
