"""Train and evaluate tiled PatchCore on large images — one command.

Trains one TiledPatchCore per category at native resolution (images are cut
into overlapping backbone-sized tiles; the memory bank coreset is taken ONCE
over the pooled tile features), evaluates the test split, and writes the full
analytics pack next to the usual results files.

Examples
--------
# Everything in / out of the box (DINOv3 ViT-B, 384px tiles, 50% overlap):
uv run python tiled_patchcore/run_tiled_patchcore.py \
    --data_path dataset --subdatasets bottle --results_path results_tiled

# All categories, GPU memory control, keep example overlays:
uv run python tiled_patchcore/run_tiled_patchcore.py \
    --data_path dataset --backbone dinov3_vitl \
    --tile_size 384 --tile_overlap 192 --tile_batch_size 4 \
    --coreset_percentage 0.05 --save_segmentation_images

Run folder layout (compatible with scripts/patchcore tooling):
    <results_path>/<log_group>/models/mvtec_<cat>/   (params, faiss, tiling_params)
    <results_path>/<log_group>/results.csv, confusion_matrix.csv,
    per_image_scores.csv, memory_bank.json, timing.json, curves/, plots/
"""
import logging
import os
import sys
import time

import click
import numpy as np
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import shared  # noqa: F401,E402  — must come first: puts repo src/ on sys.path

import patchcore.backbones
import patchcore.common
import patchcore.sampler
import patchcore.utils
from patchcore.tiled_patchcore import TiledPatchCore

LOGGER = logging.getLogger("run_tiled_patchcore")


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
# Data
@click.option("--data_path", required=True, type=click.Path(exists=True, file_okay=False),
              help="Dataset root with MVTec-format categories.")
@click.option("--subdatasets", "-d", multiple=True,
              help="Categories to train (default: every folder under --data_path).")
@click.option("--train_val_split", type=float, default=1.0, show_default=True)
@click.option("--batch_size", type=int, default=1, show_default=True,
              help="Images per dataloader batch (native-resolution images are big).")
@click.option("--num_workers", type=int, default=8, show_default=True)
# Model
@click.option("--backbone", default="dinov3_vitb", show_default=True,
              help="Any patchcore backbone; DINOv3 dims/layers auto-derived.")
@click.option("--layers", default=None,
              help="Comma-separated feature layers (default per family: "
              "'7,8' DINOv3, 'layer2,layer3' WR).")
@click.option("--tile_size", type=int, default=384, show_default=True,
              help="Square tile edge px; multiple of 16. Also the backbone input.")
@click.option("--tile_overlap", type=int, default=None,
              help="Tile overlap px (default tile_size // 2, i.e. 50%).")
@click.option("--tile_batch_size", type=int, default=8, show_default=True,
              help="Tiles per backbone forward pass (GPU peak knob).")
@click.option("--pad_value", type=float, default=0.0, show_default=True,
              help="Border padding value in normalized space (0.0 = dataset mean).")
@click.option("--coreset_percentage", "-p", type=float, default=0.01, show_default=True,
              help="Memory-bank keep ratio (PatchCore paper: 1% ≈ 10% AUROC).")
@click.option("--anomaly_scorer_num_nn", type=int, default=5, show_default=True)
@click.option("--patchsize", type=int, default=1, show_default=True,
              help="1 = per-patch features (DINOv3); 3 for WR-style aggregation.")
@click.option("--pretrain_embed_dimension", type=int, default=None,
              help="Auto-derived for DINOv3/WR; required for exotic backbones.")
@click.option("--target_embed_dimension", type=int, default=None)
# Compute / output
@click.option("--gpu", type=int, default=None,
              help="CUDA device id. Default: auto — CUDA, then Apple MPS, then CPU.")
@click.option("--cpu", is_flag=True,
              help="Force CPU (auto prefers CUDA, then Apple MPS).")
@click.option("--faiss_num_workers", type=int, default=8, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--results_path", default="results_tiled", show_default=True)
@click.option("--log_project", default="tiled_patchcore", show_default=True)
@click.option("--log_group", default="tiled", show_default=True)
@click.option("--save_segmentation_images", is_flag=True,
              help="Per-image original/GT/prediction panels (more disk).")
@click.option("--top_anomalous", type=int, default=4, show_default=True,
              help="# panels with tile-grid overlay per category (0 disables).")
@click.option("--no_save_model", is_flag=True, help="Skip saving the memory bank.")
def main(**kwargs):
    """Train + evaluate tiled PatchCore (one command, full analytics pack).

    \b
    Examples:
      uv run python tiled_patchcore/run_tiled_patchcore.py \
          --data_path dataset --subdatasets bottle --results_path results_tiled
      uv run python tiled_patchcore/run_tiled_patchcore.py \
          --data_path dataset --backbone dinov3_vitl --tile_size 384 \
          --tile_batch_size 4 --coreset_percentage 0.05

    Tiled defaults: native resolution, good-only training, approx-greedy
    coreset taken ONCE globally, DINOv3 layers/dims auto-derived.
    """
    opts = kwargs
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    device, faiss_on_gpu = shared.resolve_device(gpu=opts["gpu"], cpu=opts["cpu"])
    patchcore.utils.fix_seeds(opts["seed"], device)
    torch.manual_seed(opts["seed"])
    np.random.seed(opts["seed"])

    categories = list(opts["subdatasets"]) or shared.discover_categories(opts["data_path"])
    if not categories:
        raise click.UsageError(
            f"No MVTec-format categories (folders with train/) under {opts['data_path']}."
        )
    layers = opts["layers"]
    layers = layers.split(",") if layers else shared.default_layers(opts["backbone"])
    if not layers:
        raise click.UsageError(
            f"--layers is required for backbone '{opts['backbone']}'."
        )
    pretrain_dim, target_dim = shared.resolve_embed_dims(
        opts["backbone"], opts["pretrain_embed_dimension"], opts["target_embed_dimension"]
    )

    run_dir = patchcore.utils.create_storage_folder(
        opts["results_path"], opts["log_project"], opts["log_group"], mode="iterate"
    )
    models_dir = os.path.join(run_dir, "models")
    curves_dir = os.path.join(run_dir, "curves")
    plots_dir = os.path.join(run_dir, "plots")
    for folder in (models_dir, curves_dir, plots_dir):
        os.makedirs(folder, exist_ok=True)

    LOGGER.info(
        "Tiled PatchCore | device=%s backbone=%s tile=%d stride=%s categories=%s",
        device, opts["backbone"], opts["tile_size"],
        opts["tile_overlap"] is not None and opts["tile_size"] - opts["tile_overlap"] or "auto",
        ",".join(categories),
    )

    result_rows, memory_bank, timings = [], {}, {}
    for category in categories:
        LOGGER.info("=== Category: %s ===", category)
        train_set, dropped = shared.build_dataset(
            opts["data_path"], category, "TRAIN",
            resize=None, imagesize=None,  # tiled => native resolution
            train_val_split=opts["train_val_split"], seed=opts["seed"],
            good_only=True,
        )
        test_set, _ = shared.build_dataset(
            opts["data_path"], category, "TEST", resize=None, imagesize=None,
            seed=opts["seed"],
        )
        if dropped:
            LOGGER.info("Good-only filter dropped %d train/bad samples.", dropped)
        train_loader = shared.build_dataloader(train_set, opts["batch_size"], opts["num_workers"])
        test_loader = shared.build_dataloader(test_set, opts["batch_size"], opts["num_workers"])

        image_h, image_w = shared.dataset_image_hw(test_set)
        # Seed before backbone construction: untrained test backbones (random
        # init) must match what load_models_from_dir() reproduces at eval time.
        torch.manual_seed(opts["seed"])
        backbone = patchcore.backbones.load(opts["backbone"])
        backbone.name, backbone.seed = opts["backbone"], None

        sampler = shared.RecordingSampler(
            patchcore.sampler.ApproximateGreedyCoresetSampler(
                opts["coreset_percentage"], device
            )
        )
        model = TiledPatchCore.load(
            backbone=backbone,
            layers_to_extract_from=layers,
            device=device,
            tile_size=opts["tile_size"],
            overlap=opts["tile_overlap"],
            pad_value=opts["pad_value"],
            tile_batch_size=opts["tile_batch_size"],
            pretrain_embed_dimension=pretrain_dim,
            target_embed_dimension=target_dim,
            patchsize=opts["patchsize"],
            featuresampler=sampler,
            anomaly_scorer_num_nn=opts["anomaly_scorer_num_nn"],
            nn_method=patchcore.common.FaissNN(faiss_on_gpu, opts["faiss_num_workers"]),
        )
        tile_layout = model.tile_maker.compute_layout(image_h, image_w)

        start = time.perf_counter()
        model.fit(train_loader)
        fit_seconds = time.perf_counter() - start

        evaluation = shared.evaluate_models([(model, True)], test_loader)
        scores = evaluation["scores"]
        segmentations = evaluation["segmentations"]
        labels = [int(l) for l in evaluation["labels_gt"]]
        masks_gt = evaluation["masks_gt"]
        image_metrics = evaluation["image_metrics"]
        pixel_metrics = evaluation["pixel_metrics"]
        anomaly_pixel_metrics = evaluation["anomaly_pixel_metrics"]
        threshold = float(image_metrics["optimal_threshold"])
        records = shared.classify_predictions(scores, labels, threshold)

        # ---- results row + analytics -------------------------------------
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
            os.path.join(run_dir, "per_image_scores.csv"),
            category, test_set, scores, records,
        )
        memory_bank[f"mvtec_{category}"] = {
            "tile_size": opts["tile_size"],
            "stride": model.tile_maker.stride,
            "tiles_per_image": tile_layout.num_tiles,
            "image_hw": [image_h, image_w],
            "pooled_feature_rows": sampler.pooled_rows,
            "coreset_rows": sampler.coreset_rows,
            "feature_dim": int(model.anomaly_scorer.detection_features.shape[1]),
        }
        timings[f"mvtec_{category}"] = {
            "fit_seconds": round(fit_seconds, 3),
            "predict_seconds": round(evaluation["predict_seconds"], 3),
            "n_train_images": len(train_set),
            "n_test_images": len(test_set),
        }

        np.savez(
            os.path.join(curves_dir, f"mvtec_{category}_curves.npz"),
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

        first_test_image = test_set.data_to_iterate[0][2]
        shared.plot_tile_grid(
            first_test_image, tile_layout,
            os.path.join(plots_dir, f"mvtec_{category}_tile_grid.png"),
            title=f"mvtec_{category} tile grid",
        )
        if shared.plot_score_distribution(
            scores, labels, threshold,
            os.path.join(plots_dir, f"mvtec_{category}_score_dist.png"),
            title=f"mvtec_{category}: score distribution",
        ):
            shared.plot_roc(
                image_metrics,
                os.path.join(plots_dir, f"mvtec_{category}_roc.png"),
                title=f"mvtec_{category}: image-level ROC",
            )
        if opts["top_anomalous"] > 0:
            image_paths = [t[2] for t in test_set.data_to_iterate]
            shared.plot_top_anomalous(
                image_paths, scores, segmentations, tile_layout,
                os.path.join(plots_dir, f"mvtec_{category}_top_anomalous.png"),
                n=opts["top_anomalous"],
                title=f"mvtec_{category}: top scores + tile grid (check seams/borders)",
            )
        if opts["save_segmentation_images"]:
            shared.save_segmentation_images_for_category(
                os.path.join(run_dir, "segmentation_images", f"mvtec_{category}"),
                test_set, scores, segmentations, records, threshold,
            )

        if not opts["no_save_model"]:
            model_dir = os.path.join(models_dir, f"mvtec_{category}")
            model.save_to_path(model_dir)
            index_mb = os.path.getsize(
                os.path.join(model_dir, "nnscorer_search_index.faiss")
            ) / 1e6
            memory_bank[f"mvtec_{category}"]["faiss_index_mb"] = round(index_mb, 2)
            LOGGER.info("Saved model (+tiling_params.pkl) to %s", model_dir)

        LOGGER.info(
            "mvtec_%s: instance_auroc=%.3f full_pixel_auroc=%s anomaly_pixel_auroc=%s "
            "threshold=%.3f (fit %.1fs, predict %.1fs)",
            category, image_metrics["auroc"],
            f"{pixel_metrics['auroc']:.3f}" if pixel_metrics else "n/a",
            f"{anomaly_pixel_metrics['auroc']:.3f}" if anomaly_pixel_metrics else "n/a",
            threshold, fit_seconds, evaluation["predict_seconds"],
        )

    shared.write_results_and_confusion(run_dir, result_rows, model_name=opts["log_group"])
    shared.write_json(os.path.join(run_dir, "memory_bank.json"), memory_bank)
    shared.write_json(os.path.join(run_dir, "timing.json"), timings)
    LOGGER.info("All artifacts in %s", run_dir)


if __name__ == "__main__":
    main()
