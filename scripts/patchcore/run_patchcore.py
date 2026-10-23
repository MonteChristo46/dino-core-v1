import csv
import contextlib
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.backbones
import patchcore.common
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.tiled_patchcore
import patchcore.utils

LOGGER = logging.getLogger(__name__)

_DATASETS = {"mvtec": ["patchcore.datasets.mvtec", "MVTecDataset"]}


@click.group(chain=True)
@click.argument("results_path", type=str)
@click.option("--gpu", type=int, default=[0], multiple=True, show_default=True)
@click.option("--seed", type=int, default=0, show_default=True)
@click.option("--log_group", type=str, default="group")
@click.option("--log_project", type=str, default="project")
@click.option("--save_segmentation_images", is_flag=True)
@click.option("--save_patchcore_model", is_flag=True)
def main(**kwargs):
    pass


@main.result_callback()
def run(
    methods,
    results_path,
    gpu,
    seed,
    log_group,
    log_project,
    save_segmentation_images,
    save_patchcore_model,
):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    list_of_dataloaders = methods["get_dataloaders"](seed)

    device = patchcore.utils.set_torch_device(gpu)
    # Device context here is specifically set and used later
    # because there was GPU memory-bleeding which I could only fix with
    # context managers.
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    result_collect = []

    for dataloader_count, dataloaders in enumerate(list_of_dataloaders):
        LOGGER.info(
            "Evaluating dataset [{}] ({}/{})...".format(
                dataloaders["training"].name,
                dataloader_count + 1,
                len(list_of_dataloaders),
            )
        )

        patchcore.utils.fix_seeds(seed, device)

        dataset_name = dataloaders["training"].name

        with device_context:
            torch.cuda.empty_cache()
            imagesize = dataloaders["training"].dataset.imagesize
            sampler = methods["get_sampler"](
                device,
            )
            PatchCore_list = methods["get_patchcore"](imagesize, sampler, device)
            if len(PatchCore_list) > 1:
                LOGGER.info(
                    "Utilizing PatchCore Ensemble (N={}).".format(len(PatchCore_list))
                )
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                if PatchCore.backbone.seed is not None:
                    patchcore.utils.fix_seeds(PatchCore.backbone.seed, device)
                LOGGER.info(
                    "Training models ({}/{})".format(i + 1, len(PatchCore_list))
                )
                torch.cuda.empty_cache()
                PatchCore.fit(dataloaders["training"])

            torch.cuda.empty_cache()
            aggregator = {"scores": [], "segmentations": []}
            for i, PatchCore in enumerate(PatchCore_list):
                torch.cuda.empty_cache()
                LOGGER.info(
                    "Embedding test data with models ({}/{})".format(
                        i + 1, len(PatchCore_list)
                    )
                )
                scores, segmentations, labels_gt, masks_gt = PatchCore.predict(
                    dataloaders["testing"]
                )
                aggregator["scores"].append(scores)
                aggregator["segmentations"].append(segmentations)

            scores = np.array(aggregator["scores"])
            raw_min_scores = scores.min(axis=-1).reshape(-1, 1)
            raw_max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - raw_min_scores) / (raw_max_scores - raw_min_scores)
            scores = np.mean(scores, axis=0)

            segmentations = np.array(aggregator["segmentations"])
            min_scores = (
                segmentations.reshape(len(segmentations), -1)
                .min(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            max_scores = (
                segmentations.reshape(len(segmentations), -1)
                .max(axis=-1)
                .reshape(-1, 1, 1, 1)
            )
            segmentations = (segmentations - min_scores) / (max_scores - min_scores)
            segmentations = np.mean(segmentations, axis=0)

            anomaly_labels = [
                x[1] != "good" for x in dataloaders["testing"].dataset.data_to_iterate
            ]

            LOGGER.info("Computing evaluation metrics.")
            image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )
            auroc = image_metrics["auroc"]
            optimal_threshold = image_metrics["optimal_threshold"]

            # Compute PRO score & PW Auroc for all images
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                segmentations, masks_gt
            )
            full_pixel_auroc = pixel_scores["auroc"]

            # Compute PRO score & PW Auroc only images with anomalies
            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations[i] for i in sel_idxs],
                [masks_gt[i] for i in sel_idxs],
            )
            anomaly_pixel_auroc = pixel_scores["auroc"]

            result_collect.append(
                {
                    "model_name": log_group,
                    "dataset_name": dataset_name,
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "optimal_threshold": optimal_threshold,
                }
            )

            for key, item in result_collect[-1].items():
                if key not in ("dataset_name", "model_name"):
                    LOGGER.info("{0}: {1:3.3f}".format(key, item))

            # (Optional) Plot example images categorized as TP/FP/TN/FN.
            if save_segmentation_images:
                image_paths = [
                    x[2] for x in dataloaders["testing"].dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in dataloaders["testing"].dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array(
                        [0.229, 0.224, 0.225]
                    ).reshape(-1, 1, 1)
                    in_mean = np.array(
                        [0.485, 0.456, 0.406]
                    ).reshape(-1, 1, 1)
                    image = dataloaders["testing"].dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return dataloaders["testing"].dataset.transform_mask(mask).numpy()

                categories = []
                for i in range(len(scores)):
                    is_anomaly = anomaly_labels[i]
                    predicted_anomaly = scores[i] >= optimal_threshold
                    if is_anomaly and predicted_anomaly:
                        categories.append("TP")
                    elif not is_anomaly and predicted_anomaly:
                        categories.append("FP")
                    elif not is_anomaly and not predicted_anomaly:
                        categories.append("TN")
                    else:
                        categories.append("FN")

                image_save_path = os.path.join(
                    run_save_path, "segmentation_images", dataset_name
                )
                os.makedirs(image_save_path, exist_ok=True)
                patchcore.utils.plot_segmentation_images(
                    image_save_path,
                    image_paths,
                    segmentations,
                    scores,
                    mask_paths,
                    image_transform=image_transform,
                    mask_transform=mask_transform,
                    categories=categories,
                    optimal_threshold=optimal_threshold,
                )

                category_counts = {}
                for cat in categories:
                    category_counts[cat] = category_counts.get(cat, 0) + 1
                LOGGER.info(
                    f"Segmentation images: TP={category_counts.get('TP', 0)}, "
                    f"FP={category_counts.get('FP', 0)}, "
                    f"TN={category_counts.get('TN', 0)}, "
                    f"FN={category_counts.get('FN', 0)}"
                )

                tp = category_counts.get("TP", 0)
                fp = category_counts.get("FP", 0)
                tn = category_counts.get("TN", 0)
                fn = category_counts.get("FN", 0)
                precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
                recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

                result_collect[-1].update(
                    {
                        "TP": tp,
                        "FP": fp,
                        "FN": fn,
                        "TN": tn,
                        "Precision": precision,
                        "Recall": recall,
                    }
                )

            curves_save_path = os.path.join(run_save_path, "curves")
            os.makedirs(curves_save_path, exist_ok=True)
            np.savez(
                os.path.join(curves_save_path, f"{dataset_name}_curves.npz"),
                image_fpr=image_metrics["fpr"],
                image_tpr=image_metrics["tpr"],
                image_thresholds=image_metrics["threshold"],
                pixel_fpr=pixel_scores["fpr"],
                pixel_tpr=pixel_scores["tpr"],
                image_auroc=auroc,
                pixel_auroc=full_pixel_auroc,
                anomaly_pixel_auroc=anomaly_pixel_auroc,
                optimal_threshold=optimal_threshold,
            )

            # Save raw (score, label) pairs for dynamic F1 threshold recalculation.
            # Training-job reads this to populate model_score_labels with source='training',
            # so the ThresholdWorker can compute F1 from all individual pairs instead of
            # relying on pre-computed TPR/FPR rates combined with fake counts.
            np.savez(
                os.path.join(curves_save_path, f"{dataset_name}_scores.npz"),
                scores=scores,
                labels=np.array(anomaly_labels, dtype=bool),
            )

            # (Optional) Store PatchCore model for later re-use.
            # SAVE all patchcores only if mean_threshold is passed?
            if save_patchcore_model:
                patchcore_save_path = os.path.join(
                    run_save_path, "models", dataset_name
                )
                os.makedirs(patchcore_save_path, exist_ok=True)
                for i, PatchCore in enumerate(PatchCore_list):
                    prepend = (
                        "Ensemble-{}-{}_".format(i + 1, len(PatchCore_list))
                        if len(PatchCore_list) > 1
                        else ""
                    )
                    PatchCore.raw_score_min = float(raw_min_scores[i, 0])
                    PatchCore.raw_score_max = float(raw_max_scores[i, 0])
                    PatchCore.save_to_path(patchcore_save_path, prepend)

        LOGGER.info("\n\n-----\n")

    # Store all results and mean scores to a csv-file.
    all_keys = list(result_collect[-1].keys())
    result_metric_names = all_keys[2:]
    result_dataset_names = [results["dataset_name"] for results in result_collect]
    result_model_names = [results["model_name"] for results in result_collect]
    result_scores = [list(results.values())[2:] for results in result_collect]
    patchcore.utils.compute_and_store_final_results(
        run_save_path,
        result_scores,
        column_names=result_metric_names,
        row_names=result_dataset_names,
        model_names=result_model_names,
    )

    # Write confusion matrix CSV
    cm_columns = ["TP", "FP", "FN", "TN", "Precision", "Recall"]
    cm_path = os.path.join(run_save_path, "confusion_matrix.csv")
    with open(cm_path, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["Model", "Category"] + cm_columns)
        for results in result_collect:
            row = [results["model_name"], results["dataset_name"]] + [results.get(col, 0) for col in cm_columns]
            csv_writer.writerow(row)
        # Mean row
        mean_tp = np.mean([r.get("TP", 0) for r in result_collect])
        mean_fp = np.mean([r.get("FP", 0) for r in result_collect])
        mean_fn = np.mean([r.get("FN", 0) for r in result_collect])
        mean_tn = np.mean([r.get("TN", 0) for r in result_collect])
        mean_precision = np.mean([r.get("Precision", 0) for r in result_collect])
        mean_recall = np.mean([r.get("Recall", 0) for r in result_collect])
        csv_writer.writerow(
            [log_group, "Mean", mean_tp, mean_fp, mean_fn, mean_tn, mean_precision, mean_recall]
        )
    LOGGER.info(f"Confusion matrix saved to {cm_path}")


@main.command("patch_core")
# Pretraining-specific parameters.
@click.option("--backbone_names", "-b", type=str, multiple=True, default=[])
@click.option("--layers_to_extract_from", "-le", type=str, multiple=True, default=[])
# Parameters for Glue-code (to merge different parts of the pipeline.
@click.option("--pretrain_embed_dimension", type=int, default=1024)
@click.option("--target_embed_dimension", type=int, default=1024)
@click.option("--preprocessing", type=click.Choice(["mean", "conv"]), default="mean")
@click.option("--aggregation", type=click.Choice(["mean", "mlp"]), default="mean")
# Nearest-Neighbour Anomaly Scorer parameters.
@click.option("--anomaly_scorer_num_nn", type=int, default=5)
# Patch-parameters.
@click.option("--patchsize", type=int, default=3)
@click.option("--patchscore", type=str, default="max")
@click.option("--patchoverlap", type=float, default=0.0)
@click.option("--patchsize_aggregate", "-pa", type=int, multiple=True, default=[])
# NN on GPU.
@click.option("--faiss_on_gpu", is_flag=True)
@click.option("--faiss_num_workers", type=int, default=8)
# Tiling (very large images). --tile_size None keeps current untiled behavior.
@click.option("--tile_size", type=int, default=None, help="Square tile edge in px (multiple of 16). Omit to disable tiling.")
@click.option("--tile_overlap", type=int, default=None, help="Tile overlap in px (default tile_size // 2).")
@click.option("--tile_batch_size", type=int, default=8, help="Max tiles per backbone forward pass.")
def patch_core(
    backbone_names,
    layers_to_extract_from,
    pretrain_embed_dimension,
    target_embed_dimension,
    preprocessing,
    aggregation,
    patchsize,
    patchscore,
    patchoverlap,
    anomaly_scorer_num_nn,
    patchsize_aggregate,
    faiss_on_gpu,
    faiss_num_workers,
    tile_size,
    tile_overlap,
    tile_batch_size,
):
    backbone_names = list(backbone_names)

    def _is_dinov3_backbone(name):
        return "dinov3" in name.lower()

    if len(backbone_names) > 1:
        layers_to_extract_from_coll = [[] for _ in range(len(backbone_names))]
        for layer in layers_to_extract_from:
            idx = int(layer.split(".")[0])
            layer = ".".join(layer.split(".")[1:])
            layers_to_extract_from_coll[idx].append(layer)
    else:
        layers_to_extract_from_coll = [layers_to_extract_from]

    for idx, backbone_name in enumerate(backbone_names):
        if not layers_to_extract_from_coll[idx] and _is_dinov3_backbone(backbone_name):
            layers_to_extract_from_coll[idx] = ["7", "8"]

    def get_patchcore(input_shape, sampler, device):
        loaded_patchcores = []
        for backbone_name, layers_to_extract_from in zip(
            backbone_names, layers_to_extract_from_coll
        ):
            backbone_seed = None
            if ".seed-" in backbone_name:
                backbone_name, backbone_seed = backbone_name.split(".seed-")[0], int(
                    backbone_name.split("-")[-1]
                )
            backbone = patchcore.backbones.load(backbone_name)
            backbone.name, backbone.seed = backbone_name, backbone_seed

            nn_method = patchcore.common.FaissNN(faiss_on_gpu, faiss_num_workers)

            if tile_size is not None:
                # Tiled mode: inner PatchCore is configured for tile-sized
                # inputs; the dataloader imagesize is irrelevant (the wrapper
                # fixes input_shape to the tile size). Requires the dataset
                # to deliver native-resolution images (--native_resolution).
                tiled_instance = patchcore.tiled_patchcore.TiledPatchCore.load(
                    backbone=backbone,
                    layers_to_extract_from=layers_to_extract_from,
                    device=device,
                    tile_size=tile_size,
                    overlap=tile_overlap,
                    tile_batch_size=tile_batch_size,
                    pretrain_embed_dimension=pretrain_embed_dimension,
                    target_embed_dimension=target_embed_dimension,
                    patchsize=patchsize,
                    featuresampler=sampler,
                    anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                    nn_method=nn_method,
                )
                loaded_patchcores.append(tiled_instance)
                continue

            if input_shape is None:
                raise ValueError(
                    "Native-resolution dataloader (imagesize=None) requires "
                    "--tile_size on the patch_core command."
                )
            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load(
                backbone=backbone,
                layers_to_extract_from=layers_to_extract_from,
                device=device,
                input_shape=input_shape,
                pretrain_embed_dimension=pretrain_embed_dimension,
                target_embed_dimension=target_embed_dimension,
                patchsize=patchsize,
                featuresampler=sampler,
                anomaly_scorer_num_nn=anomaly_scorer_num_nn,
                nn_method=nn_method,
            )
            loaded_patchcores.append(patchcore_instance)
        return loaded_patchcores

    return ("get_patchcore", get_patchcore)


@main.command("sampler")
@click.argument("name", type=str)
@click.option("--percentage", "-p", type=float, default=0.1, show_default=True)
def sampler(name, percentage):
    def get_sampler(device):
        if name == "identity":
            return patchcore.sampler.IdentitySampler()
        elif name == "greedy_coreset":
            return patchcore.sampler.GreedyCoresetSampler(percentage, device)
        elif name == "approx_greedy_coreset":
            return patchcore.sampler.ApproximateGreedyCoresetSampler(percentage, device)

    return ("get_sampler", get_sampler)


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--train_val_split", type=float, default=1, show_default=True)
@click.option("--batch_size", default=2, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
@click.option(
    "--native_resolution",
    is_flag=True,
    help="Deliver images at original size (no resize/crop). Required for tiled "
    "processing of large images. Images within a category must have equal size.",
)
@click.option("--augment", is_flag=True)
def dataset(
    name,
    data_path,
    subdatasets,
    train_val_split,
    batch_size,
    resize,
    imagesize,
    num_workers,
    native_resolution,
    augment,
):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    if native_resolution:
        resize = None
        imagesize = None

    def get_dataloaders(seed):
        dataloaders = []
        for subdataset in subdatasets:
            train_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                train_val_split=train_val_split,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TRAIN,
                seed=seed,
                augment=augment,
            )

            test_dataset = dataset_library.__dict__[dataset_info[1]](
                data_path,
                classname=subdataset,
                resize=resize,
                imagesize=imagesize,
                split=dataset_library.DatasetSplit.TEST,
                seed=seed,
            )

            train_dataloader = torch.utils.data.DataLoader(
                train_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            test_dataloader = torch.utils.data.DataLoader(
                test_dataset,
                batch_size=batch_size,
                shuffle=False,
                num_workers=num_workers,
                pin_memory=True,
            )

            train_dataloader.name = name
            if subdataset is not None:
                train_dataloader.name += "_" + subdataset

            if train_val_split < 1:
                val_dataset = dataset_library.__dict__[dataset_info[1]](
                    data_path,
                    classname=subdataset,
                    resize=resize,
                    train_val_split=train_val_split,
                    imagesize=imagesize,
                    split=dataset_library.DatasetSplit.VAL,
                    seed=seed,
                )

                val_dataloader = torch.utils.data.DataLoader(
                    val_dataset,
                    batch_size=batch_size,
                    shuffle=False,
                    num_workers=num_workers,
                    pin_memory=True,
                )
            else:
                val_dataloader = None
            dataloader_dict = {
                "training": train_dataloader,
                "validation": val_dataloader,
                "testing": test_dataloader,
            }

            dataloaders.append(dataloader_dict)
        return dataloaders

    return ("get_dataloaders", get_dataloaders)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
