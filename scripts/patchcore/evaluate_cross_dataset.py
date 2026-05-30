import contextlib
import csv
import gc
import logging
import os
import sys

import click
import numpy as np
import torch

import patchcore.common
import patchcore.metrics
import patchcore.patchcore
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
def main(**kwargs):
    pass


@main.result_callback()
def run(methods, results_path, gpu, seed, log_group, log_project, save_segmentation_images):
    methods = {key: item for (key, item) in methods}

    run_save_path = patchcore.utils.create_storage_folder(
        results_path, log_project, log_group, mode="iterate"
    )

    device = patchcore.utils.set_torch_device(gpu)
    device_context = (
        torch.cuda.device("cuda:{}".format(device.index))
        if "cuda" in device.type.lower()
        else contextlib.suppress()
    )

    model_info = methods["get_models"]
    data_info = methods["get_datasets"]
    model_name = model_info["model_name"]

    model_categories = model_info["categories"]
    dataset_categories = data_info["categories"]
    matching_categories = sorted(set(model_categories) & set(dataset_categories))

    if not matching_categories:
        LOGGER.error("No matching categories between models and dataset!")
        sys.exit(1)

    LOGGER.info(f"Found {len(matching_categories)} matching categories: {matching_categories}")

    result_collect = []

    for category in matching_categories:
        model_path = model_info["model_paths"][category]
        saved_threshold = model_info["thresholds"].get(category)

        if saved_threshold is None:
            LOGGER.warning(f"No saved threshold for {category}, skipping.")
            continue

        LOGGER.info(f"Evaluating [{category}] with saved threshold {saved_threshold:.4f}...")

        dataset_name = f"mvtec_{category}"

        with device_context:
            torch.cuda.empty_cache()

            nn_method = patchcore.common.FaissNN(False, 4)
            patchcore_instance = patchcore.patchcore.PatchCore(device)
            patchcore_instance.load_from_path(
                load_path=model_path, device=device, nn_method=nn_method
            )

            test_dataloader = data_info["dataloaders"][category]
            imagesize = test_dataloader.dataset.imagesize

            LOGGER.info("Embedding test data...")
            scores, segmentations, labels_gt, masks_gt = patchcore_instance.predict(
                test_dataloader
            )

            scores = np.array(scores)
            segmentations = np.array(segmentations)

            min_scores = scores.min(axis=-1).reshape(-1, 1)
            max_scores = scores.max(axis=-1).reshape(-1, 1)
            scores = (scores - min_scores) / (max_scores - min_scores)
            scores = scores.flatten()

            min_seg = segmentations.reshape(segmentations.shape[0], -1).min(axis=-1).reshape(-1, 1, 1, 1)
            max_seg = segmentations.reshape(segmentations.shape[0], -1).max(axis=-1).reshape(-1, 1, 1, 1)
            segmentations = (segmentations - min_seg) / (max_seg - min_seg)
            segmentations = segmentations[0]

            anomaly_labels = [
                x[1] != "good" for x in test_dataloader.dataset.data_to_iterate
            ]

            LOGGER.info("Computing evaluation metrics.")
            image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores, anomaly_labels
            )
            auroc = image_metrics["auroc"]
            optimal_threshold = image_metrics["optimal_threshold"]

            pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                [segmentations], [masks_gt]
            )
            full_pixel_auroc = pixel_scores["auroc"]

            sel_idxs = []
            for i in range(len(masks_gt)):
                if np.sum(masks_gt[i]) > 0:
                    sel_idxs.append(i)
            if sel_idxs:
                pixel_scores = patchcore.metrics.compute_pixelwise_retrieval_metrics(
                    [segmentations[i] for i in sel_idxs],
                    [masks_gt[i] for i in sel_idxs],
                )
                anomaly_pixel_auroc = pixel_scores["auroc"]
            else:
                anomaly_pixel_auroc = 0.0

            tp = fp = fn = tn = 0
            categories = []
            for i in range(len(scores)):
                is_anomaly = anomaly_labels[i]
                predicted_anomaly = scores[i] >= saved_threshold
                if is_anomaly and predicted_anomaly:
                    categories.append("TP")
                    tp += 1
                elif not is_anomaly and predicted_anomaly:
                    categories.append("FP")
                    fp += 1
                elif not is_anomaly and not predicted_anomaly:
                    categories.append("TN")
                    tn += 1
                else:
                    categories.append("FN")
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

            LOGGER.info(
                f"Segmentation images: TP={tp}, FP={fp}, TN={tn}, FN={fn}"
            )

            result_collect.append(
                {
                    "model_name": model_name,
                    "dataset_name": f"{model_name} | {dataset_name}",
                    "instance_auroc": auroc,
                    "full_pixel_auroc": full_pixel_auroc,
                    "anomaly_pixel_auroc": anomaly_pixel_auroc,
                    "saved_threshold": saved_threshold,
                    "computed_threshold": optimal_threshold,
                    "TP": tp,
                    "FP": fp,
                    "FN": fn,
                    "TN": tn,
                    "Precision": precision,
                    "Recall": recall,
                }
            )

            for key, item in result_collect[-1].items():
                if key != "dataset_name":
                    LOGGER.info(f"{key}: {item}")

            if save_segmentation_images:
                image_paths = [
                    x[2] for x in test_dataloader.dataset.data_to_iterate
                ]
                mask_paths = [
                    x[3] for x in test_dataloader.dataset.data_to_iterate
                ]

                def image_transform(image):
                    in_std = np.array([0.229, 0.224, 0.225]).reshape(-1, 1, 1)
                    in_mean = np.array([0.485, 0.456, 0.406]).reshape(-1, 1, 1)
                    image = test_dataloader.dataset.transform_img(image)
                    return np.clip(
                        (image.numpy() * in_std + in_mean) * 255, 0, 255
                    ).astype(np.uint8)

                def mask_transform(mask):
                    return test_dataloader.dataset.transform_mask(mask).numpy()

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
                    optimal_threshold=saved_threshold,
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
                optimal_threshold=saved_threshold,
            )

            del patchcore_instance
            gc.collect()

        LOGGER.info("\n\n-----\n")

    # Store results.csv
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

    # Write confusion_matrix.csv
    cm_columns = ["TP", "FP", "FN", "TN", "Precision", "Recall"]
    cm_path = os.path.join(run_save_path, "confusion_matrix.csv")
    with open(cm_path, "w") as csv_file:
        csv_writer = csv.writer(csv_file, delimiter=",")
        csv_writer.writerow(["Model", "Category"] + cm_columns)
        for results in result_collect:
            row = [results["model_name"], results["dataset_name"]] + [results[col] for col in cm_columns]
            csv_writer.writerow(row)
        if result_collect:
            mean_tp = np.mean([r["TP"] for r in result_collect])
            mean_fp = np.mean([r["FP"] for r in result_collect])
            mean_fn = np.mean([r["FN"] for r in result_collect])
            mean_tn = np.mean([r["TN"] for r in result_collect])
            mean_precision = np.mean([r["Precision"] for r in result_collect])
            mean_recall = np.mean([r["Recall"] for r in result_collect])
            csv_writer.writerow(
                [model_name, "Mean", mean_tp, mean_fp, mean_fn, mean_tn, mean_precision, mean_recall]
            )
    LOGGER.info(f"Confusion matrix saved to {cm_path}")


@main.command("cross_eval")
@click.option("--model_root", type=str, required=True, help="Path to training results containing models/ and curves/")
def cross_eval(model_root):
    model_dir = os.path.join(model_root, "models")
    curves_dir = os.path.join(model_root, "curves")

    if not os.path.isdir(model_dir):
        LOGGER.error(f"Models directory not found: {model_dir}")
        sys.exit(1)
    if not os.path.isdir(curves_dir):
        LOGGER.error(f"Curves directory not found: {curves_dir}")
        sys.exit(1)

    model_name = os.path.basename(os.path.normpath(model_root))

    model_paths = {}
    thresholds = {}
    categories = []

    for entry in sorted(os.listdir(model_dir)):
        entry_path = os.path.join(model_dir, entry)
        if not os.path.isdir(entry_path):
            continue
        if not entry.startswith("mvtec_"):
            continue
        category = entry.replace("mvtec_", "")
        model_paths[category] = entry_path

        curves_file = os.path.join(curves_dir, f"mvtec_{category}_curves.npz")
        if os.path.exists(curves_file):
            curves_data = np.load(curves_file)
            thresholds[category] = float(curves_data["optimal_threshold"])
            categories.append(category)
        else:
            LOGGER.warning(f"No curves file for {category}, skipping.")

    def get_models():
        return {
            "model_name": model_name,
            "categories": categories,
            "model_paths": model_paths,
            "thresholds": thresholds,
        }

    return ("get_models", get_models())


@main.command("dataset")
@click.argument("name", type=str)
@click.argument("data_path", type=click.Path(exists=True, file_okay=False))
@click.option("--subdatasets", "-d", multiple=True, type=str, required=True)
@click.option("--batch_size", default=1, type=int, show_default=True)
@click.option("--num_workers", default=8, type=int, show_default=True)
@click.option("--resize", default=256, type=int, show_default=True)
@click.option("--imagesize", default=224, type=int, show_default=True)
def dataset(name, data_path, subdatasets, batch_size, resize, imagesize, num_workers):
    dataset_info = _DATASETS[name]
    dataset_library = __import__(dataset_info[0], fromlist=[dataset_info[1]])

    dataloaders = {}
    categories = []

    for subdataset in subdatasets:
        test_dataset = dataset_library.__dict__[dataset_info[1]](
            data_path,
            classname=subdataset,
            resize=resize,
            imagesize=imagesize,
            split=dataset_library.DatasetSplit.TEST,
            seed=0,
        )

        test_dataloader = torch.utils.data.DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
        )

        test_dataloader.name = name + "_" + subdataset
        dataloaders[subdataset] = test_dataloader
        categories.append(subdataset)

    def get_datasets():
        return {
            "categories": categories,
            "dataloaders": dataloaders,
        }

    return ("get_datasets", get_datasets())


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    LOGGER.info("Command line arguments: {}".format(" ".join(sys.argv)))
    main()
