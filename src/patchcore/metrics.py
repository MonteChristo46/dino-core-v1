"""Anomaly metrics."""
import numpy as np
from sklearn import metrics


def compute_imagewise_retrieval_metrics(
    anomaly_prediction_weights, anomaly_ground_truth_labels
):
    """
    Computes retrieval statistics (AUROC, FPR, TPR).

    Args:
        anomaly_prediction_weights: [np.array or list] [N] Assignment weights
                                    per image. Higher indicates higher
                                    probability of being an anomaly.
        anomaly_ground_truth_labels: [np.array or list] [N] Binary labels - 1
                                    if image is an anomaly, 0 if not.
    """
    fpr, tpr, thresholds = metrics.roc_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    auroc = metrics.roc_auc_score(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )

    precision, recall, thresholds_pr = metrics.precision_recall_curve(
        anomaly_ground_truth_labels, anomaly_prediction_weights
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )
    optimal_threshold = float(thresholds_pr[np.argmax(F1_scores)])

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "threshold": thresholds,
        "optimal_threshold": optimal_threshold,
    }


def _subsample_pixels(scores_2d, gt_2d, max_pixels, seed):
    """Deterministic per-image Bernoulli subsample of pixel scores/labels.

    Stratified per image so every image (including the few anomaly images in
    anomaly-only pixel AUROC) contributes; per-image RNG buffers keep peak
    memory at one image's pixels. Seeded => bit-reproducible across runs.
    """
    rng = np.random.default_rng(seed)
    quota = max(1, max_pixels // scores_2d.shape[0])
    kept_scores, kept_gt = [], []
    for row_scores, row_gt in zip(scores_2d, gt_2d):
        if row_scores.size <= quota:
            kept_scores.append(np.asarray(row_scores))
            kept_gt.append(np.asarray(row_gt))
            continue
        keep = rng.random(row_scores.size) < (quota / row_scores.size)
        kept_scores.append(np.asarray(row_scores[keep]))
        kept_gt.append(np.asarray(row_gt[keep]))
    return np.concatenate(kept_scores), np.concatenate(kept_gt)


def compute_pixelwise_retrieval_metrics(
    anomaly_segmentations, ground_truth_masks, max_pixels=None, seed=0
):
    """
    Computes pixel-wise statistics (AUROC, FPR, TPR) for anomaly segmentations
    and ground truth segmentation masks.

    Args:
        anomaly_segmentations: [list of np.arrays or np.array] [NxHxW] Contains
                                generated segmentation masks.
        ground_truth_masks: [list of np.arrays or np.array] [NxHxW] Contains
                            predefined ground truth segmentation masks
        max_pixels: optional cap on the total number of pixels evaluated.
                            Above the cap a deterministic (seeded) subsample is
                            used — full-fidelity roc_curve on >1G native-res
                            pixels costs tens of GB for a rank statistic whose
                            estimate noise at 250M samples is far below 1e-3.
                            None (default) evaluates every pixel.
        seed: RNG seed for the subsample.

    The returned dict carries ``pixel_metric_samples``, the number of pixels
    actually evaluated (== N*H*W when below the cap).
    """
    if isinstance(anomaly_segmentations, list):
        anomaly_segmentations = np.stack(anomaly_segmentations)
    if isinstance(ground_truth_masks, list):
        ground_truth_masks = np.stack(ground_truth_masks)

    if max_pixels is not None and anomaly_segmentations.size > max_pixels:
        flat_anomaly_segmentations, flat_ground_truth_masks = _subsample_pixels(
            anomaly_segmentations.reshape(len(anomaly_segmentations), -1),
            ground_truth_masks.reshape(len(ground_truth_masks), -1),
            max_pixels,
            seed,
        )
    else:
        flat_anomaly_segmentations = anomaly_segmentations.ravel()
        flat_ground_truth_masks = ground_truth_masks.ravel()
    pixel_metric_samples = int(flat_anomaly_segmentations.size)

    fpr, tpr, thresholds = metrics.roc_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    auroc = metrics.roc_auc_score(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )

    precision, recall, thresholds = metrics.precision_recall_curve(
        flat_ground_truth_masks.astype(int), flat_anomaly_segmentations
    )
    F1_scores = np.divide(
        2 * precision * recall,
        precision + recall,
        out=np.zeros_like(precision),
        where=(precision + recall) != 0,
    )

    optimal_threshold = thresholds[np.argmax(F1_scores)]
    predictions = (flat_anomaly_segmentations >= optimal_threshold).astype(int)
    fpr_optim = np.mean(predictions > flat_ground_truth_masks)
    fnr_optim = np.mean(predictions < flat_ground_truth_masks)

    return {
        "auroc": auroc,
        "fpr": fpr,
        "tpr": tpr,
        "optimal_threshold": optimal_threshold,
        "optimal_fpr": fpr_optim,
        "optimal_fnr": fnr_optim,
        "pixel_metric_samples": pixel_metric_samples,
    }
