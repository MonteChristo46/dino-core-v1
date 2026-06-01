import contextlib
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.metrics import f1_score

import patchcore.backbones
import patchcore.common
import patchcore.datasets.mvtec as mvtec
import patchcore.metrics
import patchcore.patchcore
import patchcore.sampler
import patchcore.utils
from patchcore_optuna.search_spaces import BackboneConfig, CORESET_PCT, BATCH_SIZE, NUM_WORKERS

LOGGER = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent


def _resolve_weights_dir(weights_dir: str) -> str:
    if os.path.isabs(weights_dir):
        return weights_dir
    return str(SCRIPT_DIR / weights_dir)

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvalParams:
    pretrain_dim: int
    target_dim: int
    image_size: int
    resize: int
    coreset_pct: float
    batch_size: int
    num_workers: int


class PatchCoreEvaluator:
    def __init__(
        self,
        config: BackboneConfig,
        dataset_path: str,
        device: torch.device,
    ):
        self.config = config
        self.dataset_path = dataset_path
        self.device = device
        self.params = EvalParams(
            pretrain_dim=config.pretrain_dim,
            target_dim=config.target_dim,
            image_size=config.image_size,
            resize=config.resize,
            coreset_pct=CORESET_PCT,
            batch_size=BATCH_SIZE,
            num_workers=NUM_WORKERS,
        )

        weights_dir = _resolve_weights_dir(config.weights_dir)
        LOGGER.info(f"Loading backbone {config.name} from {weights_dir}")
        self.backbone = patchcore.dinov3.DinoV3Backbone(weights_dir=weights_dir)
        self.backbone.name = config.name
        self.backbone.to(device)

    def _create_dataloaders(
        self, category: str, seed: int
    ) -> dict[str, torch.utils.data.DataLoader]:
        p = self.params
        train_dataset = mvtec.MVTecDataset(
            source=self.dataset_path,
            classname=category,
            resize=p.resize,
            imagesize=p.image_size,
            split=mvtec.DatasetSplit.TRAIN,
            train_val_split=1.0,
        )
        test_dataset = mvtec.MVTecDataset(
            source=self.dataset_path,
            classname=category,
            resize=p.resize,
            imagesize=p.image_size,
            split=mvtec.DatasetSplit.TEST,
        )
        return {
            "training": torch.utils.data.DataLoader(
                train_dataset,
                batch_size=p.batch_size,
                shuffle=False,
                num_workers=p.num_workers,
                pin_memory=True,
            ),
            "testing": torch.utils.data.DataLoader(
                test_dataset,
                batch_size=p.batch_size,
                shuffle=False,
                num_workers=p.num_workers,
                pin_memory=True,
            ),
        }

    def _create_patchcore(
        self,
        layers: tuple[str, ...],
        patch_size: int,
        num_nn: int,
        sampler,
        device: torch.device,
    ) -> patchcore.patchcore.PatchCore:
        p = self.params
        nn_method = patchcore.common.FaissNN(False, p.num_workers)

        pc = patchcore.patchcore.PatchCore(device)
        pc.load(
            backbone=self.backbone,
            layers_to_extract_from=list(layers),
            device=device,
            input_shape=(3, p.image_size, p.image_size),
            pretrain_embed_dimension=p.pretrain_dim,
            target_embed_dimension=p.target_dim,
            patchsize=patch_size,
            featuresampler=sampler,
            anomaly_scorer_num_nn=num_nn,
            nn_method=nn_method,
        )
        return pc

    def _compute_f1_at_threshold(
        self, scores: list[float], labels: list[int], threshold: float
    ) -> dict[str, float]:
        preds = [1 if s >= threshold else 0 for s in scores]
        f1 = f1_score(labels, preds)
        precision = (
            sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
            / max(sum(preds), 1)
        )
        recall = (
            sum(1 for p, l in zip(preds, labels) if p == 1 and l == 1)
            / max(sum(labels), 1)
        )
        return {"f1": f1, "precision": precision, "recall": recall}

    def evaluate(
        self,
        layers: tuple[str, ...],
        patch_size: int,
        num_nn: int,
        categories: list[str],
        seed: int = 0,
    ) -> dict[str, Any]:
        device_context = (
            torch.cuda.device("cuda:{}".format(self.device.index))
            if "cuda" in self.device.type.lower()
            else contextlib.suppress()
        )

        sampler = patchcore.sampler.ApproximateGreedyCoresetSampler(
            self.params.coreset_pct, self.device
        )

        per_category = []
        for category in categories:
            LOGGER.info(f"Evaluating category: {category}")
            dataloaders = self._create_dataloaders(category, seed)

            with device_context:
                torch.cuda.empty_cache()

                pc = self._create_patchcore(
                    layers=layers,
                    patch_size=patch_size,
                    num_nn=num_nn,
                    sampler=sampler,
                    device=self.device,
                )

                LOGGER.info(f"  Fitting PatchCore for {category}...")
                pc.fit(dataloaders["training"])

                LOGGER.info(f"  Predicting for {category}...")
                scores, segmentations, labels_gt, masks_gt = pc.predict(
                    dataloaders["testing"]
                )

                torch.cuda.empty_cache()

            scores_np = np.array(scores)
            min_s = scores_np.min()
            max_s = scores_np.max()
            if max_s - min_s > 0:
                scores_np = (scores_np - min_s) / (max_s - min_s)

            image_metrics = patchcore.metrics.compute_imagewise_retrieval_metrics(
                scores_np, labels_gt
            )
            auroc = image_metrics["auroc"]
            optimal_threshold = image_metrics["optimal_threshold"]

            f1_metrics = self._compute_f1_at_threshold(
                scores_np.tolist(), labels_gt, optimal_threshold
            )

            per_category.append(
                {
                    "category": category,
                    "f1": f1_metrics["f1"],
                    "auroc": auroc,
                    "optimal_threshold": optimal_threshold,
                    "precision": f1_metrics["precision"],
                    "recall": f1_metrics["recall"],
                }
            )

            LOGGER.info(
                f"  {category}: F1={f1_metrics['f1']:.4f}, "
                f"AUROC={auroc:.4f}, "
                f"P={f1_metrics['precision']:.4f}, "
                f"R={f1_metrics['recall']:.4f}"
            )

        mean_f1 = np.mean([r["f1"] for r in per_category])
        mean_auroc = np.mean([r["auroc"] for r in per_category])

        return {
            "mean_f1": float(mean_f1),
            "mean_auroc": float(mean_auroc),
            "per_category": per_category,
        }
