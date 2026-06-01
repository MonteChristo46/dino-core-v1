#!/usr/bin/env python
import argparse
import csv
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path

import optuna
import torch

SCRIPT_DIR = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(SCRIPT_DIR / "src"))

from patchcore_optuna.evaluator import PatchCoreEvaluator
from patchcore_optuna.objective import PatchCoreObjective
from patchcore_optuna.search_spaces import (
    DEFAULT_CATEGORIES,
    get_backbone_config,
)

LOGGER = logging.getLogger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(
        description="Optuna hyperparameter search for PatchCore with DINOv3"
    )
    parser.add_argument(
        "--n-trials", type=int, default=100, help="Number of Optuna trials"
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--dataset-path",
        type=str,
        default=os.environ.get(
            "DATASET_PATH",
            "/home/daniel/jetbrains-projects/glitch-hunt-monolith/services/glitch-hunt-ml/dataset/mvtec_small",
        ),
        help="Path to MVTec dataset",
    )
    parser.add_argument("--gpu", type=int, default=0, help="GPU ID")
    parser.add_argument(
        "--categories",
        type=str,
        nargs="+",
        default=DEFAULT_CATEGORIES,
        help="Categories to evaluate on",
    )
    parser.add_argument(
        "--backbone",
        type=str,
        default="dinov3_vits",
        help="Backbone name (dinov3_vits, dinov3_vitb, dinov3_vitl, dinov3_vithplus)",
    )
    parser.add_argument(
        "--timestamp",
        type=str,
        default=None,
        help="Timestamp for study name (default: YYYYMMDD)",
    )
    parser.add_argument(
        "--study-name",
        type=str,
        default=None,
        help="Optuna study name (default: {backbone}_layer_search)",
    )
    parser.add_argument(
        "--results-dir",
        type=str,
        default=str(SCRIPT_DIR / "results" / "optuna"),
        help="Base directory for Optuna results",
    )
    return parser.parse_args()


def export_results(study: optuna.Study, results_dir: Path, categories: list[str]):
    trials = study.trials
    rows = []
    for t in trials:
        row = {
            "trial_number": t.number,
            "state": t.state.name,
            "layers": str(t.params.get("layers", "")),
            "patch_size": t.params.get("patch_size", ""),
            "num_nn": t.params.get("num_nn", ""),
            "mean_f1": t.user_attrs.get("mean_f1", None),
            "mean_auroc": t.user_attrs.get("mean_auroc", None),
        }
        for cat in categories:
            row[f"{cat}_f1"] = t.user_attrs.get(f"{cat}_f1", None)
            row[f"{cat}_auroc"] = t.user_attrs.get(f"{cat}_auroc", None)
        rows.append(row)

    csv_path = results_dir / "best_trials.csv"
    with open(csv_path, "w", newline="") as f:
        if rows:
            writer = csv.DictWriter(f, fieldnames=rows[0].keys())
            writer.writeheader()
            writer.writerows(rows)
    LOGGER.info(f"Results CSV saved to {csv_path}")

    best = study.best_trial
    summary = {
        "study_name": study.study_name,
        "n_trials": len(trials),
        "best_trial": {
            "number": best.number,
            "mean_f1": best.value,
            "mean_auroc": best.user_attrs.get("mean_auroc"),
            "params": best.params,
            "per_category": best.user_attrs.get("per_category", []),
        },
        "top_10_trials": [],
    }

    sorted_trials = sorted(
        [t for t in trials if t.state == optuna.trial.TrialState.COMPLETE],
        key=lambda t: t.value if t.value is not None else 0,
        reverse=True,
    )
    for t in sorted_trials[:10]:
        summary["top_10_trials"].append(
            {
                "number": t.number,
                "mean_f1": t.value,
                "mean_auroc": t.user_attrs.get("mean_auroc"),
                "params": t.params,
            }
        )

    summary_path = results_dir / "summary.json"
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2, default=str)
    LOGGER.info(f"Summary JSON saved to {summary_path}")


def main():
    args = parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    if not os.path.isdir(args.dataset_path):
        LOGGER.error(f"Dataset path does not exist: {args.dataset_path}")
        sys.exit(1)

    config = get_backbone_config(args.backbone)
    timestamp = args.timestamp or datetime.now().strftime("%Y%m%d")
    study_name = args.study_name or f"{config.name}_layer_search_{timestamp}"

    results_dir = Path(args.results_dir) / study_name
    results_dir.mkdir(parents=True, exist_ok=True)

    storage_url = f"sqlite:///{results_dir / 'study.db'}"

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    LOGGER.info(f"Backbone: {config.name} (dim={config.pretrain_dim}, layers={config.num_layers})")
    LOGGER.info(f"Device: {device}")
    LOGGER.info(f"Categories: {args.categories}")
    LOGGER.info(f"N trials: {args.n_trials}")
    LOGGER.info(f"Results dir: {results_dir}")

    evaluator = PatchCoreEvaluator(
        config=config,
        dataset_path=args.dataset_path,
        device=device,
    )

    objective = PatchCoreObjective(
        evaluator=evaluator,
        categories=args.categories,
    )

    sampler = optuna.samplers.TPESampler(seed=args.seed)

    try:
        study = optuna.load_study(
            study_name=study_name,
            storage=storage_url,
            sampler=sampler,
        )
        LOGGER.info(f"Resuming existing study '{study_name}'")
    except KeyError:
        study = optuna.create_study(
            direction="maximize",
            study_name=study_name,
            storage=storage_url,
            sampler=sampler,
        )
        LOGGER.info(f"Created new study '{study_name}'")

    start_time = time.time()

    study.optimize(
        objective,
        n_trials=args.n_trials,
        gc_after_trial=True,
        show_progress_bar=True,
    )

    elapsed = time.time() - start_time
    LOGGER.info(f"Optimization complete in {elapsed:.1f}s ({elapsed / 60:.1f}min)")

    best = study.best_trial
    LOGGER.info(f"Best trial: #{best.number}")
    LOGGER.info(f"  Mean F1: {best.value:.4f}")
    LOGGER.info(f"  Params: {best.params}")

    export_results(study, results_dir, args.categories)

    LOGGER.info(f"All results saved to {results_dir}")


if __name__ == "__main__":
    main()
