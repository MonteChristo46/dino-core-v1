import logging

import optuna

from patchcore_optuna.evaluator import PatchCoreEvaluator
from patchcore_optuna.search_spaces import (
    DEFAULT_CATEGORIES,
    NUM_NN_CHOICES,
    PATCH_SIZES,
    parse_layers,
)

LOGGER = logging.getLogger(__name__)


class PatchCoreObjective:
    def __init__(
        self,
        evaluator: PatchCoreEvaluator,
        categories: list[str] | None = None,
    ):
        self.evaluator = evaluator
        self.categories = categories or DEFAULT_CATEGORIES
        self.layer_combos = evaluator.config.layer_combos

    def __call__(self, trial: optuna.Trial) -> float:
        layers_str = trial.suggest_categorical("layers", self.layer_combos)
        layers = parse_layers(layers_str)
        patch_size = trial.suggest_categorical("patch_size", PATCH_SIZES)
        num_nn = trial.suggest_categorical("num_nn", NUM_NN_CHOICES)

        LOGGER.info(
            f"Trial {trial.number}: layers={layers}, "
            f"patch_size={patch_size}, num_nn={num_nn}"
        )

        result = self.evaluator.evaluate(
            layers=layers,
            patch_size=patch_size,
            num_nn=num_nn,
            categories=self.categories,
            seed=trial.suggest_int("seed", 0, 0),
        )

        trial.set_user_attr("layers", layers_str)
        trial.set_user_attr("patch_size", patch_size)
        trial.set_user_attr("num_nn", num_nn)
        trial.set_user_attr("mean_f1", result["mean_f1"])
        trial.set_user_attr("mean_auroc", result["mean_auroc"])
        trial.set_user_attr("per_category", result["per_category"])

        for cat_result in result["per_category"]:
            cat = cat_result["category"]
            trial.set_user_attr(f"{cat}_f1", cat_result["f1"])
            trial.set_user_attr(f"{cat}_auroc", cat_result["auroc"])

        LOGGER.info(
            f"Trial {trial.number} complete: mean_f1={result['mean_f1']:.4f}, "
            f"mean_auroc={result['mean_auroc']:.4f}"
        )

        return result["mean_f1"]
