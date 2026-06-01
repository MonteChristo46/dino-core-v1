#!/usr/bin/env python
"""Merge all study.db files from results/optuna/ into a single all_studies.db using Optuna's API."""
import argparse
import logging
import sys
from pathlib import Path

import optuna

LOGGER = logging.getLogger(__name__)


def merge_studies(optuna_dir: Path, output_db: Path) -> int:
    study_dbs = sorted(optuna_dir.rglob("study.db"))
    study_dbs = [db for db in study_dbs if db != output_db]

    if not study_dbs:
        LOGGER.info("No study.db files found in %s", optuna_dir)
        return 0

    if output_db.exists():
        LOGGER.info("Removing existing %s", output_db)
        output_db.unlink()

    storage_url = f"sqlite:///{output_db}"

    merged_count = 0
    for db_path in study_dbs:
        rel = db_path.relative_to(optuna_dir)
        src_url = f"sqlite:///{db_path}"

        try:
            src_study_names = optuna.get_all_study_names(src_url)
        except Exception as e:
            LOGGER.warning("Skipping %s: %s", rel, e)
            continue

        for study_name in src_study_names:
            try:
                src_study = optuna.load_study(study_name=study_name, storage=src_url)
            except Exception as e:
                LOGGER.warning("Could not load study '%s' from %s: %s", study_name, rel, e)
                continue

            LOGGER.info("Copying study '%s' from %s", study_name, rel)

            dst_study = optuna.create_study(
                study_name=study_name,
                storage=storage_url,
                sampler=src_study.sampler,
                direction=src_study.direction,
            )

            for trial in src_study.trials:
                dst_study.add_trial(
                    optuna.trial.create_trial(
                        params=trial.params,
                        distributions=trial.distributions,
                        value=trial.value,
                        state=trial.state,
                        user_attrs=trial.user_attrs,
                        system_attrs=trial.system_attrs,
                        intermediate_values=trial.intermediate_values,
                    )
                )

            merged_count += 1

    return merged_count


def main():
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--optuna-dir",
        type=str,
        default=None,
        help="Path to results/optuna directory",
    )
    args = parser.parse_args()

    script_dir = Path(__file__).resolve().parent.parent.parent
    optuna_dir = Path(args.optuna_dir) if args.optuna_dir else script_dir / "results" / "optuna"

    if not optuna_dir.is_dir():
        LOGGER.error("Directory not found: %s", optuna_dir)
        sys.exit(1)

    output_db = optuna_dir / "all_studies.db"
    count = merge_studies(optuna_dir, output_db)

    if count > 0:
        LOGGER.info("Merged %d studies into %s", count, output_db)
    else:
        LOGGER.info("No studies to merge.")


if __name__ == "__main__":
    main()
