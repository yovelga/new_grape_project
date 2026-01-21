"""
Optuna optimization script for HSI crack detection post-processing parameters.

This script replaces the brute-force grid search with an Optuna-driven
Bayesian optimization over post-processing hyperparameters.

Usage (PowerShell):
    python optuna_blob_patch_opt.py --results_csv path/to/late_detection_with_prob_maps.csv --n_trials 150 --out_prefix optuna_blob

Outputs:
- <out_prefix>_best_params.json
- <out_prefix>_trials.csv

The objective maximizes F2 score (recall prioritized).
"""
import argparse
import json
import os
import logging

import numpy as np
import pandas as pd
import optuna

# Import helper functions from existing grid-search module
from grid_search_blob_patch import (
    load_probability_maps_from_results,
    evaluate_single_combination,
    GridSearchConfig
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("optuna.blobpatch")


def compute_f2(precision: float, recall: float) -> float:
    """Compute F2 score given precision and recall."""
    if precision is None or recall is None:
        return float("nan")
    if precision + recall == 0:
        return 0.0
    beta2 = 4.0
    return (1 + beta2) * (precision * recall) / (beta2 * precision + recall)


def make_objective(prob_maps_dict, labels_dict):
    """Return an Optuna objective function bound to loaded probability maps.

    The objective suggests hyperparameters, calls evaluate_single_combination and
    returns the F2 score (to maximize).
    """
    def objective(trial: optuna.Trial) -> float:
        # Suggest hyperparameters per requirements
        prob_thr = trial.suggest_float("prob_thr", 0.80, 0.99)

        # Morph size: allow 0 or odd sizes 1..15
        morph_size = trial.suggest_categorical("morph_size", [0] + list(range(1, 16, 2)))

        min_blob_size = trial.suggest_int("min_blob_size", 10, 1000)

        # Circularity: enable/disable via categorical, if enabled sample a min value and set max to 1.0
        use_circ = trial.suggest_categorical("use_circularity", [False, True])
        circularity_min = None
        circularity_max = None
        if use_circ:
            circularity_min = trial.suggest_float("circularity_min", 0.0, 0.5)
            circularity_max = trial.suggest_float("circularity_max", max(circularity_min, 0.6), 1.0)

        # Aspect ratio: optionally enabled
        use_ar = trial.suggest_categorical("use_aspect_ratio", [False, True])
        aspect_ratio_min = None
        aspect_ratio_limit = None
        if use_ar:
            aspect_ratio_min = trial.suggest_float("aspect_ratio_min", 1.0, 3.0)
            aspect_ratio_limit = trial.suggest_float("aspect_ratio_limit", max(aspect_ratio_min, 2.0), 10.0)

        # Solidity: optionally enabled
        use_sol = trial.suggest_categorical("use_solidity", [False, True])
        solidity_min = None
        solidity_max = None
        if use_sol:
            solidity_min = trial.suggest_float("solidity_min", 0.0, 0.8)
            solidity_max = trial.suggest_float("solidity_max", max(solidity_min, 0.8), 1.0)

        patch_size = trial.suggest_categorical("patch_size", [32, 64, 128])
        patch_pixel_ratio = trial.suggest_float("patch_pixel_ratio", 0.01, 0.30)
        global_threshold = trial.suggest_float("global_threshold", 0.001, 0.1)

        # Evaluate the chosen combination
        try:
            metrics = evaluate_single_combination(
                prob_maps_dict,
                labels_dict,
                prob_thr=prob_thr,
                min_blob_size=min_blob_size,
                circularity_min=circularity_min,
                circularity_max=circularity_max,
                aspect_ratio_min=aspect_ratio_min,
                aspect_ratio_limit=aspect_ratio_limit,
                solidity_min=solidity_min,
                solidity_max=solidity_max,
                patch_size=int(patch_size),
                patch_pixel_ratio=float(patch_pixel_ratio),
                global_threshold=float(global_threshold),
                morph_size=int(morph_size)
            )
        except Exception as e:
            logger.exception("Evaluation failed for trial: %s", e)
            # Return a very low score so Optuna avoids this region
            return 0.0

        precision = metrics.get("precision", float("nan"))
        recall = metrics.get("recall", float("nan"))
        f2 = compute_f2(precision, recall)

        # Report intermediate values to Optuna for logging
        trial.set_user_attr("precision", float(precision) if not pd.isna(precision) else None)
        trial.set_user_attr("recall", float(recall) if not pd.isna(recall) else None)
        trial.set_user_attr("f2", float(f2) if not pd.isna(f2) else None)

        # Optuna maximizes the returned value
        return float(f2)

    return objective


def run_study(results_csv: str, n_trials: int = 150, out_prefix: str = "optuna_blob"):
    # Load results dataframe and probability maps (dev set)
    if not os.path.exists(results_csv):
        raise FileNotFoundError(results_csv)

    results_df = pd.read_csv(results_csv)
    prob_maps_dict, labels_dict = load_probability_maps_from_results(results_df, row_filter=1)

    logger.info("Loaded %d probability maps for optimization", len(prob_maps_dict))

    study = optuna.create_study(direction="maximize")

    objective = make_objective(prob_maps_dict, labels_dict)

    logger.info("Starting Optuna study with %d trials", n_trials)
    study.optimize(objective, n_trials=n_trials)

    # Save best params
    best = study.best_trial
    best_params = best.params
    out_json = f"{out_prefix}_best_params.json"
    with open(out_json, "w") as f:
        json.dump({"best_value": best.value, "best_params": best_params}, f, indent=2)
    logger.info("Saved best params to %s", out_json)

    # Save trials dataframe
    df = study.trials_dataframe()
    out_csv = f"{out_prefix}_trials.csv"
    df.to_csv(out_csv, index=False)
    logger.info("Saved trials dataframe to %s", out_csv)

    return study


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Optuna optimization for HSI blob+patch detection post-processing")
    parser.add_argument("--results_csv", type=str, required=True,
                        help="CSV produced by prepare_and_run_inference (contains prob_map_path column)")
    parser.add_argument("--n_trials", type=int, default=150, help="Number of Optuna trials (default:150)")
    parser.add_argument("--out_prefix", type=str, default="optuna_blob", help="Output prefix for results files")

    args = parser.parse_args()
    run_study(args.results_csv, n_trials=args.n_trials, out_prefix=args.out_prefix)
