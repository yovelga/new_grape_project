"""
run_bfs_prauc_stability_5runs.py

Run the Backward Feature Selection (BFS) experiment 5 times with different
model seeds, using PR-AUC as the SOLE objective.  This quantifies how stable
the selected wavelength subset is when ONLY algorithm randomness varies.

Key design (v2.0):
  - The data split is FIXED across all runs (SPLIT_SEED), so every run sees
    exactly the same train/test partition.
  - Only the model / BFS seed (MODEL_SEEDS) varies, isolating algorithm
    instability from dataset variability.
  - A "split signature" guardrail asserts that train/test class counts and
    grape-ID counts are identical across runs.

Folder layout produced
======================
experiments/feature_selection/stability_bfs_prauc_fixedsplit/
├── master_index.csv                         # one row per run
├── prauc_split123_model1_2026-02-08_12-00-00/
│   ├── run_manifest.json
│   ├── split_manifest.json
│   ├── split_signature.json
│   ├── bfs_log_prauc.csv
│   ├── best_features_prauc.json
│   ├── selected_wavelengths_at30.csv   ★ TRUE set at n=30
│   ├── selected_wavelengths_at11.csv   ★ TRUE set at n=11 (if reachable)
│   └── selected_wavelengths_best.csv   ★ set at best_n_features
├── prauc_split123_model2_2026-02-08_12-05-00/
│   └── …
└── …

Reuses the data-loading, splitting, and BFS logic from
``run_bfs_80_20_grapeid.py`` via direct import (no code duplication).

Author : Stability Experiment Pipeline
Date   : February 2026
"""

import sys
import json
import time
import random
import argparse
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime, timezone
from dataclasses import asdict

# ---------------------------------------------------------------------------
# Project root (same convention as the original script)
# ---------------------------------------------------------------------------
_PROJECT_ROOT = Path(__file__).resolve().parents[5]
if str(_PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(_PROJECT_ROOT))

# ---------------------------------------------------------------------------
# Import BFS helpers from the original script
# ---------------------------------------------------------------------------
from src.models.classification.pixel_level.feature_selection.run_bfs_80_20_grapeid import (
    load_and_prepare_data,
    create_80_20_split,
    run_bfs,
    CSV_PATH_MULTICLASS,
)


# ====================== CONFIGURATION (edit here) =========================
CONFIG = {
    # ---- SPLIT MODE ----
    # "fixed"   = FIXED_SPLIT_SEED (42) for all runs -> isolates model randomness
    # "varying" = split_seed = model_seed for each run -> tests split sensitivity
    "SPLIT_MODE": "fixed",             # <- "fixed" for Exp A, "varying" for Exp B

    # ---- FIXED_SPLIT_SEED (used when SPLIT_MODE="fixed") ----
    "FIXED_SPLIT_SEED": 42,

    # ---- MODEL_SEEDS (one BFS run per seed) ----
    "MODEL_SEEDS": [4, 5],  # Running only missing seeds

    # ---- data ----
    "csv_path": str(CSV_PATH_MULTICLASS),
    "wl_min": 450,
    "wl_max": 925,

    # ---- output ----
    "experiments_base": str(
        _PROJECT_ROOT / "experiments" / "feature_selection" / "stability_bfs_prauc"
    ),

    # ---- BFS settings ----
    "objective": "prauc",              # PR-AUC only
    "min_features": 1,                # stop BFS at 1 (thesis-relevant region)
    "use_gpu": True,

    # ---- stability parameter ----
    "top_k_for_stability": 30,         # evaluate stability on top-30 features

    # ---- target n_features to extract from BFS path ----
    # These are the exact BFS steps we want to capture (not slicing best_info)
    "target_n_features": [30, 11],

    # ---- script version tag (git-like) ----
    "script_version": "run_bfs_prauc_stability_5runs@v2.4.0",
}
# ===========================================================================


def _build_run_manifest(
    split_seed: int,
    model_seed: int,
    split_mode: str,
    timestamp_str: str,
    output_dir: Path,
    best_info: dict,
    features_paths: dict,
) -> dict:
    """Create a small JSON manifest that uniquely identifies one BFS run."""
    return {
        "split_mode": split_mode,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "timestamp": timestamp_str,
        "objective": CONFIG["objective"],
        "wl_range": [CONFIG["wl_min"], CONFIG["wl_max"]],
        "csv_path": CONFIG["csv_path"],
        "script_version": CONFIG["script_version"],
        "output_dir": str(output_dir),
        "top_k_for_stability": CONFIG["top_k_for_stability"],
        "target_n_features": CONFIG["target_n_features"],
        "min_features": CONFIG["min_features"],
        "use_gpu": CONFIG["use_gpu"],
        "best_score": best_info["best_score"],
        "best_n_features": best_info["best_n_features"],
        # Paths to feature set files
        "features_at_30_path": features_paths.get(30),
        "features_at_11_path": features_paths.get(11),
        "features_best_path": features_paths.get("best"),
    }


def _parse_wl_nm(wl_str: str) -> float:
    """Parse wavelength string like '723.5nm' or '723.5' to float."""
    return float(str(wl_str).strip().lower().replace("nm", ""))


def _reconstruct_features_at_n(
    bfs_log_path: Path,
    initial_features: list,
    target_n: int,
) -> list:
    """Reconstruct the exact feature set when BFS had `target_n` features.

    Algorithm:
      1. Start with a copy of the initial full feature list.
      2. Iterate through BFS log rows in order.
      3. For each row where 'removed_feature' is non-empty, remove that
         feature from the current set.
      4. After processing each row, check if len(current_set) == target_n.
         If so, return the current set (sorted by wavelength for readability).
      5. If target_n is never reached (e.g., BFS stopped before n=target_n),
         return None.

    Returns:
        Sorted list of wavelength strings at n=target_n, or None if not reached.
    """
    log_df = pd.read_csv(bfs_log_path)
    current_features = list(initial_features)  # mutable copy

    for _, row in log_df.iterrows():
        n_feat = int(row["n_features"])
        removed = row.get("removed_feature", "")

        # Check BEFORE removing (this row represents the state at n_features)
        if n_feat == target_n:
            # Sort by wavelength for readability
            current_features_sorted = sorted(
                current_features, key=lambda w: _parse_wl_nm(w)
            )
            return current_features_sorted

        # Remove the feature that was eliminated after this step
        if pd.notna(removed) and removed != "" and removed in current_features:
            current_features.remove(removed)

    return None  # target_n was never reached


def _save_features_at_n(
    features: list,
    n: int,
    output_dir: Path,
    label: str = None,
) -> Path:
    """Save a feature set CSV with rank and wavelength columns."""
    if label is None:
        label = str(n)
    df = pd.DataFrame({
        "rank": range(1, len(features) + 1),
        "wavelength": features,
    })
    path = output_dir / f"selected_wavelengths_at{label}.csv"
    df.to_csv(path, index=False)
    return path


def _save_best_features(
    best_info: dict,
    output_dir: Path,
) -> Path:
    """Save the best feature set (at best_n_features) to CSV."""
    features = best_info["selected_features"]
    # Sort by wavelength for consistency
    features_sorted = sorted(features, key=lambda w: _parse_wl_nm(w))
    df = pd.DataFrame({
        "rank": range(1, len(features_sorted) + 1),
        "wavelength": features_sorted,
    })
    path = output_dir / "selected_wavelengths_best.csv"
    df.to_csv(path, index=False)
    return path


def _compute_split_signature(
    y_train: np.ndarray,
    y_test: np.ndarray,
    manifest,
) -> dict:
    """Compute a compact signature of the train/test split.

    Used as a guardrail to assert the split is identical across runs.
    """
    from collections import Counter
    train_counts = dict(sorted(Counter(y_train.tolist()).items()))
    test_counts = dict(sorted(Counter(y_test.tolist()).items()))
    sig = {
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
        "train_class_counts": {str(k): v for k, v in train_counts.items()},
        "test_class_counts": {str(k): v for k, v in test_counts.items()},
        "n_train_grape_ids": len(manifest.train_grape_ids),
        "n_test_grape_ids": len(manifest.test_grape_ids),
        "train_grape_ids_sorted": sorted(manifest.train_grape_ids),
        "test_grape_ids_sorted": sorted(manifest.test_grape_ids),
    }
    return sig


def run_single_seed(
    split_seed: int,
    model_seed: int,
    split_mode: str,
    stability_root: Path,
) -> tuple:
    """Execute one full BFS-PR-AUC run.

    The data split uses *split_seed* while BFS/model training uses *model_seed*.
    - In 'fixed' mode: split_seed == FIXED_SPLIT_SEED for all runs
    - In 'varying' mode: split_seed == model_seed (deterministic per seed)

    Returns (row_dict, split_signature).
    """
    # ---- SAFETY ASSERTIONS ----
    if split_mode == "fixed":
        assert split_seed == CONFIG["FIXED_SPLIT_SEED"], \
            f"[SAFETY] fixed mode but split_seed={split_seed} != FIXED_SPLIT_SEED={CONFIG['FIXED_SPLIT_SEED']}"
    elif split_mode == "varying":
        assert split_seed == model_seed, \
            f"[SAFETY] varying mode but split_seed={split_seed} != model_seed={model_seed}"
    else:
        raise ValueError(f"Unknown split_mode: {split_mode}")

    timestamp_str = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    run_dir = stability_root / f"prauc_{split_mode}_split{split_seed}_model{model_seed}_{timestamp_str}"
    run_dir.mkdir(parents=True, exist_ok=True)

    print(f"\n{'#' * 70}")
    print(f"# MODE={split_mode} | SPLIT_SEED={split_seed} | MODEL_SEED={model_seed}")
    print(f"# output -> {run_dir}")
    print(f"{'#' * 70}")

    # ---- load & split (deterministic via split_seed) ----
    random.seed(split_seed)
    np.random.seed(split_seed)

    df, feature_names, class_mapping, crack_idx = load_and_prepare_data(
        csv_path=Path(CONFIG["csv_path"]),
        wl_min=CONFIG["wl_min"],
        wl_max=CONFIG["wl_max"],
        seed=split_seed,
    )

    X_train, X_test, y_train, y_test, manifest = create_80_20_split(
        df=df,
        feature_names=feature_names,
        class_mapping=class_mapping,
        seed=split_seed,
    )
    n_classes = len(class_mapping)

    # save split manifest
    manifest_path = run_dir / "split_manifest.json"
    with open(manifest_path, "w") as f:
        json.dump(asdict(manifest), f, indent=2, default=str)
    print(f"[SAVE] {manifest_path}")

    # ---- split signature (guardrail) ----
    split_sig = _compute_split_signature(y_train, y_test, manifest)
    sig_path = run_dir / "split_signature.json"
    with open(sig_path, "w") as f:
        json.dump(split_sig, f, indent=2)
    print(f"[SAVE] {sig_path}")
    print(f"[GUARDRAIL] Split signature: n_train={split_sig['n_train']}, "
          f"n_test={split_sig['n_test']}, "
          f"train_grape_ids={split_sig['n_train_grape_ids']}, "
          f"test_grape_ids={split_sig['n_test_grape_ids']}")

    # ---- BFS (PR-AUC only) – uses VARYING model_seed ----
    random.seed(model_seed)
    np.random.seed(model_seed)

    _log_df, best_info = run_bfs(
        X_train, X_test, y_train, y_test,
        feature_names, crack_idx, n_classes,
        objective=CONFIG["objective"],
        min_features=CONFIG["min_features"],
        seed=model_seed,
        use_gpu=CONFIG["use_gpu"],
        output_dir=run_dir,
    )

    # save best features JSON
    best_path = run_dir / "best_features_prauc.json"
    with open(best_path, "w") as f:
        json.dump(best_info, f, indent=2)
    print(f"[SAVE] {best_path}")

    # ---- Reconstruct TRUE feature sets at target n_features ----
    bfs_log_path = run_dir / "bfs_log_prauc.csv"
    features_paths = {}  # n -> path (or "best" -> path)

    for target_n in CONFIG["target_n_features"]:
        features_at_n = _reconstruct_features_at_n(
            bfs_log_path, feature_names, target_n
        )
        if features_at_n is not None:
            path_at_n = _save_features_at_n(features_at_n, target_n, run_dir)
            features_paths[target_n] = str(path_at_n)
            print(f"[SAVE] {path_at_n}")

            # Validation print for n=30
            if target_n == 30:
                wls_nm = [_parse_wl_nm(w) for w in features_at_n]
                print(f"[VALIDATE] FEATURES_AT_30: count={len(features_at_n)}, "
                      f"min={min(wls_nm):.2f}nm, max={max(wls_nm):.2f}nm")
        else:
            features_paths[target_n] = None
            print(f"[WARN] n_features={target_n} was not reached during BFS (min_features={CONFIG['min_features']})")

    # Save best feature set (at best_n_features)
    best_features_path = _save_best_features(best_info, run_dir)
    features_paths["best"] = str(best_features_path)
    print(f"[SAVE] {best_features_path}")

    # save run manifest (includes split_mode, BOTH seeds, and feature paths)
    run_manifest = _build_run_manifest(
        split_seed, model_seed, split_mode, timestamp_str, run_dir, best_info, features_paths,
    )
    rm_path = run_dir / "run_manifest.json"
    with open(rm_path, "w") as f:
        json.dump(run_manifest, f, indent=2)
    print(f"[SAVE] {rm_path}")

    # row for master index
    row = {
        "run_dir": str(run_dir),
        "split_mode": split_mode,
        "split_seed": split_seed,
        "model_seed": model_seed,
        "best_score": best_info["best_score"],
        "best_n_features": best_info["best_n_features"],
        "timestamp": timestamp_str,
        "bfs_log_path": str(bfs_log_path),
        "best_features_path": str(best_path),
        "split_manifest_path": str(manifest_path),
        "run_manifest_path": str(rm_path),
        "features_at_30_path": features_paths.get(30),
        "features_at_11_path": features_paths.get(11),
        "features_best_path": features_paths.get("best"),
    }
    return row, split_sig


# ========================== MAIN ==========================================

def main():
    split_mode = CONFIG["SPLIT_MODE"]
    fixed_split_seed = CONFIG["FIXED_SPLIT_SEED"]
    model_seeds = CONFIG["MODEL_SEEDS"]

    # Validate split_mode
    assert split_mode in ("fixed", "varying"), f"Invalid SPLIT_MODE: {split_mode}"

    # Output folder includes split_mode for clarity
    base_path = CONFIG["experiments_base"]
    stability_root = Path(base_path) / f"{split_mode}split"
    stability_root.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"BFS PR-AUC STABILITY EXPERIMENT - {split_mode.upper()} SPLIT MODE")
    print("=" * 70)
    if split_mode == "fixed":
        print(f"Split mode   : FIXED (split_seed={fixed_split_seed} for ALL runs)")
    else:
        print(f"Split mode   : VARYING (split_seed = model_seed, deterministic per seed)")
    print(f"Model seeds  : {model_seeds}")
    print(f"Objective    : {CONFIG['objective']}")
    print(f"Min features : {CONFIG['min_features']}")
    print(f"Top-K        : {CONFIG['top_k_for_stability']}")
    print(f"Output root  : {stability_root}")
    print("=" * 70)

    t0 = time.time()
    index_rows = []
    reference_sig = None  # guardrail: first run's split signature (only for fixed mode)

    for model_seed in model_seeds:
        # Determine split_seed based on mode
        if split_mode == "fixed":
            split_seed = fixed_split_seed
        else:  # "varying"
            split_seed = model_seed  # deterministic: same model_seed always gets same split

        row, split_sig = run_single_seed(split_seed, model_seed, split_mode, stability_root)
        index_rows.append(row)

        # ---- guardrail: assert split is identical across runs (only for fixed mode) ----
        if split_mode == "fixed":
            if reference_sig is None:
                reference_sig = split_sig
                print("[GUARDRAIL] Reference split signature captured from first run.")
            else:
                for key in ["n_train", "n_test", "train_class_counts",
                            "test_class_counts", "n_train_grape_ids",
                            "n_test_grape_ids", "train_grape_ids_sorted",
                            "test_grape_ids_sorted"]:
                    if split_sig[key] != reference_sig[key]:
                        raise RuntimeError(
                            f"[GUARDRAIL FAILED] Split signature mismatch on '{key}' "
                            f"for model_seed={model_seed}!\n"
                            f"  reference : {reference_sig[key]}\n"
                            f"  this run  : {split_sig[key]}"
                        )
                print(f"[GUARDRAIL] Split signature MATCHES reference (model_seed={model_seed}).")
        else:
            print(f"[INFO] Varying split mode - split_seed={split_seed} for this run.")

    # ---- master index ----
    master_df = pd.DataFrame(index_rows)
    master_path = stability_root / "master_index.csv"
    master_df.to_csv(master_path, index=False)
    print(f"\n[SAVE] Master index: {master_path}")

    elapsed = time.time() - t0
    print(f"\n{'=' * 70}")
    print(f"ALL {len(model_seeds)} RUNS COMPLETED in {elapsed / 60:.1f} min")
    if split_mode == "fixed":
        print(f"Split was FIXED (seed={fixed_split_seed}) - only model randomness varied.")
    else:
        print(f"Split VARIED per run (split_seed=model_seed) - tests split sensitivity.")
    print(f"{'=' * 70}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="BFS PR-AUC Stability Experiment"
    )
    parser.add_argument(
        "--split-mode",
        choices=["fixed", "varying"],
        default=None,
        help="Override SPLIT_MODE from CONFIG. Use 'fixed' or 'varying'."
    )
    args = parser.parse_args()

    # Override CONFIG if command line arg provided
    if args.split_mode:
        CONFIG["SPLIT_MODE"] = args.split_mode
        print(f"[CLI] SPLIT_MODE overridden to: {args.split_mode}")

    main()
