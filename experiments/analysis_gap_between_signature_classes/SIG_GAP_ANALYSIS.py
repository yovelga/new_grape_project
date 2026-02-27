import os
import json
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.lines as mlines
from tqdm.auto import tqdm
from sklearn.feature_selection import mutual_info_classif
from sklearn.ensemble import RandomForestClassifier
from pathlib import Path
from dotenv import load_dotenv
import logging
_PROJECT_ROOT = Path(__file__).resolve().parents[2]

# ------------------------------------------------------------
# Logging setup
# ------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

# ------------------------------------------------------------
# File paths and setup
# ------------------------------------------------------------
load_dotenv()
CSV_PATH = str(_PROJECT_ROOT / r"experiments/analysis_gap_between_signature_classes/all_origin_signatures_results_2026-01-06.csv")
RESULT_DIR = Path(__file__).parent / "result"
RESULT_DIR.mkdir(exist_ok=True)
TXT_PATH = RESULT_DIR / "most_effective_wl_new.txt"
CSV_PATH_OUT = RESULT_DIR / "most_effective_wl_new.csv"
JSON_PATH = RESULT_DIR / "most_effective_wl_new.json"

# ------------------------------------------------------------
# Feature selection and save functions
# ------------------------------------------------------------
def save_txt(fisher_top30, mi_top30, rf_top30, counts):
    logger.info("Saving TXT summary...")
    with open(TXT_PATH, "w", encoding="utf-8") as f:
        for header, series in [
            ("FISHER_TOP_30", fisher_top30),
            ("MI_TOP_30", mi_top30),
            ("RF_TOP_30", rf_top30),
        ]:
            f.write(f"[{header}]\n")
            f.write(", ".join(map(str, series.index)) + "\n\n")
        for header, mask in [
            ("SELECTED_BY_3_METHODS", counts == 3),
            ("SELECTED_BY_2_METHODS", counts == 2),
            ("SELECTED_BY_1_METHOD", counts == 1),
        ]:
            f.write(f"[{header}]\n")
            idx = counts[mask].index
            f.write(", ".join(map(str, idx)) if len(idx) else "(none)")
            f.write("\n\n")
    logger.info(f"TXT summary saved to {TXT_PATH}")

def save_structured(fisher_top30, mi_top30, rf_top30):
    logger.info("Saving structured CSV and JSON...")
    wls = sorted(set(fisher_top30.index) | set(mi_top30.index) | set(rf_top30.index))
    df = pd.DataFrame(
        {
            "wavelength": wls,
            "selected_by_fisher": [int(w in fisher_top30.index) for w in wls],
            "selected_by_mi": [int(w in mi_top30.index) for w in wls],
            "selected_by_rf": [int(w in rf_top30.index) for w in wls],
        }
    )
    df["selection_count"] = df[
        ["selected_by_fisher", "selected_by_mi", "selected_by_rf"]
    ].sum(axis=1)
    df = df.sort_values("selection_count", ascending=False)
    df.to_csv(CSV_PATH_OUT, index=False)
    counts = df.set_index("wavelength")["selection_count"]
    json_dict = {
        "fisher_top30": fisher_top30.index.tolist(),
        "mi_top30": mi_top30.index.tolist(),
        "rf_top30": rf_top30.index.tolist(),
        "selected_by_3": counts[counts == 3].index.tolist(),
        "selected_by_2": counts[counts == 2].index.tolist(),
        "selected_by_1": counts[counts == 1].index.tolist(),
        "wavelengths_summary": df.to_dict(orient="records"),
    }
    with open(JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(json_dict, f, indent=2)
    logger.info(f"CSV saved to {CSV_PATH_OUT}")
    logger.info(f"JSON saved to {JSON_PATH}")
    return df

# ------------------------------------------------------------
# Main analysis pipeline
# ------------------------------------------------------------
def main():
    logger.info("Starting SIG_GAP_ANALYSIS pipeline...")
    # 1) Load data
    logger.info("Loading data...")
    df = pd.read_csv(CSV_PATH)
    feature_cols = [c for c in df.columns if c.endswith("nm")]
    X = df[feature_cols].copy()
    X.columns = [float(c.replace("nm", "")) for c in feature_cols]
    X = X.reindex(sorted(X.columns), axis=1)
    y = (
        df["label"]
        .replace({"REGULAR": 0, "CRACK": 1, "healthy": 0, "sick": 1})
        .astype(int)
    )
    logger.info(f"Loaded {len(df)} samples with {len(X.columns)} features.")

    # 2) Fisher score
    logger.info("Computing Fisher scores...")
    with tqdm(total=2, desc="Fisher Score", leave=False) as pbar:
        m1 = X[y == 1].mean()
        s1 = X[y == 1].std()
        pbar.update()
        m0 = X[y == 0].mean()
        s0 = X[y == 0].std()
        fisher = (m1 - m0) ** 2 / (s1**2 + s0**2)
        fisher_top30 = fisher.sort_values(ascending=False).head(30)
        pbar.update()
    logger.info("Fisher score calculation complete.")

    # 3) Mutual Information
    logger.info("Computing Mutual Information...")
    with tqdm(total=1, desc="Mutual Info", leave=False) as pbar:
        mi_vals = mutual_info_classif(X, y, discrete_features=False, random_state=42)
        mi_top30 = (
            pd.Series(mi_vals, index=X.columns).sort_values(ascending=False).head(30)
        )
        pbar.update()
    logger.info("Mutual Information calculation complete.")

    # 4) Random Forest importance
    logger.info("Training Random Forest for feature importances...")
    with tqdm(total=2, desc="Random Forest", leave=False) as pbar:
        rf = RandomForestClassifier(n_estimators=100, random_state=42)
        rf.fit(X, y)
        pbar.update()
        rf_top30 = (
            pd.Series(rf.feature_importances_, index=X.columns)
            .sort_values(ascending=False)
            .head(30)
        )
        pbar.update()
    logger.info("Random Forest feature importance calculation complete.")

    # 5) Count how many methods picked each wl
    logger.info("Counting wavelength selections across methods...")
    with tqdm(total=1, desc="Counting", leave=False) as pbar:
        combined = pd.concat([fisher_top30, mi_top30, rf_top30])
        counts = combined.index.value_counts().sort_values(ascending=False)
        pbar.update()
    logger.info("Counting complete.")

    # 6) Save the TXT summary
    save_txt(fisher_top30, mi_top30, rf_top30, counts)

    # 7) Save structured CSV + JSON
    sel_df = save_structured(fisher_top30, mi_top30, rf_top30)

    # 8) Plots
    logger.info("Generating plots...")
    with tqdm(total=1, desc="Plotting", leave=False) as pbar:
        plot_all(X, y, sel_df)
        pbar.update()
    logger.info("All plots saved.")

    logger.info(f"✓ All outputs saved in: {RESULT_DIR}")

# ------------------------------------------------------------
# Plotting functions
# ------------------------------------------------------------
def plot_all(X, y, sel_df):
    # Prepare means and stds
    mean_reg, std_reg = X[y == 0].mean(), X[y == 0].std()
    mean_crk, std_crk = X[y == 1].mean(), X[y == 1].std()

    # Percent of samples within one STD
    within_1std_reg = (
        ((X[y == 0] >= (mean_reg - std_reg)) & (X[y == 0] <= (mean_reg + std_reg)))
        .mean()
        .mean()
    )
    within_1std_crk = (
        ((X[y == 1] >= (mean_crk - std_crk)) & (X[y == 1] <= (mean_crk + std_crk)))
        .mean()
        .mean()
    )
    within_1std_reg_pct = within_1std_reg * 100
    within_1std_crk_pct = within_1std_crk * 100
    annotation_text = (
        f"{within_1std_reg_pct:.1f}% Regular\n"
        f"{within_1std_crk_pct:.1f}% Crack\n"
        "within ±1 STD"
    )

    # Selected wavelengths
    sel_wl = sel_df["wavelength"]
    sel_cnt = sel_df["selection_count"]

    # --- Plot 1: Mean ± STD ---
    fig1, ax1 = plt.subplots(figsize=(14, 6))
    ax1.plot(X.columns, mean_reg, color="#007ACC", label="Regular Mean")
    ax1.fill_between(
        X.columns, mean_reg - std_reg, mean_reg + std_reg, alpha=0.25, color="#007ACC"
    )
    ax1.plot(X.columns, mean_crk, color="#D55E00", label="Crack Mean")
    ax1.fill_between(
        X.columns, mean_crk - std_crk, mean_crk + std_crk, alpha=0.25, color="#D55E00"
    )
    ax1.text(
        0.98,
        0.90,
        annotation_text,
        transform=ax1.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )
    ax1.set(
        title="Average ± STD Spectral Signatures",
        xlabel="Wavelength (nm)",
        ylabel="Reflectance",
    )
    ax1.grid(True)
    ax1.legend()
    fig1.tight_layout()
    fig1.savefig(RESULT_DIR / "avg_std_signatures.png", dpi=300)

    # --- Plot 2: Means + Selected Wavelengths ---
    fig2, ax2 = plt.subplots(figsize=(14, 6))
    ax2.plot(X.columns, mean_reg, color="blue", label="Healthy (mean)")
    ax2.plot(X.columns, mean_crk, color="red", label="Sick (mean)")
    ax2.vlines(
        sel_wl[sel_cnt == 3],
        ymin=0,
        ymax=1,
        color="black",
        lw=0.8,
        alpha=0.9,
        label="Selected by 3 methods",
    )
    ax2.vlines(
        sel_wl[sel_cnt == 2],
        ymin=0,
        ymax=1,
        color="purple",
        lw=0.8,
        alpha=0.6,
        label="Selected by 2 methods",
    )
    ax2.text(
        0.98,
        0.90,
        annotation_text,
        transform=ax2.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )
    ax2.set(
        title="Average Spectral Signatures: Healthy vs Sick\n(Top discriminative wavelengths)",
        xlabel="Wavelength (nm)",
        ylabel="Reflectance (0–1)",
    )
    ax2.set_ylim(0, 1)
    ax2.grid(True)
    ax2.legend(loc="upper left")
    fig2.tight_layout()
    fig2.savefig(RESULT_DIR / "avg_with_selected_wl.png", dpi=300)

    # --- Plot 3: Combined (side-by-side) ---
    fig3, axs = plt.subplots(1, 2, figsize=(18, 6), sharey=True)
    # Left – Mean ± STD
    axs[0].plot(X.columns, mean_reg, color="#007ACC", label="Regular Mean")
    axs[0].fill_between(
        X.columns, mean_reg - std_reg, mean_reg + std_reg, alpha=0.25, color="#007ACC"
    )
    axs[0].plot(X.columns, mean_crk, color="#D55E00", label="Crack Mean")
    axs[0].fill_between(
        X.columns, mean_crk - std_crk, mean_crk + std_crk, alpha=0.25, color="#D55E00"
    )
    axs[0].set(title="Average ± STD", xlabel="Wavelength (nm)", ylabel="Reflectance")
    axs[0].grid(True)
    axs[0].legend()
    # Right – Means + Selected Wavelengths
    axs[1].plot(X.columns, mean_reg, color="blue", label="Healthy (mean)")
    axs[1].plot(X.columns, mean_crk, color="red", label="Sick (mean)")
    axs[1].vlines(
        sel_wl[sel_cnt == 3],
        ymin=0,
        ymax=1,
        color="black",
        lw=0.8,
        alpha=0.9,
        label="Selected by 3 methods",
    )
    axs[1].vlines(
        sel_wl[sel_cnt == 2],
        ymin=0,
        ymax=1,
        color="purple",
        lw=0.8,
        alpha=0.6,
        label="Selected by 2 methods",
    )
    axs[1].set(title="Means + Selected Wavelengths", xlabel="Wavelength (nm)")
    axs[1].grid(True)
    axs[1].legend(loc="upper left")
    for ax in axs:
        ax.text(
            0.98,
            0.90,
            annotation_text,
            transform=ax.transAxes,
            fontsize=12,
            verticalalignment="top",
            horizontalalignment="right",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
        )
    fig3.suptitle("Combined View", fontsize=16)
    fig3.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig3.savefig(RESULT_DIR / "combined_signatures.png", dpi=300)

    # --- Plot 4: Single Combined Plot ---
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(X.columns, mean_reg, color="#007ACC", lw=2, label="Regular Mean")
    ax.fill_between(
        X.columns, mean_reg - std_reg, mean_reg + std_reg, color="#007ACC", alpha=0.25
    )
    ax.plot(X.columns, mean_crk, color="#D55E00", lw=2, label="Crack Mean")
    ax.fill_between(
        X.columns, mean_crk - std_crk, mean_crk + std_crk, color="#D55E00", alpha=0.25
    )
    ax.vlines(
        sel_wl[sel_cnt == 3],
        ymin=0,
        ymax=1,
        color="black",
        lw=0.8,
        alpha=0.9,
        label="Selected by 3 methods",
    )
    ax.vlines(
        sel_wl[sel_cnt == 2],
        ymin=0,
        ymax=1,
        color="purple",
        lw=0.8,
        alpha=0.6,
        label="Selected by 2 methods",
    )
    ax.set(
        title="Average ± STD with Discriminative Wavelengths",
        xlabel="Wavelength (nm)",
        ylabel="Reflectance (0–1)",
    )
    ax.set_ylim(0, 1)
    ax.grid(True)
    ax.text(
        0.98,
        0.90,
        annotation_text,
        transform=ax.transAxes,
        fontsize=12,
        verticalalignment="top",
        horizontalalignment="right",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", alpha=0.8),
    )
    handles, labels = ax.get_legend_handles_labels()
    uniq = dict(zip(labels, handles))
    ax.legend(uniq.values(), uniq.keys(), loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULT_DIR / "combined_single_plot.png", dpi=300)
    plt.show()

if __name__ == "__main__":
    main()
