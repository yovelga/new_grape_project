"""
Generate a before/after SNV normalization figure for the thesis.
4 classes × 2 columns (raw | SNV-normalized).
Each panel shows the mean spectrum ± 1 std shaded band,
computed over many sampled pixels of that class.
Wavelength range restricted to 450–925 nm.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
matplotlib.use('Agg')

# ── CONFIG ──
CSV_PATH = r"C:\Users\yovel\OneDrive\Desktop\Grape_Project\src\preprocessing\dataset_builder_grapes\detection\raw_exported_data\all_origin_signatures_results_multiclass_2026-01-16.csv"
OUT_PATH = r"C:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis\figures\literature\snv_normalization_effect.png"

CLASSES_TO_SHOW = ["CRACK", "REGULAR", "PLASTIC", "BRANCH"]
MAX_PIXELS_PER_CLASS = 500  # sample up to 500 pixels per class for mean/std

CLASS_COLORS = {
    "CRACK":   "#d62728",  # red
    "REGULAR": "#2ca02c",  # green
    "PLASTIC": "#ff7f0e",  # orange
    "BRANCH":  "#8c564b",  # brown
}

CLASS_LABELS = {
    "CRACK":   "Crack",
    "REGULAR": "Regular Berry",
    "PLASTIC": "Plastic",
    "BRANCH":  "Branch",
}

# ── SNV TRANSFORM (per-row) ──
def snv(spectra_2d):
    """SNV on each row: (x - row_mean) / row_std."""
    means = spectra_2d.mean(axis=1, keepdims=True)
    stds  = spectra_2d.std(axis=1, ddof=0, keepdims=True)
    stds[stds == 0] = 1.0
    return (spectra_2d - means) / stds

# ── LOAD DATA ──
print("Loading CSV in chunks to collect pixels per class...")
wl_cols_restricted = None
wavelengths = None
class_data = {cls: [] for cls in CLASSES_TO_SHOW}
class_done = {cls: False for cls in CLASSES_TO_SHOW}

for chunk in pd.read_csv(CSV_PATH, chunksize=50000):
    if wl_cols_restricted is None:
        wl_cols_all = [c for c in chunk.columns if c.endswith("nm")]
        wavelengths_all = np.array([float(c.replace("nm", "")) for c in wl_cols_all])
        mask = (wavelengths_all >= 450) & (wavelengths_all <= 925)
        wl_cols_restricted = [c for c, m in zip(wl_cols_all, mask) if m]
        wavelengths = wavelengths_all[mask]

    for cls in CLASSES_TO_SHOW:
        if class_done[cls]:
            continue
        rows = chunk.loc[chunk["label"] == cls, wl_cols_restricted]
        if len(rows) > 0:
            need = MAX_PIXELS_PER_CLASS - len(class_data[cls])
            class_data[cls].append(rows.iloc[:need].values.astype(float))
            total = sum(a.shape[0] for a in class_data[cls])
            if total >= MAX_PIXELS_PER_CLASS:
                class_done[cls] = True

    if all(class_done.values()):
        break

# Concatenate collected arrays
for cls in CLASSES_TO_SHOW:
    class_data[cls] = np.concatenate(class_data[cls], axis=0)
    print(f"  {cls}: {class_data[cls].shape[0]} pixels")

# ── COMPUTE STATS ──
raw_stats = {}
snv_stats = {}
for cls in CLASSES_TO_SHOW:
    arr = class_data[cls]
    raw_stats[cls] = (arr.mean(axis=0), arr.std(axis=0, ddof=0))
    arr_snv = snv(arr)
    snv_stats[cls] = (arr_snv.mean(axis=0), arr_snv.std(axis=0, ddof=0))

# ── PLOT: 4 rows × 2 cols ──
fig, axes = plt.subplots(4, 2, figsize=(13, 14), sharex=True)

for i, cls in enumerate(CLASSES_TO_SHOW):
    color = CLASS_COLORS[cls]
    label = CLASS_LABELS[cls]

    # --- Left: raw ---
    ax_raw = axes[i, 0]
    mean_r, std_r = raw_stats[cls]
    ax_raw.plot(wavelengths, mean_r, color=color, linewidth=1.5, label="Mean")
    ax_raw.fill_between(wavelengths, mean_r - std_r, mean_r + std_r,
                        color=color, alpha=0.25, label="± 1 SD")
    ax_raw.set_ylabel("Reflectance", fontsize=10)
    ax_raw.set_xlim(450, 925)
    ax_raw.grid(True, alpha=0.3)
    ax_raw.tick_params(labelsize=9)
    ax_raw.legend(loc="upper right", fontsize=8, framealpha=0.7)
    if i == 0:
        ax_raw.set_title("Raw Reflectance", fontsize=12, fontweight="bold")
    # Row label on the left
    ax_raw.annotate(label, xy=(0, 0.5), xytext=(-55, 0),
                    textcoords="offset points", xycoords="axes fraction",
                    fontsize=11, fontweight="bold", ha="center", va="center",
                    rotation=90, color=color)

    # --- Right: SNV ---
    ax_snv = axes[i, 1]
    mean_s, std_s = snv_stats[cls]
    ax_snv.plot(wavelengths, mean_s, color=color, linewidth=1.5, label="Mean")
    ax_snv.fill_between(wavelengths, mean_s - std_s, mean_s + std_s,
                        color=color, alpha=0.25, label="± 1 SD")
    ax_snv.set_ylabel("SNV Value", fontsize=10)
    ax_snv.set_xlim(450, 925)
    ax_snv.grid(True, alpha=0.3)
    ax_snv.tick_params(labelsize=9)
    ax_snv.legend(loc="upper right", fontsize=8, framealpha=0.7)
    if i == 0:
        ax_snv.set_title("After SNV Normalization", fontsize=12, fontweight="bold")

# Bottom x-labels
axes[-1, 0].set_xlabel("Wavelength (nm)", fontsize=11)
axes[-1, 1].set_xlabel("Wavelength (nm)", fontsize=11)

fig.suptitle("Effect of SNV Normalization on Hyperspectral Signatures",
             fontsize=14, fontweight="bold", y=0.995)
plt.tight_layout(rect=[0.04, 0, 1, 0.97], h_pad=1.5)
plt.savefig(OUT_PATH, dpi=300, bbox_inches="tight", facecolor="white")
print(f"\nFigure saved to: {OUT_PATH}")
plt.close()
