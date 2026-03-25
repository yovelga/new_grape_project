"""
Generate a professional consensus wavelength bins figure for the thesis.
Shows 5/5 and 4/5 seed consensus bins with five scientifically
conservative spectral interpretation overlays.
"""
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np

# ── Data ──────────────────────────────────────────────────────────────
unanimous = [450, 535, 545, 550, 570, 580, 635, 730, 740, 755, 760, 780, 915, 920]
highly_stable = [515, 605, 735, 770, 840, 900]

# ── Interpretation regions (easy to modify) ──────────────────────────
# Each entry: (start_nm, end_nm, colour, label, label_colour)

# 1. Visible pigment region (430–560 nm): carotenoid, anthocyanin and
#    chlorophyll pigments absorb broadly across this range.
#    Refs: Wang2015FruitQualitySpectroscopy, Gitelson1994ChlorophyllReflectance
VIS_PIGMENT = (430, 560, "#66C2A5", "Visible pigment", "#1B7837")

# 2. Chl-a red absorption (640–680 nm): narrow zone around the
#    well-documented chlorophyll-a red absorption peak (~660–680 nm).
#    Refs: Gitelson1994ChlorophyllReflectance
CHL_RED = (640, 680, "#D73027", "Chl-a red\nabsorption", "#B2182B")

# 3. Red-edge transition (680–750 nm): steep reflectance rise driven
#    by the transition from chlorophyll absorption to cellular scattering.
#    Refs: Imran2020RedEdgeNIRShoulder
RED_EDGE = (680, 750, "#FC8D62", "Red-edge transition", "#D95F02")

# 4. NIR structural / scattering plateau (750–850 nm): reflectance
#    dominated by internal cellular structure and air–cell-wall interfaces.
#    Refs: Lu2020OpticalPropertiesReview, Imran2020RedEdgeNIRShoulder
NIR_STRUCT = (750, 850, "#B2ABD2", "NIR structural\nscattering", "#5E4FA2")

# 5. Pre-970 nm water shoulder (890–930 nm): ascending flank toward the
#    ~970 nm O-H overtone water absorption band; NOT the main water band.
#    Refs: Cheng2019AppleDRS, Sun2021PeachBruisingOptical, Munera2021PersimmonDamageHSI
WATER_NIR = (890, 930, "#8DA0CB", "Pre-970 nm\nwater shoulder", "#2166AC")

regions = [VIS_PIGMENT, RED_EDGE, NIR_STRUCT, WATER_NIR]
# CHL_RED handled separately as a narrow annotation

# ── Figure setup ──────────────────────────────────────────────────────
fig, ax = plt.subplots(figsize=(16, 7), dpi=300)

bar_width = 4.0

# Colorblind-friendly palette
color_5 = "#2166AC"   # strong blue for unanimous
color_4 = "#92C5DE"   # lighter blue for highly stable

# ── Shaded interpretation regions (drawn first, behind bars) ─────────
for start, end, col, _lbl, _lc in regions:
    ax.axvspan(start, end, color=col, alpha=0.10, zorder=1)

# Narrow Chl-a red absorption annotation (subtler shading)
ax.axvspan(CHL_RED[0], CHL_RED[1], color=CHL_RED[2], alpha=0.08, zorder=1)

# ── Data bars (foreground) ───────────────────────────────────────────
ax.bar(unanimous, [5] * len(unanimous), width=bar_width,
       color=color_5, edgecolor="white", linewidth=0.5,
       label="Unanimous (5/5 seeds)", zorder=3)
ax.bar(highly_stable, [4] * len(highly_stable), width=bar_width,
       color=color_4, edgecolor="white", linewidth=0.5,
       label="Highly stable (4/5 seeds)", zorder=3)

# ── Region labels ────────────────────────────────────────────────────
label_y = 5.55
label_props = dict(
    ha="center", va="bottom", fontsize=9, fontstyle="italic",
    fontweight="semibold", zorder=5,
    bbox=dict(boxstyle="round,pad=0.25", facecolor="white",
              edgecolor="none", alpha=0.85),
)

# Visible pigment label – centred on the 430–560 range
ax.text(495, label_y, VIS_PIGMENT[3], color=VIS_PIGMENT[4], **label_props)

# Red-edge label – centred on 680–750
ax.text(715, label_y, RED_EDGE[3], color=RED_EDGE[4], **label_props)

# NIR structural scattering label – centred on 750–850
ax.text(800, label_y, NIR_STRUCT[3], color=NIR_STRUCT[4], **label_props)

# Chl-a annotation – arrow pointing into the narrow 640–680 zone
ax.annotate(
    CHL_RED[3], xy=(660, 5.15), xytext=(660, 6.10),
    ha="center", va="bottom", fontsize=8, fontstyle="italic",
    fontweight="semibold", color=CHL_RED[4],
    arrowprops=dict(arrowstyle="-|>", color=CHL_RED[4], lw=0.8),
    bbox=dict(boxstyle="round,pad=0.2", facecolor="white",
              edgecolor="none", alpha=0.85),
    zorder=5,
)

# Pre-970 nm water shoulder label
ax.text(910, label_y, WATER_NIR[3], color=WATER_NIR[4], **label_props)

# ── Axes ─────────────────────────────────────────────────────────────
ax.set_xlim(400, 980)
ax.set_ylim(0, 6.8)

ax.set_xlabel("Wavelength (nm)", fontsize=13, labelpad=8)
ax.set_ylabel("Consensus (seeds)", fontsize=13, labelpad=8)
ax.set_title("Consensus Wavelength Bins with Spectral Interpretation",
             fontsize=15, fontweight="bold", pad=14)

ax.xaxis.set_major_locator(ticker.MultipleLocator(100))
ax.xaxis.set_minor_locator(ticker.MultipleLocator(50))
ax.yaxis.set_major_locator(ticker.FixedLocator([0, 1, 2, 3, 4, 5]))
ax.set_yticklabels([0, 1, 2, 3, 4, 5])

ax.tick_params(axis="both", labelsize=11)

# Gridlines – horizontal only, subtle
ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, zorder=0)
ax.xaxis.grid(False)
ax.set_axisbelow(True)

# Spine styling
for spine in ["top", "right"]:
    ax.spines[spine].set_visible(False)

# Legend
ax.legend(loc="upper left", fontsize=11,
          frameon=True, framealpha=0.95, edgecolor="#cccccc", fancybox=False)

plt.tight_layout()

# ── Save ──────────────────────────────────────────────────────────────
out_path = r"c:\Users\yovel\OneDrive\Desktop\Grape_Project\thesis\figures\results\feature_selection\consensus_bins_physical_interpretation.png"
fig.savefig(out_path, dpi=300, bbox_inches="tight", facecolor="white")
print(f"Saved to {out_path}")
plt.close()
