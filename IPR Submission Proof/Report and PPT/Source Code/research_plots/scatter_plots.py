import pandas as pd
import matplotlib.pyplot as plt
import matplotlib
import seaborn as sns
from scipy import stats

# ── 1. Load your data ─────────────────────────────────────────────────────────
df = pd.read_csv("your_data.csv")   # <-- replace with your actual filename

# ── 2. Global style ───────────────────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "axes.titlesize": 11,
    "axes.titleweight": "bold",
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

BLUE = "#0f3460"
RED  = "#e94560"
GREY = "#8d99ae"

# ── Helper: draw one scatter with regression line ─────────────────────────────
def scatter_with_regression(ax, x, y, xlabel, ylabel, title, color):
    # scatter points
    ax.scatter(x, y, color=color, alpha=0.55, edgecolors="white",
               linewidth=0.5, s=55, zorder=3)

    # regression line
    slope, intercept, r, p, _ = stats.linregress(x, y)
    x_line = sorted(x)
    y_line = [slope * xi + intercept for xi in x_line]
    ax.plot(x_line, y_line, color="black", linewidth=1.5, zorder=4)

    # r and p annotation
    ptext = "p < 0.001" if p < 0.001 else f"p = {p:.3f}"
    ax.text(0.97, 0.05,
            f"r = {r:.2f}\n{ptext}",
            transform=ax.transAxes,
            ha="right", va="bottom", fontsize=9,
            bbox=dict(boxstyle="round,pad=0.3",
                      facecolor="white", edgecolor=GREY, linewidth=0.8))

    ax.set_xlabel(xlabel, labelpad=5)
    ax.set_ylabel(ylabel, labelpad=5)
    ax.set_title(title, pad=8)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.grid(color="#e0e0e0", linewidth=0.5, zorder=1)


# ── PLOT 1: Digital Connectivity vs Decision Fatigue ─────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
scatter_with_regression(
    ax,
    x=df["dc_score"],
    y=df["fatigue_score"],
    xlabel="Digital Connectivity Intensity (1–5)",
    ylabel="Decision Fatigue Score (1–5)",
    title="Fig. A  Digital Connectivity vs Decision Fatigue\n(n = 200)",
    color=BLUE
)
plt.tight_layout()
plt.savefig("scatter_dc_vs_fatigue.png")
plt.show()
print("Plot 1 saved.")


# ── PLOT 2: Decision Fatigue vs Decision Quality ──────────────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
scatter_with_regression(
    ax,
    x=df["fatigue_score"],
    y=df["dq_score"],
    xlabel="Decision Fatigue Score (1–5)",
    ylabel="Decision Quality Score (1–5)",
    title="Fig. B  Decision Fatigue vs Perceived Decision Quality\n(n = 200)",
    color=RED
)
plt.tight_layout()
plt.savefig("scatter_fatigue_vs_dq.png")
plt.show()
print("Plot 2 saved.")


# ── PLOT 3: After-Hours Engagement vs Decision Fatigue ───────────────────────
fig, ax = plt.subplots(figsize=(5, 4))
scatter_with_regression(
    ax,
    x=df["after_hours_score"],
    y=df["fatigue_score"],
    xlabel="After-Hours Digital Engagement (1–5)",
    ylabel="Decision Fatigue Score (1–5)",
    title="Fig. C  After-Hours Engagement vs Decision Fatigue\n(n = 200)",
    color="#2d6a4f"
)
plt.tight_layout()
plt.savefig("scatter_afterhours_vs_fatigue.png")
plt.show()
print("Plot 3 saved.")


# ── PLOT 4: All 3 side-by-side (publication-ready) ───────────────────────────
fig, axes = plt.subplots(1, 3, figsize=(13, 4.2))

scatter_with_regression(axes[0], df["dc_score"], df["fatigue_score"],
    "Digital Connectivity (1–5)", "Decision Fatigue (1–5)",
    "DC Intensity vs Fatigue", BLUE)

scatter_with_regression(axes[1], df["fatigue_score"], df["dq_score"],
    "Decision Fatigue (1–5)", "Decision Quality (1–5)",
    "Fatigue vs Decision Quality", RED)

scatter_with_regression(axes[2], df["after_hours_score"], df["fatigue_score"],
    "After-Hours Engagement (1–5)", "Decision Fatigue (1–5)",
    "After-Hours vs Fatigue", "#2d6a4f")

fig.suptitle(
    "Scatter Plot Analysis — Digital Connectivity, Decision Fatigue & Decision Quality (n = 200)",
    fontsize=11, fontweight="bold", y=1.02
)
plt.tight_layout()
plt.savefig("scatter_all_combined.png", bbox_inches="tight")
plt.show()
print("Combined plot saved.")

# ── PLOT 5: Correlation Heatmap ───────────────────────────────────────────────
import seaborn as sns
import numpy as np

fig, ax = plt.subplots(figsize=(6, 5))

labels = ["DC\nIntensity", "Decision\nFatigue", "Decision\nQuality", "After-Hours\nEngagement"]
corr_matrix = df[["dc_score", "fatigue_score", "dq_score", "after_hours_score"]].corr()

sns.heatmap(corr_matrix,
            annot=True, fmt=".2f",
            cmap="RdBu_r", center=0, vmin=-1, vmax=1,
            linewidths=0.5, linecolor="white",
            square=True, ax=ax,
            xticklabels=labels, yticklabels=labels,
            annot_kws={"size": 11, "weight": "bold"})

ax.set_title("Correlation Matrix — Key Study Variables\n(n = 200)", 
             pad=10, fontsize=11, fontweight="bold")

plt.tight_layout()
plt.savefig("scatter_correlation_heatmap.png", dpi=180, bbox_inches="tight")
plt.show()
print("Plot 5 (heatmap) saved.")

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.size": 10,
    "figure.dpi": 180,
    "savefig.dpi": 180,
    "savefig.bbox": "tight",
})

BLUE = "#0f3460"
RED  = "#e94560"
GREY = "#8d99ae"

fig, ax = plt.subplots(figsize=(5.5, 4.5))
fig.patch.set_facecolor("white")
ax.set_facecolor("white")

predictors = ["Digital Connectivity\nIntensity (\u03b2\u2081)", 
              "After-Hours Digital\nEngagement (\u03b2\u2082)"]
betas      = [0.42, 0.36]
pvals      = ["p < 0.01", "p < 0.05"]
colors     = [BLUE, RED]

bars = ax.bar(predictors, betas, color=colors, width=0.38,
              edgecolor="none", zorder=3)

ax.set_ylim(0, 0.6)
ax.set_ylabel("Standardized Coefficient (\u03b2)", labelpad=6)
ax.grid(axis="y", color="#e0e0e0", linewidth=0.6, zorder=1)
ax.spines["top"].set_visible(False)
ax.spines["right"].set_visible(False)

# Beta + p-value labels on top of each bar
for bar, val, pv in zip(bars, betas, pvals):
    ax.text(bar.get_x() + bar.get_width() / 2, val + 0.015,
            f"\u03b2 = {val}\n{pv}",
            ha="center", fontsize=9, fontweight="bold", color="black")

# R² annotation box (top right)
ax.text(0.98, 0.96,
        "Adj. R\u00b2 \u2248 0.61\np < 0.001\nn = 200",
        transform=ax.transAxes, ha="right", va="top", fontsize=9,
        bbox=dict(boxstyle="round,pad=0.35", facecolor="white",
                  edgecolor=GREY, linewidth=0.9))

ax.set_title("Regression Predictor Coefficients \u2014 Dependent Variable: Decision Fatigue",
             fontsize=9, pad=8, loc="left", color="black")

plt.tight_layout()
plt.savefig("scatter_regression_coefficients.png")
plt.show()
print("Regression plot saved.")