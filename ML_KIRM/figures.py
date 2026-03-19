#####################################################################
# Date:         March 2026
# Developer:  Paulina Portales, ppicazo@uw.edu
# Institution:  University of Washington, Seattle, WA
#####################################################################

import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter


files = [
    "Results/predicted_data_eggyolk_SV.csv",
    # "Results/predicted_data_eggalbumen_SV.csv",
]

panel_labels = ["(a)", "(b)"]

# Optional axis limits
xlims = [None, None]
ylims = [None, None]

# xlims = [
#     (0, 8000),
#     (0, 4000),
# ]
# ylims = [
#     (0, 0.01),
#     (-0.0002, 0.0014),
# ]

def clean_tick(x, pos):
    if abs(x) < 1e-12:
        return "0"
    return f"{x:g}"


def load_results(filepath):
    df = pd.read_csv(filepath)

    required_cols = ["t", "xdot_original", "xdot_pred"]
    missing = [col for col in required_cols if col not in df.columns]
    if missing:
        raise ValueError(f"{filepath} is missing required columns: {missing}")

    return df.sort_values("t").reset_index(drop=True)


datasets = [load_results(f) for f in files]
n_panels = len(datasets)

if len(panel_labels) < n_panels:
    raise ValueError("Not enough panel_labels for the number of files.")
if len(xlims) < n_panels:
    xlims = list(xlims) + [None] * (n_panels - len(xlims))
if len(ylims) < n_panels:
    ylims = list(ylims) + [None] * (n_panels - len(ylims))

plt.rcParams.update({
    "font.size": 13,
    "font.family": "sans-serif",
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#d9d9d9",
    "axes.linewidth": 1.2,
    "axes.labelweight": "bold",
    "axes.titleweight": "bold",
    "xtick.color": "#333333",
    "ytick.color": "#333333",
})

fig, axes = plt.subplots(1, n_panels, figsize=(5.6 * n_panels, 4.8))
fig.patch.set_facecolor("white")

if n_panels == 1:
    axes = [axes]

for i, (ax, df) in enumerate(zip(axes, datasets)):
    ax.set_facecolor("#f2f2f2")

    ax.plot(
        df["t"],
        df["xdot_pred"],
        linestyle="--",
        linewidth=1.6,
        color="#4C9BE8",
        label="Predicted Rate",
        zorder=3,
    )

    ax.plot(
        df["t"],
        df["xdot_original"],
        linestyle="-",
        linewidth=1.7,
        color="black",
        label="Original Rate",
        zorder=2,
    )

    ax.set_xlabel("Time(s)", fontweight="bold")
    ax.set_ylabel("dx/dt", fontweight="bold")

    ax.legend(
        loc="upper right",
        frameon=False,
        handlelength=1.1,
        handletextpad=0.4,
        borderpad=0.2,
        labelspacing=0.3,
    )

    if xlims[i] is not None:
        ax.set_xlim(xlims[i])
    if ylims[i] is not None:
        ax.set_ylim(ylims[i])

    ax.xaxis.set_major_formatter(FuncFormatter(clean_tick))
    ax.yaxis.set_major_formatter(FuncFormatter(clean_tick))
    ax.tick_params(length=0)

    ax.text(
        0.5,
        -0.18,
        panel_labels[i],
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=16,
    )

plt.subplots_adjust(wspace=0.32, bottom=0.27)
plt.show()