"""
CDF comparison for four models:
- LSTM Upstream
- LSTM Combined
- Transformer Upstream
- Transformer Combined

    python 03_summarize/cdf_compare.py --lstm_up exp/lstm1/lstm_upstream_1311_222213/resume_from001/test/model_epoch001/test_metrics.csv --lstm_comb exp/lstm1/lstm_combined_1311_204458/resume_from001/test/model_epoch001/test_metrics.csv   --trans_up exp/transformer1/transformer_upstream_1311_124932/resume_from001/test/model_epoch001/test_metrics.csv --trans_comb exp/transformer1/transformer_combined_1311_125139/resume_from001/test/model_epoch001/test_metrics.csv -o 03_summarize/output/
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


parser = argparse.ArgumentParser(description="Compare four models using CDF plots.")
parser.add_argument("--lstm_up", required=True)
parser.add_argument("--lstm_comb", required=True)
parser.add_argument("--trans_up", required=True)
parser.add_argument("--trans_comb", required=True)
parser.add_argument("--out", "-o", default="cdf_four_models")
args = parser.parse_args()


metrics = ["NNSE", "KGE", "Pearson-r", "RMSE", ]

colors = {
    "LSTM Upstream": "#1F77B4",      # blue
    "LSTM Combined": "#D62728",      # red
    "Trans Upstream": "#2CA02C",     # green
    "Trans Combined": "#9467BD",     # purple
}

plt.rcParams.update({
    "font.size": 7,
    "axes.labelsize": 8,
    "axes.titlesize": 7,
    "legend.fontsize": 7,
})

# ----------------------------------------------------
# Load all datasets
# ----------------------------------------------------
df_lstm_up = pd.read_csv(args.lstm_up)
df_lstm_comb = pd.read_csv(args.lstm_comb)
df_trans_up = pd.read_csv(args.trans_up)
df_trans_comb = pd.read_csv(args.trans_comb)

dfs = {
    "LSTM Upstream": df_lstm_up,
    "LSTM Combined": df_lstm_comb,
    "Trans Upstream": df_trans_up,
    "Trans Combined": df_trans_comb,
}

for k in dfs:
    dfs[k].columns = dfs[k].columns.str.strip()

# ----------------------------------------------------
# Helper function
# ----------------------------------------------------
def get_cdf_data(x):
    x = pd.to_numeric(x, errors="coerce").dropna().values
    if len(x) == 0:
        return np.array([]), np.array([])
    x = np.sort(x)
    y = np.linspace(0, 1, len(x))
    return x, y

# ----------------------------------------------------
# Figure layout
# ----------------------------------------------------
fig, axes = plt.subplots(2, 3, figsize=(10, 4.2))
axes = axes.flatten()

for i, metric in enumerate(metrics):
    ax = axes[i]

    for model_name, df in dfs.items():
        if metric not in df.columns:
            continue

        x, y = get_cdf_data(df[metric])

        # X-limits per metric
        if metric in ["NSE", "KGE"]:
            xmin, xmax = -1, 1
        elif metric in ["RMSE", "MSE"]:
            xmin = 0
            xmax = np.nanpercentile(x, 99)  # avoid long tails
        else:
            xmin, xmax = np.nanmin(x), np.nanmax(x)

        ax.plot(x, y, label=model_name, color=colors[model_name], linewidth=1.2)

    ax.set_xlim(xmin, xmax)
    ax.set_ylim(0, 1)
    ax.set_xlabel(f"({chr(97+i)}) {metric}")
    ax.set_ylabel("CDF")
    ax.grid(alpha=0.3)

# Title + Legend
fig.text(0.5, 0.97, "CDF Comparison of Hydrologic Metrics (4 Models)",
         ha="center", va="top", fontsize=10,
         bbox=dict(boxstyle="round,pad=0.3", facecolor="white", edgecolor="#CCCCCC"))

handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=8)

plt.subplots_adjust(top=0.90, bottom=0.12, wspace=0.30, hspace=0.30)

# Save
out_base = Path(args.out)
plt.savefig(out_base.with_suffix(".png"), dpi=400, bbox_inches="tight")
plt.savefig(out_base.with_suffix(".pdf"), dpi=400, bbox_inches="tight")

print("Saved:", out_base.with_suffix(".png"))
