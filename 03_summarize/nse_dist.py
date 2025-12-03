#!/usr/bin/env python3
"""
Compare NSE distributions between upstream and combined datasets
using side-by-side bar charts.

Usage:
    python compare_nse_distribution.py \
        --upstream lstm_upstream_valbas_metrics.csv \
        --combined trans_comb_valbas_metrics.csv \
        --out nse_distribution_comparison
    python 03_summarize/nse_dist.py -u exp/lstm/transformer_upstream_1011_144629/test/model_epoch001/test_metrics.csv -c exp/lstm/transformer_combined_1011_091910/test/model_epoch001/test_metrics.csv -o 03_summarize/output/nse_dist
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

parser = argparse.ArgumentParser(description="Plot NSE and KGE basin count comparison (Upstream vs Combined).")
parser.add_argument("--upstream", "-u", required=True, help="CSV file containing metrics for upstream run.")
parser.add_argument("--combined", "-c", required=True, help="CSV file containing metrics for combined run.")
parser.add_argument("--out", "-o", default="nse_kge_distribution", help="Output file base name (no extension).")
args = parser.parse_args()

plt.rcParams.update({
    "font.size": 9,
    "axes.labelsize": 10,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "figure.dpi": 400,
})

colors = ["#a7c7e7", "#b0c4de"]  # blue shades for combined and upstream


def load_metrics(path):
    df = pd.read_csv(path)
    df.columns = df.columns.str.strip()
    for col in ["NSE", "KGE"]:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce").replace([np.inf, -np.inf], np.nan)
    return df

df_up = load_metrics(args.upstream)
df_comb = load_metrics(args.combined)


bins = [-np.inf, 0.0, 0.5, 0.8, np.inf]
labels = ["< 0.0", "0.0–0.5", "0.5–0.8", "> 0.8"]

def summarize_bins(df, metric):
    df_clean = df.dropna(subset=[metric]).copy()
    df_clean[f"{metric}_bin"] = pd.cut(df_clean[metric], bins=bins, labels=labels, 
                                        include_lowest=True, right=False)
    return (
        df_clean.groupby(f"{metric}_bin")[metric]
          .agg(["count", "mean"])
          .reset_index()
          .assign(mean=lambda d: d["mean"].round(2))
    )


summary_nse_up = summarize_bins(df_up, "NSE")
summary_nse_comb = summarize_bins(df_comb, "NSE")


summary_kge_up = summarize_bins(df_up, "KGE")
summary_kge_comb = summarize_bins(df_comb, "KGE")


fig, axes = plt.subplots(2, 2, figsize=(6, 4.5), sharex=True)

fig.suptitle(
    "Basin Count by NSE and KGE Range for Combined and Upstream Models",
    fontsize=11,
    y=0.98,
    bbox=dict(
        boxstyle="round,pad=0.35",
        facecolor="white",
        edgecolor="#CCCCCC",
        alpha=0.9
    )
)


ax = axes[0, 0]
bars = ax.bar(summary_nse_comb["NSE_bin"], summary_nse_comb["count"],
              color=colors[1], edgecolor="gray", width=0.6)
for rect, mean_val in zip(bars, summary_nse_comb["mean"]):
    h = rect.get_height()
    if np.isfinite(mean_val):
        ax.text(rect.get_x() + rect.get_width() / 2, h + summary_nse_comb["count"].max() * 0.02,
                f"{mean_val:.2f}", ha="center", fontsize=8)
ax.set_ylabel("Count of basins")
ax.set_title("(a) NSE Combined", fontsize=8, loc="center")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)


ax = axes[0, 1]
bars = ax.bar(summary_nse_up["NSE_bin"], summary_nse_up["count"],
              color=colors[0], edgecolor="gray", width=0.6)
for rect, mean_val in zip(bars, summary_nse_up["mean"]):
    h = rect.get_height()
    if np.isfinite(mean_val):
        ax.text(rect.get_x() + rect.get_width() / 2, h + summary_nse_up["count"].max() * 0.02,
                f"{mean_val:.2f}", ha="center", fontsize=8)
ax.set_ylabel("Count of basins")
ax.set_title("(b) NSE Upstream", fontsize=8, loc="center")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

ax = axes[1, 0]
bars = ax.bar(summary_kge_comb["KGE_bin"], summary_kge_comb["count"],
              color=colors[1], edgecolor="gray", width=0.6)
for rect, mean_val in zip(bars, summary_kge_comb["mean"]):
    h = rect.get_height()
    if np.isfinite(mean_val):
        ax.text(rect.get_x() + rect.get_width() / 2, h + summary_kge_comb["count"].max() * 0.02,
                f"{mean_val:.2f}", ha="center", fontsize=8)
ax.set_xlabel("KGE range")
ax.set_ylabel("Count of basins")
ax.set_title("(c) KGE Combined", fontsize=8, loc="center")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)


ax = axes[1, 1]
bars = ax.bar(summary_kge_up["KGE_bin"], summary_kge_up["count"],
              color=colors[0], edgecolor="gray", width=0.6)
for rect, mean_val in zip(bars, summary_kge_up["mean"]):
    h = rect.get_height()
    if np.isfinite(mean_val):
        ax.text(rect.get_x() + rect.get_width() / 2, h + summary_kge_up["count"].max() * 0.02,
                f"{mean_val:.2f}", ha="center", fontsize=8)
ax.set_xlabel("KGE range")
ax.set_ylabel("Count of basins")
ax.set_title("(d) KGE Upstream", fontsize=8, loc="center")
ax.grid(axis="y", alpha=0.3)
ax.set_axisbelow(True)

plt.tight_layout()

out_base = Path(args.out)
plt.savefig(out_base.with_suffix(".pdf"), bbox_inches="tight")
plt.savefig(out_base.with_suffix(".png"), bbox_inches="tight")
print(f"Saved: {out_base.with_suffix('.pdf')} and {out_base.with_suffix('.png')}")
plt.show()
