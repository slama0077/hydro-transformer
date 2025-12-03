"""
Plot NNSE Distributions for Combined vs Upstream Models

Usage:
    python nnse_dist.py \
        --upstream lstm_upstream_valbas_metrics.csv \
        --combined trans_comb_valbas_metrics.csv \
        --out nse_distribution_comparison
    python 03_summarize/nnse_dist.py -u exp/transformer1/transformer_upstream_1311_124932/resume_from001/test/model_epoch001/test_metrics.csv -c exp/transformer1/transformer_combined_1311_125139/resume_from001/test/model_epoch001/test_metrics.csv -o 03_summarize/output/nnse_dist

Author: Taye Akinrele
"""

import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path


# Academic color palette (colorblind-safe, muted)
COLOR_COMBINED = "#4C72B0"   # muted blue
COLOR_UPSTREAM = "#DD8452"   # muted orange
EDGE_COLOR = "#4f4f4f"       # gray edges


def compute_bins(values):
    """Compute counts and means for the 4 fixed NNSE categories."""
    bins = { "0.00–0.50": values[(values >= 0.0) & (values < 0.5)],
        "0.50–0.70": values[(values >= 0.3) & (values < 0.6)],
        "0.70–0.85": values[(values >= 0.7) & (values < 0.85)],
        "0.85-1.00": values[values >= 0.85]
    }
    labels = list(bins.keys())
    counts = [len(v) for v in bins.values()]
    means  = [np.nanmean(v) if len(v) else np.nan for v in bins.values()]
    return labels, counts, means


def load_nnse(path):
    df = pd.read_csv(path)
    if "NNSE" not in df.columns:
        raise ValueError(f"{path} does not contain column 'NNSE'")
    return df["NNSE"].dropna().values


def plot_grouped_bars(combined, upstream, labels, outdir):
    x = np.arange(len(labels))
    width = 0.37  # width of bars

    fig, ax = plt.subplots(figsize=(8, 5))

    # --- Bar plots ---
    bars1 = ax.bar(
        x - width/2,
        combined["counts"],
        width,
        label="Combined",
        color=COLOR_COMBINED,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        alpha=0.9
    )

    bars2 = ax.bar(
        x + width/2,
        upstream["counts"],
        width,
        label="Upstream",
        color=COLOR_UPSTREAM,
        edgecolor=EDGE_COLOR,
        linewidth=1.0,
        alpha=0.9
    )

    # --- Annotate means above bars ---
    for bars, means in [(bars1, combined["means"]), (bars2, upstream["means"])]:
        for bar, mean_val in zip(bars, means):
            if np.isfinite(mean_val):
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + max(combined["counts"] + upstream["counts"]) * 0.02,
                    f"{mean_val:.2f}",
                    ha="center",
                    va="bottom",
                    fontsize=9,
                    color="black"
                )

    # --- Labels & Title ---
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=10)
    ax.set_ylabel("Count of basins", fontsize=11)
    ax.set_xlabel("NNSE Range", fontsize=11)
    # ax.set_title("NNSE Distribution Comparison", fontsize=13)
    ax.legend(frameon=False, fontsize=10)

    ax.grid(axis="y", linestyle="--", alpha=0.3)

    plt.tight_layout()
    outpath = Path(outdir) / "nnse_distribution_comparison_grouped.png"
    plt.savefig(outpath, dpi=400, bbox_inches="tight")
    plt.close()

    print(f"Saved figure: {outpath}")


def main():
    parser = argparse.ArgumentParser(description="NNSE distribution comparison.")
    parser.add_argument("--combined", "-c", required=True, help="Path to combined CSV")
    parser.add_argument("--upstream", "-u", required=True, help="Path to upstream CSV")
    parser.add_argument("--outdir", "-o", required=True, help="Output directory")
    args = parser.parse_args()

    Path(args.outdir).mkdir(parents=True, exist_ok=True)

    # Load NNSE values
    nnse_combined = load_nnse(args.combined)
    nnse_upstream = load_nnse(args.upstream)

    # Bin both datasets
    labels, comb_counts, comb_means = compute_bins(nnse_combined)
    _, up_counts, up_means = compute_bins(nnse_upstream)

    combined = {"counts": comb_counts, "means": comb_means}
    upstream = {"counts": up_counts, "means": up_means}

    # Plot
    plot_grouped_bars(combined, upstream, labels, args.outdir)


if __name__ == "__main__":
    main()