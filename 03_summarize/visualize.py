#!/usr/bin/env python3
"""
Merge hydrologic metrics with basin coordinates and visualize spatial performance.

Usage:
    python visualize.py --metrics metrics.csv --coords latlon.csv \
                          --metric NSE --title "(b) Upstream-Only NSE" \
                          --out upstream_nse_binned.pdf

python 03_summarize/visualize.py --metrics exp/transformer1/transformer_upstream_1311_124932/resume_from001/test/model_epoch001/test_metrics.csv --coords data/camels_link.csv --metric NNSE --title "(b) Spatial Distribution of NSNE for Upstream Configuration" --out 03_summarize/output/upstream_nnse.png           
python 03_summarize/visualize.py --metrics exp/transformer1/transformer_combined_1311_125139/resume_from001/test/model_epoch001/test_metrics.csv --coords data/camels_link.csv --metric NNSE --title "(a) Spatial Distribution of NNSE for Combined Configuration" --out 03_summarize/output/combined_nnse.pdf

"""

import argparse
import re
import numpy as np
import pandas as pd
import geopandas as gpd
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle
from mpl_toolkits.axes_grid1.inset_locator import inset_axes
from pathlib import Path


parser = argparse.ArgumentParser(description="Visualize basin metrics across CONUS.")
parser.add_argument("--metrics", "-m", required=True, help="CSV file with basin metrics (must have 'basin' column).")
parser.add_argument("--coords", "-c", required=True, help="CSV file with basin coordinates (must have gages, lon, lat).")
parser.add_argument("--metric", "-x", required=True, help="Metric column to visualize (e.g., NSE, KGE, RMSE).")
parser.add_argument("--title", "-t", required=True, help="Title to display on the figure.")
parser.add_argument("--out", "-o", default="basin_map.pdf", help="Output file for the map (PDF or PNG).")
parser.add_argument("--shapefile", "-s", default="data/us_states_shapefile/tl_2023_us_state.shp",
                    help="Path to CONUS shapefile.")
args = parser.parse_args()


def normalize_id(obj) -> str:
    s = str(obj)
    digits = re.sub(r"\D", "", s)
    if len(digits) < 8:
        digits = digits.zfill(8)
    elif len(digits) > 8:
        digits = digits[-8:]
    return digits



df_metrics = pd.read_csv(args.metrics)
df_coords = pd.read_csv(args.coords)

df_metrics.columns = df_metrics.columns.str.strip()
df_coords.columns = df_coords.columns.str.strip()

if "basin" not in df_metrics.columns:
    raise ValueError("Metrics file must contain a 'basin' column.")

if not any(c in df_coords.columns for c in ["gages", "gagestr"]):
    raise ValueError("Coordinates file must have a 'gages' or 'gagestr' column.")

gage_col = "gages" if "gages" in df_coords.columns else "gagestr"
df_metrics["basin_id"] = df_metrics["basin"].map(normalize_id)
df_coords["gage_id"] = df_coords[gage_col].map(normalize_id)

df_coords_dedup = df_coords.drop_duplicates(subset=["gage_id"], keep="first")
merged = df_metrics.merge(df_coords_dedup[["gage_id", "lat", "lon"]],
                          left_on="basin_id", right_on="gage_id", how="left").drop(columns=["gage_id"])

for c in ["lat", "lon"]:
    merged[c] = pd.to_numeric(merged[c], errors="coerce")

total = len(merged)
matched = merged["lat"].notna().sum()
print(f"Matched {matched} of {total} basins ({matched/total:.1%}).")


metric = args.metric
if metric not in merged.columns:
    raise ValueError(f"Metric '{metric}' not found in file. Available: {list(merged.columns)}")

merged[metric] = pd.to_numeric(merged[metric], errors="coerce").replace([np.inf, -np.inf], np.nan)
vals = merged[metric].dropna().to_numpy()
median_val = float(np.nanmedian(vals)) if vals.size else np.nan

gdf = gpd.GeoDataFrame(merged, geometry=gpd.points_from_xy(merged['lon'], merged['lat']))
conus = gpd.read_file(args.shapefile)


bins = [0.0, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
labels = ['0.0-0.4', '0.4-0.5', '0.5-0.6', '0.6-0.7', '0.7-0.8', '0.8-0.9', '0.9-1.0']
colors = ['#8B0000', '#D32F2F', '#A7C7E7', '#6FA8DC', '#4682B4', '#2E5C8A', '#1A3A5C']


def get_color(val):
    if pd.isna(val):
        return None
    if val < 0:
        return colors[0]
    for i in range(len(bins) - 1):
        if bins[i] <= val < bins[i + 1]:
            return colors[i + 1]
    if val >= bins[-1]:
        return colors[-1]
    return colors[1]


gdf["color"] = gdf[metric].apply(get_color)

fig, ax = plt.subplots(figsize=(9.2, 5))
ax.set_title(args.title, fontsize=10, loc="left")

conus.plot(ax=ax, facecolor="#D9D9D9", edgecolor="#7A7A7A", linewidth=0.5)

for color in colors:
    mask = gdf["color"] == color
    if mask.any():
        ax.scatter(gdf.loc[mask, "lon"], gdf.loc[mask, "lat"],
                   c=color, s=18, edgecolors="white", linewidths=0.5)

ax.set_xlim(-125, -66)
ax.set_ylim(24, 50)
ax.axis("off")


ax_in = inset_axes(ax, width="30%", height="30%", loc="lower left", borderpad=0.3)
hxmin, hxmax = -1.0, 1.0
ax_in.hist(vals, bins=np.linspace(hxmin, hxmax, 25),
           color="#e6eef6", edgecolor="#8aa7c4", linewidth=0.6)
ax_in.set_xlim(hxmin, hxmax)
ax_in.set_xticks([-1, 0, 1])
# ax_in.set_xticks([0, 1])
ax_in.set_yticks([])
ax_in.patch.set_alpha(0.0)
for s in ax_in.spines.values():
    s.set_alpha(0.35)

if np.isfinite(median_val):
    med_color = get_color(median_val)
    ax_in.axvline(median_val, color=med_color, linestyle="--", linewidth=1.2)
    y_top = ax_in.get_ylim()[1]
    ax_in.text(median_val, y_top * 1.03, f"Median: {median_val:.2f}",
               fontsize=8, ha="center", va="bottom",
               bbox=dict(facecolor="white", alpha=0.85, edgecolor="none"),
               clip_on=False)

legend_elems = [Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="white", linewidth=0.5)
                for c in colors]
ax.legend(legend_elems, labels, title=metric,
          loc="lower right", frameon=True, facecolor="white",
          edgecolor="#7A7A7A", fontsize=10, title_fontsize=10).get_frame().set_alpha(0.9)

plt.tight_layout()
Path(args.out).parent.mkdir(parents=True, exist_ok=True)
plt.savefig(args.out, bbox_inches="tight", dpi=400)
print(f"Saved map: {args.out}")
plt.show()
