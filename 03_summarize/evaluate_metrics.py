#!/usr/bin/env python3
"""
Evaluate Hydrological Model Test Results (Pickle-based)

Description
-----------
This script loads model prediction results from a NeuralHydrology-style pickle
file (e.g., `test_results.p`), computes performance metrics for each basin,
filters out low-variance basins, and generates evaluation outputs.

It produces:
- `valid_basins_metrics.csv` — metrics for valid basins
- `low_variance_basins.csv` — metrics for filtered-out basins
- `evaluation_summary_valid.csv` — summary (mean, median) of metrics
- Violin plot of NSE (`fig_nse_violin_valid.png`)
- Random sample of hydrographs (`fig_hydrograph_<id>.png`)
- Text summary report (`evaluation_report_valid.txt`)

Usage
-----
    python evaluate_metrics.py --pickle <path_to_test_results.p> --outdir <output_dir>

Arguments
---------
--pickle, -p : str
    Path to the input pickle file containing model test results.

--outdir, -o : str
    Output directory where metrics, plots, and reports will be saved.

Example
-------
    python evaluate_metrics.py \
        --pickle ../exp/lstm/lstm_upstream_0610_140026/test/model_epoch001/test_results.p \
        --outdir metrics/lstm

Author
------
Taye Akinrele
"""

import argparse
import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# ---------------- Argument Parser ----------------
parser = argparse.ArgumentParser(description="Evaluate hydrological model results.")
parser.add_argument("--pickle", "-p", required=True, help="Path to the input pickle file.")
parser.add_argument("--outdir", "-o", required=True, help="Directory to save outputs.")
args = parser.parse_args()

# Assign arguments
PICKLE_RESULTS_PATH = args.pickle
OUTDIR = args.outdir
TIME_KEY = "1H"
OBS_VAR = "streamflow_u_obs"
SIM_VAR = "streamflow_u_sim"
# Try these time coordinate candidates in this order (first match wins)
TIME_COORD_CANDIDATES = ["date", "time", "Time", "datetime"]
LOW_VAR_STD_TOL = 0.01
FIG_DPI = 300
N_RANDOM_HYDROGRAPHS = 5
RANDOM_SEED = 152
# ---------------------------------------------

Path(OUTDIR).mkdir(parents=True, exist_ok=True)
sns.set_theme(style="whitegrid", context="talk")

# ---------- Metric functions ----------
def _to_1d(x): return np.asarray(x, float).ravel()

def nse(y_obs, y_sim):
    y_obs = _to_1d(y_obs); y_sim = _to_1d(y_sim)
    m = np.isfinite(y_obs) & np.isfinite(y_sim)
    if not m.any(): return np.nan
    y_obs, y_sim = y_obs[m], y_sim[m]
    denom = np.sum((y_obs - y_obs.mean()) ** 2)
    if denom <= 0: return np.nan
    return 1 - np.sum((y_sim - y_obs) ** 2) / denom

def nnse(val):
    return np.nan if not np.isfinite(val) else 1.0 / (2.0 - val)

def kge(y_obs, y_sim):
    y_obs, y_sim = _to_1d(y_obs), _to_1d(y_sim)
    m = np.isfinite(y_obs) & np.isfinite(y_sim)
    if not m.any(): return np.nan
    y_obs, y_sim = y_obs[m], y_sim[m]
    if np.std(y_obs) == 0 or np.std(y_sim) == 0: return np.nan
    r = np.corrcoef(y_obs, y_sim)[0, 1]
    alpha = np.std(y_sim) / np.std(y_obs)
    beta = np.mean(y_sim) / np.mean(y_obs)
    return 1 - np.sqrt((r - 1) ** 2 + (alpha - 1) ** 2 + (beta - 1) ** 2)

def pearson_r(y_obs, y_sim):
    y_obs, y_sim = _to_1d(y_obs), _to_1d(y_sim)
    m = np.isfinite(y_obs) & np.isfinite(y_sim)
    if not m.any(): return np.nan
    y_obs, y_sim = y_obs[m], y_sim[m]
    if np.std(y_obs) == 0 or np.std(y_sim) == 0: return np.nan
    return np.corrcoef(y_obs, y_sim)[0, 1]

def get_time_coord(ds):
    # pick first matching coordinate or index-like coordinate
    for cand in TIME_COORD_CANDIDATES:
        if cand in ds.coords or cand in ds:
            return cand
    # fallback to any 1D coord with same length as variables
    for c in list(ds.coords):
        if getattr(ds[c], "ndim", 0) == 1:
            return c
    return None

# ---------- Load pickle and compute ----------
with open(PICKLE_RESULTS_PATH, "rb") as f:
    results = pickle.load(f)

rows = []
# stash per-basin time series for plotting later without recomputing
basin_timeseries = {}

for basin, top in results.items():
    if not isinstance(top, dict) or TIME_KEY not in top or "xr" not in top[TIME_KEY]:
        continue
    ds = top[TIME_KEY]["xr"]
    if OBS_VAR not in ds.data_vars or SIM_VAR not in ds.data_vars:
        # try to guess variables if names differ
        obs_cand = [v for v in ds.data_vars if "obs" in v.lower() and ds[v].ndim == 1]
        sim_cand = [v for v in ds.data_vars if any(k in v.lower() for k in ["sim", "pred", "yhat"]) and ds[v].ndim == 1]
        if not obs_cand or not sim_cand:
            continue
        _obs, _sim = obs_cand[0], sim_cand[0]
    else:
        _obs, _sim = OBS_VAR, SIM_VAR

    # collapse time_step=1 if present
    if "time_step" in ds.dims and ds.dims["time_step"] == 1:
        ds = ds.isel(time_step=0)

    qobs = _to_1d(ds[_obs].values)
    qsim = _to_1d(ds[_sim].values)

    m = np.isfinite(qobs) & np.isfinite(qsim)
    qobs, qsim = qobs[m], qsim[m]
    if qobs.size == 0:
        continue

    std_obs = float(np.std(qobs))
    mse = float(np.mean((qsim - qobs) ** 2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(qsim - qobs)))
    nse_val = float(nse(qobs, qsim))
    nnse_val = float(nnse(nse_val)) if np.isfinite(nse_val) else np.nan
    kge_val = float(kge(qobs, qsim))
    r_val = float(pearson_r(qobs, qsim))

    # store metrics
    rows.append({
        "basin": basin,
        "std_obs": std_obs,
        "NSE": nse_val,
        "NNSE": nnse_val,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        "KGE": kge_val,
        "Pearson-r": r_val
    })

    # store time series for potential plotting
    tcoord = get_time_coord(ds)
    if tcoord is not None:
        try:
            tvals = np.array(pd.to_datetime(np.asarray(ds[tcoord].values)))
        except Exception:
            tvals = np.asarray(ds[tcoord].values)
        # apply same finite mask if aligning lengths matters
        # if tvals length mismatches, try to slice
        if tvals.shape[0] != _to_1d(ds[_obs].values).shape[0]:
            # best-effort: skip saving time series for this basin
            continue
        # apply mask to time
        tvals = tvals[m]
        if tvals.size:
            basin_timeseries[basin] = dict(time=tvals, obs=qobs, sim=qsim)

df = pd.DataFrame(rows)
if df.empty:
    raise RuntimeError("No basins with usable data were found in the pickle.")

# ---------- Filter only valid basins ----------
df["low_var_flag"] = df["std_obs"] <= LOW_VAR_STD_TOL

n_total = len(df)
df_valid = df.loc[~df["low_var_flag"]].copy()
n_valid = len(df_valid)
n_lowvar = n_total - n_valid

pct_lowvar = 100 * n_lowvar / n_total if n_total else 0
pct_valid = 100 * n_valid / n_total if n_total else 0

print(f"Total basins: {n_total}")
print(f"Low-variance basins removed (std_obs <= {LOW_VAR_STD_TOL}): {n_lowvar} ({pct_lowvar:.2f}%)")
print(f"Valid basins kept: {n_valid} ({pct_valid:.2f}%)")

# ---------- Save valid basins only ----------
df_valid.to_csv(Path(OUTDIR) / "valid_basins_metrics.csv", index=False)
df.loc[df["low_var_flag"]].to_csv(Path(OUTDIR) / "low_variance_basins.csv", index=False)

# ---------- Compute summary (valid only) ----------
metrics = ["NSE", "NNSE", "MSE", "RMSE", "MAE", "KGE", "Pearson-r"]
summary = pd.DataFrame({
    "Metric": metrics,
    "Mean": [df_valid[m].mean(skipna=True) for m in metrics],
    "Median": [df_valid[m].median(skipna=True) for m in metrics],
    "Count": [df_valid[m].notna().sum() for m in metrics]
})
summary.to_csv(Path(OUTDIR) / "evaluation_summary_valid.csv", index=False)
print("Saved evaluation_summary_valid.csv")

# ---------- Violin plot (NSE valid only) ----------
sns.violinplot(y=df_valid["NSE"], color="skyblue", inner="box")
plt.axhline(0, color="k", ls="--", lw=1)
plt.axhline(0.5, color="gray", ls="--", lw=1)
plt.axhline(0.75, color="gray", ls="--", lw=1)
plt.title(f"NSE distribution, valid basins only (n={n_valid})")
plt.ylabel("NSE")
plt.tight_layout()
plt.savefig(Path(OUTDIR) / "fig_nse_violin_valid.png", dpi=FIG_DPI)
plt.close()

# ---------- Random hydrographs for valid basins ----------
# Pick basins that we actually have stored time series for
valid_ids_with_ts = [b for b in df_valid["basin"].tolist() if b in basin_timeseries]
if valid_ids_with_ts:
    rng = np.random.default_rng(RANDOM_SEED)
    k = min(N_RANDOM_HYDROGRAPHS, len(valid_ids_with_ts))
    sample_basins = rng.choice(valid_ids_with_ts, size=k, replace=False)

    for b in sample_basins:
        ts = basin_timeseries[b]
        nse_val = df_valid.loc[df_valid["basin"].eq(b), "NSE"].values[0]

        plt.figure(figsize=(12, 4))
        plt.plot(ts["time"], ts["obs"], label="Observed", linewidth=1.6)
        plt.plot(ts["time"], ts["sim"], label="Simulated", linewidth=1.6)
        plt.xlabel("Date")
        plt.ylabel("Discharge (m³/s)")
        plt.title(f"Hydrograph {b}  (NSE={nse_val:.2f})")
        plt.legend()
        plt.tight_layout()
        out_path = Path(OUTDIR) / f"fig_hydrograph_{b}.png"
        plt.savefig(out_path, dpi=FIG_DPI, bbox_inches="tight")
        plt.close()
    print(f"Saved {k} hydrographs for valid basins in {OUTDIR}")
else:
    print("No valid basins with available time series to plot hydrographs.")
    
# ---------- Text summary ----------
with open(Path(OUTDIR) / "evaluation_report_valid.txt", "w") as f:
    f.write("Evaluation Summary — Valid Basins Only\n")
    f.write("=====================================\n")
    f.write(f"Total basins: {n_total}\n")
    f.write(f"Low-variance basins removed (std_obs <= {LOW_VAR_STD_TOL}): {n_lowvar} ({pct_lowvar:.2f}%)\n")
    f.write(f"Valid basins kept: {n_valid} ({pct_valid:.2f}%)\n\n")
    for m in metrics:
        f.write(f"{m:<10} Mean={df_valid[m].mean():.4f}  Median={df_valid[m].median():.4f}\n")