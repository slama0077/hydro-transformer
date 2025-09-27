"""
This script
- Scans ../data/time_series for NetCDF files and loads the first file found.
   - Computes correlations among dynamic variables (e.g., meteorological forcings, streamflow).
   - Saves heatmap as correlation_matrix.png
- Loads ../data/attributes/static_attributes.csv
   - Computes correlations among numeric static attributes (e.g., basin length, area).
   - Saves heatmap as static_attributes_correlation.png

Inputs
------
- Folder: ../data/time_series/*.nc (first file used)
- File: ../data/attributes/static_attributes.csv

Outputs
-------
- correlation_matrix.png (dynamic inputs)
- static_attributes_correlation.png (static attributes)
"""

import xarray as xr
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path

# Dynamic inputs correlation
ts_dir = Path("../data/time_series")
nc_files = sorted(ts_dir.glob("*.nc"))

if not nc_files:
    raise FileNotFoundError(f"No .nc files found in {ts_dir}")

# Pick the first file in the folder
first_file = nc_files[0]
print(f"Using file: {first_file.name}")

ds = xr.open_dataset(first_file)

dynamic_inputs = [
    "UGRD_10maboveground_d", "TMP_2maboveground_d", "VGRD_10maboveground_d",
    "APCP_surface_d", "streamflow_d", "DSWRF_surface_d", "precip_rate_d",
    "PRES_surface_d", "SPFH_2maboveground_d", "DLWRF_surface_d"
]

# Convert to DataFrame and drop NaNs
df_dyn = ds[dynamic_inputs].to_dataframe().dropna()

# Compute correlation matrix
corr_dyn = df_dyn.corr()

# Plot heatmap
plt.figure(figsize=(10, 8))
sns.heatmap(corr_dyn, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={'label': 'Correlation'})
plt.title(f"Correlation Matrix of Dynamic Inputs ({first_file.name})")
plt.tight_layout()
plt.savefig("output/dynamic_correlation_matrix.png", dpi=300)
plt.close()
print("Saved dynamic correlation heatmap as correlation_matrix.png")

# Static attributes correlation
static_file = Path("../data/attributes/static_attributes.csv")
df_static = pd.read_csv(static_file)

# Drop non-numeric (like gage ID) and NaNs
numeric_df = df_static.drop(columns=["gage"], errors="ignore").select_dtypes(include=["number"]).dropna()

# Compute correlation matrix
corr_static = numeric_df.corr()

# Plot heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr_static, annot=True, fmt=".2f", cmap="Blues", square=True, cbar_kws={'label': 'Correlation'})
plt.title("Correlation Matrix of Static Catchment Attributes")
plt.tight_layout()
plt.savefig("output/static_correlation_matrix.png", dpi=300)
plt.close()
print("Saved static correlation heatmap as static_attributes_correlation.png")
