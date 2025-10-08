# This script prints out the validation and test metrics (check if model is overfitting and underfitting)

"""
Example usage:
python summarize_from_csv.py --test test_metrics.csv --val validation_metrics.csv --outdir metrics_out
python summarize_from_csv.py --test ../exp/lstm/lstm_combined_0610_193144/test/model_epoch001/test_metrics.csv --val ../exp/lstm/lstm_combined_0610_193144/validation/model_epoch001/validation_metrics.csv --outdir output

python 03_summarize/summarize_from_csv.py --test exp/lstm/lstm_combined_0610_193144/test/model_epoch001/test_metrics.csv --val exp/lstm/lstm_combined_0610_193144/validation/model_epoch001/validation_metrics.csv --outdir 03_summarize/output/lstm_combined1

"""

import argparse
import os
import pandas as pd
import numpy as np

# Exact metrics to summarize
TARGET_METRICS = ["NSE", "NNSE", "MSE", "RMSE", "MAE", "KGE", "Pearson-r"]

def summarize_frame(df: pd.DataFrame) -> pd.DataFrame:
    """Compute mean, median, and count for each metric in TARGET_METRICS."""
    data = {}
    for metric in TARGET_METRICS:
        if metric not in df.columns:
            data[metric] = {"mean": np.nan, "median": np.nan, "count": 0}
            continue
        series = pd.to_numeric(df[metric], errors="coerce")
        data[metric] = {
            "mean": float(series.mean(skipna=True)),
            "median": float(series.median(skipna=True)),
            "count": int(series.count())
        }
    return pd.DataFrame(data).T

def main():
    parser = argparse.ArgumentParser(description="Combine test and validation metric summaries into one CSV.")
    parser.add_argument("--test", required=True, help="Path to test_metrics.csv")
    parser.add_argument("--val", required=True, help="Path to validation_metrics.csv")
    parser.add_argument("--outdir", default="metrics_out", help="Output directory")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # Load and summarize
    test_df = pd.read_csv(args.test)
    val_df = pd.read_csv(args.val)

    test_summary = summarize_frame(test_df)
    val_summary = summarize_frame(val_df)

    # Merge summaries
    combined = pd.concat(
        [test_summary.add_prefix("test_"), val_summary.add_prefix("val_")],
        axis=1
    )
    combined.index.name = "metric"

    out_csv = os.path.join(args.outdir, "combined_summary.csv")
    combined.to_csv(out_csv, float_format="%.6g")

    print(f"✅ Combined summary saved to: {out_csv}")

if __name__ == "__main__":
    main()
