"""
This script does an end-to-end utility for basin pair preprocessing:

- Scan ../data/n{reach_value}/time_series for NetCDF files named <downstream>_<upstream>.nc
- Extract downstream and upstream IDs into basin pairs
- Join each downstream ID to its CAMELS gage ID via ../data/camels_link.csv
- Write to output/basin_pair_with_gages.csv
- Write ../data/n{reach_value}/gage_list.txt
- Rename the NetCDF files from <downstream>_<upstream>.nc to <gage>.nc

Inputs
------
- Pair discovery:
    Folder: ../data/n{reach_value}/time_series
    Files: *.nc named as <downstream>_<upstream>.nc
- Gage link table:
    ../data/camels_link.csv with columns:
      to (int-like downstream reach ID), gages (string or bytes-like gage)

Outputs
-------
- ../data/n{reach_value}/basin_pair_with_gages.csv
- ../data/n{reach_value}/gage_list.txt
- Renamed files in ../data/n{reach_value}/time_series
    Before: <downstream>_<upstream>.nc
    After:  <gage>.nc

Usage examples
--------------
- Run end-to-end with rename:
    python 01_basin_pair_preprocessor.py --reach-value 10

- Dry run the rename step to preview:
    python 01_basin_pair_preprocessor.py --dry-run

- Skip renaming entirely:
    python 01_basin_pair_preprocessor.py --skip-rename
"""

from pathlib import Path
import argparse
import pandas as pd
import os


def discover_basin_pairs(ts_dir: Path) -> pd.DataFrame:
    """Find *.nc files named <downstream>_<upstream>.nc and return DataFrame."""
    nc_files = list(ts_dir.glob("*.nc"))
    pairs = []
    for f in nc_files:
        try:
            downstream, upstream = f.stem.split("_", 1)
            pairs.append((int(downstream), int(upstream)))
        except ValueError:
            print(f"Skipping file with unexpected name: {f.name}")
    return pd.DataFrame(pairs, columns=["downstream", "upstream"])


def load_and_clean_link(link_csv: Path) -> pd.DataFrame:
    """Load camels_link.csv and extract clean numeric gage strings with leading zeros preserved."""
    link_df = pd.read_csv(link_csv)
    link_df["to"] = link_df["to"].astype(int)
    # Extract only digits from gages, handle b'01234567' or "01234567"
    gage_series = (
        link_df["gages"]
        .astype(str)
        .str.strip()
        .str.extract(r"b?['\"]?\s*([0-9]+)\s*['\"]?", expand=False)
        .astype("string")
    )
    link_df = link_df.assign(gages_clean=gage_series)
    return link_df[["to", "gages_clean"]]


def merge_pairs_with_gages(pairs_df: pd.DataFrame, link_df: pd.DataFrame) -> pd.DataFrame:
    """Left-join downstream to gage, keep only rows with a matched gage."""
    merged = (
        pairs_df.merge(link_df, how="left", left_on="downstream", right_on="to")
        .drop(columns=["to"])
        .rename(columns={"gages_clean": "gage"})
    )
    with_gage = merged[merged["gage"].notna()].copy()
    with_gage["gage"] = with_gage["gage"].astype(str)
    return with_gage[["downstream", "upstream", "gage"]]


def save_gage_lists(pairs_csv: Path,
                    out_txt: Path = Path("output/gage_list.txt"),
                    zero_pad: bool = True) -> pd.DataFrame:
    """
    Read basin_pair_with_gages.csv and write unique gage lists to TXT and CSV.
    - Ensures gages are strings, stripped, optional zero-padding to 8 digits.
    - Returns the DataFrame of unique gages for further use.
    """
    df = pd.read_csv(pairs_csv, dtype={"gage": str})
    g = (
        df["gage"]
        .dropna()
        .astype(str)
        .str.strip()
    )
    if zero_pad:
        g = g.str.zfill(8)

    unique_gages = pd.DataFrame(sorted(g.unique()), columns=["gage"])

    out_txt.parent.mkdir(parents=True, exist_ok=True)
    out_txt.write_text("\n".join(unique_gages["gage"].tolist()) + "\n", encoding="utf-8")

    print(f"Saved {len(unique_gages)} unique gages to {out_txt}")
    return unique_gages


def rename_nc_files(ts_dir: Path, pairs_with_gage_df: pd.DataFrame, dry_run: bool = False) -> None:
    """Rename <downstream>_<upstream>.nc to <gage>.nc in ts_dir. Pads gage to 8 digits."""
    df = pairs_with_gage_df.copy()
    df["gage"] = df["gage"].astype(str).str.zfill(8)

    ts_dir.mkdir(parents=True, exist_ok=True)
    renamed = 0
    missing = 0

    for _, row in df.iterrows():
        old_name = f"{row['downstream']}_{row['upstream']}.nc"
        new_name = f"{row['gage']}.nc"
        old_path = ts_dir / old_name
        new_path = ts_dir / new_name

        if old_path.exists():
            if dry_run:
                print(f"[DRY RUN] Would rename {old_name} -> {new_name}")
            else:
                old_path.rename(new_path)
                print(f"Renamed {old_name} -> {new_name}")
            renamed += 1
        else:
            print(f"File not found: {old_name}")
            missing += 1

    print(f"Rename summary - renamed: {renamed}, missing: {missing}")


def main():
    parser = argparse.ArgumentParser(description="End-to-end basin pair pipeline.")

    # reach value decides which nXX folder to use
    parser.add_argument("--reach-value", type=int, default=10,
                        help="Reach value used to build data paths (e.g., 10 for n10).")

    parser.add_argument("--link-csv", type=Path, default=Path("data/camels_link.csv"),
                        help="CSV mapping downstream reach to gage.")
    parser.add_argument("--atts-csv", type=Path, default=Path("data/camelsatts.csv"),
                        help="CSV with basin attributes.")
    parser.add_argument("--skip-rename", action="store_true",
                        help="Skip renaming NetCDF files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned renames without changing files.")
    args = parser.parse_args()

    
    # derive paths from reach-value
    base_dir = Path(f"data/n{args.reach_value}")
    os.mkdir(f"{base_dir}/gages")

    args.ts_dir    = base_dir / "time_series"
    args.gages_txt = base_dir / gages / "gage_list.txt"
    args.pairs_out = base_dir / "basin_pair_with_gages.csv"

    # ensure dirs exist
    args.ts_dir.mkdir(parents=True, exist_ok=True)

    # Discover basin pairs from filenames
    print("Discovering basin pairs from filenames...")
    pairs_df = discover_basin_pairs(args.ts_dir)
    if pairs_df.empty:
        print("No valid pairs found in the time series directory.")
        return
    print(f"Found {len(pairs_df)} filename pairs")

    # Load and clean link table
    print("Loading gage link table...")
    link_df = load_and_clean_link(args.link_csv)

    # Merge to get basin pairs with gages
    print("Merging pairs with gages...")
    pairs_with_gage = merge_pairs_with_gages(pairs_df, link_df)
    pairs_with_gage.to_csv(args.pairs_out, index=False)
    print(f"Wrote basin pairs with gages to {args.pairs_out} - rows: {len(pairs_with_gage)}")

    # Save gage lists
    print("Building gage lists...")
    save_gage_lists(args.pairs_out, args.gages_txt, zero_pad=True)

    # Rename files
    if not args.skip_rename:
        print("Renaming NetCDF files...")
        rename_nc_files(args.ts_dir, pairs_with_gage, dry_run=args.dry_run)
    else:
        print("Skipping rename step by request.")


if __name__ == "__main__":
    main()
