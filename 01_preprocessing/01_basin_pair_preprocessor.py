"""
This script does an end-to-end utility for basin pair preprocessing:

- Scan ../data/time_series for NetCDF files named <downstream>_<upstream>.nc
- Extract downstream and upstream IDs into basin pairs
- Join each downstream ID to its CAMELS gage ID via ../data/camels_link.csv
- Write to output/basin_pair_with_gages.csv
- Load ../data/camelsatts.csv and merge downstream and upstream attributes
   - Outputs ../data/attributes/static_attributes.csv with columns:
     gage, basin_length_d, basin_area_d, reach_length_d,
     basin_length_u, basin_area_u, reach_length_u
- Rename the NetCDF files from <downstream>_<upstream>.nc to <gage>.nc

Inputs
------
- Pair discovery:
    Folder: ../data/time_series
    Files: *.nc named as <downstream>_<upstream>.nc
- Gage link table:
    ../data/camels_link.csv with columns:
      to (int-like downstream reach ID), gages (string or bytes-like gage)
- Attributes:
    ../data/camelsatts.csv with columns:
      id, basin_length, basin_area, reach_length

Outputs
-------
- output/basin_pair_with_gages.csv
- ../data/attributes/static_attributes.csv
- Renamed files in ../data/time_series (optional)
    Before: <downstream>_<upstream>.nc
    After:  <gage>.nc

Usage examples
--------------
- Run end-to-end with rename:
    python pipeline_basin_pairs.py

- Dry run the rename step to preview:
    python pipeline_basin_pairs.py --dry-run

- Skip renaming entirely:
    python pipeline_basin_pairs.py --skip-rename
"""


from pathlib import Path
import argparse
import pandas as pd


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
    """Load alabama_link.csv and extract clean numeric gage strings with leading zeros preserved."""
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


def build_static_attributes(basin_pairs_with_gage_csv: Path,
                            atts_csv: Path,
                            out_static_csv: Path) -> pd.DataFrame:
    """Merge downstream and upstream attributes and write static_attributes.csv."""
    atts_df = pd.read_csv(atts_csv, dtype={"id": str})
    crosswalk_df = pd.read_csv(basin_pairs_with_gage_csv,
                               dtype={"gage": str, "downstream": str, "upstream": str})

    down_attrs = atts_df.rename(columns={
        "id": "downstream",
        "basin_length": "basin_length_d",
        "basin_area": "basin_area_d",
        "reach_length": "reach_length_d"
    })
    merged = crosswalk_df.merge(down_attrs, on="downstream", how="left")

    up_attrs = atts_df.rename(columns={
        "id": "upstream",
        "basin_length": "basin_length_u",
        "basin_area": "basin_area_u",
        "reach_length": "reach_length_u"
    })
    merged = merged.merge(up_attrs, on="upstream", how="left")

    static_attributes = merged[[
        "gage",
        "basin_length_d", "basin_area_d", "reach_length_d",
        "basin_length_u", "basin_area_u", "reach_length_u"
    ]].drop_duplicates()

    out_static_csv.parent.mkdir(parents=True, exist_ok=True)
    static_attributes.to_csv(out_static_csv, index=False)
    print(f"Saved {len(static_attributes)} unique gage records to {out_static_csv}")

    # Duplicate checks
    df_check = pd.read_csv(out_static_csv, dtype={"gage": str})
    dup_cols = df_check.columns[df_check.columns.duplicated()].tolist()
    print("Duplicate columns:", dup_cols)
    dup_ids = df_check[df_check.duplicated(subset=["gage"], keep=False)]
    print("Number of duplicate gage IDs:", dup_ids["gage"].nunique())
    if not dup_ids.empty:
        print("Examples of duplicate gages:\n", dup_ids.head())

    return static_attributes


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
    parser.add_argument("--ts-dir", type=Path, default=Path("../data/time_series"),
                        help="Path to time series directory with *.nc files.")
    parser.add_argument("--link-csv", type=Path, default=Path("../data/camels_link.csv"),
                        help="CSV mapping downstream reach to gage.")
    parser.add_argument("--pairs-out", type=Path, default=Path("output/basin_pair_with_gages.csv"),
                        help="Output CSV for basin pairs with gages.")
    parser.add_argument("--gages-txt", type=Path, default=Path("../data/gage_list.txt"),
                        help="Output TXT file with one gage per line.")
    parser.add_argument("--atts-csv", type=Path, default=Path("../data/camelsatts.csv"),
                        help="CSV with basin attributes.")
    parser.add_argument("--static-out", type=Path, default=Path("../data/attributes/static_attributes.csv"),
                        help="Output CSV for static attributes.")
    parser.add_argument("--skip-rename", action="store_true",
                        help="Skip renaming NetCDF files.")
    parser.add_argument("--dry-run", action="store_true",
                        help="Print planned renames without changing files.")
    args = parser.parse_args()

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

    # Build static attributes
    print("Building static attributes...")
    build_static_attributes(args.pairs_out, args.atts_csv, args.static_out)

    # Rename files
    if not args.skip_rename:
        print("Renaming NetCDF files...")
        rename_nc_files(args.ts_dir, pairs_with_gage, dry_run=args.dry_run)
    else:
        print("Skipping rename step by request.")


if __name__ == "__main__":
    main()
