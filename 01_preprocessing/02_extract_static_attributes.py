"""
Static Attributes Builder
-------------------------

This script merges downstream and upstream basin attributes into a single file.

Note: make sure the nwm_attributes csv is in the right path

Inputs
------
- Basin pairs with gages:
    data/n{reach_value}/basin_pair_with_gages.csv
    (columns: downstream, upstream, gage)

- Attributes:
    data/camelsatts.csv or data/nwm_attributes.csv

Outputs
-------
- data/n{reach_value}/attributes/static_attributes.csv

Usage
-----
# Build static attributes for reach-value = 10
python 02_extract_static_attributes.py --reach-value 10 --atts-csv data/nwm_attributes.csv
"""

from pathlib import Path
import argparse
import pandas as pd


def build_static_attributes(pairs_csv: Path,
                            atts_csv: Path,
                            out_csv: Path) -> pd.DataFrame:
    """Merge downstream and upstream attributes and write to static_attributes.csv."""
    # Load attributes
    atts_df = pd.read_csv(atts_csv, dtype={"ID": str})
    crosswalk_df = pd.read_csv(pairs_csv,
                               dtype={"gage": str, "downstream": str, "upstream": str})

    # Downstream attributes
    down_attrs = atts_df.rename(columns={
        "ID": "downstream",
        "Contributing_Area": "Contributing_Area_d",
        "alt": "alt_d",
        "order": "order_d",
        "n":"n_d",
        "So": "So_d",
        "Kchan": "Kchan_d",
        "nCC": "nCC_d",
        "TopWdth": "TopWdth_d",
        "TopWdthCC": "TopWdthCC_d",
        "ChSlp": "ChSlp_d",
        "BtmWdth": "BtmWdth_d",
        "lon": "lon_d",
        "lat": "lat_d",
        "Shape_Area": "Shape_Area_d",
        "Shape_Length": "Shape_Length_d",
        "Reach_Length":"Reach_Length_d"
    })
    merged = crosswalk_df.merge(down_attrs, on="downstream", how="left")

    # Upstream attributes
    up_attrs = atts_df.rename(columns={
        "ID": "upstream",
        "Contributing_Area": "Contributing_Area_u",
        "alt": "alt_u",
        "order": "order_u",
        "n":"n_u",
        "So": "So_u",
        "Kchan": "Kchan_u",
        "nCC": "nCC_u",
        "TopWdth": "TopWdth_u",
        "TopWdthCC": "TopWdthCC_u",
        "ChSlp": "ChSlp_u",
        "BtmWdth": "BtmWdth_u",
        "lon": "lon_u",
        "lat": "lat_u",
        "Shape_Area": "Shape_Area_u",
        "Shape_Length": "Shape_Length_u",
        "Reach_Length":"Reach_Length_u"
    })
    merged = merged.merge(up_attrs, on="upstream", how="left")

    # Select final columns
    static_attributes = merged[[
        "gage",
        "Contributing_Area_u", "alt_u", "order_u", "n_u", "So_u", 
        "Kchan_u", "nCC_u", "TopWdth_u", "TopWdthCC_u", "ChSlp_u", 
        "BtmWdth_u", "lon_u", "lat_u", "Shape_Area_u", "Shape_Length_u", 
        "Reach_Length_u", "Contributing_Area_d", "alt_d", "order_d", "n_d", "So_d", 
        "Kchan_d", "nCC_d", "TopWdth_d", "TopWdthCC_d", "ChSlp_d", 
        "BtmWdth_d", "lon_d", "lat_d", "Shape_Area_d", "Shape_Length_d", 
        "Reach_Length_d"
    ]].drop_duplicates()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    static_attributes.to_csv(out_csv, index=False)
    print(f"Saved {len(static_attributes)} static attribute rows to {out_csv}")

    return static_attributes


def main():
    parser = argparse.ArgumentParser(description="Build static attributes for basin pairs.")
    parser.add_argument("--reach-value", type=int, default=10,
                        help="Reach value (e.g., 10 for n10).")
    parser.add_argument("--atts-csv", type=Path, default=Path("data/nwm_attributes.csv"),
                        help="Path to CAMELS attributes CSV.")
    args = parser.parse_args()

    base_dir = Path(f"data/n{args.reach_value}")
    pairs_csv = base_dir / "basin_pair_with_gages.csv"
    out_csv   = base_dir / "attributes" / "static_attributes.csv"

    if not pairs_csv.exists():
        print(f"Basin pairs file not found: {pairs_csv}")
        return

    build_static_attributes(pairs_csv, args.atts_csv, out_csv)


if __name__ == "__main__":
    main()
