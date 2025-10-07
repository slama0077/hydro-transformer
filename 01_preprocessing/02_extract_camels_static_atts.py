"""
Static Attributes Builder
-------------------------

This script merges downstream and upstream basin attributes into a single file.

Inputs
------
- Basin pairs with gages:
    data/n{reach_value}/basin_pair_with_gages.csv
    (columns: downstream, upstream, gage)

- Attributes:
    data/camelsatts.csv
    (columns: id, basin_length, basin_area, reach_length)

Outputs
-------
- data/n{reach_value}/attributes/static_attributes.csv
    Columns: gage, basin_length_d, basin_area_d, reach_length_d,
             basin_length_u, basin_area_u, reach_length_u

Usage
-----
# Build static attributes for reach-value = 10
python 02_static_attributes.py --reach-value 10
"""

from pathlib import Path
import argparse
import pandas as pd


def build_static_attributes(pairs_csv: Path,
                            atts_csv: Path,
                            out_csv: Path) -> pd.DataFrame:
    """Merge downstream and upstream attributes and write to static_attributes.csv."""
    # Load attributes
    atts_df = pd.read_csv(atts_csv, dtype={"id": str})
    crosswalk_df = pd.read_csv(pairs_csv,
                               dtype={"gage": str, "downstream": str, "upstream": str})

    # Downstream attributes
    down_attrs = atts_df.rename(columns={
        "id": "downstream",
        "basin_length": "basin_length_d",
        "basin_area": "basin_area_d",
        "reach_length": "reach_length_d"
    })
    merged = crosswalk_df.merge(down_attrs, on="downstream", how="left")

    # Upstream attributes
    up_attrs = atts_df.rename(columns={
        "id": "upstream",
        "basin_length": "basin_length_u",
        "basin_area": "basin_area_u",
        "reach_length": "reach_length_u"
    })
    merged = merged.merge(up_attrs, on="upstream", how="left")

    # Select final columnsr_len
    static_attributes = merged[[
        "gage",
        "basin_length_d", "basin_length_u", "basin_area_d","basin_area_u",
        "reach_length_u", "reach_length_d"
    ]].drop_duplicates()

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    static_attributes.to_csv(out_csv, index=False)
    print(f"Saved {len(static_attributes)} static attribute rows to {out_csv}")

    return static_attributes


def main():
    parser = argparse.ArgumentParser(description="Build static attributes for basin pairs.")
    parser.add_argument("--reach-value", type=int, default=10,
                        help="Reach value (e.g., 10 for n10).")
    parser.add_argument("--atts-csv", type=Path, default=Path("data/camelsatts.csv"),
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
