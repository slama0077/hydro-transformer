# -------------------------------------------------------------------
# This script removes invalid gage IDs from basin chunk files.
#
# Context
# -------
# Observed errors during preprocessing/evaluation:
#
# 1) Missing upstream attributes:
# RuntimeError: The following basins/attributes are NaN, which can't be used as input:
# 01123000: ['basin_length_u', 'basin_area_u']
# 02177000: ['basin_length_u', 'basin_area_u']
# 09447800: ['basin_length_u', 'basin_area_u']
# 13337000: ['basin_length_u', 'basin_area_u']
# 14141500: ['basin_length_u', 'basin_area_u']
# 14185000: ['basin_length_u', 'basin_area_u']
#
# 2) No valid samples in train period:
# These basins do not have a single valid sample in the train period:
# ['01121000', '01187300', '01350000', '01365000', '01435000', '02011400',
#  '02178400', '06221400', '06409000', '09107000', '10336660', '11124500',
#  '11141280', '12115000', '14139800', '14185900']
#
# - Input:
#   * Folder: ../data/gages
#   * Files: basin_chunk_000.txt through basin_chunk_004.txt
#   * Each file contains one gage ID per line (8-digit strings).
#   * A predefined list of invalid gage IDs to be removed.
#
# - Process:
#   1. Iterates over basin_chunk_000.txt to basin_chunk_004.txt.
#   2. Reads all gage IDs from each file.
#   3. Filters out any gages listed in the invalid_gages set.
#   4. Overwrites the file with the cleaned list.
#   5. Prints a summary of how many gages were removed.
#
# - Output:
#   * Updated chunk files with invalid gages removed.
#   * Console messages showing original vs cleaned counts.
# -------------------------------------------------------------------

from pathlib import Path

# Folder containing the basin_chunk files
chunk_dir = Path("../data/gages")

# Invalid gages (zero-padded to 8 digits)
invalid_gages = {
    # From missing static attributes
    "09447800", "14185000", "13337000", "14141500", "02177000", "01123000",
    # From no valid samples in train period
    "01121000", "01187300", "01350000", "01365000", "01435000",
    "02011400", "02178400", "06221400", "06409000", "09107000",
    "10336660", "11124500", "11141280", "12115000", "14139800", "14185900"
}

# Loop through basin_chunk_000.txt ... basin_chunk_004.txt
for i in range(5):
    fpath = chunk_dir / f"basin_chunk_{i}.txt"  # keep consistent zero-padding
    if not fpath.exists():
        print(f"Skipping {fpath}, does not exist")
        continue

    # Read gages
    with fpath.open() as f:
        gages = [line.strip() for line in f if line.strip()]

    # Normalize to 8-digit strings
    gages = [g.zfill(8) for g in gages]

    # Filter out invalid gages
    filtered = [g for g in gages if g not in invalid_gages]

    # Overwrite file with cleaned list
    fpath.write_text("\n".join(filtered) + "\n", encoding="utf-8")

    print(f"Updated {fpath}: {len(gages)} -> {len(filtered)} gages (removed {len(gages)-len(filtered)})")
