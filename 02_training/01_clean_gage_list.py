from pathlib import Path

"""
Purpose
-------
Removes invalid gage IDs from a master list to prevent training failures due to
missing attributes or lack of valid samples in the train period.

Context
-------
Observed errors during preprocessing/evaluation:

1) Missing upstream attributes:
RuntimeError: The following basins/attributes are NaN, which can't be used as input:
01123000: ['basin_length_u', 'basin_area_u']
02177000: ['basin_length_u', 'basin_area_u']
09447800: ['basin_length_u', 'basin_area_u']
13337000: ['basin_length_u', 'basin_area_u']
14141500: ['basin_length_u', 'basin_area_u']
14185000: ['basin_length_u', 'basin_area_u']

2) No valid samples in train period:
These basins do not have a single valid sample in the train period:
['01121000', '01187300', '01350000', '01365000', '01435000', '02011400',
 '02178400', '06221400', '06409000', '09107000', '10336660', '11124500',
 '11141280', '12115000', '14139800', '14185900']

Inputs
------
- input_file: TXT with one gage ID per line (zero padded to 8 digits recommended).

Process
-------
1. Read all gage IDs from input_file.
2. Remove any IDs listed in invalid_gages.
3. Write the cleaned list to output_file.

Outputs
-------
- output_file: TXT with invalid gages removed.

Usage
-----
- Adjust input_file and output_file paths if needed and run the script.
"""

# Configure paths
input_file = Path("../data/gage_list.txt")
output_file = Path("../data/gage_list_clean.txt")

# Invalid gages identified from RuntimeErrors and train-period checks
invalid_gages = {
    # From missing upstream attributes
    "01123000", "02177000", "09447800",
    "13337000", "14141500", "14185000",

    # From no valid samples in train period
    "01121000", "01187300", "01350000", "01365000", "01435000",
    "02011400", "02178400", "06221400", "06409000", "09107000",
    "10336660", "11124500", "11141280", "12115000", "14139800", "14185900"
}

# Load, strip, and zero pad to ensure consistent 8-digit formatting
with input_file.open("r", encoding="utf-8") as f:
    gages = [line.strip() for line in f if line.strip()]

gages_norm = [g.zfill(8) for g in gages]

# Filter out invalids
cleaned = [g for g in gages_norm if g not in invalid_gages]

# Write cleaned list
output_file.parent.mkdir(parents=True, exist_ok=True)
output_file.write_text("\n".join(cleaned) + "\n", encoding="utf-8")

print(f"Loaded {len(gages_norm)} gages from {input_file}")
print(f"Removed {len(gages_norm) - len(cleaned)} invalid gages")
print(f"Wrote {len(cleaned)} gages to {output_file}")
