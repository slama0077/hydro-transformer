"""
This script splits the list of gages into smaller chunks to make training more manageable 
and to reduce the risk of crashes from processing too many gages at once. 
The resulting files are renamed using the prefix 'basin' to simplify file naming conventions.

Input - gage_list.txt
Output - a folder with chunks in .txt
"""



from pathlib import Path

# Input and output
input_file = Path("../data/gage_list.txt")
out_dir = Path("../data/gages")
chunk_size = 120

# Read gages
with input_file.open() as f:
    gages = [line.strip().zfill(8) for line in f if line.strip()]

# Make output folder
out_dir.mkdir(parents=True, exist_ok=True)

# Write chunks
for i in range(0, len(gages), chunk_size):
    chunk = gages[i:i+chunk_size]
    out_file = out_dir / f"basin_chunk_{i//chunk_size:01d}.txt"
    out_file.write_text("\n".join(chunk) + "\n", encoding="utf-8")
    print(f"Wrote {len(chunk)} gages to {out_file}")
