#!/usr/bin/env python3
"""
Train NeuralHydrology models either
1) in **chunks** of gages (multiple TXT files like basin_chunk_000.txt …), or
2) on the **entire** gage list (a single TXT file).

Why
----
Chunked training helps on low-memory machines by limiting the number of basins
loaded at once. Whole-set training is convenient when resources allow.

Inputs
------
- YAML config: Base NeuralHydrology configuration file.
- Chunk files (for chunk mode): TXT files with one gage ID per line, typically
  named `basin_chunk_X.txt` under a chunk directory.
- Whole gage file (for all mode): A single TXT file with one gage ID per line.

Modes
-----
- chunks: iterate basin_chunk_*.txt files and (optionally) chain checkpoints from
  the previous chunk to the next.
- all: train once using the whole gage list file.


Outputs
-------
- Each run writes to: <exp-dir>/<experiment_name>/
- experiment_name is derived from the base config name + timestamp (and chunk index in chunk mode).

Usage Examples
--------------
# Train in chunks (default mode) from data/gages/
python train.py transformer_combined.yml

# Train in chunks with 5 epochs per chunk, store runs in a custom folder
python train.py transformer_combined.yml --epochs 5 --exp-dir exp/custom_runs

# Train all gages from a single list
python train.py transformer_upstream.yml --mode all --gage-file data/gage_list_clean.txt --epochs 3

# Chain checkpoints across chunks (resume each chunk from the previous chunk’s final epoch)
python train.py transformer_combined.yml --resume-chain
"""

from pathlib import Path
from datetime import datetime
import argparse
import torch

from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training


def train_one(cfg: Config, exp_dir: Path, exp_name: str, device: str):
    cfg._cfg["experiment_name"] = exp_name
    cfg._cfg["run_dir"] = exp_dir / exp_name
    cfg._cfg["device"] = device
    start_training(cfg)


def main():
    parser = argparse.ArgumentParser(description="Chunked or whole-set training for NeuralHydrology")
    parser.add_argument("config", type=str, help="Base YAML config file")
    parser.add_argument("--mode", choices=["chunks", "all"], default="chunks",
                        help="Train by chunks (multiple TXT files) or on the whole gage list (single TXT).")
    parser.add_argument("--epochs", type=int, default=1, help="Epochs per run (per chunk in chunk mode).")
    parser.add_argument("--exp-dir", type=Path, default=Path("exp/transformer"),
                        help="Experiment base output directory.")
    parser.add_argument("--chunk-dir", type=Path, default=Path("data/gages"),
                        help="Directory containing basin_chunk_*.txt files (chunk mode).")
    parser.add_argument("--chunk-pattern", type=str, default="basin_chunk_*.txt",
                        help="Glob pattern for chunk files inside --chunk-dir.")
    parser.add_argument("--gage-file", type=Path, default=Path("data/gage_list_clean.txt"),
                        help="Whole gage list file (all mode).")
    parser.add_argument("--resume-chain", action="store_true",
                        help="In chunk mode, resume each chunk from the previous chunk’s final checkpoint.")
    parser.add_argument("--device", type=str, default=None,
                        help="Force device, e.g., 'cuda:0' or 'cpu'. Defaults to auto.")
    args = parser.parse_args()

    timestamp = datetime.now().strftime("%d%m_%H%M%S")
    base_config = Path(args.config)
    exp_dir: Path = args.exp_dir
    exp_dir.mkdir(parents=True, exist_ok=True)

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    if args.mode == "chunks":
        chunk_files = sorted((args.chunk_dir).glob(args.chunk_pattern))
        if not chunk_files:
            raise FileNotFoundError(f"No chunk files found in {args.chunk_dir} matching '{args.chunk_pattern}'")

        for i, basin_file in enumerate(chunk_files):
            print(f"\n=== Training on chunk {i}: {basin_file.name} ===")
            cfg = Config(base_config)

            # Override basin files per chunk
            cfg._cfg["train_basin_file"] = Path(basin_file)
            cfg._cfg["validation_basin_file"] = Path(basin_file)
            cfg._cfg["test_basin_file"] = Path(basin_file)

            # Training params
            cfg._cfg["epochs"] = int(args.epochs)

            # Experiment naming
            exp_name = f"{base_config.stem}_{timestamp}_{i}"

            # Optional: resume from previous chunk's final checkpoint
            if args.resume_chain and i > 0:
                prev_name = f"{base_config.stem}_{timestamp}_{i-1}"
                prev_run = exp_dir / prev_name
                prev_ckpt = prev_run / f"model_epoch{int(args.epochs):03d}.pt"
                if prev_ckpt.exists():
                    cfg._cfg["checkpoint_path"] = str(prev_ckpt)
                    print(f"Resuming chunk {i} from checkpoint: {prev_ckpt}")
                else:
                    print(f"[WARN] Expected checkpoint not found for chunk {i}: {prev_ckpt}")

            train_one(cfg, exp_dir, exp_name, device)

    else:  # args.mode == "all"
        basin_file = args.gage_file
        if not basin_file.exists():
            raise FileNotFoundError(f"Gage file not found: {basin_file}")

        print(f"\n=== Training on ALL gages from: {basin_file} ===")
        cfg = Config(base_config)

        cfg._cfg["train_basin_file"] = Path(basin_file)
        cfg._cfg["validation_basin_file"] = Path(basin_file)
        cfg._cfg["test_basin_file"] = Path(basin_file)

        cfg._cfg["epochs"] = int(args.epochs)

        exp_name = f"{base_config.stem}_{timestamp}"
        train_one(cfg, exp_dir, exp_name, device)


if __name__ == "__main__":
    main()
