#!/usr/bin/env python3
"""
Resume NeuralHydrology training from epoch 001 of an existing run.

This script resumes training *in place* for an existing run directory
(e.g., exp/lstm_upstream_0710_131354).

Usage:
------
python resume_train.py CONFIG.yml --run-dir exp/lstm_upstream_0710_131354 --epochs 20

"""

from pathlib import Path
import argparse
import torch

from neuralhydrology.utils.config import Config
from neuralhydrology.training.train import start_training

def find_checkpoint(run_dir: Path, epoch: int = 1) -> Path | None:
    """Prefer root-level model_epochXXX.pt; else search recursively."""
    target = f"model_epoch{epoch:03d}.pt"
    root = run_dir / target
    if root.exists():
        return root
    matches = list(run_dir.rglob(target))
    return matches[0] if matches else None


def main():
    parser = argparse.ArgumentParser(description="Resume NeuralHydrology training from epoch 001")
    parser.add_argument("config", type=Path, help="Path to the YAML config file used for this run")
    parser.add_argument("--run-dir", type=Path, required=True, help="Existing run directory to resume FROM")
    parser.add_argument("--epochs", type=int, required=True, help="Total number of epochs to train up to")
    parser.add_argument("--device", type=str, default="cuda:0", help="Force device, e.g. 'cuda:0' or 'cpu'")
    args = parser.parse_args()

    run_dir: Path = args.run_dir
    if not run_dir.exists():
        raise FileNotFoundError(f"Run directory not found: {run_dir}")

    ckpt = find_checkpoint(run_dir, epoch=1)
    if not ckpt:
        raise FileNotFoundError(f"Could not find model_epoch001.pt under {run_dir}")

    device = args.device or ("cuda:0" if torch.cuda.is_available() else "cpu")

    # Build config
    cfg = Config(args.config)
    cfg._cfg["allow_existing_run_dir"] = True
    cfg._cfg["checkpoint_path"] = str(ckpt)
    cfg._cfg["device"] = device
    cfg._cfg["epochs"] = int(args.epochs)


    # Safe default: write to a child folder to avoid NH's "folder exists" error.
    out_dir = run_dir / "resume_from001"
    out_dir.mkdir(exist_ok=True, parents=True)
    cfg._cfg["run_dir"] = out_dir
    cfg._cfg["experiment_name"] = out_dir.name
    print(f"[resume] Resuming FROM: {ckpt}")
    print(f"[resume] Writing TO:   {out_dir}")

    start_training(cfg)


if __name__ == "__main__":
    main()
