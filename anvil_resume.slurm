#!/bin/bash
#SBATCH --job-name=lc2 # edit as needed
#SBATCH --account=ees240087          # <-- REQUIRED
#SBATCH --qos=gpu                    # <-- REQUIRED for GPU jobs
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=32G                    # pick a value; uncommented
#SBATCH --gpus=1                     # Anvil usually accepts --gpus or --gres=gpu:1
#SBATC H --gres=gpu:1                # (alt syntax) use one of these, not both
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -euo pipefail
mkdir -p logs exp/lstm

# Activate your env
module load anaconda
cd /home/x-takinrele/hydro-transformer

source activate neuralhydrology
#conda activate /home/x-takinrele/.conda/envs/2024.02-py311/neuralhydrology

# CPU threading hygiene
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# Quick sanity
nvidia-smi || true
python - <<'PY'
import torch
print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())
print("GPU:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else None)
PY

# Paths - edit as needed
CONFIG="exp/lstm/lstm_combined_1310_141159/config.yml"
RUN_DIR="exp/lstm/lstm_combined_1310_141159"


# Train one model on the full gage list
python neuralhydrology_py/resume_train.py "${CONFIG}" \
  --run-dir "${RUN_DIR}" \
  --epochs 1 \
  --device cuda:0
