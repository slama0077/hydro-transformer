#!/bin/bash
#SBATCH --job-name=test_lu1 # eidt as needed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=5
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=11:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -euo pipefail
mkdir -p logs exp/lstm

# Activate your env
module load Anaconda3
cd /home/toakinrele/toakinrele/hydro-transformer

#source ~/miniconda3/etc/profile.d/conda.sh
source activate neuralhydrology
#conda activate /home/toakinrele/toakinrele/hydro-transformer/toakinrele/.conda/envs/neuralhydrology

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
GAGE_FILE="data/n5/gages/gage_list_clean.txt"
RUN_DIR="exp/lstm/lstm_upstream_0610_140026"


# Evaluate model on the full gage list
python neuralhydrology_py/evaluate.py \
  --gage-file "${GAGE_FILE}" \
  --run-dir "${RUN_DIR}"
