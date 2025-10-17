#!/bin/bash
#SBATCH --job-name=lu1 # edit as needed
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=3
#SBATCH --mem=64G
#SBATCH --gres=gpu:v100:1
#SBATCH --time=11:59:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err


set -euo pipefail
mkdir -p logs exp/transformer

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
CONFIG="02_train_configs/lstm/lstm_upstream.yml"
GAGE_FILE="data/n10/gages/gage_list_clean.txt"
EXP_DIR="exp/lstm"


# Train one model on the full gage list
python neuralhydrology_py/train.py "${CONFIG}" \
  --mode all \
  --gage-file "${GAGE_FILE}" \
  --epochs 1 \
  --exp-dir "${EXP_DIR}" \
  --device cuda:0
