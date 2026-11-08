#!/bin/bash

#SBATCH --account=curso04
#SBATCH --job-name=YOLO26_ALPR
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --partition=gpu
#SBATCH --gres=gpu:MI210:2
#SBATCH --mem=64G

set -euo pipefail

module load rocm/7.2.2

CONDA_BASE="$(conda info --base)"
source "${CONDA_BASE}/etc/profile.d/conda.sh"
conda activate yolo-rocm

export MIOPEN_LOG_LEVEL=3

echo "[INFO] Entorno Conda: ${CONDA_PREFIX}" >&2
rocm-smi || true
python -c 'import sys; print(f"[INFO] Python ejecutado: {sys.executable}")'
python -c 'import torch; print("[INFO] PyTorch:", torch.__version__, "| HIP:", torch.version.hip, "| GPUs:", torch.cuda.device_count())'

python yolo.py
