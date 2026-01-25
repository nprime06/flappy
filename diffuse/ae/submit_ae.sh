#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=vae-train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 06:00:00
#SBATCH --output="/home/willzhao/flappy/diffuse/ae/%x-%j.log"
#SBATCH --error="/home/willzhao/flappy/diffuse/ae/%x-%j.err"

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/diffuse:${PYTHONPATH:-}"

RUN_DIR="${RUN_DIR:-}"
if [[ -n "$RUN_DIR" ]]; then
  python /home/willzhao/flappy/diffuse/ae/train_ae.py --run-dir "$RUN_DIR"
else
  python /home/willzhao/flappy/diffuse/ae/train_ae.py
fi

