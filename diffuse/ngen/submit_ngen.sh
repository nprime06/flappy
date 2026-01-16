#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=ngen-train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 9:00:00
#SBATCH --output="/home/willzhao/flappy/diffuse/ngen/%x-%j.log"
#SBATCH --error="/home/willzhao/flappy/diffuse/ngen/%x-%j.err"

# USAGE: 
# Standard flow matching:
# sbatch submit_ngen.sh

# Resume training:
# RUN_DIR=/home/willzhao/flappy/diffuse/ngen/runs/ngen_xxx sbatch submit_ngen.sh

# Reflow (rectified flow) training:
# REFLOW=/home/willzhao/flappy/diffuse/ngen/runs/ngen_xxx/checkpoints/latest.pt sbatch submit_ngen.sh

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/diffuse:${PYTHONPATH:-}"

# Optional: specify run directory to resume
RUN_DIR="${RUN_DIR:-}"
# Optional: specify reflow checkpoint for rectified flow training
REFLOW="${REFLOW:-}"

ARGS=""
if [[ -n "$RUN_DIR" ]]; then
  ARGS="$ARGS --run-dir $RUN_DIR"
fi
if [[ -n "$REFLOW" ]]; then
  ARGS="$ARGS --reflow $REFLOW"
fi

python /home/willzhao/flappy/diffuse/ngen/train_ngen.py $ARGS
