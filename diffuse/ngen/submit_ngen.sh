#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=ngen-train
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 6:00:00
#SBATCH --output="/home/willzhao/flappy/diffuse/ngen/%x-%j.log"
#SBATCH --error="/home/willzhao/flappy/diffuse/ngen/%x-%j.err"

# USAGE:
# Single GPU (default):
#   sbatch submit_ngen.sh
#
# Multi-GPU (must also update SBATCH --gres and -c):
#   NUM_GPUS=4 sbatch submit_ngen.sh
#
# Resume training:
#   RUN_DIR=/path/to/run sbatch submit_ngen.sh
#
# Reflow training:
#   REFLOW=/path/to/checkpoint.pt sbatch submit_ngen.sh

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/diffuse:${PYTHONPATH:-}"

# Configuration
NUM_GPUS="${NUM_GPUS:-1}"
RUN_DIR="${RUN_DIR:-}"
REFLOW="${REFLOW:-}"

ARGS=""
if [[ -n "$RUN_DIR" ]]; then
    ARGS="$ARGS --run-dir $RUN_DIR"
fi
if [[ -n "$REFLOW" ]]; then
    ARGS="$ARGS --reflow $REFLOW"
fi

# Use torchrun for DDP support (works with single GPU too)
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    /home/willzhao/flappy/diffuse/ngen/train_ngen.py $ARGS
