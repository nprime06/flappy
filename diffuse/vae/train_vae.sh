#!/bin/bash
# SLURM job script for VAE training
# This script is submitted by submit_vae.sh - do not run directly
#
# Expected environment variables (set by submit_vae.sh):
#   TRAIN_ARGS - Arguments for train_vae.py

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/diffuse:${PYTHONPATH:-}"

echo "Starting VAE training"
echo "  TRAIN_ARGS: $TRAIN_ARGS"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

python /home/willzhao/flappy/diffuse/vae/train_vae.py $TRAIN_ARGS
