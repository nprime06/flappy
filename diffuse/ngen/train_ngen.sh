#!/bin/bash
# SLURM job script for NGEN training
# This script is submitted by submit_ngen.sh - do not run directly
#
# Expected environment variables (set by submit_ngen.sh):
#   NUM_GPUS   - Number of GPUs to use
#   TRAIN_ARGS - Arguments for train_ngen.py

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/diffuse:${PYTHONPATH:-}"

echo "Starting NGEN training"
echo "  NUM_GPUS: $NUM_GPUS"
echo "  TRAIN_ARGS: $TRAIN_ARGS"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

# Use torchrun for DDP support (works with single GPU too)
torchrun --standalone --nproc_per_node=$NUM_GPUS \
    --show-error-traceback \
    /home/willzhao/flappy/diffuse/ngen/train_ngen.py $TRAIN_ARGS
