#!/bin/bash
# SLURM job script for VOD encoding
# This script is submitted by submit_encode.sh - do not run directly
#
# Expected environment variables (set by submit_encode.sh):
#   ENCODE_ARGS - Arguments for encode_vod.py

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/diffuse:${PYTHONPATH:-}"

echo "Starting VOD encoding"
echo "  ENCODE_ARGS: $ENCODE_ARGS"
echo "  SLURM_JOB_ID: $SLURM_JOB_ID"
echo "  CUDA_VISIBLE_DEVICES: ${CUDA_VISIBLE_DEVICES:-not set}"

python /home/willzhao/flappy/diffuse/ngen/encode_vod.py $ENCODE_ARGS
