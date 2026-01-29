#!/bin/bash
# slurm job script for ngen training
# USE THE SUBMIT WRAPPER!
#
# expected environment variables (set by submit_ngen.sh):
#   NUM_GPUS   - number of gpus to use
#   TRAIN_ARGS - arguments for train_ngen.py

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/diffuse:${PYTHONPATH:-}"

echo "starting ngen training"
echo "  num gpus: $NUM_GPUS"
echo "  train args: $TRAIN_ARGS"
echo "  slurm job id: $SLURM_JOB_ID"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

torchrun --standalone --nproc_per_node=$NUM_GPUS \
    /home/willzhao/flappy/diffuse/ngen/train_ngen.py $TRAIN_ARGS
