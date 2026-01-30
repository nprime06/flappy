#!/bin/bash
# slurm job script for RL training
# USE THE SUBMIT WRAPPER!
#
# expected environment variables (set by submit_main.sh):
#   FILE       - Python training script to run (e.g., train_ppo.py)
#   TRAIN_ARGS - arguments for the training script

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/game:${PYTHONPATH:-}"

echo "starting RL training"
echo "  file: $FILE"
echo "  train args: $TRAIN_ARGS"
echo "  slurm job id: $SLURM_JOB_ID"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

python "/home/willzhao/flappy/game/rl/$FILE" $TRAIN_ARGS
