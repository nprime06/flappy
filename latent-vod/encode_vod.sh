#!/bin/bash
# slurm job script for vod encoding
# USE THE SUBMIT WRAPPER!
#
# expected environment variables (set by submit_encode.sh):
#   ENCODE_ARGS - arguments for encode_vod.py

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/latent-vod:${PYTHONPATH:-}"

echo "starting vod encoding"
echo "  encode args: $ENCODE_ARGS"
echo "  slurm job id: $SLURM_JOB_ID"
echo "  cuda visible devices: ${CUDA_VISIBLE_DEVICES:-not set}"

python /home/willzhao/flappy/latent-vod/encode_vod.py $ENCODE_ARGS
