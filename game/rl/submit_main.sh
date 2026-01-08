#!/bin/bash
#SBATCH -p mit_normal_gpu
#SBATCH --job-name=flappy-bot
#SBATCH -N 1
#SBATCH -n 1
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH --gres=gpu:h200:1
#SBATCH -t 06:00:00
#SBATCH --output="/home/willzhao/flappy/game/rl/%x-%j.log"
#SBATCH --error="/home/willzhao/flappy/game/rl/%x-%j.err"

set -euo pipefail

cd /home/willzhao/flappy

module load miniforge
eval "$(conda shell.bash hook)"
conda activate /home/willzhao/flappy/.conda/py31114

export PYTHONUNBUFFERED=1
export PYTHONPATH="/home/willzhao/flappy/game:${PYTHONPATH:-}"

FILE="${FILE:-}"
if [[ -z "$FILE" ]]; then
  echo "ERROR: FILE is required; FILE=train_ppo.py sbatch submit_main.sh" >&2
  exit 1
fi

RUN_PATH="/home/willzhao/flappy/game/rl/${FILE}"
if [[ ! -f "$RUN_PATH" ]]; then
  echo "ERROR: FILE '$FILE' not found at '$RUN_PATH'" >&2
  exit 1
fi

# Optional: pass --run-dir to resume from existing run
RUN_DIR="${RUN_DIR:-}"
if [[ -n "$RUN_DIR" ]]; then
  python "$RUN_PATH" --run-dir "$RUN_DIR" | tee /tmp/train_output_$$.txt
else
  python "$RUN_PATH" | tee /tmp/train_output_$$.txt
fi

# Extract run directory from output and move SLURM logs there
DETECTED_RUN_DIR=$(grep "^Run directory:" /tmp/train_output_$$.txt | head -1 | cut -d' ' -f3)
rm -f /tmp/train_output_$$.txt

if [[ -n "$DETECTED_RUN_DIR" && -d "$DETECTED_RUN_DIR" ]]; then
  SLURM_LOG="/home/willzhao/flappy/game/rl/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.log"
  SLURM_ERR="/home/willzhao/flappy/game/rl/${SLURM_JOB_NAME}-${SLURM_JOB_ID}.err"

  # Copy logs to run directory (can't move since SLURM still writing)
  cp "$SLURM_LOG" "$DETECTED_RUN_DIR/slurm.log" 2>/dev/null || true
  cp "$SLURM_ERR" "$DETECTED_RUN_DIR/slurm.err" 2>/dev/null || true

  echo "Copied SLURM logs to $DETECTED_RUN_DIR"
fi
