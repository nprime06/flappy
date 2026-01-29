#!/bin/bash
# wrapper script to submit vod encoding job
#
# USAGE:
#   ./submit_encode.sh --vae-checkpoint PATH --vod-dir DIR  # use full absolute paths

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
VOD_DIR=""
OUTPUT_DIR="$SCRIPT_DIR"
VAE_CHECKPOINT=""
BATCH_SIZE=256
TIME="6:00:00"

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --vod-dir)
            VOD_DIR="$2"
            shift 2
            ;;
        --output-dir)
            OUTPUT_DIR="$2"
            shift 2
            ;;
        --vae-checkpoint)
            VAE_CHECKPOINT="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --vae-checkpoint PATH --vod-dir DIR [--output-dir DIR]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VAE_CHECKPOINT" ]]; then
    echo "Error: --vae-checkpoint is required"
    echo "Usage: $0 --vae-checkpoint PATH --vod-dir DIR [--output-dir DIR]"
    exit 1
fi
if [[ -z "$VOD_DIR" ]]; then
    echo "Error: --vod-dir is required"
    echo "Usage: $0 --vae-checkpoint PATH --vod-dir DIR [--output-dir DIR]"
    exit 1
fi

ENCODE_ARGS="--vod-dir $VOD_DIR --output-dir $OUTPUT_DIR --vae-checkpoint $VAE_CHECKPOINT --batch-size $BATCH_SIZE"

TIMESTAMP=$(date +%Y%m%d_%H%M%S)
LOG_BASE_DIR="${SCRIPT_DIR}/encode-logs"
LOG_DIR="${LOG_BASE_DIR}/encode_${TIMESTAMP}"
mkdir -p "$LOG_DIR"

sbatch \
    --job-name=encode-vod \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=16 \
    --mem=128G \
    --gres=gpu:h200:1 \
    --time=$TIME \
    --output="${LOG_DIR}/encode-slurm-%j.log" \
    --error="${LOG_DIR}/encode-slurm-%j.err" \
    --export=ALL,ENCODE_ARGS="$ENCODE_ARGS" \
    "${SCRIPT_DIR}/encode_vod.sh"

echo "logs in $LOG_DIR"
echo "note: $OUTPUT_DIR will be wiped"