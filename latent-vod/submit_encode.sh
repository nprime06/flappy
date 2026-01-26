#!/bin/bash
# Wrapper script to submit VOD encoding job
#
# USAGE:
#   ./submit_encode.sh --vae-checkpoint PATH              # Required: specify VAE checkpoint
#   ./submit_encode.sh --vae-checkpoint PATH --batch-size 128   # Custom batch size
#   ./submit_encode.sh --vae-checkpoint PATH --time 2:00:00     # Custom time limit

set -euo pipefail

# Defaults
BATCH_SIZE=64
TIME="6:00:00"
VOD_DIR="/home/willzhao/flappy/vod"
OUTPUT_DIR="/home/willzhao/flappy/latent-vod"
VAE_CHECKPOINT=""

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --batch-size)
            BATCH_SIZE="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
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
            echo "Usage: $0 --vae-checkpoint PATH [--batch-size N] [--time HH:MM:SS] [--vod-dir DIR] [--output-dir DIR]"
            exit 1
            ;;
    esac
done

# Validate required arguments
if [[ -z "$VAE_CHECKPOINT" ]]; then
    echo "Error: --vae-checkpoint is required"
    echo "Usage: $0 --vae-checkpoint PATH [--batch-size N] [--time HH:MM:SS] [--vod-dir DIR] [--output-dir DIR]"
    exit 1
fi

# Build encode script arguments
ENCODE_ARGS="--vod-dir $VOD_DIR --output-dir $OUTPUT_DIR --vae-checkpoint $VAE_CHECKPOINT --batch-size $BATCH_SIZE"

# Create log directory (separate from output dir since encoding wipes output dir)
LOG_DIR="/home/willzhao/flappy/latent-vod/encode-logs"
mkdir -p "$LOG_DIR"

echo "Submitting VOD encoding job:"
echo "  Batch size: $BATCH_SIZE"
echo "  Time limit: $TIME"
echo "  VOD dir: $VOD_DIR"
echo "  Output dir: $OUTPUT_DIR"
echo "  VAE checkpoint: $VAE_CHECKPOINT"

# Submit job - single GPU, encoding is fast
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
    /home/willzhao/flappy/latent-vod/encode_vod.sh

echo "Job submitted. Logs will be in: $LOG_DIR"
echo "(Note: output dir $OUTPUT_DIR will be wiped and recreated during encoding)"
