#!/bin/bash
# wrapper to submit ngen training job
#
# USAGE:
#   ./submit_ngen.sh --latent-vod /path/to/dir
#   ./submit_ngen.sh --latent-vod /path/to/dir --tag k8-dyn03 --context-size 8 --dynamics-loss-weight 0.3
#   ./submit_ngen.sh --latent-vod /path/to/dir --gpus 2
#   ./submit_ngen.sh --latent-vod /path/to/dir --run-dir /path/to/run    # resume existing run (use full absolute path)
#   ./submit_ngen.sh --latent-vod /path/to/dir --reflow /path/to/ckpt    # reflow

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_DIR=""
TAG=""
REFLOW=""
LATENT_VOD=""
NUM_GPUS=1
CPUS_PER_GPU=8
MEM_PER_GPU=128
TIME="6:00:00"
EXTRA_TRAIN_ARGS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --tag|--run-tag)
            TAG="$2"
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --run-tag $2"
            shift 2
            ;;
        --gpus)
            NUM_GPUS="$2"
            shift 2
            ;;
        --time)
            TIME="$2"
            shift 2
            ;;
        --run-dir)
            RUN_DIR="$2"
            shift 2
            ;;
        --reflow)
            REFLOW="$2"
            shift 2
            ;;
        --latent-vod)
            LATENT_VOD="$2"
            shift 2
            ;;
        --context-size)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --context-size $2"
            shift 2
            ;;
        --hidden-channels)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --hidden-channels $2"
            shift 2
            ;;
        --num-layers)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --num-layers $2"
            shift 2
            ;;
        --embed-dim)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --embed-dim $2"
            shift 2
            ;;
        --act-embed-dim)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --act-embed-dim $2"
            shift 2
            ;;
        --num-aug-bins)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --num-aug-bins $2"
            shift 2
            ;;
        --dynamics-dim)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --dynamics-dim $2"
            shift 2
            ;;
        --epochs)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --epochs $2"
            shift 2
            ;;
        --batch-size)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --batch-size $2"
            shift 2
            ;;
        --lr)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --lr $2"
            shift 2
            ;;
        --max-aug-std)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --max-aug-std $2"
            shift 2
            ;;
        --cfg-dropout-prob)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --cfg-dropout-prob $2"
            shift 2
            ;;
        --done-loss-weight)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --done-loss-weight $2"
            shift 2
            ;;
        --done-t-power)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --done-t-power $2"
            shift 2
            ;;
        --dynamics-loss-weight)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --dynamics-loss-weight $2"
            shift 2
            ;;
        --checkpoint-interval)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --checkpoint-interval $2"
            shift 2
            ;;
        --num-workers)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --num-workers $2"
            shift 2
            ;;
        --reflow-steps)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --reflow-steps $2"
            shift 2
            ;;
        --filter-target-action)
            EXTRA_TRAIN_ARGS="$EXTRA_TRAIN_ARGS --filter-target-action $2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1"
            echo "Usage: $0 --latent-vod DIR [--tag NAME] [--context-size K] [--dynamics-loss-weight W] [--gpus N] [--run-dir DIR] [--reflow CKPT]"
            exit 1
            ;;
    esac
done

NUM_CPUS=$((NUM_GPUS * CPUS_PER_GPU))
TOTAL_MEM=$((NUM_GPUS * MEM_PER_GPU))

if [[ -z "$LATENT_VOD" ]]; then
    echo "Error: --latent-vod is required"
    echo "Usage: $0 --latent-vod DIR [--tag NAME] [--context-size K] [--dynamics-loss-weight W] [--gpus N] [--run-dir DIR] [--reflow CKPT]"
    exit 1
fi

# create run directory if not resuming
RUNS_DIR="${SCRIPT_DIR}/runs"
if [[ -z "$RUN_DIR" ]]; then
    TIMESTAMP=$(date +%Y%m%d_%H%M%S)
    if [[ -n "$REFLOW" ]]; then
        PREFIX="reflow"
    else
        PREFIX="ngen"
    fi
    if [[ -n "$TAG" ]]; then
        RUN_DIR="${RUNS_DIR}/${PREFIX}_${TIMESTAMP}_${TAG}"
    else
        RUN_DIR="${RUNS_DIR}/${PREFIX}_${TIMESTAMP}"
    fi
fi
mkdir -p "$RUN_DIR"

# build train script arguments
TRAIN_ARGS="--run-dir $RUN_DIR"
if [[ -n "$REFLOW" ]]; then
    TRAIN_ARGS="$TRAIN_ARGS --reflow $REFLOW"
fi
TRAIN_ARGS="$TRAIN_ARGS --latent-vod $LATENT_VOD$EXTRA_TRAIN_ARGS"

sbatch \
    --job-name=ngen-train \
    --partition=mit_normal_gpu \
    --nodes=1 \
    --ntasks=1 \
    --cpus-per-task=$NUM_CPUS \
    --mem=${TOTAL_MEM}G \
    --gres=gpu:h200:$NUM_GPUS \
    --time=$TIME \
    --output="${RUN_DIR}/slurm-%j.log" \
    --error="${RUN_DIR}/slurm-%j.err" \
    --export=ALL,NUM_GPUS=$NUM_GPUS,TRAIN_ARGS="$TRAIN_ARGS" \
    "${SCRIPT_DIR}/train_ngen.sh"

echo "logs in $RUN_DIR"
