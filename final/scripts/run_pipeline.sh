#!/usr/bin/env bash
#
# Full pipeline: train -> evaluate -> ablation
#
# Usage:
#   # Quick test (small batch, few epochs):
#   bash scripts/run_pipeline.sh --quick
#
#   # Full training run:
#   bash scripts/run_pipeline.sh
#
#   # Ablation study on a single axis:
#   bash scripts/run_pipeline.sh --ablation backbone
#
#   # A100 benchmark settings:
#   bash scripts/run_pipeline.sh --a100

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG="configs/default.yaml"
CHECKPOINT_DIR="checkpoints"
RESULTS_DIR="results"
MODE="full"
ABLATION_AXIS=""

while [[ $# -gt 0 ]]; do
    case $1 in
        --quick)
            MODE="quick"
            shift ;;
        --a100)
            MODE="a100"
            shift ;;
        --ablation)
            MODE="ablation"
            ABLATION_AXIS="${2:-}"
            shift; shift ;;
        --config)
            CONFIG="$2"
            shift; shift ;;
        *)
            echo "Unknown option: $1"
            exit 1 ;;
    esac
done

echo "=== BreastDCEDL ViT Pipeline ==="
echo "Mode: $MODE"
echo "Config: $CONFIG"
echo ""

# Quick test: reduced epochs and batch size for local laptop testing
if [ "$MODE" = "quick" ]; then
    echo "--- Quick test mode (3 epochs, batch=4) ---"
    python scripts/train.py \
        --config "$CONFIG" \
        --epochs 3 \
        --batch-size 4 \
        --checkpoint-dir "${CHECKPOINT_DIR}/quick"

    if [ -f "${CHECKPOINT_DIR}/quick/best.pth" ]; then
        python scripts/evaluate.py \
            --config "$CONFIG" \
            --weights "${CHECKPOINT_DIR}/quick/best.pth" \
            --output "${RESULTS_DIR}/quick"
    fi

# A100 benchmark: larger batch, full training
elif [ "$MODE" = "a100" ]; then
    echo "--- A100 benchmark mode ---"
    python scripts/train.py \
        --config "$CONFIG" \
        --batch-size 32 \
        --checkpoint-dir "${CHECKPOINT_DIR}/a100"

    python scripts/evaluate.py \
        --config "$CONFIG" \
        --weights "${CHECKPOINT_DIR}/a100/best.pth" \
        --output "${RESULTS_DIR}/a100"

# Ablation study
elif [ "$MODE" = "ablation" ]; then
    if [ -n "$ABLATION_AXIS" ]; then
        echo "--- Ablation: $ABLATION_AXIS ---"
        python scripts/ablation.py \
            --config "$CONFIG" \
            --axis "$ABLATION_AXIS" \
            --output "${RESULTS_DIR}/ablation"
    else
        echo "--- Full ablation study ---"
        python scripts/ablation.py \
            --config "$CONFIG" \
            --output "${RESULTS_DIR}/ablation"
    fi

# Full training
else
    echo "--- Full training ---"
    python scripts/train.py \
        --config "$CONFIG" \
        --checkpoint-dir "$CHECKPOINT_DIR"

    python scripts/evaluate.py \
        --config "$CONFIG" \
        --weights "${CHECKPOINT_DIR}/best.pth" \
        --output "$RESULTS_DIR"
fi

echo ""
echo "=== Pipeline complete ==="
