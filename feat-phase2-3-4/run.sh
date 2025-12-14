#!/bin/bash
# AutoEncoder Phase 3 - Run Script

set -e

BUILD_DIR="build"
MODE="train"
EPOCHS=20
SAMPLES=0
DATA_DIR="./data"
WEIGHTS_FILE=""
SAVE_WEIGHTS=""

show_help() {
    cat << 'EOF'
Usage: ./run.sh [COMMAND] [OPTIONS]

COMMANDS:
    train           Train autoencoder only (default)
    pipeline        Full: train autoencoder -> train SVM -> evaluate
    train-svm       Train SVM using pre-trained encoder weights
    evaluate        Evaluate with pre-trained weights

OPTIONS:
    --epochs N          Training epochs (default: 20)
    --samples N         Limit samples, 0=all (default: 0)
    --data-dir PATH     CIFAR-10 data directory (default: ./data)
    --weights PATH      Input weights file for SVM training/evaluation
    --save PATH         Save trained weights to this path
    --help              Show this help

EXAMPLES:
    ./run.sh train --epochs 20
    ./run.sh pipeline --epochs 10 --samples 1000
    ./run.sh train-svm --weights autoencoder_gpu.weights
EOF
}

# Parse command
case ${1:-} in
    train) MODE="train"; shift ;;
    pipeline) MODE="pipeline"; shift ;;
    train-svm) MODE="train-svm"; shift ;;
    evaluate) MODE="evaluate"; shift ;;
    --*) ;;
    -h|--help) show_help; exit 0 ;;
    "") ;;
    *) echo "Unknown command: $1"; show_help; exit 1 ;;
esac

# Parse options
while [[ $# -gt 0 ]]; do
    case $1 in
        --epochs) EPOCHS="$2"; shift 2 ;;
        --samples) SAMPLES="$2"; shift 2 ;;
        --data-dir) DATA_DIR="$2"; shift 2 ;;
        --weights) WEIGHTS_FILE="$2"; shift 2 ;;
        --save) SAVE_WEIGHTS="$2"; shift 2 ;;
        --help|-h) show_help; exit 0 ;;
        *) echo "Unknown option: $1"; show_help; exit 1 ;;
    esac
done

# Check executables
GPU_TRAIN="$BUILD_DIR/bin/gpu_train"
FULL_PIPELINE="$BUILD_DIR/bin/full_pipeline"

if [[ ! -f "$GPU_TRAIN" ]] && [[ ! -f "$FULL_PIPELINE" ]]; then
    echo "Error: Executables not found. Run ./build.sh first."
    exit 1
fi

# Build arguments
ARGS="--epochs $EPOCHS --data $DATA_DIR"
[[ $SAMPLES -gt 0 ]] && ARGS="$ARGS --max-train $SAMPLES --max-test $((SAMPLES / 5))"
[[ -n "$WEIGHTS_FILE" ]] && ARGS="$ARGS --weights $WEIGHTS_FILE"
[[ -n "$SAVE_WEIGHTS" ]] && ARGS="$ARGS --save $SAVE_WEIGHTS"

echo "[$MODE] epochs=$EPOCHS samples=$SAMPLES"
echo ""

case "$MODE" in
    train)
        echo "=== Training Autoencoder ==="
        $GPU_TRAIN $ARGS
        ;;
    pipeline)
        echo "=== Full Pipeline (Autoencoder + SVM) ==="
        $FULL_PIPELINE $ARGS
        ;;
    train-svm)
        if [[ -z "$WEIGHTS_FILE" ]]; then
            echo "Error: --weights required for train-svm mode"
            exit 1
        fi
        echo "=== Training SVM with encoder: $WEIGHTS_FILE ==="
        $FULL_PIPELINE $ARGS --skip-train
        ;;
    evaluate)
        if [[ -z "$WEIGHTS_FILE" ]]; then
            echo "Error: --weights required for evaluate mode"
            exit 1
        fi
        echo "=== Evaluating with weights: $WEIGHTS_FILE ==="
        $FULL_PIPELINE $ARGS --eval-only
        ;;
esac
