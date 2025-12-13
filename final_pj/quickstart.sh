#!/bin/bash

# Quick Start Script for CIFAR-10 Autoencoder
# This demonstrates the complete workflow

set -e

echo "=========================================="
echo "CIFAR-10 Autoencoder - Complete Pipeline"
echo "=========================================="
echo ""

# Configuration
DATA_DIR="data/cifar-10-batches-bin"
BUILD_DIR="build"
EPOCHS=20
BATCH_SIZE=128
LEARNING_RATE=0.001

echo "Configuration:"
echo "  Data Directory: $DATA_DIR"
echo "  Build Directory: $BUILD_DIR"
echo "  Epochs: $EPOCHS"
echo "  Batch Size: $BATCH_SIZE"
echo "  Learning Rate: $LEARNING_RATE"
echo ""

# Check dataset
if [ ! -f "$DATA_DIR/data_batch_1.bin" ]; then
    echo "ERROR: CIFAR-10 dataset not found at $DATA_DIR"
    echo "Please download from: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    exit 1
fi

echo "✓ CIFAR-10 dataset found"

# Check build directory
if [ ! -f "$BUILD_DIR/train_gpu" ]; then
    echo "ERROR: Executables not found. Run ./build.sh first"
    exit 1
fi

echo "✓ Executables found"
echo ""

# Step 1: Train GPU Autoencoder
echo "=========================================="
echo "STEP 1: Training GPU Autoencoder"
echo "=========================================="
echo "This will take ~10 minutes..."
echo ""

cd "$BUILD_DIR"
./train_gpu --data-dir "../$DATA_DIR" \
            --epochs $EPOCHS \
            --batch-size $BATCH_SIZE \
            --lr $LEARNING_RATE

echo ""
echo "✓ Training complete"
echo "Weights saved to: autoencoder_gpu.weights"
echo ""

# Step 2: Extract Features
echo "=========================================="
echo "STEP 2: Extracting Features"
echo "=========================================="
echo "This will take < 20 seconds..."
echo ""

./extract_features --data-dir "../$DATA_DIR" \
                   --output ./cifar10_features.bin \
                   --weights ./autoencoder_gpu.weights

echo ""
echo "✓ Feature extraction complete"
echo "Features saved to: cifar10_features.bin"
echo ""

# Step 3: SVM Classification (if LIBSVM available)
if command -v svm-predict &> /dev/null || [ -f "svm_classifier" ]; then
    echo "=========================================="
    echo "STEP 3: SVM Classification"
    echo "=========================================="
    echo "This will take ~5-10 minutes..."
    echo ""
    
    if [ -f "svm_classifier" ]; then
        ./svm_classifier --data-dir "../$DATA_DIR" \
                        --features ./cifar10_features.bin
    else
        echo "LIBSVM not available, skipping SVM classification"
    fi
    
    echo ""
else
    echo "=========================================="
    echo "STEP 3: SVM Classification (SKIPPED)"
    echo "=========================================="
    echo "LIBSVM not installed. To enable:"
    echo "  macOS: brew install libsvm"
    echo "  Linux: apt-get install libsvm-dev"
    echo ""
fi

cd ..

echo ""
echo "=========================================="
echo "PIPELINE COMPLETE!"
echo "=========================================="
echo ""
echo "Results:"
echo "  Trained weights: $BUILD_DIR/autoencoder_gpu.weights"
echo "  Extracted features: $BUILD_DIR/cifar10_features.bin"
echo ""
echo "Performance Metrics:"
echo "  - Check training time in step 1 output"
echo "  - Check feature extraction speed in step 2 output"
echo "  - Check test accuracy in step 3 output"
echo ""
