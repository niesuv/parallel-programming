#!/bin/bash

# CIFAR-10 Autoencoder Build Script
# This script automates the build process

set -e  # Exit on error

echo "========================================"
echo "CIFAR-10 Autoencoder Build Script"
echo "========================================"

# Check if cmake is installed
if ! command -v cmake &> /dev/null; then
    echo "ERROR: CMake is not installed"
    echo "Install with: brew install cmake (macOS) or apt-get install cmake (Linux)"
    exit 1
fi

# Check if CUDA is available
if ! command -v nvcc &> /dev/null; then
    echo "ERROR: CUDA is not installed or not in PATH"
    echo "Install from: https://developer.nvidia.com/cuda-downloads"
    exit 1
fi

echo "CMake version: $(cmake --version | head -1)"
echo "CUDA version: $(nvcc --version | grep release)"

# Check dataset
if [ ! -f "data/cifar-10-batches-bin/data_batch_1.bin" ]; then
    echo ""
    echo "WARNING: CIFAR-10 dataset not found!"
    echo "Please download from: https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz"
    echo "Extract to: data/cifar-10-batches-bin/"
    echo ""
fi

# Create build directory
echo ""
echo "Creating build directory..."
mkdir -p build
cd build

# Configure
echo "Configuring CMake..."
cmake ..

# Build
echo ""
echo "Building project..."
echo "Note: First build may take 2-5 minutes with CUDA compilation"
make -j$(nproc)

echo ""
echo "========================================"
echo "Build completed successfully!"
echo "========================================"
echo ""
echo "Executables created:"
echo "  - build/train_cpu           (CPU training)"
echo "  - build/train_gpu           (GPU training)"
echo "  - build/extract_features    (Feature extraction)"
if command -v libsvm-predict &> /dev/null || [ -f "/usr/lib/libsvm.so" ] || [ -f "/usr/local/lib/libsvm.so" ]; then
    echo "  - build/svm_classifier      (SVM classification)"
else
    echo "  - svm_classifier            (Optional, requires LIBSVM)"
fi
echo ""
echo "Quick start:"
echo "  cd build"
echo "  ./train_gpu --data-dir ../data/cifar-10-batches-bin --epochs 20"
echo ""
