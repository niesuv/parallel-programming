#!/bin/bash

# Build Verification Script
# This script performs diagnostic checks and builds the CIFAR-10 autoencoder project

set -e

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
BUILD_DIR="$PROJECT_DIR/build"

echo "============================================"
echo "CIFAR-10 Autoencoder Build Verification"
echo "============================================"
echo ""

# 1. Check prerequisites
echo "[1/6] Checking prerequisites..."
echo "  CMake: $(cmake --version | head -1)"
echo "  GCC: $(g++ --version | head -1)"
echo "  NVCC: $(nvcc --version | tail -1)"
echo ""

# 2. Verify source files
echo "[2/6] Verifying source files..."
required_headers=("include/data_loader.h" "include/cpu_layers.h" "include/autoencoder.h" "include/gpu_autoencoder.h")
required_sources=("src/data_loader.cpp" "src/cpu_layers.cpp" "src/autoencoder.cpp" "src/gpu_autoencoder.cu" "src/train_cpu.cpp" "src/train_gpu.cu" "src/feature_extraction.cu" "src/svm_classifier.cpp")

for file in "${required_headers[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        exit 1
    fi
done

for file in "${required_sources[@]}"; do
    if [ -f "$PROJECT_DIR/$file" ]; then
        echo "  ✓ $file"
    else
        echo "  ✗ MISSING: $file"
        exit 1
    fi
done
echo ""

# 3. Verify includes
echo "[3/6] Verifying header includes..."
errors=0

# Check that headers have required includes
check_include() {
    local file=$1
    local include=$2
    if grep -q "#include $include" "$file"; then
        echo "  ✓ $file has $include"
    else
        echo "  ✗ MISSING: $file needs $include"
        errors=$((errors + 1))
    fi
}

check_include "$PROJECT_DIR/include/autoencoder.h" "<string>"
check_include "$PROJECT_DIR/include/gpu_autoencoder.h" "<cuda_runtime.h>"
check_include "$PROJECT_DIR/src/data_loader.cpp" "\"data_loader.h\""
check_include "$PROJECT_DIR/src/autoencoder.cpp" "\"autoencoder.h\""

if [ $errors -gt 0 ]; then
    echo "  ✗ Found $errors missing includes"
    exit 1
fi
echo ""

# 4. Verify data directory
echo "[4/6] Verifying CIFAR-10 dataset..."
if [ -d "$PROJECT_DIR/data/cifar-10-batches-bin" ]; then
    echo "  ✓ Dataset directory found"
    batch_count=$(find "$PROJECT_DIR/data/cifar-10-batches-bin" -name "*.bin" 2>/dev/null | wc -l)
    echo "    Binary files: $batch_count"
else
    echo "  ! Dataset directory not found (optional for build, required for training)"
fi
echo ""

# 5. Clean and configure build
echo "[5/6] Configuring build..."
if [ -d "$BUILD_DIR" ]; then
    echo "  Cleaning previous build..."
    rm -rf "$BUILD_DIR"
fi

mkdir -p "$BUILD_DIR"
cd "$BUILD_DIR"

cmake .. 2>&1 | tail -20
echo ""

# 6. Compile
echo "[6/6] Compiling..."
make -j$(nproc) 2>&1 | tail -30
echo ""

# Verify executables were created
echo "============================================"
echo "Build Verification Results"
echo "============================================"
echo ""

success=true
for exe in train_cpu train_gpu extract_features; do
    if [ -f "$BUILD_DIR/$exe" ]; then
        echo "  ✓ $exe"
    else
        echo "  ✗ MISSING: $exe"
        success=false
    fi
done

# SVM is optional
if [ -f "$BUILD_DIR/svm_classifier" ]; then
    echo "  ✓ svm_classifier (optional)"
else
    echo "  ! svm_classifier not built (LIBSVM not found - this is optional)"
fi

echo ""
if [ "$success" = true ]; then
    echo "✓ Build successful! All executables created."
    echo ""
    echo "Next steps:"
    echo "  1. Run quick test: ./build/train_cpu --epochs 1 --num-samples 10"
    echo "  2. Full training: ./build/train_cpu --epochs 20"
    echo "  3. See README.md for detailed usage"
else
    echo "✗ Build failed. See errors above."
    exit 1
fi
