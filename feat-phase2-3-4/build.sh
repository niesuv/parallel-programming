#!/bin/bash
# AutoEncoder Phase 3 - Build Script

set -e

BUILD_DIR="build"
BUILD_TYPE="Release"
CLEAN=false

# Parse arguments
while [[ $# -gt 0 ]]; do
    case $1 in
        --clean) CLEAN=true; shift ;;
        --debug) BUILD_TYPE="Debug"; shift ;;
        --help)
            echo "Usage: ./build.sh [--clean] [--debug]"
            echo "  --clean   Remove build directory and rebuild"
            echo "  --debug   Build with debug symbols"
            exit 0 ;;
        *) echo "Unknown option: $1"; exit 1 ;;
    esac
done

# Clean if requested
[[ "$CLEAN" == true ]] && rm -rf "$BUILD_DIR"

# Check submodule
if [[ ! -f "external/thundersvm/CMakeLists.txt" ]]; then
    echo "ThunderSVM submodule not found. Initializing..."
    git submodule update --init --recursive
fi

# Build
echo "Building with CMake..."
mkdir -p "$BUILD_DIR"
cmake -S . -B "$BUILD_DIR" -DCMAKE_BUILD_TYPE="$BUILD_TYPE"
cmake --build "$BUILD_DIR" -j$(nproc 2>/dev/null || echo 4)

# Summary
echo ""
echo "Build complete! Executables:"
ls -1 "$BUILD_DIR/bin/" 2>/dev/null | sed 's/^/  - /'
echo ""
echo "Usage:"
echo "  ./build/bin/gpu_train --help       # Train autoencoder only"
echo "  ./build/bin/full_pipeline --help   # Full pipeline with SVM"
