# CMakeLists.txt
# CIFAR-10 Autoencoder with Feature Extraction and SVM Classification
#
# ============================================================================
# Dataset Organization:
# ============================================================================
#   Stage 1 - Autoencoder Training:
#     - Input: 50,000 training images (labels IGNORED)
#     - Task: Unsupervised reconstruction (target = input)
#     - Output: Trained encoder weights -> weights/
#
#   Stage 2 - Feature Extraction:
#     - Training set: 50,000 images -> 50,000 x 8192 features + labels
#     - Test set: 10,000 images -> 10,000 x 8192 features + labels
#     - Output: Feature files -> features/
#
#   Stage 3 - SVM Classification:
#     - Training: 50,000 feature vectors with labels
#     - Evaluation: 10,000 feature vectors with labels
#     - Task: 10-class classification (CIFAR-10 categories)
#     - Output: SVM model -> weights/
#
# ============================================================================
# Directory Structure:
# ============================================================================
#   include/           - Header files
#   src/kernel/        - CUDA kernel implementations
#   src/layer/         - Autoencoder layer implementation
#   src/train/         - Training scripts (autoencoder + SVM)
#   src/inference/     - Feature extraction
#   weights/           - Saved model weights (AE + SVM)
#   features/          - Extracted feature files
#   data/              - CIFAR-10 dataset (after download)
#
# ============================================================================
# Build Targets:
# ============================================================================
#   train_autoencoder  - Train autoencoder only
#   extract_features   - Extract features only
#   train_svm          - Train SVM (Python script)
#   all_stages         - Run all stages sequentially
#
# Usage:
#   mkdir build && cd build
#   cmake .. -DCMAKE_CUDA_ARCHITECTURES=75
#   make -j4

cmake_minimum_required(VERSION 3.18)
project(CIFAR10_Autoencoder CUDA CXX)

# ============================================================================
# CUDA Configuration
# ============================================================================

set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CXX_STANDARD 14)

# Default to sm_75 (T4) if not specified
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 75)
endif()

# CUDA flags
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3 --use_fast_math")
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -Xcompiler -O3")

# ============================================================================
# Directory Structure
# ============================================================================

set(INCLUDE_DIR ${CMAKE_SOURCE_DIR}/include)
set(KERNEL_DIR ${CMAKE_SOURCE_DIR}/src/kernel)
set(LAYER_DIR ${CMAKE_SOURCE_DIR}/src/layer)
set(TRAIN_DIR ${CMAKE_SOURCE_DIR}/src/train)
set(INFERENCE_DIR ${CMAKE_SOURCE_DIR}/src/inference)

# Output directories (created in build folder)
set(WEIGHTS_DIR ${CMAKE_BINARY_DIR}/weights)
set(FEATURES_DIR ${CMAKE_BINARY_DIR}/features)
set(DATA_DIR ${CMAKE_BINARY_DIR}/data)
set(BIN_DIR ${CMAKE_BINARY_DIR}/bin)

# Create output directories
file(MAKE_DIRECTORY ${WEIGHTS_DIR})
file(MAKE_DIRECTORY ${FEATURES_DIR})
file(MAKE_DIRECTORY ${DATA_DIR})
file(MAKE_DIRECTORY ${BIN_DIR})

# ============================================================================
# Source Files
# ============================================================================

# Core CUDA kernels
set(KERNEL_SOURCES
    ${KERNEL_DIR}/gpu_conv2d_fp16_forward_v6.cu
    ${KERNEL_DIR}/gpu_conv2d_fp16_backward_v8.cu
    ${KERNEL_DIR}/autoencoder_ops.cu
)

# Autoencoder layer
set(LAYER_SOURCES
    ${LAYER_DIR}/autoencoder.cu
)

# Training
set(TRAIN_SOURCES
    ${TRAIN_DIR}/train_AE.cu
)

# Inference
set(INFERENCE_SOURCES
    ${INFERENCE_DIR}/extract_features.cu
)

# ============================================================================
# Include Directories
# ============================================================================

include_directories(${INCLUDE_DIR})

# ============================================================================
# Libraries
# ============================================================================

# Static library for kernels
add_library(cuda_kernels STATIC ${KERNEL_SOURCES})
target_include_directories(cuda_kernels PUBLIC ${INCLUDE_DIR})
set_target_properties(cuda_kernels PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

# Static library for autoencoder
add_library(autoencoder_lib STATIC ${LAYER_SOURCES})
target_include_directories(autoencoder_lib PUBLIC ${INCLUDE_DIR})
target_link_libraries(autoencoder_lib cuda_kernels)
set_target_properties(autoencoder_lib PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
)

# ============================================================================
# Executables
# ============================================================================

# 1. Train Autoencoder
add_executable(train_autoencoder ${TRAIN_SOURCES})
target_include_directories(train_autoencoder PRIVATE ${INCLUDE_DIR})
target_link_libraries(train_autoencoder autoencoder_lib cuda_kernels)
set_target_properties(train_autoencoder PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# 2. Extract Features
add_executable(extract_features ${INFERENCE_SOURCES})
target_include_directories(extract_features PRIVATE ${INCLUDE_DIR})
target_link_libraries(extract_features autoencoder_lib cuda_kernels)
set_target_properties(extract_features PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# ============================================================================
# Naive Implementations (for comparison)
# ============================================================================

set(NAIVE_DIR ${CMAKE_SOURCE_DIR}/src/naive)

# Naive GPU Autoencoder
add_executable(naive_gpu_autoencoder ${NAIVE_DIR}/naive_gpu_autoencoder.cu)
set_target_properties(naive_gpu_autoencoder PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)

# Naive CPU Autoencoder (C++ only, no CUDA)
add_executable(naive_cpu_autoencoder ${NAIVE_DIR}/naive_cpu_autoencoder.cpp)
set_target_properties(naive_cpu_autoencoder PROPERTIES
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin
)
target_link_libraries(naive_cpu_autoencoder m)

# ============================================================================
# Python Script Setup (train_svm.py)
# ============================================================================

# Copy Python script to build directory
configure_file(
    ${TRAIN_DIR}/train_svm.py
    ${CMAKE_BINARY_DIR}/bin/train_svm.py
    COPYONLY
)

# Create a target that "builds" the SVM trainer
add_custom_target(train_svm
    COMMAND ${CMAKE_COMMAND} -E echo "SVM training script ready: ${CMAKE_BINARY_DIR}/bin/train_svm.py"
    COMMAND ${CMAKE_COMMAND} -E echo "Usage: python bin/train_svm.py --features cifar10_features --epochs 20"
    DEPENDS ${CMAKE_BINARY_DIR}/bin/train_svm.py
)

# ============================================================================
# All Stages Target
# ============================================================================

# Create a shell script that runs all stages
file(WRITE ${CMAKE_BINARY_DIR}/bin/run_all_stages.sh
"#!/bin/bash
set -e

echo '=============================================='
echo '  CIFAR-10 Autoencoder Pipeline'
echo '=============================================='
echo ''

# Get script directory (where executables are)
SCRIPT_DIR=\"$( cd \"$( dirname \"\${BASH_SOURCE[0]}\" )\" && pwd )\"
BUILD_DIR=\"$( cd \"$SCRIPT_DIR/..\" && pwd )\"

# Configuration with proper directory structure
DATA_DIR=\${DATA_DIR:-$BUILD_DIR/data}
WEIGHTS_DIR=\${WEIGHTS_DIR:-$BUILD_DIR/weights}
FEATURES_DIR=\${FEATURES_DIR:-$BUILD_DIR/features}
LR=\${LR:-0.003}
EPOCHS_AE=\${EPOCHS_AE:-20}
EPOCHS_SVM=\${EPOCHS_SVM:-10}

# Derived paths
AE_WEIGHTS=\"$WEIGHTS_DIR/autoencoder_best.bin\"
AE_FINAL=\"$WEIGHTS_DIR/autoencoder_final.bin\"
SVM_MODEL=\"$WEIGHTS_DIR/svm_model.pkl\"
FEATURES_PREFIX=\"$FEATURES_DIR/cifar10\"

echo 'Configuration:'
echo \"  DATA_DIR=$DATA_DIR\"
echo \"  WEIGHTS_DIR=$WEIGHTS_DIR\"
echo \"  FEATURES_DIR=$FEATURES_DIR\"
echo \"  LR=$LR\"
echo \"  EPOCHS_AE=$EPOCHS_AE\"
echo \"  EPOCHS_SVM=$EPOCHS_SVM\"
echo ''
echo 'Output paths:'
echo \"  AE weights: $AE_WEIGHTS\"
echo \"  Features: $FEATURES_PREFIX\"
echo ''

# Create directories if they don't exist
mkdir -p \"$WEIGHTS_DIR\"
mkdir -p \"$FEATURES_DIR\"

# Stage 1: Train Autoencoder
echo '=============================================='
echo '  Stage 1: Training Autoencoder'
echo '=============================================='
\"$SCRIPT_DIR/train_autoencoder\" --data \"$DATA_DIR\" --lr $LR --epochs $EPOCHS_AE --weights_dir \"$WEIGHTS_DIR\"

# Stage 2: Extract Features  
echo ''
echo '=============================================='
echo '  Stage 2: Extracting Features'
echo '=============================================='
\"$SCRIPT_DIR/extract_features\" --data \"$DATA_DIR\" --weights \"$AE_WEIGHTS\" --output \"$FEATURES_PREFIX\"

# Stage 3: Train Classifier
echo ''
echo '=============================================='
echo '  Stage 3: Training Classifier'
echo '=============================================='
python3 \"$SCRIPT_DIR/train_svm.py\" --features \"$FEATURES_PREFIX\" --epochs $EPOCHS_SVM --classifier sgd --save_model \"$SVM_MODEL\"

echo ''
echo '=============================================='
echo '  Pipeline Complete!'
echo '=============================================='
echo ''
echo 'Output files:'
echo \"  Autoencoder weights: $AE_WEIGHTS\"
echo \"  Features: ${FEATURES_PREFIX}_train_features.bin\"
echo \"            ${FEATURES_PREFIX}_test_features.bin\"
echo \"  SVM model: $SVM_MODEL\"
")

# Make script executable and create target
add_custom_target(all_stages
    COMMAND chmod +x ${CMAKE_BINARY_DIR}/bin/run_all_stages.sh
    COMMAND ${CMAKE_COMMAND} -E echo "All stages script ready: ${CMAKE_BINARY_DIR}/bin/run_all_stages.sh"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Usage:"
    COMMAND ${CMAKE_COMMAND} -E echo "  cd bin && ./run_all_stages.sh"
    COMMAND ${CMAKE_COMMAND} -E echo ""
    COMMAND ${CMAKE_COMMAND} -E echo "Or with custom settings:"
    COMMAND ${CMAKE_COMMAND} -E echo "  DATA_DIR=../data LR=0.003 ./run_all_stages.sh"
    DEPENDS train_autoencoder extract_features train_svm
)

# ============================================================================
# Convenience Targets
# ============================================================================

# Build all executables (optimized)
add_custom_target(build_all
    DEPENDS train_autoencoder extract_features train_svm
)

# Build naive implementations
add_custom_target(build_naive
    DEPENDS naive_gpu_autoencoder naive_cpu_autoencoder
)

# Build everything
add_custom_target(build_everything
    DEPENDS train_autoencoder extract_features train_svm naive_gpu_autoencoder naive_cpu_autoencoder
)

# Clean data files
add_custom_target(clean_data
    COMMAND ${CMAKE_COMMAND} -E echo "Cleaning generated data files..."
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/bin/autoencoder_best.bin
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/bin/autoencoder_weights.bin
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/bin/cifar10_features_train_features.bin
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/bin/cifar10_features_train_labels.bin
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/bin/cifar10_features_test_features.bin
    COMMAND ${CMAKE_COMMAND} -E remove -f ${CMAKE_BINARY_DIR}/bin/cifar10_features_test_labels.bin
)

# Download CIFAR-10 data
add_custom_target(download_data
    COMMAND ${CMAKE_COMMAND} -E echo "Downloading CIFAR-10 dataset..."
    COMMAND ${CMAKE_COMMAND} -E make_directory ${CMAKE_BINARY_DIR}/data
    COMMAND curl -L -o ${CMAKE_BINARY_DIR}/data/cifar-10-binary.tar.gz 
            https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
    COMMAND tar -xzf ${CMAKE_BINARY_DIR}/data/cifar-10-binary.tar.gz -C ${CMAKE_BINARY_DIR}/data
    COMMAND ${CMAKE_COMMAND} -E rename 
            ${CMAKE_BINARY_DIR}/data/cifar-10-batches-bin/data_batch_1.bin
            ${CMAKE_BINARY_DIR}/data/data_batch_1.bin
    COMMAND ${CMAKE_COMMAND} -E rename 
            ${CMAKE_BINARY_DIR}/data/cifar-10-batches-bin/data_batch_2.bin
            ${CMAKE_BINARY_DIR}/data/data_batch_2.bin
    COMMAND ${CMAKE_COMMAND} -E rename 
            ${CMAKE_BINARY_DIR}/data/cifar-10-batches-bin/data_batch_3.bin
            ${CMAKE_BINARY_DIR}/data/data_batch_3.bin
    COMMAND ${CMAKE_COMMAND} -E rename 
            ${CMAKE_BINARY_DIR}/data/cifar-10-batches-bin/data_batch_4.bin
            ${CMAKE_BINARY_DIR}/data/data_batch_4.bin
    COMMAND ${CMAKE_COMMAND} -E rename 
            ${CMAKE_BINARY_DIR}/data/cifar-10-batches-bin/data_batch_5.bin
            ${CMAKE_BINARY_DIR}/data/data_batch_5.bin
    COMMAND ${CMAKE_COMMAND} -E rename 
            ${CMAKE_BINARY_DIR}/data/cifar-10-batches-bin/test_batch.bin
            ${CMAKE_BINARY_DIR}/data/test_batch.bin
    COMMAND ${CMAKE_COMMAND} -E echo "CIFAR-10 downloaded to ${CMAKE_BINARY_DIR}/data/"
)

# ============================================================================
# Install
# ============================================================================

install(TARGETS train_autoencoder extract_features
    RUNTIME DESTINATION bin
)

install(FILES ${TRAIN_DIR}/train_svm.py
    DESTINATION bin
    PERMISSIONS OWNER_READ OWNER_WRITE OWNER_EXECUTE
                GROUP_READ GROUP_EXECUTE
                WORLD_READ WORLD_EXECUTE
)

install(DIRECTORY ${INCLUDE_DIR}/
    DESTINATION include
    FILES_MATCHING PATTERN "*.h"
)

# ============================================================================
# Print Configuration
# ============================================================================

message(STATUS "")
message(STATUS "============================================")
message(STATUS "  CIFAR-10 Autoencoder Build Configuration")
message(STATUS "============================================")
message(STATUS "")
message(STATUS "Directory Structure:")
message(STATUS "  include/              ${INCLUDE_DIR}")
message(STATUS "  src/kernel/           ${KERNEL_DIR}")
message(STATUS "  src/layer/            ${LAYER_DIR}")
message(STATUS "  src/train/            ${TRAIN_DIR}")
message(STATUS "  src/inference/        ${INFERENCE_DIR}")
message(STATUS "  src/naive/            ${NAIVE_DIR}")
message(STATUS "")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")
message(STATUS "Targets (Optimized):")
message(STATUS "  train_autoencoder     - Train the autoencoder (FP16 + Tensor Cores)")
message(STATUS "  extract_features      - Extract latent features")
message(STATUS "  train_svm             - Setup SVM training script")
message(STATUS "  all_stages            - Run complete pipeline")
message(STATUS "  build_all             - Build optimized executables")
message(STATUS "")
message(STATUS "Targets (Naive - for comparison):")
message(STATUS "  naive_gpu_autoencoder - Simple GPU implementation")
message(STATUS "  naive_cpu_autoencoder - Pure C++ CPU implementation")
message(STATUS "  build_naive           - Build naive executables")
message(STATUS "")
message(STATUS "Utility Targets:")
message(STATUS "  download_data         - Download CIFAR-10 dataset")
message(STATUS "  clean_data            - Remove generated files")
message(STATUS "  build_everything      - Build all targets")
message(STATUS "============================================")
message(STATUS "")
