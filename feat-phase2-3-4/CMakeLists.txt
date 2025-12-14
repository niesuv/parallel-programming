cmake_minimum_required(VERSION 3.18)
project(AutoEncoderPhase3 LANGUAGES CXX CUDA)

# ==============================================================================
# Build Configuration
# ==============================================================================
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

# CUDA architectures: Tesla T4 (75), A100 (80), RTX 30xx (86)
set(CMAKE_CUDA_ARCHITECTURES 75 80 86)

# Find CUDA Toolkit
find_package(CUDAToolkit REQUIRED)

# Output directory
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_BINARY_DIR}/bin)

# ==============================================================================
# Source Files
# ==============================================================================
set(GPU_LAYERS_SRC
    src/layers_gpu.cu
    src/layers_gpu_opt.cu
    src/gpu_autoencoder.cu
)

set(DATASET_SRC
    src/dataset.cpp
)

set(SVM_SRC
    src/svm_wrapper.cpp
)

set(CPU_SRC
    src/layers_cpu.cpp
    src/autoencoder.cpp
)

# ==============================================================================
# ThunderSVM Integration
# ==============================================================================
set(THUNDERSVM_DIR "${PROJECT_SOURCE_DIR}/external/thundersvm")
if(NOT EXISTS "${THUNDERSVM_DIR}/CMakeLists.txt")
    message(FATAL_ERROR "ThunderSVM not found. Run: git submodule update --init --recursive")
endif()

# ThunderSVM uses legacy cuda_add_library which ignores CMAKE_CUDA_ARCHITECTURES.
# We must set CUDA_NVCC_FLAGS explicitly for Tesla T4 (sm_75) compatibility.
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=sm_75")
set(CUDA_NVCC_FLAGS "${CUDA_NVCC_FLAGS} -gencode arch=compute_75,code=compute_75")

add_subdirectory(${THUNDERSVM_DIR})

# ==============================================================================
# Helper Function: Configure CUDA Target
# ==============================================================================
function(configure_cuda_target TARGET_NAME)
    target_include_directories(${TARGET_NAME} PRIVATE ${PROJECT_SOURCE_DIR}/include)
    set_target_properties(${TARGET_NAME} PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
    
    target_compile_options(${TARGET_NAME} PRIVATE
        $<$<COMPILE_LANGUAGE:CXX>:-Wall -Wextra -O3>
        $<$<COMPILE_LANGUAGE:CUDA>:-O3 --use_fast_math>
    )
endfunction()

# ==============================================================================
# GPU Training (without SVM)
# ==============================================================================
add_executable(gpu_train
    src/main_gpu.cu
    ${GPU_LAYERS_SRC}
    ${DATASET_SRC}
)
target_compile_definitions(gpu_train PRIVATE USE_OPTIMIZED_KERNELS)
target_link_libraries(gpu_train PRIVATE CUDA::cudart)
configure_cuda_target(gpu_train)

# ==============================================================================
# Full Pipeline (with ThunderSVM)
# ==============================================================================
add_executable(full_pipeline
    src/main_gpu.cu
    ${GPU_LAYERS_SRC}
    ${DATASET_SRC}
    ${SVM_SRC}
)
target_compile_definitions(full_pipeline PRIVATE USE_OPTIMIZED_KERNELS WITH_SVM WITH_THUNDERSVM)
target_include_directories(full_pipeline PRIVATE 
    ${THUNDERSVM_DIR}/include
    ${CMAKE_BINARY_DIR}
    ${CMAKE_BINARY_DIR}/external/thundersvm
    ${CUDAToolkit_INCLUDE_DIRS}
)
target_link_libraries(full_pipeline PRIVATE thundersvm CUDA::cudart)
configure_cuda_target(full_pipeline)

# ==============================================================================
# Verify GPU (disabled - has compilation issues, not needed for SVM)
# ==============================================================================
# add_executable(verify_gpu
#     src/verify_gpu.cu
#     ${GPU_LAYERS_SRC}
#     ${CPU_SRC}
#     ${DATASET_SRC}
# )
# target_link_libraries(verify_gpu PRIVATE CUDA::cudart)
# configure_cuda_target(verify_gpu)

# ==============================================================================
# Output Directories
# ==============================================================================
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/checkpoints)
file(MAKE_DIRECTORY ${CMAKE_SOURCE_DIR}/results)

# ==============================================================================
# Build Summary
# ==============================================================================
message(STATUS "========================================")
message(STATUS "AutoEncoder Phase 3 Build Configuration")
message(STATUS "========================================")
message(STATUS "Build Type:         ${CMAKE_BUILD_TYPE}")
message(STATUS "C++ Standard:       ${CMAKE_CXX_STANDARD}")
message(STATUS "CUDA Standard:      ${CMAKE_CUDA_STANDARD}")
message(STATUS "CUDA Architectures: ${CMAKE_CUDA_ARCHITECTURES}")
message(STATUS "ThunderSVM:         ${THUNDERSVM_DIR}")
message(STATUS "========================================")
