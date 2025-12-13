cmake_minimum_required(VERSION 3.20)
project(CIFAR10_Autoencoder LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 17)

# Find CUDA
find_package(CUDA REQUIRED)

# Include directories
include_directories(${PROJECT_SOURCE_DIR}/include)

# Data loader library
add_library(data_loader src/data_loader.cpp)
target_include_directories(data_loader PUBLIC ${PROJECT_SOURCE_DIR}/include)

# CPU layers library
add_library(cpu_layers src/cpu_layers.cpp)
target_include_directories(cpu_layers PUBLIC ${PROJECT_SOURCE_DIR}/include)

# CPU autoencoder library
add_library(cpu_autoencoder src/autoencoder.cpp)
target_link_libraries(cpu_autoencoder cpu_layers)
target_include_directories(cpu_autoencoder PUBLIC ${PROJECT_SOURCE_DIR}/include)

# CPU training executable
add_executable(train_cpu src/train_cpu.cpp)
target_link_libraries(train_cpu cpu_autoencoder data_loader cpu_layers)

# GPU autoencoder library
add_library(gpu_autoencoder src/gpu_autoencoder.cu)
set_target_properties(gpu_autoencoder PROPERTIES LANGUAGE CUDA)
target_include_directories(gpu_autoencoder PUBLIC ${PROJECT_SOURCE_DIR}/include)

# GPU training executable
add_executable(train_gpu src/train_gpu.cu)
set_target_properties(train_gpu PROPERTIES LANGUAGE CUDA)
target_link_libraries(train_gpu gpu_autoencoder cpu_autoencoder data_loader cpu_layers)

# Feature extraction executable
add_executable(extract_features src/feature_extraction.cu)
set_target_properties(extract_features PROPERTIES LANGUAGE CUDA)
target_link_libraries(extract_features gpu_autoencoder cpu_autoencoder data_loader cpu_layers)

# SVM classifier executable (requires LIBSVM)
find_path(LIBSVM_INCLUDE_DIR svm.h HINTS /usr/local/include /opt/local/include)
find_library(LIBSVM_LIB svm HINTS /usr/local/lib /opt/local/lib)

if(LIBSVM_INCLUDE_DIR AND LIBSVM_LIB)
    add_executable(svm_classifier src/svm_classifier.cpp)
    target_include_directories(svm_classifier PRIVATE ${LIBSVM_INCLUDE_DIR})
    target_link_libraries(svm_classifier ${LIBSVM_LIB} data_loader cpu_autoencoder cpu_layers)
    message(STATUS "LIBSVM found: ${LIBSVM_LIB}")
else()
    message(WARNING "LIBSVM not found - SVM classifier will not be built")
endif()

# Set CUDA compute capability
set_property(TARGET gpu_autoencoder PROPERTY CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real)
set_property(TARGET train_gpu PROPERTY CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real)
set_property(TARGET extract_features PROPERTY CUDA_ARCHITECTURES 75-real 80-real 86-real 89-real)
