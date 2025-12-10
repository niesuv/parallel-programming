cmake_minimum_required(VERSION 3.10)
project(cifar_autoencoder LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O2 -march=native")

include_directories(${PROJECT_SOURCE_DIR}/include)
add_subdirectory(src)
