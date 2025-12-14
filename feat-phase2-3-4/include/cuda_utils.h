/**
 * @file cuda_utils.h
 * @brief CUDA Error Checking Utilities
 *
 * Provides macros for CUDA error handling and debugging.
 * The CUDA_CHECK macro wraps CUDA API calls and prints errors with location.
 */

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <iostream>

/**
 * @brief CUDA Error Checking Macro
 *
 * Usage: CUDA_CHECK(cudaMalloc(&ptr, size));
 *
 * If the CUDA call fails, prints error message with file and line number.
 */
#define CUDA_CHECK(expr)                                                       \
  do {                                                                         \
    cudaError_t err__ = (expr);                                                \
    if (err__ != cudaSuccess) {                                                \
      std::cerr << "CUDA error: " << cudaGetErrorString(err__) << " at "       \
                << __FILE__ << ":" << __LINE__ << std::endl;                   \
    }                                                                          \
  } while (0)

#else
// No-op when not compiling with CUDA
#define CUDA_CHECK(expr) (expr)
#endif

#endif // CUDA_UTILS_H
