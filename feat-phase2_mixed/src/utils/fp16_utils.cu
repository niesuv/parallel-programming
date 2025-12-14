#include <cuda_runtime.h>
#include <cuda_fp16.h>

#include "cuda_utils.h"

__global__ void fp32_to_fp16_kernel(
    const float *src,
    __half *dst,
    size_t n) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __float2half(src[i]);
    }
}

void gpu_fp32_to_fp16(const float *src,
                      __half *dst,
                        size_t n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    fp32_to_fp16_kernel<<<blocks, threads>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}

__global__ void fp16_to_fp32_kernel(
    const __half *src,
    float *dst,
    size_t n) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        dst[i] = __half2float(src[i]);
    }
}

void gpu_fp16_to_fp32(const __half *src,
                    float *dst,
                    size_t n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    fp16_to_fp32_kernel<<<blocks, threads>>>(src, dst, n);
    CUDA_CHECK(cudaGetLastError());
}
