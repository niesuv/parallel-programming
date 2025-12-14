#include "cuda_utils.h"
#include "gpu_layer.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>

__global__ void sgd_update_fp32_kernel(
    float *weights,
    const float *grads,
    float lr,
    size_t n) {

  size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        weights[i] -= lr * grads[i];
    }
}

void sgd_update_fp32(float *weights,
                    const float *grads,
                    float lr,
                    size_t n) {
    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    sgd_update_fp32_kernel<<<blocks, threads>>>(
        weights, grads, lr, n);
    CUDA_CHECK(cudaGetLastError());
}
