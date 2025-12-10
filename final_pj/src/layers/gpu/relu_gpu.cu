#include "layers.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ReLU forward kernel (in-place)
__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// ReLU backward kernel
__global__ void relu_backward_kernel(const float* input, const float* d_output,
                                     float* d_input, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_input[idx] = (input[idx] > 0.0f) ? d_output[idx] : 0.0f;
    }
}

// ReLU forward on GPU (in-place)
void relu_forward_cuda(float* data, int size) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_forward_kernel<<<blocks, threads>>>(data, size);

    CUDA_CHECK(cudaGetLastError());
}

// ReLU backward on GPU
void relu_backward_cuda(const float* input, const float* d_output,
                        float* d_input, int size) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;

    relu_backward_kernel<<<blocks, threads>>>(input, d_output, d_input, size);

    CUDA_CHECK(cudaGetLastError());
}
