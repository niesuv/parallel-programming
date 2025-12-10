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

// MSE loss forward kernel with parallel reduction
__global__ void mse_loss_forward_kernel(
    const float* output, const float* target, float* partial_sums, int size) {

    __shared__ float sdata[256];

    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    // Load data and compute squared error
    float diff = 0.0f;
    if (idx < size) {
        diff = output[idx] - target[idx];
        diff = diff * diff;
    }
    sdata[tid] = diff;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }

    // Write result for this block to global memory
    if (tid == 0) {
        partial_sums[blockIdx.x] = sdata[0];
    }
}

// MSE loss backward kernel (gradient)
__global__ void mse_loss_backward_kernel(
    const float* output, const float* target, float* d_output, int size) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Gradient of MSE: 2 * (output - target) / N
        // We'll divide by N on the CPU after summing
        d_output[idx] = 2.0f * (output[idx] - target[idx]);
    }
}

// MSE loss forward on GPU
float mse_loss_forward_cuda(const float* output, const float* target, int size) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;

    // Allocate device memory for partial sums
    float* d_partial_sums;
    CUDA_CHECK(cudaMalloc(&d_partial_sums, blocks * sizeof(float)));

    // Compute partial sums
    mse_loss_forward_kernel<<<blocks, threads>>>(output, target, d_partial_sums, size);

    // Copy partial sums to host and complete reduction
    float* h_partial_sums = (float*)malloc(blocks * sizeof(float));
    CUDA_CHECK(cudaMemcpy(h_partial_sums, d_partial_sums, blocks * sizeof(float),
                          cudaMemcpyDeviceToHost));

    float total_loss = 0.0f;
    for (int i = 0; i < blocks; i++) {
        total_loss += h_partial_sums[i];
    }

    // Average loss
    float mse = total_loss / size;

    // Cleanup
    free(h_partial_sums);
    CUDA_CHECK(cudaFree(d_partial_sums));

    return mse;
}

// MSE loss backward on GPU
void mse_loss_backward_cuda(const float* output, const float* target,
                            float* d_output, int size) {
    const int threads = 256;
    int blocks = (size + threads - 1) / threads;

    mse_loss_backward_kernel<<<blocks, threads>>>(output, target, d_output, size);

    CUDA_CHECK(cudaGetLastError());

    // Divide by N to get average gradient (done in-place)
    float scale = 2.0f / size;
    cudaMemcpy(d_output, d_output, size * sizeof(float), cudaMemcpyDeviceToDevice);
}
