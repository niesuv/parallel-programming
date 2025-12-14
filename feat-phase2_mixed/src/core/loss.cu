#include "cuda_utils.h"
#include "gpu_layer.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>

__global__ void mse_loss_fp32_kernel(
    const __half *output,
    const __half *target,
    float *loss,
    size_t n) {

    __shared__ float buf[256];

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    float val = 0.0f;

    if (i < n) {
        float diff =
            __half2float(output[i]) -
            __half2float(target[i]);
        val = diff * diff;
    }

    buf[threadIdx.x] = val;
    __syncthreads();

    // block reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (threadIdx.x < s)
        buf[threadIdx.x] += buf[threadIdx.x + s];
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        atomicAdd(loss, buf[0]);
    }
}

float gpu_mse_loss_fp32(const GPUTensor4D &output_fp16,
                        const GPUTensor4D &target_fp16) {
    size_t n = output_fp16.size();

    float *d_loss = nullptr;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    mse_loss_fp32_kernel<<<blocks, threads>>>(
        static_cast<const __half *>(output_fp16.d_data),
        static_cast<const __half *>(target_fp16.d_data),
        d_loss,
        n);

    CUDA_CHECK(cudaGetLastError());

    float h_loss = 0.0f;
    CUDA_CHECK(cudaMemcpy(&h_loss,
                            d_loss,
                            sizeof(float),
                            cudaMemcpyDeviceToHost));

    cudaFree(d_loss);

    return h_loss / static_cast<float>(n);
}

__global__ void mse_backward_fp16_scaled_kernel(
        const __half *output,
        const __half *target,
        __half *grad_output,
        float loss_scale,
        size_t n) {

    size_t i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        float diff =
            __half2float(output[i]) -
            __half2float(target[i]);

        float grad = (2.0f * diff / static_cast<float>(n))
                    * loss_scale;

        grad_output[i] = __float2half(grad);
    }
}


void gpu_mse_loss_backward_fp16_scaled(
    const GPUTensor4D &output_fp16,
    const GPUTensor4D &target_fp16,
    GPUTensor4D &grad_output_fp16,
    float loss_scale) {

    size_t n = output_fp16.size();

    const int threads = 256;
    const int blocks = (n + threads - 1) / threads;

    mse_backward_fp16_scaled_kernel<<<blocks, threads>>>(
        static_cast<const __half *>(output_fp16.d_data),
        static_cast<const __half *>(target_fp16.d_data),
        static_cast<__half *>(grad_output_fp16.d_data),
        loss_scale,
        n);

    CUDA_CHECK(cudaGetLastError());
}
