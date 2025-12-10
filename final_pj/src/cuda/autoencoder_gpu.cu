#include "autoencoder_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// 2.1 GPU Memory Management
// ============================================================================

GPUAutoencoder* gpu_autoencoder_create(float learning_rate, int batch_size, int num_epochs) {
    GPUAutoencoder* gpu_ae = (GPUAutoencoder*)malloc(sizeof(GPUAutoencoder));
    if (!gpu_ae) {
        fprintf(stderr, "Failed to allocate GPUAutoencoder structure\n");
        return NULL;
    }

    gpu_ae->learning_rate = learning_rate;
    gpu_ae->batch_size = batch_size;
    gpu_ae->num_epochs = num_epochs;

    // Calculate sizes for all activations
    gpu_ae->enc1_out_size = batch_size * 256 * 32 * 32;
    gpu_ae->pool1_out_size = batch_size * 256 * 16 * 16;
    gpu_ae->enc2_out_size = batch_size * 128 * 16 * 16;
    gpu_ae->latent_size = batch_size * 128 * 8 * 8;
    gpu_ae->dec1_out_size = batch_size * 128 * 8 * 8;
    gpu_ae->up1_out_size = batch_size * 128 * 16 * 16;
    gpu_ae->dec2_out_size = batch_size * 256 * 16 * 16;
    gpu_ae->up2_out_size = batch_size * 256 * 32 * 32;
    gpu_ae->output_size = batch_size * 3 * 32 * 32;

    // Allocate device memory for weights
    // enc1: 3 -> 256, kernel 3x3
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc1_weights, 256 * 3 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc1_bias, 256 * sizeof(float)));

    // enc2: 256 -> 128, kernel 3x3
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc2_weights, 128 * 256 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc2_bias, 128 * sizeof(float)));

    // dec1: 128 -> 128, kernel 3x3
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec1_weights, 128 * 128 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec1_bias, 128 * sizeof(float)));

    // dec2: 128 -> 256, kernel 3x3
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec2_weights, 256 * 128 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec2_bias, 256 * sizeof(float)));

    // dec3: 256 -> 3, kernel 3x3
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec3_weights, 3 * 256 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec3_bias, 3 * sizeof(float)));

    // Allocate device memory for weight gradients
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc1_d_weights, 256 * 3 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc1_d_bias, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc2_d_weights, 128 * 256 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc2_d_bias, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec1_d_weights, 128 * 128 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec1_d_bias, 128 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec2_d_weights, 256 * 128 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec2_d_bias, 256 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec3_d_weights, 3 * 256 * 3 * 3 * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec3_d_bias, 3 * sizeof(float)));

    // Allocate device memory for activations
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_input, gpu_ae->output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc1_out, gpu_ae->enc1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_pool1_out, gpu_ae->pool1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_enc2_out, gpu_ae->enc2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_latent, gpu_ae->latent_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec1_out, gpu_ae->dec1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_up1_out, gpu_ae->up1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_dec2_out, gpu_ae->dec2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_up2_out, gpu_ae->up2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_output, gpu_ae->output_size * sizeof(float)));

    // Allocate device memory for gradients
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_output, gpu_ae->output_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_up2_out, gpu_ae->up2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_dec2_out, gpu_ae->dec2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_up1_out, gpu_ae->up1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_dec1_out, gpu_ae->dec1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_latent, gpu_ae->latent_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_enc2_out, gpu_ae->enc2_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_pool1_out, gpu_ae->pool1_out_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_grad_enc1_out, gpu_ae->enc1_out_size * sizeof(float)));

    // Allocate device memory for target
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_target, gpu_ae->output_size * sizeof(float)));

    // Allocate device memory for MaxPool indices
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_pool1_indices, gpu_ae->pool1_out_size * sizeof(int)));
    CUDA_CHECK(cudaMalloc(&gpu_ae->d_pool2_indices, gpu_ae->latent_size * sizeof(int)));

    printf("✅ GPU Autoencoder created:\n");
    printf("   Batch size: %d\n", batch_size);
    printf("   Learning rate: %.4f\n", learning_rate);
    printf("   Total GPU memory allocated: ~%.2f MB\n",
           (gpu_ae->enc1_out_size * 2 + gpu_ae->pool1_out_size * 2 +
            gpu_ae->enc2_out_size * 2 + gpu_ae->latent_size * 2 +
            gpu_ae->dec1_out_size * 2 + gpu_ae->up1_out_size * 2 +
            gpu_ae->dec2_out_size * 2 + gpu_ae->up2_out_size * 2 +
            gpu_ae->output_size * 3) * sizeof(float) / (1024.0f * 1024.0f));

    return gpu_ae;
}

void gpu_autoencoder_free(GPUAutoencoder* gpu_ae) {
    if (!gpu_ae) return;

    // Free weight memory
    CUDA_CHECK(cudaFree(gpu_ae->d_enc1_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc1_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc2_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc2_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec1_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec1_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec2_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec2_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec3_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec3_bias));

    // Free weight gradient memory
    CUDA_CHECK(cudaFree(gpu_ae->d_enc1_d_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc1_d_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc2_d_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc2_d_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec1_d_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec1_d_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec2_d_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec2_d_bias));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec3_d_weights));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec3_d_bias));

    // Free activation memory
    CUDA_CHECK(cudaFree(gpu_ae->d_input));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc1_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_pool1_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_enc2_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_latent));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec1_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_up1_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_dec2_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_up2_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_output));

    // Free gradient memory
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_output));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_up2_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_dec2_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_up1_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_dec1_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_latent));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_enc2_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_pool1_out));
    CUDA_CHECK(cudaFree(gpu_ae->d_grad_enc1_out));

    // Free target memory
    CUDA_CHECK(cudaFree(gpu_ae->d_target));

    // Free MaxPool index memory
    CUDA_CHECK(cudaFree(gpu_ae->d_pool1_indices));
    CUDA_CHECK(cudaFree(gpu_ae->d_pool2_indices));

    free(gpu_ae);
    printf("✅ GPU Autoencoder freed\n");
}

void gpu_autoencoder_copy_weights_to_device(GPUAutoencoder* gpu_ae,
                                            Conv2DLayer* enc1, Conv2DLayer* enc2,
                                            Conv2DLayer* dec1, Conv2DLayer* dec2, Conv2DLayer* dec3) {
    // Copy enc1 weights (256 * 3 * 3 * 3)
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_enc1_weights, enc1->weights,
                          256 * 3 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_enc1_bias, enc1->bias,
                          256 * sizeof(float), cudaMemcpyHostToDevice));

    // Copy enc2 weights (128 * 256 * 3 * 3)
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_enc2_weights, enc2->weights,
                          128 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_enc2_bias, enc2->bias,
                          128 * sizeof(float), cudaMemcpyHostToDevice));

    // Copy dec1 weights (128 * 128 * 3 * 3)
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_dec1_weights, dec1->weights,
                          128 * 128 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_dec1_bias, dec1->bias,
                          128 * sizeof(float), cudaMemcpyHostToDevice));

    // Copy dec2 weights (256 * 128 * 3 * 3)
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_dec2_weights, dec2->weights,
                          256 * 128 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_dec2_bias, dec2->bias,
                          256 * sizeof(float), cudaMemcpyHostToDevice));

    // Copy dec3 weights (3 * 256 * 3 * 3)
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_dec3_weights, dec3->weights,
                          3 * 256 * 3 * 3 * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_dec3_bias, dec3->bias,
                          3 * sizeof(float), cudaMemcpyHostToDevice));

    printf("✅ Weights copied to GPU\n");
}

void gpu_autoencoder_copy_weights_to_host(GPUAutoencoder* gpu_ae,
                                          Conv2DLayer* enc1, Conv2DLayer* enc2,
                                          Conv2DLayer* dec1, Conv2DLayer* dec2, Conv2DLayer* dec3) {
    // Copy enc1 weights back
    CUDA_CHECK(cudaMemcpy(enc1->weights, gpu_ae->d_enc1_weights,
                          256 * 3 * 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(enc1->bias, gpu_ae->d_enc1_bias,
                          256 * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy enc2 weights back
    CUDA_CHECK(cudaMemcpy(enc2->weights, gpu_ae->d_enc2_weights,
                          128 * 256 * 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(enc2->bias, gpu_ae->d_enc2_bias,
                          128 * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy dec1 weights back
    CUDA_CHECK(cudaMemcpy(dec1->weights, gpu_ae->d_dec1_weights,
                          128 * 128 * 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dec1->bias, gpu_ae->d_dec1_bias,
                          128 * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy dec2 weights back
    CUDA_CHECK(cudaMemcpy(dec2->weights, gpu_ae->d_dec2_weights,
                          256 * 128 * 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dec2->bias, gpu_ae->d_dec2_bias,
                          256 * sizeof(float), cudaMemcpyDeviceToHost));

    // Copy dec3 weights back
    CUDA_CHECK(cudaMemcpy(dec3->weights, gpu_ae->d_dec3_weights,
                          3 * 256 * 3 * 3 * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(dec3->bias, gpu_ae->d_dec3_bias,
                          3 * sizeof(float), cudaMemcpyDeviceToHost));

    printf("✅ Weights copied to CPU\n");
}

// ============================================================================
// 2.2 Naive GPU Kernels
// ============================================================================

// Naive Convolution Kernel
// Each thread computes one output pixel
// Uses global memory for all reads/writes
__global__ void naive_conv2d_forward_kernel(
    const float* input,      // (batch, in_c, in_h, in_w)
    const float* weights,    // (out_c, in_c, k, k)
    const float* bias,       // (out_c)
    float* output,           // (batch, out_c, out_h, out_w)
    int batch_size,
    int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    // Each thread computes one output pixel
    int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int total_outputs = batch_size * out_c * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode index to (b, oc, oh, ow)
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_c;
    int b = idx / (out_w * out_h * out_c);

    // Compute convolution for this output pixel
    float sum = bias[oc];

    for (int ic = 0; ic < in_c; ic++) {
        for (int kh = 0; kh < k; kh++) {
            for (int kw = 0; kw < k; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                // Check boundaries (padding)
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                    int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

// ReLU Kernel
// Each thread processes one element
__global__ void naive_relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = fmaxf(0.0f, data[idx]);
    }
}

// ReLU Backward Kernel
// Gradient: d_input = d_output * (input > 0)
__global__ void naive_relu_backward_kernel(
    const float* output,     // Forward pass output (after ReLU)
    const float* d_output,   // Gradient w.r.t output
    float* d_input,          // Gradient w.r.t input
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        d_input[idx] = (output[idx] > 0.0f) ? d_output[idx] : 0.0f;
    }
}

// MaxPooling 2x2 Forward Kernel
// Each thread computes one output element
__global__ void naive_maxpool2d_forward_kernel(
    const float* input,      // (batch, c, in_h, in_w)
    float* output,           // (batch, c, out_h, out_w)
    int* indices,            // Store max indices for backward
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode index
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int b = idx / (out_w * out_h * channels);

    // Find maximum in 2x2 window
    int ih_start = oh * 2;
    int iw_start = ow * 2;

    float max_val = -1e38f;
    int max_idx = 0;

    for (int kh = 0; kh < 2; kh++) {
        for (int kw = 0; kw < 2; kw++) {
            int ih = ih_start + kh;
            int iw = iw_start + kw;

            if (ih < in_h && iw < in_w) {
                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                if (input[input_idx] > max_val) {
                    max_val = input[input_idx];
                    max_idx = kh * 2 + kw;  // Local index within 2x2 window
                }
            }
        }
    }

    output[idx] = max_val;
    indices[idx] = max_idx;
}

// MaxPooling 2x2 Backward Kernel
__global__ void naive_maxpool2d_backward_kernel(
    const float* d_output,   // Gradient w.r.t output (batch, c, out_h, out_w)
    const int* indices,      // Max indices from forward pass
    float* d_input,          // Gradient w.r.t input (batch, c, in_h, in_w)
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode index
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int b = idx / (out_w * out_h * channels);

    // Get max index from forward pass
    int max_idx = indices[idx];
    int kh = max_idx / 2;
    int kw = max_idx % 2;

    int ih = oh * 2 + kh;
    int iw = ow * 2 + kw;

    if (ih < in_h && iw < in_w) {
        int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
        atomicAdd(&d_input[input_idx], d_output[idx]);
    }
}

// Upsampling Kernel (Nearest Neighbor)
// Each thread computes one output pixel
__global__ void naive_upsample2d_forward_kernel(
    const float* input,      // (batch, c, in_h, in_w)
    float* output,           // (batch, c, out_h, out_w)
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode index
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int b = idx / (out_w * out_h * channels);

    // Map output coordinates back to input (divide by 2)
    int ih = oh / 2;
    int iw = ow / 2;

    int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
    output[idx] = input[input_idx];
}

// Upsampling Backward Kernel
__global__ void naive_upsample2d_backward_kernel(
    const float* d_output,   // Gradient w.r.t output (batch, c, out_h, out_w)
    float* d_input,          // Gradient w.r.t input (batch, c, in_h, in_w)
    int batch_size, int channels,
    int in_h, int in_w,
    int out_h, int out_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode index
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int b = idx / (out_w * out_h * channels);

    // Map to input coordinates
    int ih = oh / 2;
    int iw = ow / 2;

    int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;

    // Accumulate gradients (multiple output pixels map to same input)
    atomicAdd(&d_input[input_idx], d_output[idx]);
}

// MSE Loss Kernel with Parallel Reduction
__global__ void naive_mse_loss_kernel(
    const float* predicted,  // (batch, c, h, w)
    const float* target,     // (batch, c, h, w)
    float* loss,             // Single output value
    int size
) {
    __shared__ float partial_sum[256];

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int tid = threadIdx.x;

    // Compute partial sum for this thread
    float sum = 0.0f;
    if (idx < size) {
        float diff = predicted[idx] - target[idx];
        sum = diff * diff;
    }
    partial_sum[tid] = sum;
    __syncthreads();

    // Parallel reduction in shared memory
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            partial_sum[tid] += partial_sum[tid + s];
        }
        __syncthreads();
    }

    // First thread in block writes result
    if (tid == 0) {
        atomicAdd(loss, partial_sum[0]);
    }
}

// MSE Loss Gradient Kernel
__global__ void naive_mse_loss_gradient_kernel(
    const float* predicted,  // (batch, c, h, w)
    const float* target,     // (batch, c, h, w)
    float* d_predicted,      // Gradient output
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        // Gradient of MSE: 2 * (predicted - target) / size
        d_predicted[idx] = 2.0f * (predicted[idx] - target[idx]) / (float)size;
    }
}

// Convolution Backward Kernel - Compute gradient w.r.t input
__global__ void naive_conv2d_backward_input_kernel(
    const float* d_output,   // Gradient w.r.t output (batch, out_c, out_h, out_w)
    const float* weights,    // (out_c, in_c, k, k)
    float* d_input,          // Gradient w.r.t input (batch, in_c, in_h, in_w)
    int batch_size,
    int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * in_c * in_h * in_w;
    if (idx >= total_inputs) return;

    // Decode index to (b, ic, ih, iw)
    int iw = idx % in_w;
    int ih = (idx / in_w) % in_h;
    int ic = (idx / (in_w * in_h)) % in_c;
    int b = idx / (in_w * in_h * in_c);

    float sum = 0.0f;

    // Sum over all output positions that used this input pixel
    for (int oc = 0; oc < out_c; oc++) {
        for (int kh = 0; kh < k; kh++) {
            for (int kw = 0; kw < k; kw++) {
                // Compute which output position (oh, ow) used this input (ih, iw)
                int oh = ih + padding - kh;
                int ow = iw + padding - kw;

                // Check if this output position is valid
                if (oh % stride == 0 && ow % stride == 0) {
                    oh /= stride;
                    ow /= stride;

                    if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
                        int output_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                        int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                        sum += d_output[output_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }

    d_input[idx] = sum;
}

// Convolution Backward Kernel - Compute gradient w.r.t weights
__global__ void naive_conv2d_backward_weights_kernel(
    const float* input,      // Forward pass input (batch, in_c, in_h, in_w)
    const float* d_output,   // Gradient w.r.t output (batch, out_c, out_h, out_w)
    float* d_weights,        // Gradient w.r.t weights (out_c, in_c, k, k)
    float* d_bias,           // Gradient w.r.t bias (out_c)
    int batch_size,
    int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_c * in_c * k * k;
    if (idx >= total_weights) return;

    // Decode index to (oc, ic, kh, kw)
    int kw = idx % k;
    int kh = (idx / k) % k;
    int ic = (idx / (k * k)) % in_c;
    int oc = idx / (k * k * in_c);

    float sum = 0.0f;

    // Sum gradients over all batch samples and spatial positions
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                    int output_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    sum += input[input_idx] * d_output[output_idx];
                }
            }
        }
    }

    d_weights[idx] = sum;
}

// Compute bias gradient
__global__ void naive_conv2d_backward_bias_kernel(
    const float* d_output,   // Gradient w.r.t output (batch, out_c, out_h, out_w)
    float* d_bias,           // Gradient w.r.t bias (out_c)
    int batch_size,
    int out_c, int out_h, int out_w
) {
    int oc = blockIdx.x * blockDim.x + threadIdx.x;
    if (oc >= out_c) return;

    float sum = 0.0f;
    for (int b = 0; b < batch_size; b++) {
        for (int oh = 0; oh < out_h; oh++) {
            for (int ow = 0; ow < out_w; ow++) {
                int idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                sum += d_output[idx];
            }
        }
    }

    d_bias[oc] = sum;
}
