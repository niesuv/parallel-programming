#include "autoencoder_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>

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

// Forward declarations of kernels from autoencoder_gpu.cu
extern "C" {
__global__ void naive_conv2d_forward_kernel(const float*, const float*, const float*, float*,
                                            int, int, int, int, int, int, int, int, int, int);
__global__ void naive_relu_forward_kernel(float*, int);
__global__ void naive_maxpool2d_forward_kernel(const float*, float*, int*, int, int, int, int, int, int);
__global__ void naive_upsample2d_forward_kernel(const float*, float*, int, int, int, int, int, int);
__global__ void naive_mse_loss_kernel(const float*, const float*, float*, int);
__global__ void naive_mse_loss_gradient_kernel(const float*, const float*, float*, int);
__global__ void naive_relu_backward_kernel(const float*, const float*, float*, int);
__global__ void naive_maxpool2d_backward_kernel(const float*, const int*, float*, int, int, int, int, int, int);
__global__ void naive_upsample2d_backward_kernel(const float*, float*, int, int, int, int, int, int);
__global__ void naive_conv2d_backward_input_kernel(const float*, const float*, float*,
                                                    int, int, int, int, int, int, int, int, int, int);
__global__ void naive_conv2d_backward_weights_kernel(const float*, const float*, float*, float*,
                                                     int, int, int, int, int, int, int, int, int, int);
__global__ void naive_conv2d_backward_bias_kernel(const float*, float*, int, int, int, int);
}

// ============================================================================
// 2.3 GPU Forward Pass
// ============================================================================

void autoencoder_gpu_forward(Autoencoder_GPU* gpu_ae, const float* h_input, int batch_size) {
    const int threads = 256;

    // Copy input from host to device
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_input, h_input,
                          gpu_ae->output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Layer 1: enc_conv1 (3 -> 256, 32x32 -> 32x32)
    int blocks = (gpu_ae->enc1_out_size + threads - 1) / threads;
    naive_conv2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_input, gpu_ae->d_enc1_weights, gpu_ae->d_enc1_bias, gpu_ae->d_enc1_out,
        batch_size, 3, 32, 32, 256, 32, 32, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    // ReLU
    naive_relu_forward_kernel<<<blocks, threads>>>(gpu_ae->d_enc1_out, gpu_ae->enc1_out_size);
    CUDA_CHECK(cudaGetLastError());

    // Layer 2: enc_pool1 (32x32 -> 16x16)
    blocks = (gpu_ae->pool1_out_size + threads - 1) / threads;
    naive_maxpool2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc1_out, gpu_ae->d_pool1_out, gpu_ae->d_pool1_indices,
        batch_size, 256, 32, 32, 16, 16
    );
    CUDA_CHECK(cudaGetLastError());

    // Layer 3: enc_conv2 (256 -> 128, 16x16 -> 16x16)
    blocks = (gpu_ae->enc2_out_size + threads - 1) / threads;
    naive_conv2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_pool1_out, gpu_ae->d_enc2_weights, gpu_ae->d_enc2_bias, gpu_ae->d_enc2_out,
        batch_size, 256, 16, 16, 128, 16, 16, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    // ReLU
    naive_relu_forward_kernel<<<blocks, threads>>>(gpu_ae->d_enc2_out, gpu_ae->enc2_out_size);
    CUDA_CHECK(cudaGetLastError());

    // Layer 4: enc_pool2 (16x16 -> 8x8) - LATENT
    blocks = (gpu_ae->latent_size + threads - 1) / threads;
    naive_maxpool2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc2_out, gpu_ae->d_latent, gpu_ae->d_pool2_indices,
        batch_size, 128, 16, 16, 8, 8
    );
    CUDA_CHECK(cudaGetLastError());

    // Layer 5: dec_conv1 (128 -> 128, 8x8 -> 8x8)
    blocks = (gpu_ae->dec1_out_size + threads - 1) / threads;
    naive_conv2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_latent, gpu_ae->d_dec1_weights, gpu_ae->d_dec1_bias, gpu_ae->d_dec1_out,
        batch_size, 128, 8, 8, 128, 8, 8, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    // ReLU
    naive_relu_forward_kernel<<<blocks, threads>>>(gpu_ae->d_dec1_out, gpu_ae->dec1_out_size);
    CUDA_CHECK(cudaGetLastError());

    // Layer 6: dec_up1 (8x8 -> 16x16)
    blocks = (gpu_ae->up1_out_size + threads - 1) / threads;
    naive_upsample2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec1_out, gpu_ae->d_up1_out,
        batch_size, 128, 8, 8, 16, 16
    );
    CUDA_CHECK(cudaGetLastError());

    // Layer 7: dec_conv2 (128 -> 256, 16x16 -> 16x16)
    blocks = (gpu_ae->dec2_out_size + threads - 1) / threads;
    naive_conv2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_up1_out, gpu_ae->d_dec2_weights, gpu_ae->d_dec2_bias, gpu_ae->d_dec2_out,
        batch_size, 128, 16, 16, 256, 16, 16, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    // ReLU
    naive_relu_forward_kernel<<<blocks, threads>>>(gpu_ae->d_dec2_out, gpu_ae->dec2_out_size);
    CUDA_CHECK(cudaGetLastError());

    // Layer 8: dec_up2 (16x16 -> 32x32)
    blocks = (gpu_ae->up2_out_size + threads - 1) / threads;
    naive_upsample2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec2_out, gpu_ae->d_up2_out,
        batch_size, 256, 16, 16, 32, 32
    );
    CUDA_CHECK(cudaGetLastError());

    // Layer 9: dec_conv3 (256 -> 3, 32x32 -> 32x32) - OUTPUT
    blocks = (gpu_ae->output_size + threads - 1) / threads;
    naive_conv2d_forward_kernel<<<blocks, threads>>>(
        gpu_ae->d_up2_out, gpu_ae->d_dec3_weights, gpu_ae->d_dec3_bias, gpu_ae->d_output,
        batch_size, 256, 32, 32, 3, 32, 32, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    // Synchronize to ensure all kernels complete
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// 2.4 GPU Backward Pass
// ============================================================================

void autoencoder_gpu_backward(Autoencoder_GPU* gpu_ae, const float* h_target, int batch_size) {
    const int threads = 256;

    // Copy target from host to device
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_target, h_target,
                          gpu_ae->output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Compute loss gradient: d_output
    int blocks = (gpu_ae->output_size + threads - 1) / threads;
    naive_mse_loss_gradient_kernel<<<blocks, threads>>>(
        gpu_ae->d_output, gpu_ae->d_target, gpu_ae->d_grad_output, gpu_ae->output_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 9: dec_conv3 (256 -> 3)
    // Gradient w.r.t input (d_up2_out)
    blocks = (gpu_ae->up2_out_size + threads - 1) / threads;
    naive_conv2d_backward_input_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_output, gpu_ae->d_dec3_weights, gpu_ae->d_grad_up2_out,
        batch_size, 256, 32, 32, 3, 32, 32, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    // Gradient w.r.t weights
    int weight_blocks = (3 * 256 * 3 * 3 + threads - 1) / threads;
    naive_conv2d_backward_weights_kernel<<<weight_blocks, threads>>>(
        gpu_ae->d_up2_out, gpu_ae->d_grad_output,
        gpu_ae->d_dec3_d_weights, gpu_ae->d_dec3_d_bias,
        batch_size, 256, 32, 32, 3, 32, 32, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    int bias_blocks = (3 + threads - 1) / threads;
    naive_conv2d_backward_bias_kernel<<<bias_blocks, threads>>>(
        gpu_ae->d_grad_output, gpu_ae->d_dec3_d_bias, batch_size, 3, 32, 32
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 8: dec_up2 (upsample)
    blocks = (gpu_ae->dec2_out_size + threads - 1) / threads;
    // Initialize gradient to zero
    CUDA_CHECK(cudaMemset(gpu_ae->d_grad_dec2_out, 0, gpu_ae->dec2_out_size * sizeof(float)));
    blocks = (gpu_ae->up2_out_size + threads - 1) / threads;
    naive_upsample2d_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_up2_out, gpu_ae->d_grad_dec2_out,
        batch_size, 256, 16, 16, 32, 32
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward ReLU
    blocks = (gpu_ae->dec2_out_size + threads - 1) / threads;
    naive_relu_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec2_out, gpu_ae->d_grad_dec2_out, gpu_ae->d_grad_dec2_out, gpu_ae->dec2_out_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 7: dec_conv2 (128 -> 256)
    blocks = (gpu_ae->up1_out_size + threads - 1) / threads;
    naive_conv2d_backward_input_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_dec2_out, gpu_ae->d_dec2_weights, gpu_ae->d_grad_up1_out,
        batch_size, 128, 16, 16, 256, 16, 16, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    weight_blocks = (256 * 128 * 3 * 3 + threads - 1) / threads;
    naive_conv2d_backward_weights_kernel<<<weight_blocks, threads>>>(
        gpu_ae->d_up1_out, gpu_ae->d_grad_dec2_out,
        gpu_ae->d_dec2_d_weights, gpu_ae->d_dec2_d_bias,
        batch_size, 128, 16, 16, 256, 16, 16, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    bias_blocks = (256 + threads - 1) / threads;
    naive_conv2d_backward_bias_kernel<<<bias_blocks, threads>>>(
        gpu_ae->d_grad_dec2_out, gpu_ae->d_dec2_d_bias, batch_size, 256, 16, 16
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 6: dec_up1 (upsample)
    CUDA_CHECK(cudaMemset(gpu_ae->d_grad_dec1_out, 0, gpu_ae->dec1_out_size * sizeof(float)));
    blocks = (gpu_ae->up1_out_size + threads - 1) / threads;
    naive_upsample2d_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_up1_out, gpu_ae->d_grad_dec1_out,
        batch_size, 128, 8, 8, 16, 16
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward ReLU
    blocks = (gpu_ae->dec1_out_size + threads - 1) / threads;
    naive_relu_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec1_out, gpu_ae->d_grad_dec1_out, gpu_ae->d_grad_dec1_out, gpu_ae->dec1_out_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 5: dec_conv1 (128 -> 128)
    blocks = (gpu_ae->latent_size + threads - 1) / threads;
    naive_conv2d_backward_input_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_dec1_out, gpu_ae->d_dec1_weights, gpu_ae->d_grad_latent,
        batch_size, 128, 8, 8, 128, 8, 8, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    weight_blocks = (128 * 128 * 3 * 3 + threads - 1) / threads;
    naive_conv2d_backward_weights_kernel<<<weight_blocks, threads>>>(
        gpu_ae->d_latent, gpu_ae->d_grad_dec1_out,
        gpu_ae->d_dec1_d_weights, gpu_ae->d_dec1_d_bias,
        batch_size, 128, 8, 8, 128, 8, 8, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    bias_blocks = (128 + threads - 1) / threads;
    naive_conv2d_backward_bias_kernel<<<bias_blocks, threads>>>(
        gpu_ae->d_grad_dec1_out, gpu_ae->d_dec1_d_bias, batch_size, 128, 8, 8
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 4: enc_pool2 (maxpool)
    CUDA_CHECK(cudaMemset(gpu_ae->d_grad_enc2_out, 0, gpu_ae->enc2_out_size * sizeof(float)));
    blocks = (gpu_ae->latent_size + threads - 1) / threads;
    naive_maxpool2d_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_latent, gpu_ae->d_pool2_indices, gpu_ae->d_grad_enc2_out,
        batch_size, 128, 16, 16, 8, 8
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward ReLU
    blocks = (gpu_ae->enc2_out_size + threads - 1) / threads;
    naive_relu_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc2_out, gpu_ae->d_grad_enc2_out, gpu_ae->d_grad_enc2_out, gpu_ae->enc2_out_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 3: enc_conv2 (256 -> 128)
    blocks = (gpu_ae->pool1_out_size + threads - 1) / threads;
    naive_conv2d_backward_input_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_enc2_out, gpu_ae->d_enc2_weights, gpu_ae->d_grad_pool1_out,
        batch_size, 256, 16, 16, 128, 16, 16, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    weight_blocks = (128 * 256 * 3 * 3 + threads - 1) / threads;
    naive_conv2d_backward_weights_kernel<<<weight_blocks, threads>>>(
        gpu_ae->d_pool1_out, gpu_ae->d_grad_enc2_out,
        gpu_ae->d_enc2_d_weights, gpu_ae->d_enc2_d_bias,
        batch_size, 256, 16, 16, 128, 16, 16, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    bias_blocks = (128 + threads - 1) / threads;
    naive_conv2d_backward_bias_kernel<<<bias_blocks, threads>>>(
        gpu_ae->d_grad_enc2_out, gpu_ae->d_enc2_d_bias, batch_size, 128, 16, 16
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 2: enc_pool1 (maxpool)
    CUDA_CHECK(cudaMemset(gpu_ae->d_grad_enc1_out, 0, gpu_ae->enc1_out_size * sizeof(float)));
    blocks = (gpu_ae->pool1_out_size + threads - 1) / threads;
    naive_maxpool2d_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_grad_pool1_out, gpu_ae->d_pool1_indices, gpu_ae->d_grad_enc1_out,
        batch_size, 256, 32, 32, 16, 16
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward ReLU
    blocks = (gpu_ae->enc1_out_size + threads - 1) / threads;
    naive_relu_backward_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc1_out, gpu_ae->d_grad_enc1_out, gpu_ae->d_grad_enc1_out, gpu_ae->enc1_out_size
    );
    CUDA_CHECK(cudaGetLastError());

    // Backward Layer 1: enc_conv1 (3 -> 256)
    // We don't need d_input gradient (input is data, not parameters)
    // Only compute weight gradients
    weight_blocks = (256 * 3 * 3 * 3 + threads - 1) / threads;
    naive_conv2d_backward_weights_kernel<<<weight_blocks, threads>>>(
        gpu_ae->d_input, gpu_ae->d_grad_enc1_out,
        gpu_ae->d_enc1_d_weights, gpu_ae->d_enc1_d_bias,
        batch_size, 3, 32, 32, 256, 32, 32, 3, 1, 1
    );
    CUDA_CHECK(cudaGetLastError());

    bias_blocks = (256 + threads - 1) / threads;
    naive_conv2d_backward_bias_kernel<<<bias_blocks, threads>>>(
        gpu_ae->d_grad_enc1_out, gpu_ae->d_enc1_d_bias, batch_size, 256, 32, 32
    );
    CUDA_CHECK(cudaGetLastError());

    // Synchronize
    CUDA_CHECK(cudaDeviceSynchronize());
}

// ============================================================================
// 2.4 GPU Weight Update (SGD)
// ============================================================================

__global__ void sgd_update_kernel(
    float* weights,
    const float* gradients,
    float learning_rate,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= learning_rate * gradients[idx];
    }
}

void autoencoder_gpu_update_weights(Autoencoder_GPU* gpu_ae) {
    const int threads = 256;
    float lr = gpu_ae->learning_rate;

    // Update enc1
    int blocks = (256 * 3 * 3 * 3 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc1_weights, gpu_ae->d_enc1_d_weights, lr, 256 * 3 * 3 * 3
    );
    blocks = (256 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc1_bias, gpu_ae->d_enc1_d_bias, lr, 256
    );

    // Update enc2
    blocks = (128 * 256 * 3 * 3 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc2_weights, gpu_ae->d_enc2_d_weights, lr, 128 * 256 * 3 * 3
    );
    blocks = (128 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_enc2_bias, gpu_ae->d_enc2_d_bias, lr, 128
    );

    // Update dec1
    blocks = (128 * 128 * 3 * 3 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec1_weights, gpu_ae->d_dec1_d_weights, lr, 128 * 128 * 3 * 3
    );
    blocks = (128 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec1_bias, gpu_ae->d_dec1_d_bias, lr, 128
    );

    // Update dec2
    blocks = (256 * 128 * 3 * 3 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec2_weights, gpu_ae->d_dec2_d_weights, lr, 256 * 128 * 3 * 3
    );
    blocks = (256 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec2_bias, gpu_ae->d_dec2_d_bias, lr, 256
    );

    // Update dec3
    blocks = (3 * 256 * 3 * 3 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec3_weights, gpu_ae->d_dec3_d_weights, lr, 3 * 256 * 3 * 3
    );
    blocks = (3 + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        gpu_ae->d_dec3_bias, gpu_ae->d_dec3_d_bias, lr, 3
    );

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
}

// Compute MSE loss
float autoencoder_gpu_compute_loss(Autoencoder_GPU* gpu_ae, const float* h_target, int batch_size) {
    const int threads = 256;
    int blocks = (gpu_ae->output_size + threads - 1) / threads;

    // Allocate device memory for loss
    float* d_loss;
    CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
    CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));

    // Copy target to device
    CUDA_CHECK(cudaMemcpy(gpu_ae->d_target, h_target,
                          gpu_ae->output_size * sizeof(float), cudaMemcpyHostToDevice));

    // Compute loss
    naive_mse_loss_kernel<<<blocks, threads>>>(
        gpu_ae->d_output, gpu_ae->d_target, d_loss, gpu_ae->output_size
    );
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // Copy loss back to host
    float h_loss;
    CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_loss));

    return h_loss / (float)gpu_ae->output_size;
}
