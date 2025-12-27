// autoencoder.cu
// Autoencoder implementation for CIFAR-10 (32x32 RGB images)
// Uses FP16 for activations, FP32 for master weights

#include "autoencoder.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>

// ============================================================================
// External kernel declarations
// ============================================================================

// Forward Conv2D (from gpu_conv2d_fp16_forward_v6.cu)
extern "C" {
void launch_conv2d_fp16_wmma_optimized(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H, int W, int C, int K, cudaStream_t stream);

void launch_conv2d_relu_fp16_wmma_optimized(
    const half* input, const half* weight, const half* bias,
    half* output, half* conv_output,
    int N, int H, int W, int C, int K, cudaStream_t stream);
}

// Backward kernels (from gpu_conv2d_fp16_backward_v8.cu)
extern "C" {
void launch_relu_backward_opt(const half* grad_out, const half* conv_out, half* grad_in, int size, cudaStream_t stream);
void launch_conv2d_backward_input_opt(const half* grad_out, const half* weight, half* grad_in,
    int N, int H, int W, int C, int H_out, int W_out, int K, cudaStream_t stream);
void launch_conv2d_backward_weight_sgd(const half* grad_out, const half* input,
    float* grad_weight, float* grad_bias,
    float* master_weight, half* weight, float* master_bias, half* bias,
    float lr, int N, int H, int W, int C, int H_out, int W_out, int K, cudaStream_t stream);
void launch_fused_relu_backward_input_twopass(const half* upstream_grad, const half* conv_output,
    const half* weight, half* grad_input, half* temp_buffer,
    int N, int H, int W, int C, int H_out, int W_out, int K, cudaStream_t stream);
}

// Autoencoder ops (from autoencoder_ops.cu)
extern "C" {
void launch_maxpool2d_forward(const half* input, half* output, int8_t* max_indices,
    int N, int H, int W, int C, cudaStream_t stream);
void launch_maxpool2d_backward(const half* grad_output, const int8_t* max_indices, half* grad_input,
    int N, int H, int W, int C, cudaStream_t stream);
void launch_upsample2d_forward(const half* input, half* output,
    int N, int H, int W, int C, cudaStream_t stream);
void launch_upsample2d_backward(const half* grad_output, half* grad_input,
    int N, int H, int W, int C, cudaStream_t stream);
void launch_mse_loss_grad(const half* pred, const half* target, half* grad,
    float* loss, float* partial_buffer, int size, int elements_per_sample, cudaStream_t stream);
}

// ============================================================================
// Initialization Kernels
// ============================================================================

__global__ void init_weights_kernel(float* weights, int size, float scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    // Simple LCG random
    unsigned int state = seed + idx * 1099087573u;
    state = state * 1103515245u + 12345u;
    float r1 = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    state = state * 1103515245u + 12345u;
    float r2 = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    
    // Box-Muller transform
    float u1 = fmaxf(r1, 1e-7f);
    float u2 = r2;
    float z = sqrtf(-2.0f * logf(u1)) * cosf(2.0f * 3.14159265f * u2);
    
    weights[idx] = z * scale;
}

__global__ void copy_fp32_to_fp16(const float* src, half* dst, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

__global__ void zero_fp16(half* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data[idx] = __float2half(0.0f);
    }
}

// ============================================================================
// Constructor / Destructor
// ============================================================================

Autoencoder::Autoencoder(int batch, float lr, const std::string& weights_path)
    : batch_size(batch), learning_rate(lr) {
    cudaStreamCreate(&stream);
    allocate_layers();
    
    if (!weights_path.empty() && load_weights(weights_path)) {
        printf("Loaded weights from '%s'\n", weights_path.c_str());
    } else {
        init_weights();
        if (!weights_path.empty()) {
            printf("Warning: Could not load '%s', initialized fresh weights\n", weights_path.c_str());
        }
    }
}

Autoencoder::~Autoencoder() {
    cudaStreamDestroy(stream);
    free_layers();
}

// ============================================================================
// Layer Allocation
// ============================================================================

void Autoencoder::allocate_layers() {
    int N = batch_size;
    
    // Conv1: (N, 32, 32, 3) -> (N, 32, 32, 256)
    conv1.C_in = 3; conv1.C_out = 256; conv1.H = 32; conv1.W = 32; conv1.has_relu = true;
    cudaMalloc(&conv1.weight, conv1.weight_size() * sizeof(half));
    cudaMalloc(&conv1.bias, conv1.bias_size() * sizeof(half));
    cudaMalloc(&conv1.master_weight, conv1.weight_size() * sizeof(float));
    cudaMalloc(&conv1.master_bias, conv1.bias_size() * sizeof(float));
    cudaMalloc(&conv1.grad_weight, conv1.weight_size() * sizeof(float));
    cudaMalloc(&conv1.grad_bias, conv1.bias_size() * sizeof(float));
    cudaMalloc(&conv1.input_saved, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&conv1.conv_output, N * 32 * 32 * 256 * sizeof(half));
    cudaMalloc(&conv1.output, N * 32 * 32 * 256 * sizeof(half));
    cudaMalloc(&conv1.grad_input, N * 32 * 32 * 3 * sizeof(half));
    conv1.temp_buffer = nullptr;
    
    // Pool1: (N, 32, 32, 256) -> (N, 16, 16, 256)
    pool1.N = N; pool1.H = 32; pool1.W = 32; pool1.C = 256;
    cudaMalloc(&pool1.max_indices, N * 16 * 16 * 256 * sizeof(int8_t));
    cudaMalloc(&pool1.output, N * 16 * 16 * 256 * sizeof(half));
    cudaMalloc(&pool1.grad_input, N * 32 * 32 * 256 * sizeof(half));
    
    // Conv2: (N, 16, 16, 256) -> (N, 16, 16, 128)
    conv2.C_in = 256; conv2.C_out = 128; conv2.H = 16; conv2.W = 16; conv2.has_relu = true;
    cudaMalloc(&conv2.weight, conv2.weight_size() * sizeof(half));
    cudaMalloc(&conv2.bias, conv2.bias_size() * sizeof(half));
    cudaMalloc(&conv2.master_weight, conv2.weight_size() * sizeof(float));
    cudaMalloc(&conv2.master_bias, conv2.bias_size() * sizeof(float));
    cudaMalloc(&conv2.grad_weight, conv2.weight_size() * sizeof(float));
    cudaMalloc(&conv2.grad_bias, conv2.bias_size() * sizeof(float));
    cudaMalloc(&conv2.input_saved, N * 16 * 16 * 256 * sizeof(half));
    cudaMalloc(&conv2.conv_output, N * 16 * 16 * 128 * sizeof(half));
    cudaMalloc(&conv2.output, N * 16 * 16 * 128 * sizeof(half));
    cudaMalloc(&conv2.grad_input, N * 16 * 16 * 256 * sizeof(half));
    cudaMalloc(&conv2.temp_buffer, N * 16 * 16 * 128 * sizeof(half));
    
    // Pool2: (N, 16, 16, 128) -> (N, 8, 8, 128)
    pool2.N = N; pool2.H = 16; pool2.W = 16; pool2.C = 128;
    cudaMalloc(&pool2.max_indices, N * 8 * 8 * 128 * sizeof(int8_t));
    cudaMalloc(&pool2.output, N * 8 * 8 * 128 * sizeof(half));
    cudaMalloc(&pool2.grad_input, N * 16 * 16 * 128 * sizeof(half));
    
    // Conv3: (N, 8, 8, 128) -> (N, 8, 8, 128) [latent]
    conv3.C_in = 128; conv3.C_out = 128; conv3.H = 8; conv3.W = 8; conv3.has_relu = true;
    cudaMalloc(&conv3.weight, conv3.weight_size() * sizeof(half));
    cudaMalloc(&conv3.bias, conv3.bias_size() * sizeof(half));
    cudaMalloc(&conv3.master_weight, conv3.weight_size() * sizeof(float));
    cudaMalloc(&conv3.master_bias, conv3.bias_size() * sizeof(float));
    cudaMalloc(&conv3.grad_weight, conv3.weight_size() * sizeof(float));
    cudaMalloc(&conv3.grad_bias, conv3.bias_size() * sizeof(float));
    cudaMalloc(&conv3.input_saved, N * 8 * 8 * 128 * sizeof(half));
    cudaMalloc(&conv3.conv_output, N * 8 * 8 * 128 * sizeof(half));
    cudaMalloc(&conv3.output, N * 8 * 8 * 128 * sizeof(half));
    cudaMalloc(&conv3.grad_input, N * 8 * 8 * 128 * sizeof(half));
    cudaMalloc(&conv3.temp_buffer, N * 8 * 8 * 128 * sizeof(half));
    
    // Up1: (N, 8, 8, 128) -> (N, 16, 16, 128)
    up1.N = N; up1.H = 8; up1.W = 8; up1.C = 128;
    cudaMalloc(&up1.output, N * 16 * 16 * 128 * sizeof(half));
    cudaMalloc(&up1.grad_input, N * 8 * 8 * 128 * sizeof(half));
    
    // Conv4: (N, 16, 16, 128) -> (N, 16, 16, 256)
    conv4.C_in = 128; conv4.C_out = 256; conv4.H = 16; conv4.W = 16; conv4.has_relu = true;
    cudaMalloc(&conv4.weight, conv4.weight_size() * sizeof(half));
    cudaMalloc(&conv4.bias, conv4.bias_size() * sizeof(half));
    cudaMalloc(&conv4.master_weight, conv4.weight_size() * sizeof(float));
    cudaMalloc(&conv4.master_bias, conv4.bias_size() * sizeof(float));
    cudaMalloc(&conv4.grad_weight, conv4.weight_size() * sizeof(float));
    cudaMalloc(&conv4.grad_bias, conv4.bias_size() * sizeof(float));
    cudaMalloc(&conv4.input_saved, N * 16 * 16 * 128 * sizeof(half));
    cudaMalloc(&conv4.conv_output, N * 16 * 16 * 256 * sizeof(half));
    cudaMalloc(&conv4.output, N * 16 * 16 * 256 * sizeof(half));
    cudaMalloc(&conv4.grad_input, N * 16 * 16 * 128 * sizeof(half));
    cudaMalloc(&conv4.temp_buffer, N * 16 * 16 * 256 * sizeof(half));
    
    // Up2: (N, 16, 16, 256) -> (N, 32, 32, 256)
    up2.N = N; up2.H = 16; up2.W = 16; up2.C = 256;
    cudaMalloc(&up2.output, N * 32 * 32 * 256 * sizeof(half));
    cudaMalloc(&up2.grad_input, N * 16 * 16 * 256 * sizeof(half));
    
    // Conv5: (N, 32, 32, 256) -> (N, 32, 32, 3) [no activation]
    conv5.C_in = 256; conv5.C_out = 3; conv5.H = 32; conv5.W = 32; conv5.has_relu = false;
    cudaMalloc(&conv5.weight, conv5.weight_size() * sizeof(half));
    cudaMalloc(&conv5.bias, conv5.bias_size() * sizeof(half));
    cudaMalloc(&conv5.master_weight, conv5.weight_size() * sizeof(float));
    cudaMalloc(&conv5.master_bias, conv5.bias_size() * sizeof(float));
    cudaMalloc(&conv5.grad_weight, conv5.weight_size() * sizeof(float));
    cudaMalloc(&conv5.grad_bias, conv5.bias_size() * sizeof(float));
    cudaMalloc(&conv5.input_saved, N * 32 * 32 * 256 * sizeof(half));
    cudaMalloc(&conv5.conv_output, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&conv5.output, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&conv5.grad_input, N * 32 * 32 * 256 * sizeof(half));
    conv5.temp_buffer = nullptr;
    
    // I/O buffers
    cudaMalloc(&input, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&output, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&target, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&latent, N * 8 * 8 * 128 * sizeof(half));  // For encode()
    
    // Loss buffers
    int output_size = N * 32 * 32 * 3;
    int num_blocks = (output_size + 255) / 256;
    cudaMalloc(&loss, sizeof(float));
    cudaMalloc(&loss_partial, num_blocks * sizeof(float));
    cudaMalloc(&grad_output, output_size * sizeof(half));
}

void Autoencoder::free_layers() {
    // Conv1
    cudaFree(conv1.weight); cudaFree(conv1.bias);
    cudaFree(conv1.master_weight); cudaFree(conv1.master_bias);
    cudaFree(conv1.grad_weight); cudaFree(conv1.grad_bias);
    cudaFree(conv1.input_saved); cudaFree(conv1.conv_output);
    cudaFree(conv1.output); cudaFree(conv1.grad_input);
    
    // Pool1
    cudaFree(pool1.max_indices); cudaFree(pool1.output); cudaFree(pool1.grad_input);
    
    // Conv2
    cudaFree(conv2.weight); cudaFree(conv2.bias);
    cudaFree(conv2.master_weight); cudaFree(conv2.master_bias);
    cudaFree(conv2.grad_weight); cudaFree(conv2.grad_bias);
    cudaFree(conv2.input_saved); cudaFree(conv2.conv_output);
    cudaFree(conv2.output); cudaFree(conv2.grad_input);
    cudaFree(conv2.temp_buffer);
    
    // Pool2
    cudaFree(pool2.max_indices); cudaFree(pool2.output); cudaFree(pool2.grad_input);
    
    // Conv3
    cudaFree(conv3.weight); cudaFree(conv3.bias);
    cudaFree(conv3.master_weight); cudaFree(conv3.master_bias);
    cudaFree(conv3.grad_weight); cudaFree(conv3.grad_bias);
    cudaFree(conv3.input_saved); cudaFree(conv3.conv_output);
    cudaFree(conv3.output); cudaFree(conv3.grad_input);
    cudaFree(conv3.temp_buffer);
    
    // Up1
    cudaFree(up1.output); cudaFree(up1.grad_input);
    
    // Conv4
    cudaFree(conv4.weight); cudaFree(conv4.bias);
    cudaFree(conv4.master_weight); cudaFree(conv4.master_bias);
    cudaFree(conv4.grad_weight); cudaFree(conv4.grad_bias);
    cudaFree(conv4.input_saved); cudaFree(conv4.conv_output);
    cudaFree(conv4.output); cudaFree(conv4.grad_input);
    cudaFree(conv4.temp_buffer);
    
    // Up2
    cudaFree(up2.output); cudaFree(up2.grad_input);
    
    // Conv5
    cudaFree(conv5.weight); cudaFree(conv5.bias);
    cudaFree(conv5.master_weight); cudaFree(conv5.master_bias);
    cudaFree(conv5.grad_weight); cudaFree(conv5.grad_bias);
    cudaFree(conv5.input_saved); cudaFree(conv5.conv_output);
    cudaFree(conv5.output); cudaFree(conv5.grad_input);
    
    // I/O
    cudaFree(input); cudaFree(output); cudaFree(target); cudaFree(latent);
    cudaFree(loss); cudaFree(loss_partial); cudaFree(grad_output);
}

void Autoencoder::init_weights() {
    auto init_conv = [this](Conv2DLayer& conv, unsigned int seed) {
        int w_size = conv.weight_size();
        int b_size = conv.bias_size();
        
        // Xavier/Glorot initialization
        float fan_in = conv.C_in * 9;
        float fan_out = conv.C_out * 9;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        
        int block = 256;
        int grid = (w_size + block - 1) / block;
        init_weights_kernel<<<grid, block, 0, stream>>>(conv.master_weight, w_size, scale, seed);
        copy_fp32_to_fp16<<<grid, block, 0, stream>>>(conv.master_weight, conv.weight, w_size);
        
        // Zero bias
        cudaMemsetAsync(conv.master_bias, 0, b_size * sizeof(float), stream);
        grid = (b_size + block - 1) / block;
        zero_fp16<<<grid, block, 0, stream>>>(conv.bias, b_size);
    };
    
    init_conv(conv1, 12345);
    init_conv(conv2, 23456);
    init_conv(conv3, 34567);
    init_conv(conv4, 45678);
    init_conv(conv5, 56789);
    
    cudaStreamSynchronize(stream);
}

// ============================================================================
// Forward Pass
// ============================================================================

void Autoencoder::forward() {
    int N = batch_size;
    
    // ========== ENCODER ==========
    
    // Save input for backward
    cudaMemcpyAsync(conv1.input_saved, input, N * 32 * 32 * 3 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv1 + ReLU: (N, 32, 32, 3) -> (N, 32, 32, 256)
    launch_conv2d_relu_fp16_wmma_optimized(
        input, conv1.weight, conv1.bias, conv1.output, conv1.conv_output,
        N, 32, 32, 3, 256, stream);
    
    // Pool1: (N, 32, 32, 256) -> (N, 16, 16, 256)
    launch_maxpool2d_forward(conv1.output, pool1.output, pool1.max_indices,
                             N, 32, 32, 256, stream);
    
    // Save for conv2 backward
    cudaMemcpyAsync(conv2.input_saved, pool1.output, N * 16 * 16 * 256 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv2 + ReLU: (N, 16, 16, 256) -> (N, 16, 16, 128)
    launch_conv2d_relu_fp16_wmma_optimized(
        pool1.output, conv2.weight, conv2.bias, conv2.output, conv2.conv_output,
        N, 16, 16, 256, 128, stream);
    
    // Pool2: (N, 16, 16, 128) -> (N, 8, 8, 128)
    launch_maxpool2d_forward(conv2.output, pool2.output, pool2.max_indices,
                             N, 16, 16, 128, stream);
    
    // Save for conv3 backward
    cudaMemcpyAsync(conv3.input_saved, pool2.output, N * 8 * 8 * 128 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv3 + ReLU: (N, 8, 8, 128) -> (N, 8, 8, 128) [LATENT]
    launch_conv2d_relu_fp16_wmma_optimized(
        pool2.output, conv3.weight, conv3.bias, conv3.output, conv3.conv_output,
        N, 8, 8, 128, 128, stream);
    
    // ========== DECODER ==========
    
    // Up1: (N, 8, 8, 128) -> (N, 16, 16, 128)
    launch_upsample2d_forward(conv3.output, up1.output, N, 8, 8, 128, stream);
    
    // Save for conv4 backward
    cudaMemcpyAsync(conv4.input_saved, up1.output, N * 16 * 16 * 128 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv4 + ReLU: (N, 16, 16, 128) -> (N, 16, 16, 256)
    launch_conv2d_relu_fp16_wmma_optimized(
        up1.output, conv4.weight, conv4.bias, conv4.output, conv4.conv_output,
        N, 16, 16, 128, 256, stream);
    
    // Up2: (N, 16, 16, 256) -> (N, 32, 32, 256)
    launch_upsample2d_forward(conv4.output, up2.output, N, 16, 16, 256, stream);
    
    // Save for conv5 backward
    cudaMemcpyAsync(conv5.input_saved, up2.output, N * 32 * 32 * 256 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv5 (no ReLU): (N, 32, 32, 256) -> (N, 32, 32, 3)
    launch_conv2d_fp16_wmma_optimized(
        up2.output, conv5.weight, conv5.bias, conv5.output,
        N, 32, 32, 256, 3, stream);
    
    // Copy to output buffer
    cudaMemcpyAsync(output, conv5.output, N * 32 * 32 * 3 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
}

// ============================================================================
// Encode Only (get latent features)
// ============================================================================

half* Autoencoder::encode(bool save_features) {
    int N = batch_size;
    
    // Conv1 + ReLU
    launch_conv2d_relu_fp16_wmma_optimized(
        input, conv1.weight, conv1.bias, conv1.output, conv1.conv_output,
        N, 32, 32, 3, 256, stream);
    
    // Pool1
    launch_maxpool2d_forward(conv1.output, pool1.output, pool1.max_indices,
                             N, 32, 32, 256, stream);
    
    // Conv2 + ReLU
    launch_conv2d_relu_fp16_wmma_optimized(
        pool1.output, conv2.weight, conv2.bias, conv2.output, conv2.conv_output,
        N, 16, 16, 256, 128, stream);
    
    // Pool2
    launch_maxpool2d_forward(conv2.output, pool2.output, pool2.max_indices,
                             N, 16, 16, 128, stream);
    
    // Conv3 + ReLU -> LATENT
    launch_conv2d_relu_fp16_wmma_optimized(
        pool2.output, conv3.weight, conv3.bias, conv3.output, conv3.conv_output,
        N, 8, 8, 128, 128, stream);
    
    // Copy latent to dedicated buffer
    cudaMemcpyAsync(latent, conv3.output, N * 8 * 8 * 128 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    if (save_features) {
        // Copy to host and return
        cudaStreamSynchronize(stream);
        half* h_latent = new half[latent_buffer_size()];
        cudaMemcpy(h_latent, latent, latent_buffer_size() * sizeof(half), cudaMemcpyDeviceToHost);
        return h_latent;
    }
    
    return nullptr;  // Latent stays on GPU
}

// ============================================================================
// Backward Pass
// ============================================================================

float Autoencoder::backward() {
    int N = batch_size;
    int output_size = N * 32 * 32 * 3;
    int elements_total = N * 32 * 32 * 3;
    
    // MSE loss and gradient
    launch_mse_loss_grad(output, target, grad_output, loss, loss_partial,
                         output_size, elements_total, stream);
    
    float h_loss;
    cudaMemcpyAsync(&h_loss, loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
    
    // ========== DECODER BACKWARD ==========
    
    // Conv5 backward (no ReLU)
    launch_conv2d_backward_weight_sgd(
        grad_output, conv5.input_saved,
        conv5.grad_weight, conv5.grad_bias,
        conv5.master_weight, conv5.weight, conv5.master_bias, conv5.bias,
        learning_rate, N, 32, 32, 256, 32, 32, 3, stream);
    
    launch_conv2d_backward_input_opt(
        grad_output, conv5.weight, conv5.grad_input,
        N, 32, 32, 256, 32, 32, 3, stream);
    
    // Up2 backward
    launch_upsample2d_backward(conv5.grad_input, up2.grad_input, N, 16, 16, 256, stream);
    
    // Conv4 backward
    launch_conv2d_backward_weight_sgd(
        up2.grad_input, conv4.input_saved,
        conv4.grad_weight, conv4.grad_bias,
        conv4.master_weight, conv4.weight, conv4.master_bias, conv4.bias,
        learning_rate, N, 16, 16, 128, 16, 16, 256, stream);
    
    launch_fused_relu_backward_input_twopass(
        up2.grad_input, conv4.conv_output, conv4.weight, conv4.grad_input, conv4.temp_buffer,
        N, 16, 16, 128, 16, 16, 256, stream);
    
    // Up1 backward
    launch_upsample2d_backward(conv4.grad_input, up1.grad_input, N, 8, 8, 128, stream);
    
    // ========== ENCODER BACKWARD ==========
    
    // Conv3 backward
    launch_conv2d_backward_weight_sgd(
        up1.grad_input, conv3.input_saved,
        conv3.grad_weight, conv3.grad_bias,
        conv3.master_weight, conv3.weight, conv3.master_bias, conv3.bias,
        learning_rate, N, 8, 8, 128, 8, 8, 128, stream);
    
    launch_fused_relu_backward_input_twopass(
        up1.grad_input, conv3.conv_output, conv3.weight, conv3.grad_input, conv3.temp_buffer,
        N, 8, 8, 128, 8, 8, 128, stream);
    
    // Pool2 backward
    launch_maxpool2d_backward(conv3.grad_input, pool2.max_indices, pool2.grad_input,
                              N, 16, 16, 128, stream);
    
    // Conv2 backward
    launch_conv2d_backward_weight_sgd(
        pool2.grad_input, conv2.input_saved,
        conv2.grad_weight, conv2.grad_bias,
        conv2.master_weight, conv2.weight, conv2.master_bias, conv2.bias,
        learning_rate, N, 16, 16, 256, 16, 16, 128, stream);
    
    launch_fused_relu_backward_input_twopass(
        pool2.grad_input, conv2.conv_output, conv2.weight, conv2.grad_input, conv2.temp_buffer,
        N, 16, 16, 256, 16, 16, 128, stream);
    
    // Pool1 backward
    launch_maxpool2d_backward(conv2.grad_input, pool1.max_indices, pool1.grad_input,
                              N, 32, 32, 256, stream);
    
    // Conv1 backward
    launch_conv2d_backward_weight_sgd(
        pool1.grad_input, conv1.input_saved,
        conv1.grad_weight, conv1.grad_bias,
        conv1.master_weight, conv1.weight, conv1.master_bias, conv1.bias,
        learning_rate, N, 32, 32, 3, 32, 32, 256, stream);
    
    cudaStreamSynchronize(stream);
    return h_loss;
}

// ============================================================================
// Weight I/O
// ============================================================================

void Autoencoder::save_weights(const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "wb");
    if (!f) {
        printf("Failed to open %s for writing\n", filename.c_str());
        return;
    }
    
    auto save_conv = [&](Conv2DLayer& conv, const char* name) {
        int w_size = conv.weight_size();
        int b_size = conv.bias_size();
        
        std::vector<float> h_weight(w_size);
        std::vector<float> h_bias(b_size);
        cudaMemcpy(h_weight.data(), conv.master_weight, w_size * sizeof(float), cudaMemcpyDeviceToHost);
        cudaMemcpy(h_bias.data(), conv.master_bias, b_size * sizeof(float), cudaMemcpyDeviceToHost);
        
        int name_len = strlen(name);
        fwrite(&name_len, sizeof(int), 1, f);
        fwrite(name, 1, name_len, f);
        fwrite(&conv.C_in, sizeof(int), 1, f);
        fwrite(&conv.C_out, sizeof(int), 1, f);
        fwrite(&w_size, sizeof(int), 1, f);
        fwrite(&b_size, sizeof(int), 1, f);
        fwrite(h_weight.data(), sizeof(float), w_size, f);
        fwrite(h_bias.data(), sizeof(float), b_size, f);
        
        float w_min = h_weight[0], w_max = h_weight[0], w_sum = 0;
        for (int i = 0; i < w_size; i++) {
            w_min = fminf(w_min, h_weight[i]);
            w_max = fmaxf(w_max, h_weight[i]);
            w_sum += h_weight[i];
        }
        printf("  %s: weight[%d] min=%.4f max=%.4f mean=%.6f\n",
               name, w_size, w_min, w_max, w_sum / w_size);
    };
    
    save_conv(conv1, "conv1");
    save_conv(conv2, "conv2");
    save_conv(conv3, "conv3");
    save_conv(conv4, "conv4");
    save_conv(conv5, "conv5");
    
    fclose(f);
}

bool Autoencoder::load_weights(const std::string& filename) {
    FILE* f = fopen(filename.c_str(), "rb");
    if (!f) {
        return false;
    }
    
    auto load_conv = [&](Conv2DLayer& conv, const char* expected_name) -> bool {
        int name_len;
        if (fread(&name_len, sizeof(int), 1, f) != 1) return false;
        
        char name[256];
        if (fread(name, 1, name_len, f) != (size_t)name_len) return false;
        name[name_len] = '\0';
        
        if (strcmp(name, expected_name) != 0) {
            printf("Warning: Expected layer '%s' but got '%s'\n", expected_name, name);
            return false;
        }
        
        int c_in, c_out, w_size, b_size;
        if (fread(&c_in, sizeof(int), 1, f) != 1) return false;
        if (fread(&c_out, sizeof(int), 1, f) != 1) return false;
        if (fread(&w_size, sizeof(int), 1, f) != 1) return false;
        if (fread(&b_size, sizeof(int), 1, f) != 1) return false;
        
        if (c_in != conv.C_in || c_out != conv.C_out) {
            printf("Warning: Shape mismatch for '%s'\n", name);
            return false;
        }
        
        std::vector<float> h_weight(w_size);
        std::vector<float> h_bias(b_size);
        if (fread(h_weight.data(), sizeof(float), w_size, f) != (size_t)w_size) return false;
        if (fread(h_bias.data(), sizeof(float), b_size, f) != (size_t)b_size) return false;
        
        // Copy to GPU
        cudaMemcpy(conv.master_weight, h_weight.data(), w_size * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(conv.master_bias, h_bias.data(), b_size * sizeof(float), cudaMemcpyHostToDevice);
        
        // Also update FP16 weights
        int block = 256;
        int grid = (w_size + block - 1) / block;
        copy_fp32_to_fp16<<<grid, block, 0, stream>>>(conv.master_weight, conv.weight, w_size);
        grid = (b_size + block - 1) / block;
        copy_fp32_to_fp16<<<grid, block, 0, stream>>>(conv.master_bias, conv.bias, b_size);
        
        return true;
    };
    
    bool success = true;
    success = success && load_conv(conv1, "conv1");
    success = success && load_conv(conv2, "conv2");
    success = success && load_conv(conv3, "conv3");
    success = success && load_conv(conv4, "conv4");
    success = success && load_conv(conv5, "conv5");
    
    cudaStreamSynchronize(stream);
    fclose(f);
    return success;
}
