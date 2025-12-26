// train_gpu.cu
// Train autoencoder on CIFAR-10 dataset
// Data format: NHWC (batch, height, width, channels)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

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
// CIFAR-10 Data Loading
// ============================================================================

struct CIFAR10Dataset {
    std::vector<uint8_t> images;  // [N, 32, 32, 3] in HWC format
    std::vector<uint8_t> labels;  // [N]
    int num_samples;
    
    bool load(const std::string& data_dir, bool train = true) {
        std::vector<std::string> files;
        if (train) {
            files = {
                data_dir + "/data_batch_1.bin",
                data_dir + "/data_batch_2.bin",
                data_dir + "/data_batch_3.bin",
                data_dir + "/data_batch_4.bin",
                data_dir + "/data_batch_5.bin"
            };
        } else {
            files = {data_dir + "/test_batch.bin"};
        }
        
        images.clear();
        labels.clear();
        
        for (const auto& filename : files) {
            FILE* f = fopen(filename.c_str(), "rb");
            if (!f) {
                printf("Failed to open: %s\n", filename.c_str());
                return false;
            }
            
            // Each record: 1 byte label + 3072 bytes image (CHW format in file)
            uint8_t record[3073];
            while (fread(record, 1, 3073, f) == 3073) {
                labels.push_back(record[0]);
                
                // Convert CHW to HWC
                // File format: R[1024], G[1024], B[1024]
                // We want: RGB at each pixel
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        int pixel_idx = h * 32 + w;
                        images.push_back(record[1 + pixel_idx]);           // R
                        images.push_back(record[1 + 1024 + pixel_idx]);    // G
                        images.push_back(record[1 + 2048 + pixel_idx]);    // B
                    }
                }
            }
            fclose(f);
        }
        
        num_samples = labels.size();
        printf("Loaded %d samples from %s\n", num_samples, train ? "training set" : "test set");
        return num_samples > 0;
    }
    
    // Get a batch of images as FP16, normalized to [0, 1]
    void get_batch(half* d_batch, const std::vector<int>& indices, cudaStream_t stream) {
        int batch_size = indices.size();
        std::vector<half> h_batch(batch_size * 32 * 32 * 3);
        
        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            for (int j = 0; j < 32 * 32 * 3; j++) {
                h_batch[i * 32 * 32 * 3 + j] = __float2half(images[idx * 32 * 32 * 3 + j] / 255.0f);
            }
        }
        
        cudaMemcpyAsync(d_batch, h_batch.data(), batch_size * 32 * 32 * 3 * sizeof(half),
                        cudaMemcpyHostToDevice, stream);
    }
};

// ============================================================================
// Autoencoder Layer Structures (same as autoencoder_full.cu)
// ============================================================================

struct Conv2DLayer {
    int C_in, C_out, H, W;
    half* weight;
    half* bias;
    float* master_weight;
    float* master_bias;
    float* grad_weight;
    float* grad_bias;
    half* input_saved;
    half* conv_output;
    half* output;
    half* grad_input;
    half* temp_buffer;
    bool has_relu;
    
    int weight_size() const { return C_out * 9 * C_in; }
    int bias_size() const { return C_out; }
};

struct MaxPool2DLayer {
    int N, H, W, C;
    int8_t* max_indices;
    half* output;
    half* grad_input;
};

struct Upsample2DLayer {
    int N, H, W, C;
    half* output;
    half* grad_input;
};

// ============================================================================
// Autoencoder Model
// ============================================================================

class Autoencoder {
public:
    int batch_size;
    float learning_rate;
    
    // Layers
    Conv2DLayer conv1, conv2, conv3, conv4, conv5;
    MaxPool2DLayer pool1, pool2;
    Upsample2DLayer up1, up2;
    
    // I/O
    half* input;
    half* output;
    half* target;
    
    // Loss
    float* loss;
    float* loss_partial;
    half* grad_output;
    
    cudaStream_t stream;
    
    Autoencoder(int batch, float lr = 0.01f) : batch_size(batch), learning_rate(lr) {
        cudaStreamCreate(&stream);
        allocate_layers();
        init_weights();
    }
    
    ~Autoencoder() {
        cudaStreamDestroy(stream);
        free_layers();
    }
    
    void allocate_layers();
    void free_layers();
    void init_weights();
    void forward();
    float backward();
    void save_weights(const char* filename);
};

// Xavier initialization kernel
__global__ void init_weights_kernel(float* weights, int size, float scale, unsigned int seed) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    unsigned int state = seed + idx * 1099087573u;
    state = state * 1103515245u + 12345u;
    float r1 = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    state = state * 1103515245u + 12345u;
    float r2 = (state & 0x7FFFFFFF) / (float)0x7FFFFFFF;
    
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
    
    // Conv3: (N, 8, 8, 128) -> (N, 8, 8, 128)
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
    
    // Conv5: (N, 32, 32, 256) -> (N, 32, 32, 3) - NO RELU
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
    
    // I/O
    cudaMalloc(&input, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&output, N * 32 * 32 * 3 * sizeof(half));
    cudaMalloc(&target, N * 32 * 32 * 3 * sizeof(half));
    
    // Loss
    int output_size = N * 32 * 32 * 3;
    int num_blocks = (output_size + 255) / 256;
    cudaMalloc(&loss, sizeof(float));
    cudaMalloc(&loss_partial, num_blocks * sizeof(float));
    cudaMalloc(&grad_output, output_size * sizeof(half));
}

void Autoencoder::free_layers() {
    cudaFree(conv1.weight); cudaFree(conv1.bias);
    cudaFree(conv1.master_weight); cudaFree(conv1.master_bias);
    cudaFree(conv1.grad_weight); cudaFree(conv1.grad_bias);
    cudaFree(conv1.input_saved); cudaFree(conv1.conv_output);
    cudaFree(conv1.output); cudaFree(conv1.grad_input);
    
    cudaFree(pool1.max_indices); cudaFree(pool1.output); cudaFree(pool1.grad_input);
    
    cudaFree(conv2.weight); cudaFree(conv2.bias);
    cudaFree(conv2.master_weight); cudaFree(conv2.master_bias);
    cudaFree(conv2.grad_weight); cudaFree(conv2.grad_bias);
    cudaFree(conv2.input_saved); cudaFree(conv2.conv_output);
    cudaFree(conv2.output); cudaFree(conv2.grad_input);
    cudaFree(conv2.temp_buffer);
    
    cudaFree(pool2.max_indices); cudaFree(pool2.output); cudaFree(pool2.grad_input);
    
    cudaFree(conv3.weight); cudaFree(conv3.bias);
    cudaFree(conv3.master_weight); cudaFree(conv3.master_bias);
    cudaFree(conv3.grad_weight); cudaFree(conv3.grad_bias);
    cudaFree(conv3.input_saved); cudaFree(conv3.conv_output);
    cudaFree(conv3.output); cudaFree(conv3.grad_input);
    cudaFree(conv3.temp_buffer);
    
    cudaFree(up1.output); cudaFree(up1.grad_input);
    
    cudaFree(conv4.weight); cudaFree(conv4.bias);
    cudaFree(conv4.master_weight); cudaFree(conv4.master_bias);
    cudaFree(conv4.grad_weight); cudaFree(conv4.grad_bias);
    cudaFree(conv4.input_saved); cudaFree(conv4.conv_output);
    cudaFree(conv4.output); cudaFree(conv4.grad_input);
    cudaFree(conv4.temp_buffer);
    
    cudaFree(up2.output); cudaFree(up2.grad_input);
    
    cudaFree(conv5.weight); cudaFree(conv5.bias);
    cudaFree(conv5.master_weight); cudaFree(conv5.master_bias);
    cudaFree(conv5.grad_weight); cudaFree(conv5.grad_bias);
    cudaFree(conv5.input_saved); cudaFree(conv5.conv_output);
    cudaFree(conv5.output); cudaFree(conv5.grad_input);
    
    cudaFree(input); cudaFree(output); cudaFree(target);
    cudaFree(loss); cudaFree(loss_partial); cudaFree(grad_output);
}

void Autoencoder::init_weights() {
    auto init_conv = [this](Conv2DLayer& conv, unsigned int seed) {
        int w_size = conv.weight_size();
        int b_size = conv.bias_size();
        
        float fan_in = conv.C_in * 9;
        float fan_out = conv.C_out * 9;
        float scale = sqrtf(2.0f / (fan_in + fan_out));
        
        int block = 256;
        int grid = (w_size + block - 1) / block;
        init_weights_kernel<<<grid, block, 0, stream>>>(conv.master_weight, w_size, scale, seed);
        copy_fp32_to_fp16<<<grid, block, 0, stream>>>(conv.master_weight, conv.weight, w_size);
        
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

void Autoencoder::forward() {
    int N = batch_size;
    
    // Save input for backward
    cudaMemcpyAsync(conv1.input_saved, input, N * 32 * 32 * 3 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv1 + ReLU
    launch_conv2d_relu_fp16_wmma_optimized(
        input, conv1.weight, conv1.bias, conv1.output, conv1.conv_output,
        N, 32, 32, 3, 256, stream);
    
    // Pool1
    launch_maxpool2d_forward(conv1.output, pool1.output, pool1.max_indices,
                             N, 32, 32, 256, stream);
    
    // Save for conv2 backward
    cudaMemcpyAsync(conv2.input_saved, pool1.output, N * 16 * 16 * 256 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv2 + ReLU
    launch_conv2d_relu_fp16_wmma_optimized(
        pool1.output, conv2.weight, conv2.bias, conv2.output, conv2.conv_output,
        N, 16, 16, 256, 128, stream);
    
    // Pool2
    launch_maxpool2d_forward(conv2.output, pool2.output, pool2.max_indices,
                             N, 16, 16, 128, stream);
    
    // Save for conv3 backward
    cudaMemcpyAsync(conv3.input_saved, pool2.output, N * 8 * 8 * 128 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv3 + ReLU (latent)
    launch_conv2d_relu_fp16_wmma_optimized(
        pool2.output, conv3.weight, conv3.bias, conv3.output, conv3.conv_output,
        N, 8, 8, 128, 128, stream);
    
    // Up1
    launch_upsample2d_forward(conv3.output, up1.output, N, 8, 8, 128, stream);
    
    // Save for conv4 backward
    cudaMemcpyAsync(conv4.input_saved, up1.output, N * 16 * 16 * 128 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv4 + ReLU
    launch_conv2d_relu_fp16_wmma_optimized(
        up1.output, conv4.weight, conv4.bias, conv4.output, conv4.conv_output,
        N, 16, 16, 128, 256, stream);
    
    // Up2
    launch_upsample2d_forward(conv4.output, up2.output, N, 16, 16, 256, stream);
    
    // Save for conv5 backward
    cudaMemcpyAsync(conv5.input_saved, up2.output, N * 32 * 32 * 256 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
    
    // Conv5 (no ReLU)
    launch_conv2d_fp16_wmma_optimized(
        up2.output, conv5.weight, conv5.bias, conv5.output,
        N, 32, 32, 256, 3, stream);
    
    // Copy to output
    cudaMemcpyAsync(output, conv5.output, N * 32 * 32 * 3 * sizeof(half),
                    cudaMemcpyDeviceToDevice, stream);
}

float Autoencoder::backward() {
    int N = batch_size;
    int output_size = N * 32 * 32 * 3;
    int elements_total = N * 32 * 32 * 3;  // Full batch normalization
    
    // MSE loss and gradient
    launch_mse_loss_grad(output, target, grad_output, loss, loss_partial,
                         output_size, elements_total, stream);
    
    float h_loss;
    cudaMemcpyAsync(&h_loss, loss, sizeof(float), cudaMemcpyDeviceToHost, stream);
    
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

void Autoencoder::save_weights(const char* filename) {
    FILE* f = fopen(filename, "wb");
    if (!f) {
        printf("Failed to open %s for writing\n", filename);
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

// ============================================================================
// Main Training Loop
// ============================================================================

int main(int argc, char** argv) {
    printf("=== CIFAR-10 Autoencoder Training ===\n\n");
    
    // Hyperparameters
    int batch_size = 64;
    float learning_rate = 0.003f;  // Start moderate, will decay
    int num_epochs = 30;
    std::string data_dir = "data";
    
    // Parse command line args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        }
    }
    
    printf("Hyperparameters:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Learning rate: %f\n", learning_rate);
    printf("  Epochs: %d\n", num_epochs);
    printf("  Data dir: %s\n\n", data_dir.c_str());
    
    // Load CIFAR-10
    CIFAR10Dataset train_data;
    if (!train_data.load(data_dir, true)) {
        printf("Failed to load training data from %s\n", data_dir.c_str());
        return 1;
    }
    
    // Create model
    printf("\nCreating model...\n");
    Autoencoder model(batch_size, learning_rate);
    printf("Model created.\n\n");
    
    // Training
    int num_batches = train_data.num_samples / batch_size;
    printf("Training: %d samples, %d batches per epoch\n\n", train_data.num_samples, num_batches);
    
    // Create index array for shuffling
    std::vector<int> indices(train_data.num_samples);
    for (int i = 0; i < train_data.num_samples; i++) indices[i] = i;
    
    std::mt19937 rng(42);
    
    cudaEvent_t epoch_start, epoch_end;
    cudaEventCreate(&epoch_start);
    cudaEventCreate(&epoch_end);
    
    float best_loss = 1e10f;  // Track best loss
    int epochs_without_improvement = 0;
    float current_lr = learning_rate;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Update model's learning rate
        model.learning_rate = current_lr;
        
        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), rng);
        
        float epoch_loss = 0.0f;
        int num_batches_processed = 0;
        
        cudaEventRecord(epoch_start);
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Get batch indices
            std::vector<int> batch_indices(indices.begin() + batch * batch_size,
                                           indices.begin() + (batch + 1) * batch_size);
            
            // Load batch to GPU
            train_data.get_batch(model.input, batch_indices, model.stream);
            
            // For autoencoder: target = input
            cudaMemcpyAsync(model.target, model.input, 
                           batch_size * 32 * 32 * 3 * sizeof(half),
                           cudaMemcpyDeviceToDevice, model.stream);
            
            // Forward + Backward
            model.forward();
            float loss = model.backward();
            
            epoch_loss += loss;
            num_batches_processed++;
            
            // Print progress
            if ((batch + 1) % 100 == 0 || batch == num_batches - 1) {
                printf("  Epoch %d [%4d/%4d] loss: %.6f\n", 
                       epoch + 1, batch + 1, num_batches, loss);
            }
        }
        
        cudaEventRecord(epoch_end);
        cudaEventSynchronize(epoch_end);
        
        float epoch_ms;
        cudaEventElapsedTime(&epoch_ms, epoch_start, epoch_end);
        
        float avg_loss = epoch_loss / num_batches_processed;
        float throughput = (float)(num_batches_processed * batch_size) / (epoch_ms / 1000.0f);
        
        printf("Epoch %d complete: avg_loss=%.6f, lr=%.6f, time=%.2fs, throughput=%.0f img/s",
               epoch + 1, avg_loss, current_lr, epoch_ms / 1000.0f, throughput);
        
        // Save best weights and check for improvement
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            epochs_without_improvement = 0;
            model.save_weights("autoencoder_best.bin");
            printf(" [NEW BEST - saved]");
        } else {
            epochs_without_improvement++;
            // Reduce LR if no improvement for 3 epochs
            if (epochs_without_improvement >= 3) {
                current_lr *= 0.5f;
                epochs_without_improvement = 0;
                printf(" [LR decay -> %.6f]", current_lr);
                
                // Stop if LR too small
                if (current_lr < 1e-6f) {
                    printf("\nLR too small, stopping early.\n");
                    break;
                }
            }
        }
        printf("\n\n");
    }
    
    cudaEventDestroy(epoch_start);
    cudaEventDestroy(epoch_end);
    
    // Save final weights
    printf("Saving final weights to 'autoencoder_weights.bin'...\n");
    model.save_weights("autoencoder_weights.bin");
    printf("Best loss achieved: %.6f (saved to 'autoencoder_best.bin')\n", best_loss);
    printf("Done!\n");
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\nCUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n=== Training Complete ===\n");
    return 0;
}
