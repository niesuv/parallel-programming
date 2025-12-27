/**
 * Naive GPU Autoencoder - Simple as possible
 * All kernels are straightforward, no optimizations
 * 
 * Build:
 *   nvcc -arch=sm_75 -O2 -o naive_autoencoder naive_gpu_autoencoder.cu
 * 
 * Run:
 *   ./naive_autoencoder
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>
#include <string>
#include <cstring>

// ============================================================================
// CUDA Error Check
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Naive Kernels - Simple as possible
// ============================================================================

// ----- Conv2D Forward (3x3, stride=1, padding=1) -----
__global__ void naive_conv2d_forward(
    const float* input,   // [N, H, W, C]
    const float* weight,  // [K, 3, 3, C]
    const float* bias,    // [K]
    float* output,        // [N, H, W, K]
    int N, int H, int W, int C, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * K;
    if (idx >= total) return;
    
    // Decode index
    int k = idx % K;
    int w = (idx / K) % W;
    int h = (idx / (K * W)) % H;
    int n = idx / (K * W * H);
    
    float sum = 0.0f;
    
    // 3x3 convolution
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int ih = h + kh - 1;  // padding=1
                int iw = w + kw - 1;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = ((n * H + ih) * W + iw) * C + c;
                    int w_idx = ((k * 3 + kh) * 3 + kw) * C + c;
                    sum += input[in_idx] * weight[w_idx];
                }
            }
        }
    }
    
    output[idx] = sum + bias[k];
}

// ----- Conv2D Backward Input -----
__global__ void naive_conv2d_backward_input(
    const float* grad_output,  // [N, H, W, K]
    const float* weight,       // [K, 3, 3, C]
    float* grad_input,         // [N, H, W, C]
    int N, int H, int W, int C, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int w = (idx / C) % W;
    int h = (idx / (C * W)) % H;
    int n = idx / (C * W * H);
    
    float sum = 0.0f;
    
    // Transposed convolution
    for (int k = 0; k < K; k++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int oh = h - kh + 1;
                int ow = w - kw + 1;
                
                if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
                    int go_idx = ((n * H + oh) * W + ow) * K + k;
                    int w_idx = ((k * 3 + kh) * 3 + kw) * C + c;
                    sum += grad_output[go_idx] * weight[w_idx];
                }
            }
        }
    }
    
    grad_input[idx] = sum;
}

// ----- Conv2D Backward Weight -----
__global__ void naive_conv2d_backward_weight(
    const float* input,        // [N, H, W, C]
    const float* grad_output,  // [N, H, W, K]
    float* grad_weight,        // [K, 3, 3, C]
    int N, int H, int W, int C, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = K * 9 * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int kw = (idx / C) % 3;
    int kh = (idx / (C * 3)) % 3;
    int k = idx / (C * 9);
    
    float sum = 0.0f;
    
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                int ih = h + kh - 1;
                int iw = w + kw - 1;
                
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    int in_idx = ((n * H + ih) * W + iw) * C + c;
                    int go_idx = ((n * H + h) * W + w) * K + k;
                    sum += input[in_idx] * grad_output[go_idx];
                }
            }
        }
    }
    
    grad_weight[idx] = sum;
}

// ----- Conv2D Backward Bias -----
__global__ void naive_conv2d_backward_bias(
    const float* grad_output,  // [N, H, W, K]
    float* grad_bias,          // [K]
    int N, int H, int W, int K
) {
    int k = blockIdx.x * blockDim.x + threadIdx.x;
    if (k >= K) return;
    
    float sum = 0.0f;
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                sum += grad_output[((n * H + h) * W + w) * K + k];
            }
        }
    }
    
    grad_bias[k] = sum;
}

// ----- ReLU Forward -----
__global__ void naive_relu_forward(
    const float* input,
    float* output,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    output[idx] = fmaxf(input[idx], 0.0f);
}

// ----- ReLU Backward -----
__global__ void naive_relu_backward(
    const float* grad_output,
    const float* input,
    float* grad_input,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    grad_input[idx] = input[idx] > 0.0f ? grad_output[idx] : 0.0f;
}

// ----- MaxPool2D Forward (2x2, stride=2) -----
__global__ void naive_maxpool_forward(
    const float* input,   // [N, H, W, C]
    float* output,        // [N, H/2, W/2, C]
    int* indices,         // [N, H/2, W/2, C] - for backward
    int N, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H / 2;
    int W_out = W / 2;
    int total = N * H_out * W_out * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int w_out = (idx / C) % W_out;
    int h_out = (idx / (C * W_out)) % H_out;
    int n = idx / (C * W_out * H_out);
    
    int h_in = h_out * 2;
    int w_in = w_out * 2;
    
    float max_val = -1e10f;
    int max_idx = 0;
    
    for (int dh = 0; dh < 2; dh++) {
        for (int dw = 0; dw < 2; dw++) {
            int in_idx = ((n * H + h_in + dh) * W + w_in + dw) * C + c;
            float val = input[in_idx];
            if (val > max_val) {
                max_val = val;
                max_idx = in_idx;
            }
        }
    }
    
    output[idx] = max_val;
    indices[idx] = max_idx;
}

// ----- MaxPool2D Backward -----
__global__ void naive_maxpool_backward(
    const float* grad_output,  // [N, H/2, W/2, C]
    const int* indices,        // [N, H/2, W/2, C]
    float* grad_input,         // [N, H, W, C]
    int N, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H / 2;
    int W_out = W / 2;
    int total = N * H_out * W_out * C;
    if (idx >= total) return;
    
    int max_idx = indices[idx];
    atomicAdd(&grad_input[max_idx], grad_output[idx]);
}

// ----- Upsample2D Forward (2x, nearest neighbor) -----
__global__ void naive_upsample_forward(
    const float* input,   // [N, H, W, C]
    float* output,        // [N, H*2, W*2, C]
    int N, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int H_out = H * 2;
    int W_out = W * 2;
    int total = N * H_out * W_out * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int w_out = (idx / C) % W_out;
    int h_out = (idx / (C * W_out)) % H_out;
    int n = idx / (C * W_out * H_out);
    
    int h_in = h_out / 2;
    int w_in = w_out / 2;
    
    int in_idx = ((n * H + h_in) * W + w_in) * C + c;
    output[idx] = input[in_idx];
}

// ----- Upsample2D Backward -----
__global__ void naive_upsample_backward(
    const float* grad_output,  // [N, H*2, W*2, C]
    float* grad_input,         // [N, H, W, C]
    int N, int H, int W, int C
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    if (idx >= total) return;
    
    int c = idx % C;
    int w_in = (idx / C) % W;
    int h_in = (idx / (C * W)) % H;
    int n = idx / (C * W * H);
    
    int H_out = H * 2;
    int W_out = W * 2;
    
    float sum = 0.0f;
    for (int dh = 0; dh < 2; dh++) {
        for (int dw = 0; dw < 2; dw++) {
            int h_out = h_in * 2 + dh;
            int w_out = w_in * 2 + dw;
            int out_idx = ((n * H_out + h_out) * W_out + w_out) * C + c;
            sum += grad_output[out_idx];
        }
    }
    
    grad_input[idx] = sum;
}

// ----- MSE Loss Forward -----
__global__ void naive_mse_forward(
    const float* pred,
    const float* target,
    float* loss,  // single value
    int size
) {
    __shared__ float sdata[256];
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    
    float val = 0.0f;
    if (idx < size) {
        float diff = pred[idx] - target[idx];
        val = diff * diff;
    }
    sdata[tid] = val;
    __syncthreads();
    
    // Reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            sdata[tid] += sdata[tid + s];
        }
        __syncthreads();
    }
    
    if (tid == 0) {
        atomicAdd(loss, sdata[0] / size);
    }
}

// ----- MSE Loss Backward -----
__global__ void naive_mse_backward(
    const float* pred,
    const float* target,
    float* grad,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    grad[idx] = 2.0f * (pred[idx] - target[idx]) / size;
}

// ----- SGD Update -----
__global__ void naive_sgd_update(
    float* weight,
    const float* grad,
    float lr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    weight[idx] -= lr * grad[idx];
}

// ----- Zero Tensor -----
__global__ void naive_zero(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    data[idx] = 0.0f;
}

// ============================================================================
// Helper Functions
// ============================================================================

void launch_kernel(int size, void (*kernel)(float*, int), float* data) {
    int block = 256;
    int grid = (size + block - 1) / block;
    kernel<<<grid, block>>>(data, size);
}

void zero_tensor(float* data, int size) {
    int block = 256;
    int grid = (size + block - 1) / block;
    naive_zero<<<grid, block>>>(data, size);
}

void random_init(float* d_data, int size, float scale = 0.1f) {
    std::vector<float> h_data(size);
    for (int i = 0; i < size; i++) {
        h_data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
    }
    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), size * sizeof(float), cudaMemcpyHostToDevice));
}

// ============================================================================
// Naive Autoencoder Class
// ============================================================================

class NaiveAutoencoder {
public:
    int batch_size;
    float learning_rate;
    
    // Layer dimensions (CIFAR-10: 32x32x3)
    // Encoder: 3 -> 256 -> 128 -> 128
    // Decoder: 128 -> 256 -> 3
    
    // Weights and biases
    float *w1, *b1;  // Conv1: 3 -> 256
    float *w2, *b2;  // Conv2: 256 -> 128
    float *w3, *b3;  // Conv3: 128 -> 128
    float *w4, *b4;  // Conv4: 128 -> 256
    float *w5, *b5;  // Conv5: 256 -> 3
    
    // Gradients
    float *dw1, *db1;
    float *dw2, *db2;
    float *dw3, *db3;
    float *dw4, *db4;
    float *dw5, *db5;
    
    // Activations (saved for backward)
    float *a0;  // Input: [N, 32, 32, 3]
    float *z1, *a1;  // After conv1: [N, 32, 32, 256]
    float *p1;  // After pool1: [N, 16, 16, 256]
    int *idx1;  // Pool indices
    float *z2, *a2;  // After conv2: [N, 16, 16, 128]
    float *p2;  // After pool2: [N, 8, 8, 128]
    int *idx2;
    float *z3, *a3;  // After conv3 (latent): [N, 8, 8, 128]
    float *u1;  // After upsample1: [N, 16, 16, 128]
    float *z4, *a4;  // After conv4: [N, 16, 16, 256]
    float *u2;  // After upsample2: [N, 32, 32, 256]
    float *z5;  // After conv5 (output): [N, 32, 32, 3]
    
    // Gradients for activations
    float *dz5, *du2, *da4, *dz4, *du1;
    float *da3, *dz3, *dp2, *da2, *dz2;
    float *dp1, *da1, *dz1, *da0;
    
    // Loss
    float *d_loss;
    
    NaiveAutoencoder(int batch, float lr) : batch_size(batch), learning_rate(lr) {
        printf("Creating Naive Autoencoder...\n");
        printf("  Batch size: %d\n", batch_size);
        printf("  Learning rate: %f\n", learning_rate);
        
        // Allocate weights
        int w1_size = 256 * 9 * 3;
        int w2_size = 128 * 9 * 256;
        int w3_size = 128 * 9 * 128;
        int w4_size = 256 * 9 * 128;
        int w5_size = 3 * 9 * 256;
        
        CUDA_CHECK(cudaMalloc(&w1, w1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w2, w2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w3, w3_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w4, w4_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&w5, w5_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&b1, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b2, 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b3, 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b4, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&b5, 3 * sizeof(float)));
        
        // Allocate gradients
        CUDA_CHECK(cudaMalloc(&dw1, w1_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dw2, w2_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dw3, w3_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dw4, w4_size * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dw5, w5_size * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&db1, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db2, 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db3, 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db4, 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&db5, 3 * sizeof(float)));
        
        // Allocate activations
        int N = batch_size;
        CUDA_CHECK(cudaMalloc(&a0, N * 32 * 32 * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&z1, N * 32 * 32 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a1, N * 32 * 32 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p1, N * 16 * 16 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&idx1, N * 16 * 16 * 256 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&z2, N * 16 * 16 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a2, N * 16 * 16 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&p2, N * 8 * 8 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&idx2, N * 8 * 8 * 128 * sizeof(int)));
        CUDA_CHECK(cudaMalloc(&z3, N * 8 * 8 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a3, N * 8 * 8 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&u1, N * 16 * 16 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&z4, N * 16 * 16 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&a4, N * 16 * 16 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&u2, N * 32 * 32 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&z5, N * 32 * 32 * 3 * sizeof(float)));
        
        // Allocate activation gradients
        CUDA_CHECK(cudaMalloc(&dz5, N * 32 * 32 * 3 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&du2, N * 32 * 32 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&da4, N * 16 * 16 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz4, N * 16 * 16 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&du1, N * 16 * 16 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&da3, N * 8 * 8 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz3, N * 8 * 8 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dp2, N * 8 * 8 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&da2, N * 16 * 16 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz2, N * 16 * 16 * 128 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dp1, N * 16 * 16 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&da1, N * 32 * 32 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&dz1, N * 32 * 32 * 256 * sizeof(float)));
        CUDA_CHECK(cudaMalloc(&da0, N * 32 * 32 * 3 * sizeof(float)));
        
        CUDA_CHECK(cudaMalloc(&d_loss, sizeof(float)));
        
        // Initialize weights (Xavier-ish)
        random_init(w1, w1_size, sqrtf(2.0f / (9 * 3)));
        random_init(w2, w2_size, sqrtf(2.0f / (9 * 256)));
        random_init(w3, w3_size, sqrtf(2.0f / (9 * 128)));
        random_init(w4, w4_size, sqrtf(2.0f / (9 * 128)));
        random_init(w5, w5_size, sqrtf(2.0f / (9 * 256)));
        
        CUDA_CHECK(cudaMemset(b1, 0, 256 * sizeof(float)));
        CUDA_CHECK(cudaMemset(b2, 0, 128 * sizeof(float)));
        CUDA_CHECK(cudaMemset(b3, 0, 128 * sizeof(float)));
        CUDA_CHECK(cudaMemset(b4, 0, 256 * sizeof(float)));
        CUDA_CHECK(cudaMemset(b5, 0, 3 * sizeof(float)));
        
        printf("  Model created.\n\n");
    }
    
    ~NaiveAutoencoder() {
        cudaFree(w1); cudaFree(w2); cudaFree(w3); cudaFree(w4); cudaFree(w5);
        cudaFree(b1); cudaFree(b2); cudaFree(b3); cudaFree(b4); cudaFree(b5);
        cudaFree(dw1); cudaFree(dw2); cudaFree(dw3); cudaFree(dw4); cudaFree(dw5);
        cudaFree(db1); cudaFree(db2); cudaFree(db3); cudaFree(db4); cudaFree(db5);
        cudaFree(a0); cudaFree(z1); cudaFree(a1); cudaFree(p1); cudaFree(idx1);
        cudaFree(z2); cudaFree(a2); cudaFree(p2); cudaFree(idx2);
        cudaFree(z3); cudaFree(a3); cudaFree(u1); cudaFree(z4); cudaFree(a4);
        cudaFree(u2); cudaFree(z5);
        cudaFree(dz5); cudaFree(du2); cudaFree(da4); cudaFree(dz4); cudaFree(du1);
        cudaFree(da3); cudaFree(dz3); cudaFree(dp2); cudaFree(da2); cudaFree(dz2);
        cudaFree(dp1); cudaFree(da1); cudaFree(dz1); cudaFree(da0);
        cudaFree(d_loss);
    }
    
    // Forward pass
    float forward(const float* input) {
        int N = batch_size;
        int block = 256;
        
        // Copy input
        CUDA_CHECK(cudaMemcpy(a0, input, N * 32 * 32 * 3 * sizeof(float), cudaMemcpyHostToDevice));
        
        // Encoder
        // Conv1: [N,32,32,3] -> [N,32,32,256]
        int size = N * 32 * 32 * 256;
        naive_conv2d_forward<<<(size+block-1)/block, block>>>(a0, w1, b1, z1, N, 32, 32, 3, 256);
        naive_relu_forward<<<(size+block-1)/block, block>>>(z1, a1, size);
        
        // Pool1: [N,32,32,256] -> [N,16,16,256]
        size = N * 16 * 16 * 256;
        naive_maxpool_forward<<<(size+block-1)/block, block>>>(a1, p1, idx1, N, 32, 32, 256);
        
        // Conv2: [N,16,16,256] -> [N,16,16,128]
        size = N * 16 * 16 * 128;
        naive_conv2d_forward<<<(size+block-1)/block, block>>>(p1, w2, b2, z2, N, 16, 16, 256, 128);
        naive_relu_forward<<<(size+block-1)/block, block>>>(z2, a2, size);
        
        // Pool2: [N,16,16,128] -> [N,8,8,128]
        size = N * 8 * 8 * 128;
        naive_maxpool_forward<<<(size+block-1)/block, block>>>(a2, p2, idx2, N, 16, 16, 128);
        
        // Conv3 (latent): [N,8,8,128] -> [N,8,8,128]
        size = N * 8 * 8 * 128;
        naive_conv2d_forward<<<(size+block-1)/block, block>>>(p2, w3, b3, z3, N, 8, 8, 128, 128);
        naive_relu_forward<<<(size+block-1)/block, block>>>(z3, a3, size);
        
        // Decoder
        // Upsample1: [N,8,8,128] -> [N,16,16,128]
        size = N * 16 * 16 * 128;
        naive_upsample_forward<<<(size+block-1)/block, block>>>(a3, u1, N, 8, 8, 128);
        
        // Conv4: [N,16,16,128] -> [N,16,16,256]
        size = N * 16 * 16 * 256;
        naive_conv2d_forward<<<(size+block-1)/block, block>>>(u1, w4, b4, z4, N, 16, 16, 128, 256);
        naive_relu_forward<<<(size+block-1)/block, block>>>(z4, a4, size);
        
        // Upsample2: [N,16,16,256] -> [N,32,32,256]
        size = N * 32 * 32 * 256;
        naive_upsample_forward<<<(size+block-1)/block, block>>>(a4, u2, N, 16, 16, 256);
        
        // Conv5 (output): [N,32,32,256] -> [N,32,32,3]
        size = N * 32 * 32 * 3;
        naive_conv2d_forward<<<(size+block-1)/block, block>>>(u2, w5, b5, z5, N, 32, 32, 256, 3);
        
        // Compute MSE loss
        CUDA_CHECK(cudaMemset(d_loss, 0, sizeof(float)));
        naive_mse_forward<<<(size+block-1)/block, block>>>(z5, a0, d_loss, size);
        
        float h_loss;
        CUDA_CHECK(cudaMemcpy(&h_loss, d_loss, sizeof(float), cudaMemcpyDeviceToHost));
        
        return h_loss;
    }
    
    // Backward pass
    void backward() {
        int N = batch_size;
        int block = 256;
        int size;
        
        // MSE gradient: dz5 = 2*(z5 - a0) / size
        size = N * 32 * 32 * 3;
        naive_mse_backward<<<(size+block-1)/block, block>>>(z5, a0, dz5, size);
        
        // Conv5 backward
        size = 3 * 9 * 256;
        naive_conv2d_backward_weight<<<(size+block-1)/block, block>>>(u2, dz5, dw5, N, 32, 32, 256, 3);
        naive_conv2d_backward_bias<<<(3+block-1)/block, block>>>(dz5, db5, N, 32, 32, 3);
        size = N * 32 * 32 * 256;
        naive_conv2d_backward_input<<<(size+block-1)/block, block>>>(dz5, w5, du2, N, 32, 32, 256, 3);
        
        // Upsample2 backward
        size = N * 16 * 16 * 256;
        naive_upsample_backward<<<(size+block-1)/block, block>>>(du2, da4, N, 16, 16, 256);
        
        // ReLU4 backward
        naive_relu_backward<<<(size+block-1)/block, block>>>(da4, z4, dz4, size);
        
        // Conv4 backward
        size = 256 * 9 * 128;
        naive_conv2d_backward_weight<<<(size+block-1)/block, block>>>(u1, dz4, dw4, N, 16, 16, 128, 256);
        naive_conv2d_backward_bias<<<(256+block-1)/block, block>>>(dz4, db4, N, 16, 16, 256);
        size = N * 16 * 16 * 128;
        naive_conv2d_backward_input<<<(size+block-1)/block, block>>>(dz4, w4, du1, N, 16, 16, 128, 256);
        
        // Upsample1 backward
        size = N * 8 * 8 * 128;
        naive_upsample_backward<<<(size+block-1)/block, block>>>(du1, da3, N, 8, 8, 128);
        
        // ReLU3 backward
        naive_relu_backward<<<(size+block-1)/block, block>>>(da3, z3, dz3, size);
        
        // Conv3 backward
        size = 128 * 9 * 128;
        naive_conv2d_backward_weight<<<(size+block-1)/block, block>>>(p2, dz3, dw3, N, 8, 8, 128, 128);
        naive_conv2d_backward_bias<<<(128+block-1)/block, block>>>(dz3, db3, N, 8, 8, 128);
        size = N * 8 * 8 * 128;
        naive_conv2d_backward_input<<<(size+block-1)/block, block>>>(dz3, w3, dp2, N, 8, 8, 128, 128);
        
        // MaxPool2 backward
        size = N * 16 * 16 * 128;
        zero_tensor(da2, size);
        size = N * 8 * 8 * 128;
        naive_maxpool_backward<<<(size+block-1)/block, block>>>(dp2, idx2, da2, N, 16, 16, 128);
        
        // ReLU2 backward
        size = N * 16 * 16 * 128;
        naive_relu_backward<<<(size+block-1)/block, block>>>(da2, z2, dz2, size);
        
        // Conv2 backward
        size = 128 * 9 * 256;
        naive_conv2d_backward_weight<<<(size+block-1)/block, block>>>(p1, dz2, dw2, N, 16, 16, 256, 128);
        naive_conv2d_backward_bias<<<(128+block-1)/block, block>>>(dz2, db2, N, 16, 16, 128);
        size = N * 16 * 16 * 256;
        naive_conv2d_backward_input<<<(size+block-1)/block, block>>>(dz2, w2, dp1, N, 16, 16, 256, 128);
        
        // MaxPool1 backward
        size = N * 32 * 32 * 256;
        zero_tensor(da1, size);
        size = N * 16 * 16 * 256;
        naive_maxpool_backward<<<(size+block-1)/block, block>>>(dp1, idx1, da1, N, 32, 32, 256);
        
        // ReLU1 backward
        size = N * 32 * 32 * 256;
        naive_relu_backward<<<(size+block-1)/block, block>>>(da1, z1, dz1, size);
        
        // Conv1 backward (only weights, no need for input grad)
        size = 256 * 9 * 3;
        naive_conv2d_backward_weight<<<(size+block-1)/block, block>>>(a0, dz1, dw1, N, 32, 32, 3, 256);
        naive_conv2d_backward_bias<<<(256+block-1)/block, block>>>(dz1, db1, N, 32, 32, 256);
        
        CUDA_CHECK(cudaDeviceSynchronize());
    }
    
    // SGD update
    void update() {
        int block = 256;
        
        // Update all weights and biases
        int size;
        
        size = 256 * 9 * 3;
        naive_sgd_update<<<(size+block-1)/block, block>>>(w1, dw1, learning_rate, size);
        naive_sgd_update<<<(256+block-1)/block, block>>>(b1, db1, learning_rate, 256);
        
        size = 128 * 9 * 256;
        naive_sgd_update<<<(size+block-1)/block, block>>>(w2, dw2, learning_rate, size);
        naive_sgd_update<<<(128+block-1)/block, block>>>(b2, db2, learning_rate, 128);
        
        size = 128 * 9 * 128;
        naive_sgd_update<<<(size+block-1)/block, block>>>(w3, dw3, learning_rate, size);
        naive_sgd_update<<<(128+block-1)/block, block>>>(b3, db3, learning_rate, 128);
        
        size = 256 * 9 * 128;
        naive_sgd_update<<<(size+block-1)/block, block>>>(w4, dw4, learning_rate, size);
        naive_sgd_update<<<(256+block-1)/block, block>>>(b4, db4, learning_rate, 256);
        
        size = 3 * 9 * 256;
        naive_sgd_update<<<(size+block-1)/block, block>>>(w5, dw5, learning_rate, size);
        naive_sgd_update<<<(3+block-1)/block, block>>>(b5, db5, learning_rate, 3);
    }
    
    // Train one batch
    float train_step(const float* input) {
        float loss = forward(input);
        backward();
        update();
        return loss;
    }
};

// ============================================================================
// CIFAR-10 Data Loader (simplified)
// ============================================================================

class CIFAR10Loader {
public:
    std::vector<float> images;  // [50000, 32, 32, 3]
    std::vector<int> labels;
    int num_samples;
    int current_idx;
    
    CIFAR10Loader(const std::string& data_dir) : current_idx(0), num_samples(0) {
        printf("Loading CIFAR-10 from %s...\n", data_dir.c_str());
        
        // Load all 5 batches
        for (int batch = 1; batch <= 5; batch++) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/data_batch_%d.bin", data_dir.c_str(), batch);
            
            FILE* f = fopen(filename, "rb");
            if (!f) {
                printf("Warning: Could not open %s\n", filename);
                continue;
            }
            
            // Each sample: 1 byte label + 3072 bytes image (CHW format)
            for (int i = 0; i < 10000; i++) {
                unsigned char label;
                unsigned char pixels[3072];
                
                if (fread(&label, 1, 1, f) != 1) break;
                if (fread(pixels, 1, 3072, f) != 3072) break;
                
                labels.push_back(label);
                
                // Convert CHW to HWC and normalize to [0,1]
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        for (int c = 0; c < 3; c++) {
                            float val = pixels[c * 1024 + h * 32 + w] / 255.0f;
                            images.push_back(val);
                        }
                    }
                }
                num_samples++;
            }
            fclose(f);
        }
        
        printf("  Loaded %d samples\n\n", num_samples);
    }
    
    // Get a batch of images
    void get_batch(float* batch, int batch_size) {
        for (int i = 0; i < batch_size; i++) {
            int idx = (current_idx + i) % num_samples;
            memcpy(batch + i * 3072, images.data() + idx * 3072, 3072 * sizeof(float));
        }
        current_idx = (current_idx + batch_size) % num_samples;
    }
    
    void shuffle() {
        // Simple shuffle by rotating
        current_idx = rand() % num_samples;
    }
};

// ============================================================================
// Main Training Loop
// ============================================================================

int main(int argc, char** argv) {
    printf("=== Naive GPU Autoencoder Training ===\n\n");
    
    // Hyperparameters
    int batch_size = 64;
    float learning_rate = 0.001f;
    int num_epochs = 10;
    std::string data_dir = "data";
    
    // Parse args
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
    
    // Load data
    CIFAR10Loader loader(data_dir);
    if (loader.num_samples == 0) {
        printf("Error: No data loaded!\n");
        return 1;
    }
    
    // Create model
    NaiveAutoencoder model(batch_size, learning_rate);
    
    // Host batch buffer
    std::vector<float> h_batch(batch_size * 32 * 32 * 3);
    
    int num_batches = loader.num_samples / batch_size;
    
    printf("Training: %d samples, %d batches per epoch\n\n", loader.num_samples, num_batches);
    
    // Training loop
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        loader.shuffle();
        
        float epoch_loss = 0.0f;
        
        cudaEvent_t start, end;
        cudaEventCreate(&start);
        cudaEventCreate(&end);
        cudaEventRecord(start);
        
        for (int batch = 0; batch < num_batches; batch++) {
            loader.get_batch(h_batch.data(), batch_size);
            
            float loss = model.train_step(h_batch.data());
            epoch_loss += loss;
            
            if ((batch + 1) % 100 == 0 || batch == num_batches - 1) {
                printf("  Epoch %d [%4d/%4d] loss: %.6f\n", 
                       epoch + 1, batch + 1, num_batches, loss);
            }
        }
        
        cudaEventRecord(end);
        cudaEventSynchronize(end);
        
        float epoch_ms;
        cudaEventElapsedTime(&epoch_ms, start, end);
        
        float avg_loss = epoch_loss / num_batches;
        float throughput = (float)(num_batches * batch_size) / (epoch_ms / 1000.0f);
        
        printf("Epoch %d complete: avg_loss=%.6f, time=%.2fs, throughput=%.0f img/s\n\n",
               epoch + 1, avg_loss, epoch_ms / 1000.0f, throughput);
        
        cudaEventDestroy(start);
        cudaEventDestroy(end);
    }
    
    printf("=== Training Complete ===\n");
    
    return 0;
}
