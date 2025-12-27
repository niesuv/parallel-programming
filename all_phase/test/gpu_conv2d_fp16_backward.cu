// gpu_conv2d_fp16_backward.cu
// Backward pass kernels for FP16 convolution with shared memory, tiling, and warp reduction

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <mma.h>
#include <cstdio>
using namespace nvcuda::wmma;

using namespace nvcuda;

// ============================================================================
// Constants and Configuration
// ============================================================================

constexpr int WARP_SIZE = 32;
constexpr int TILE_SIZE = 16;           // Tile dimension for spatial tiling
constexpr int CHANNEL_TILE = 32;        // Channels processed per tile
constexpr int BATCH_TILE = 4;           // Batch elements per thread block

// ============================================================================
// Utility Functions
// ============================================================================

__device__ __forceinline__ half relu_backward_element(half grad, half input) {
    // ReLU backward: grad * (input > 0)
    return __hgt(input, __float2half(0.0f)) ? grad : __float2half(0.0f);
}

// Warp-level reduction for float
__device__ __forceinline__ float warp_reduce_sum(float val) {
    #pragma unroll
    for (int offset = WARP_SIZE / 2; offset > 0; offset /= 2) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

// Block-level reduction using shared memory
template<int BLOCK_SIZE>
__device__ float block_reduce_sum(float val, float* shared_mem) {
    int lane = threadIdx.x % WARP_SIZE;
    int warp_id = threadIdx.x / WARP_SIZE;
    
    // Warp-level reduction
    val = warp_reduce_sum(val);
    
    // Write reduced value from each warp
    if (lane == 0) {
        shared_mem[warp_id] = val;
    }
    __syncthreads();
    
    // Final reduction in first warp
    constexpr int NUM_WARPS = BLOCK_SIZE / WARP_SIZE;
    if (warp_id == 0) {
        val = (lane < NUM_WARPS) ? shared_mem[lane] : 0.0f;
        val = warp_reduce_sum(val);
    }
    
    return val;
}

// ============================================================================
// ReLU Backward Kernel
// ============================================================================
// Simple element-wise kernel: grad_input = grad_output * (conv_output > 0)

// OPTIMIZED VERSION - Uses half4 for even better bandwidth
__global__ void relu_backward_fp16_kernel(
    const half* __restrict__ grad_output,
    const half* __restrict__ conv_output,
    half* __restrict__ grad_input,
    size_t size)
{
    // Process 4 elements at a time using half4
    size_t idx = (blockIdx.x * blockDim.x + threadIdx.x) * 4;
    
    if (idx + 3 < size) {
        half4 grad = *reinterpret_cast<const half4*>(grad_output + idx);
        half4 conv = *reinterpret_cast<const half4*>(conv_output + idx);
        
        // ReLU backward using compare and mask
        half4 result;
        result.x = __hgt(conv.x, __float2half(0.0f)) ? grad.x : __float2half(0.0f);
        result.y = __hgt(conv.y, __float2half(0.0f)) ? grad.y : __float2half(0.0f);
        result.z = __hgt(conv.z, __float2half(0.0f)) ? grad.z : __float2half(0.0f);
        result.w = __hgt(conv.w, __float2half(0.0f)) ? grad.w : __float2half(0.0f);
        
        *reinterpret_cast<half4*>(grad_input + idx) = result;
    } else {
        // Handle remainder
        for (int i = 0; i < 4 && idx + i < size; i++) {
            half grad = grad_output[idx + i];
            half conv = conv_output[idx + i];
            grad_input[idx + i] = __hgt(conv, __float2half(0.0f)) ? grad : __float2half(0.0f);
        }
    }
}

// ============================================================================
// Convolution Backward Input Kernel (Transposed Convolution)
// ============================================================================
// Computes gradient w.r.t. input using tiled shared memory approach
// For 3x3 convolution: grad_input[n,h,w,c] = sum over k,kh,kw of 
//   grad_output[n,h+kh-pad,w+kw-pad,k] * weight[k,2-kh,2-kw,c]

// Tiled kernel using shared memory for grad_output and weights
__global__ void conv2d_backward_input_tiled_kernel(
    const half* __restrict__ grad_output,  // [N, H_out, W_out, K] NHWC
    const half* __restrict__ weight,       // [K, 3, 3, C] - flipped for backward
    half* __restrict__ grad_input,         // [N, H, W, C] NHWC
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    // Block handles a TILE_SIZE x TILE_SIZE spatial region for one batch element
    // Each thread computes CHANNEL_TILE / blockDim.x channels
    
    extern __shared__ char shared_mem[];
    
    // Shared memory layout:
    // - grad_output tile: (TILE_SIZE + 2) * (TILE_SIZE + 2) * K_TILE
    // - weight tile: 3 * 3 * K_TILE * C_TILE
    
    constexpr int K_TILE = 32;   // Output channels per tile
    constexpr int C_TILE = 32;   // Input channels per tile
    constexpr int HALO = 2;      // Padding for 3x3 kernel
    
    half* s_grad_out = reinterpret_cast<half*>(shared_mem);
    half* s_weight = s_grad_out + (TILE_SIZE + HALO) * (TILE_SIZE + HALO) * K_TILE;
    
    int batch_idx = blockIdx.z;
    int tile_h = blockIdx.y * TILE_SIZE;
    int tile_w = blockIdx.x * TILE_SIZE;
    
    int local_h = threadIdx.y;
    int local_w = threadIdx.x;
    int global_h = tile_h + local_h;
    int global_w = tile_w + local_w;
    
    // Accumulator for gradient
    float accum[C_TILE / TILE_SIZE];  // Each thread handles multiple channels
    
    #pragma unroll
    for (int i = 0; i < C_TILE / TILE_SIZE; i++) {
        accum[i] = 0.0f;
    }
    
    // Iterate over output channel tiles
    for (int k_tile = 0; k_tile < K; k_tile += K_TILE) {
        // Load grad_output tile with halo into shared memory
        // Need (TILE_SIZE + 2) x (TILE_SIZE + 2) region
        for (int dy = local_h; dy < TILE_SIZE + HALO; dy += blockDim.y) {
            for (int dx = local_w; dx < TILE_SIZE + HALO; dx += blockDim.x) {
                int src_h = tile_h + dy - 1;  // -1 for padding offset
                int src_w = tile_w + dx - 1;
                
                for (int k = 0; k < K_TILE && k_tile + k < K; k++) {
                    half val = __float2half(0.0f);
                    if (src_h >= 0 && src_h < H_out && src_w >= 0 && src_w < W_out) {
                        int idx = batch_idx * H_out * W_out * K + 
                                  src_h * W_out * K + 
                                  src_w * K + 
                                  k_tile + k;
                        val = grad_output[idx];
                    }
                    s_grad_out[dy * (TILE_SIZE + HALO) * K_TILE + dx * K_TILE + k] = val;
                }
            }
        }
        
        // Iterate over input channel tiles
        for (int c_tile = 0; c_tile < C; c_tile += C_TILE) {
            // Load weight tile into shared memory
            // Weight layout: [K, 3, 3, C]
            int tid = threadIdx.y * blockDim.x + threadIdx.x;
            int total_weights = 9 * K_TILE * C_TILE;
            
            for (int i = tid; i < total_weights; i += blockDim.x * blockDim.y) {
                int c_local = i % C_TILE;
                int temp = i / C_TILE;
                int kw = temp % 3;
                int kh = (temp / 3) % 3;
                int k_local = temp / 9;
                
                half val = __float2half(0.0f);
                if (k_tile + k_local < K && c_tile + c_local < C) {
                    // Flip kernel for transposed convolution
                    int k_idx = (k_tile + k_local) * 9 * C + 
                                (2 - kh) * 3 * C + 
                                (2 - kw) * C + 
                                c_tile + c_local;
                    val = weight[k_idx];
                }
                s_weight[kh * 3 * K_TILE * C_TILE + kw * K_TILE * C_TILE + 
                         k_local * C_TILE + c_local] = val;
            }
            
            __syncthreads();
            
            // Compute convolution for this tile
            if (global_h < H && global_w < W) {
                for (int c_local = 0; c_local < C_TILE && c_tile + c_local < C; c_local++) {
                    float sum = 0.0f;
                    
                    #pragma unroll
                    for (int kh = 0; kh < 3; kh++) {
                        #pragma unroll
                        for (int kw = 0; kw < 3; kw++) {
                            // Position in shared grad_output
                            int sh = local_h + kh;
                            int sw = local_w + kw;
                            
                            #pragma unroll
                            for (int k_local = 0; k_local < K_TILE && k_tile + k_local < K; k_local++) {
                                half g = s_grad_out[sh * (TILE_SIZE + HALO) * K_TILE + 
                                                    sw * K_TILE + k_local];
                                half w = s_weight[kh * 3 * K_TILE * C_TILE + 
                                                  kw * K_TILE * C_TILE + 
                                                  k_local * C_TILE + c_local];
                                sum += __half2float(g) * __half2float(w);
                            }
                        }
                    }
                    
                    // Accumulate to output
                    int out_idx = batch_idx * H * W * C + global_h * W * C + global_w * C + c_tile + c_local;
                    atomicAdd(reinterpret_cast<float*>(&grad_input[out_idx / 2]), 
                              sum);  // Simplified - actual impl needs proper half atomics
                }
            }
            
            __syncthreads();
        }
    }
}

// Simpler backward input kernel without complex tiling (more reliable)
__global__ void conv2d_backward_input_simple_kernel(
    const half* __restrict__ grad_output,  // [N, H_out, W_out, K] NHWC
    const half* __restrict__ weight,       // [K, 3, 3, C]
    half* __restrict__ grad_input,         // [N, H, W, C] NHWC
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    // Each thread computes one element of grad_input
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    
    if (idx >= total) return;
    
    // Decode position
    int c = idx % C;
    int temp = idx / C;
    int w = temp % W;
    temp /= W;
    int h = temp % H;
    int n = temp / H;
    
    float sum = 0.0f;
    
    // For each kernel position
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            // Corresponding position in grad_output
            int oh = h - kh + 1;  // With padding=1
            int ow = w - kw + 1;
            
            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                // Sum over all output channels
                for (int k = 0; k < K; k++) {
                    int g_idx = n * H_out * W_out * K + oh * W_out * K + ow * K + k;
                    // Weight index - flip kernel for backward pass
                    int w_idx = k * 9 * C + (2 - kh) * 3 * C + (2 - kw) * C + c;
                    
                    sum += __half2float(grad_output[g_idx]) * __half2float(weight[w_idx]);
                }
            }
        }
    }
    
    grad_input[idx] = __float2half(sum);
}

// ============================================================================
// Convolution Backward Weight Kernel - Highly Optimized
// ============================================================================
// Strategy: Output-stationary with vectorized memory access
// Key optimizations:
// 1. Vectorized half2 loads for 2x memory bandwidth
// 2. Larger spatial tile processed without shared memory (register-only)
// 3. Better instruction-level parallelism

constexpr int WB_BLOCK_K = 32;
constexpr int WB_BLOCK_C = 32;

// Ultra-optimized version: all threads participate in loading and computing
// OPTIMIZED VERSION - Replaces the existing ultrafast kernel
__global__ void conv2d_backward_weight_ultrafast_kernel(
    const half* __restrict__ grad_output,  // [N, H_out, W_out, K] NHWC
    const half* __restrict__ input,        // [N, H, W, C] NHWC
    float* __restrict__ grad_weight,       // [K, 3, 3, C]
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    constexpr int SPATIAL_TILE = 128;  // Increased from 64
    constexpr int UNROLL_FACTOR = 4;
    
    int k_base = blockIdx.x * WB_BLOCK_K;
    int c_base = blockIdx.y * WB_BLOCK_C;
    int kh = blockIdx.z / 3;
    int kw = blockIdx.z % 3;
    
    // Early exit if this block is entirely out of bounds
    if (k_base >= K || c_base >= C) return;
    
    int tid = threadIdx.x;
    int warp_id = tid / WARP_SIZE;
    int lane_id = tid % WARP_SIZE;
    
    // Compute actual tile sizes
    int k_tile_size = min(WB_BLOCK_K, K - k_base);
    int c_tile_size = min(WB_BLOCK_C, C - c_base);
    
    // Each warp computes a 4x4 tile of KxC
    int warp_k = (warp_id % (WB_BLOCK_K / 4)) * 4;
    int warp_c = (warp_id / (WB_BLOCK_K / 4)) * 4;
    
    int k_idx0 = warp_k + (lane_id % 4);
    int c_idx0 = warp_c + (lane_id / 4);
    int k_idx1 = warp_k + ((lane_id + 8) % 4);
    int c_idx1 = warp_c + ((lane_id + 8) / 4);
    int k_idx2 = warp_k + ((lane_id + 16) % 4);
    int c_idx2 = warp_c + ((lane_id + 16) / 4);
    int k_idx3 = warp_k + ((lane_id + 24) % 4);
    int c_idx3 = warp_c + ((lane_id + 24) / 4);
    
    // Check validity
    bool valid0 = (k_idx0 < k_tile_size && c_idx0 < c_tile_size);
    bool valid1 = (k_idx1 < k_tile_size && c_idx1 < c_tile_size);
    bool valid2 = (k_idx2 < k_tile_size && c_idx2 < c_tile_size);
    bool valid3 = (k_idx3 < k_tile_size && c_idx3 < c_tile_size);
    
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    int spatial_size = N * H_out * W_out;
    
    // Shared memory - use float2 for better bandwidth
    __shared__ float2 s_grad[SPATIAL_TILE][WB_BLOCK_K/2];
    __shared__ float2 s_input[SPATIAL_TILE][WB_BLOCK_C/2];
    
    for (int spatial_base = 0; spatial_base < spatial_size; spatial_base += SPATIAL_TILE) {
        int spatial_end = min(spatial_base + SPATIAL_TILE, spatial_size);
        int spatial_count = spatial_end - spatial_base;
        
        // === LOAD GRAD OUTPUT USING VECTORIZED ACCESS ===
        for (int s = tid; s < spatial_count * k_tile_size; s += blockDim.x) {
            int spatial_idx = s / k_tile_size;
            int k_local = s % k_tile_size;
            
            int pos = spatial_base + spatial_idx;
            int w_out = pos % W_out;
            int temp = pos / W_out;
            int h_out = temp % H_out;
            int n = temp / H_out;
            
            float val = 0.0f;
            if (pos < spatial_size) {
                int g_idx = n * H_out * W_out * K + h_out * W_out * K + w_out * K + k_base + k_local;
                // Use half2 for vectorized load
                if (k_local % 2 == 0 && k_local + 1 < k_tile_size) {
                    half2 grad2 = *reinterpret_cast<const half2*>(grad_output + g_idx);
                    s_grad[spatial_idx][k_local/2] = make_float2(
                        __half2float(grad2.x), __half2float(grad2.y));
                } else if (k_local % 2 == 0) {
                    val = __half2float(grad_output[g_idx]);
                    s_grad[spatial_idx][k_local/2] = make_float2(val, 0.0f);
                }
            } else if (k_local % 2 == 0) {
                s_grad[spatial_idx][k_local/2] = make_float2(0.0f, 0.0f);
            }
        }
        
        // === LOAD INPUT USING VECTORIZED ACCESS ===
        for (int s = tid; s < spatial_count * c_tile_size; s += blockDim.x) {
            int spatial_idx = s / c_tile_size;
            int c_local = s % c_tile_size;
            
            int pos = spatial_base + spatial_idx;
            int w_out = pos % W_out;
            int temp = pos / W_out;
            int h_out = temp % H_out;
            int n = temp / H_out;
            int ih = h_out + kh;
            int iw = w_out + kw;
            
            float val = 0.0f;
            if (pos < spatial_size && ih < H && iw < W) {
                int i_idx = n * H * W * C + ih * W * C + iw * C + c_base + c_local;
                // Use half2 for vectorized load
                if (c_local % 2 == 0 && c_local + 1 < c_tile_size) {
                    half2 input2 = *reinterpret_cast<const half2*>(input + i_idx);
                    s_input[spatial_idx][c_local/2] = make_float2(
                        __half2float(input2.x), __half2float(input2.y));
                } else if (c_local % 2 == 0) {
                    val = __half2float(input[i_idx]);
                    s_input[spatial_idx][c_local/2] = make_float2(val, 0.0f);
                }
            } else if (c_local % 2 == 0) {
                s_input[spatial_idx][c_local/2] = make_float2(0.0f, 0.0f);
            }
        }
        
        __syncthreads();
        
        // === COMPUTE WITH UNROLLING ===
        if (valid0 || valid1 || valid2 || valid3) {
            #pragma unroll
            for (int s = 0; s < spatial_count; s += UNROLL_FACTOR) {
                float2 grad_vec0, grad_vec1, grad_vec2, grad_vec3;
                float2 input_vec0, input_vec1, input_vec2, input_vec3;
                
                if (valid0) {
                    grad_vec0 = s_grad[s][k_idx0/2];
                    input_vec0 = s_input[s][c_idx0/2];
                    float grad_val = (k_idx0 % 2 == 0) ? grad_vec0.x : grad_vec0.y;
                    float input_val = (c_idx0 % 2 == 0) ? input_vec0.x : input_vec0.y;
                    acc0 += grad_val * input_val;
                }
                
                if (s+1 < spatial_count) {
                    if (valid1) {
                        grad_vec1 = s_grad[s+1][k_idx1/2];
                        input_vec1 = s_input[s+1][c_idx1/2];
                        float grad_val = (k_idx1 % 2 == 0) ? grad_vec1.x : grad_vec1.y;
                        float input_val = (c_idx1 % 2 == 0) ? input_vec1.x : input_vec1.y;
                        acc1 += grad_val * input_val;
                    }
                    
                    if (s+2 < spatial_count) {
                        if (valid2) {
                            grad_vec2 = s_grad[s+2][k_idx2/2];
                            input_vec2 = s_input[s+2][c_idx2/2];
                            float grad_val = (k_idx2 % 2 == 0) ? grad_vec2.x : grad_vec2.y;
                            float input_val = (c_idx2 % 2 == 0) ? input_vec2.x : input_vec2.y;
                            acc2 += grad_val * input_val;
                        }
                        
                        if (s+3 < spatial_count) {
                            if (valid3) {
                                grad_vec3 = s_grad[s+3][k_idx3/2];
                                input_vec3 = s_input[s+3][c_idx3/2];
                                float grad_val = (k_idx3 % 2 == 0) ? grad_vec3.x : grad_vec3.y;
                                float input_val = (c_idx3 % 2 == 0) ? input_vec3.x : input_vec3.y;
                                acc3 += grad_val * input_val;
                            }
                        }
                    }
                }
            }
        }
        
        __syncthreads();
    }
    
    // Write results with warp reduction for better coalescing
    __shared__ float warp_results[WB_BLOCK_K * WB_BLOCK_C];
    
    if (valid0) warp_results[warp_id * 16 + lane_id] = acc0;
    if (valid1) warp_results[warp_id * 16 + lane_id + 8] = acc1;
    if (valid2) warp_results[warp_id * 16 + lane_id + 16] = acc2;
    if (valid3) warp_results[warp_id * 16 + lane_id + 24] = acc3;
    
    __syncthreads();
    
    // First thread in warp writes results
    if (lane_id == 0) {
        for (int i = 0; i < 4; i++) {
            for (int j = 0; j < 4; j++) {
                int idx = warp_id * 16 + i * 4 + j;
                int k_local = warp_k + i;
                int c_local = warp_c + j;
                
                if (k_local < k_tile_size && c_local < c_tile_size) {
                    int w_idx = (k_base + k_local) * 9 * C + kh * 3 * C + kw * C + c_base + c_local;
                    grad_weight[w_idx] = warp_results[idx];
                }
            }
        }
    }
}
// Version 3: Shared memory tiled GEMM-style approach
__global__ void conv2d_backward_weight_gemm_kernel(
    const half* __restrict__ grad_output,  // [N, H_out, W_out, K] NHWC
    const half* __restrict__ input,        // [N, H, W, C] NHWC
    float* __restrict__ grad_weight,       // [K, 3, 3, C]
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    // Treat as GEMM: grad_output^T @ input
    // grad_output: [spatial, K] -> transpose -> [K, spatial]
    // input: [spatial, C]
    // output: [K, C]
    
    constexpr int TILE_K = 32;
    constexpr int TILE_C = 32;
    constexpr int TILE_S = 32;  // Spatial tile
    
    int k_base = blockIdx.x * TILE_K;
    int c_base = blockIdx.y * TILE_C;
    int kh = blockIdx.z / 3;
    int kw = blockIdx.z % 3;
    
    __shared__ float As[TILE_S][TILE_K];  // grad_output tile
    __shared__ float Bs[TILE_S][TILE_C];  // input tile
    
    int tx = threadIdx.x % TILE_K;
    int ty = threadIdx.x / TILE_K;
    
    // Each thread computes one element of output
    int k_local = threadIdx.x % TILE_K;
    int c_local = threadIdx.x / TILE_K;
    
    float acc = 0.0f;
    
    int spatial_size = N * H_out * W_out;
    
    for (int s_base = 0; s_base < spatial_size; s_base += TILE_S) {
        // Cooperative load of grad_output tile
        for (int i = threadIdx.x; i < TILE_S * TILE_K; i += blockDim.x) {
            int s = i / TILE_K;
            int k = i % TILE_K;
            int pos = s_base + s;
            
            float val = 0.0f;
            if (pos < spatial_size && k_base + k < K) {
                int w_out = pos % W_out;
                int temp = pos / W_out;
                int h_out = temp % H_out;
                int n = temp / H_out;
                int g_idx = n * H_out * W_out * K + h_out * W_out * K + w_out * K + k_base + k;
                val = __half2float(grad_output[g_idx]);
            }
            As[s][k] = val;
        }
        
        // Cooperative load of input tile
        for (int i = threadIdx.x; i < TILE_S * TILE_C; i += blockDim.x) {
            int s = i / TILE_C;
            int c = i % TILE_C;
            int pos = s_base + s;
            
            float val = 0.0f;
            if (pos < spatial_size && c_base + c < C) {
                int w_out = pos % W_out;
                int temp = pos / W_out;
                int h_out = temp % H_out;
                int n = temp / H_out;
                int ih = h_out + kh;
                int iw = w_out + kw;
                
                if (ih < H && iw < W) {
                    int i_idx = n * H * W * C + ih * W * C + iw * C + c_base + c;
                    val = __half2float(input[i_idx]);
                }
            }
            Bs[s][c] = val;
        }
        
        __syncthreads();
        
        // Compute partial dot products
        if (k_local < TILE_K && c_local < TILE_C) {
            #pragma unroll
            for (int s = 0; s < TILE_S; s++) {
                acc += As[s][k_local] * Bs[s][c_local];
            }
        }
        
        __syncthreads();
    }
    
    // Write result
    int k = k_base + k_local;
    int c = c_base + c_local;
    
    if (k < K && c < C && k_local < TILE_K && c_local < TILE_C) {
        int w_idx = k * 9 * C + kh * 3 * C + kw * C + c;
        grad_weight[w_idx] = acc;
    }
}

// More practical backward weight kernel - one thread per weight element
__global__ void conv2d_backward_weight_simple_kernel(
    const half* __restrict__ grad_output,  // [N, H_out, W_out, K] NHWC
    const half* __restrict__ input,        // [N, H, W, C] NHWC
    float* __restrict__ grad_weight,       // [K, 3, 3, C]
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    // Each thread computes one weight gradient element
    // idx maps to [k, kh, kw, c]
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = K * 9 * C;
    
    if (idx >= total_weights) return;
    
    // Decode weight position
    int c = idx % C;
    int temp = idx / C;
    int kw = temp % 3;
    temp /= 3;
    int kh = temp % 3;
    int k = temp / 3;
    
    float sum = 0.0f;
    
    // Sum over batch and spatial dimensions
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H_out; h++) {
            for (int w = 0; w < W_out; w++) {
                // Input position (with padding)
                int ih = h + kh;
                int iw = w + kw;
                
                if (ih < H && iw < W) {
                    int g_idx = n * H_out * W_out * K + h * W_out * K + w * K + k;
                    int i_idx = n * H * W * C + ih * W * C + iw * C + c;
                    
                    sum += __half2float(grad_output[g_idx]) * __half2float(input[i_idx]);
                }
            }
        }
    }
    
    grad_weight[idx] = sum;
}

// Optimized backward weight kernel with tiling and shared memory
__global__ void conv2d_backward_weight_tiled_kernel(
    const half* __restrict__ grad_output,  // [N, H_out, W_out, K] NHWC
    const half* __restrict__ input,        // [N, H, W, C] NHWC
    float* __restrict__ grad_weight,       // [K, 3, 3, C]
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    // Block: (kw, kh, k_tile) - handles K_TILE output channels at once
    // Threads: 256 threads reduce over N * H_out * W_out and compute C_TILE channels
    
    constexpr int K_TILE = 4;
    constexpr int C_TILE = 32;
    constexpr int BLOCK_SIZE = 256;
    
    extern __shared__ char shared_mem[];
    float* s_accum = reinterpret_cast<float*>(shared_mem);  // [K_TILE][C_TILE]
    float* s_reduce = s_accum + K_TILE * C_TILE;            // [BLOCK_SIZE / WARP_SIZE]
    
    int kw = blockIdx.x % 3;
    int kh = (blockIdx.x / 3) % 3;
    int k_base = (blockIdx.x / 9) * K_TILE;
    int c_base = blockIdx.y * C_TILE;
    
    int tid = threadIdx.x;
    
    // Initialize accumulators
    for (int i = tid; i < K_TILE * C_TILE; i += BLOCK_SIZE) {
        s_accum[i] = 0.0f;
    }
    __syncthreads();
    
    // Each thread processes a subset of the spatial positions
    int spatial_size = N * H_out * W_out;
    
    for (int pos = tid; pos < spatial_size; pos += BLOCK_SIZE) {
        // Decode spatial position
        int w = pos % W_out;
        int temp = pos / W_out;
        int h = temp % H_out;
        int n = temp / H_out;
        
        // Input position
        int ih = h + kh;
        int iw = w + kw;
        
        if (ih < H && iw < W) {
            // Load grad_output values for this position
            half g_vals[K_TILE];
            #pragma unroll
            for (int ki = 0; ki < K_TILE && k_base + ki < K; ki++) {
                int g_idx = n * H_out * W_out * K + h * W_out * K + w * K + k_base + ki;
                g_vals[ki] = grad_output[g_idx];
            }
            
            // Load input values for this position
            half i_vals[C_TILE];
            #pragma unroll
            for (int ci = 0; ci < C_TILE && c_base + ci < C; ci++) {
                int i_idx = n * H * W * C + ih * W * C + iw * C + c_base + ci;
                i_vals[ci] = input[i_idx];
            }
            
            // Accumulate products
            #pragma unroll
            for (int ki = 0; ki < K_TILE && k_base + ki < K; ki++) {
                float gf = __half2float(g_vals[ki]);
                #pragma unroll
                for (int ci = 0; ci < C_TILE && c_base + ci < C; ci++) {
                    float prod = gf * __half2float(i_vals[ci]);
                    atomicAdd(&s_accum[ki * C_TILE + ci], prod);
                }
            }
        }
    }
    
    __syncthreads();
    
    // Write results to global memory
    for (int i = tid; i < K_TILE * C_TILE; i += BLOCK_SIZE) {
        int ki = i / C_TILE;
        int ci = i % C_TILE;
        
        if (k_base + ki < K && c_base + ci < C) {
            int w_idx = (k_base + ki) * 9 * C + kh * 3 * C + kw * C + c_base + ci;
            grad_weight[w_idx] = s_accum[i];
        }
    }
}

// Bias gradient kernel with warp reduction
__global__ void conv2d_backward_bias_kernel(
    const half* __restrict__ grad_output,  // [N, H_out, W_out, K] NHWC
    float* __restrict__ grad_bias,         // [K]
    int N, int H_out, int W_out, int K)
{
    // Each block handles one output channel
    extern __shared__ float s_reduce[];
    
    int k = blockIdx.x;
    int tid = threadIdx.x;
    constexpr int BLOCK_SIZE = 256;
    
    float sum = 0.0f;
    int spatial_size = N * H_out * W_out;
    
    for (int pos = tid; pos < spatial_size; pos += BLOCK_SIZE) {
        int w = pos % W_out;
        int temp = pos / W_out;
        int h = temp % H_out;
        int n = temp / H_out;
        
        int idx = n * H_out * W_out * K + h * W_out * K + w * K + k;
        sum += __half2float(grad_output[idx]);
    }
    
    // Warp reduction
    sum = warp_reduce_sum(sum);
    
    int lane = tid % WARP_SIZE;
    int warp_id = tid / WARP_SIZE;
    
    if (lane == 0) {
        s_reduce[warp_id] = sum;
    }
    __syncthreads();
    
    // Final reduction in first warp
    if (warp_id == 0) {
        sum = (tid < BLOCK_SIZE / WARP_SIZE) ? s_reduce[tid] : 0.0f;
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            grad_bias[k] = sum;
        }
    }
}

// ============================================================================
// Fused ReLU + Backward Input Kernel
// ============================================================================
// Combines ReLU backward with conv backward input for better memory efficiency

__global__ void fused_relu_conv2d_backward_input_kernel(
    const half* __restrict__ upstream_grad,  // [N, H_out, W_out, K]
    const half* __restrict__ conv_output,    // [N, H_out, W_out, K]
    const half* __restrict__ weight,         // [K, 3, 3, C]
    half* __restrict__ grad_input,           // [N, H, W, C]
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    // Each thread computes one element of grad_input
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    
    if (idx >= total) return;
    
    // Decode position
    int c = idx % C;
    int temp = idx / C;
    int w = temp % W;
    temp /= W;
    int h = temp % H;
    int n = temp / H;
    
    float sum = 0.0f;
    
    // For each kernel position
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            // Corresponding position in grad_output
            int oh = h - kh + 1;  // With padding=1
            int ow = w - kw + 1;
            
            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                // Sum over all output channels
                for (int k = 0; k < K; k++) {
                    int g_idx = n * H_out * W_out * K + oh * W_out * K + ow * K + k;
                    
                    // Fused ReLU backward
                    half upstream = upstream_grad[g_idx];
                    half conv_out = conv_output[g_idx];
                    half grad = relu_backward_element(upstream, conv_out);
                    
                    // Weight index - flip kernel for backward pass
                    int w_idx = k * 9 * C + (2 - kh) * 3 * C + (2 - kw) * C + c;
                    
                    sum += __half2float(grad) * __half2float(weight[w_idx]);
                }
            }
        }
    }
    
    grad_input[idx] = __float2half(sum);
}

// ============================================================================
// Launch Functions (C interface)
// ============================================================================

// OPTIMIZED LAUNCH CONFIGURATIONS
extern "C" {

void launch_relu_backward_fp16(
    const half* grad_output,
    const half* conv_output,
    half* grad_input,
    size_t size,
    cudaStream_t stream)
{
    // Process 4 elements per thread
    size_t half4_size = (size + 3) / 4;
    int block_size = 256;
    int grid_size = (half4_size + block_size - 1) / block_size;
    
    relu_backward_fp16_kernel<<<grid_size, block_size, 0, stream>>>(
        grad_output, conv_output, grad_input, size);
}

void launch_conv2d_backward_weight(
    const half* grad_output,
    const half* input,
    float* grad_weight,
    float* grad_bias,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream)
{
    // Zero gradients first
    cudaMemsetAsync(grad_weight, 0, K * 9 * C * sizeof(float), stream);
    if (grad_bias) {
        cudaMemsetAsync(grad_bias, 0, K * sizeof(float), stream);
    }
    
    // Choose optimal tile sizes based on problem dimensions
    int k_tile = 32;
    int c_tile = 32;
    
    // For small K, use smaller tiles
    if (K < 64) k_tile = 16;
    if (C < 64) c_tile = 16;
    
    // Use ultrafast kernel with optimized tile sizes
    dim3 grid((K + k_tile - 1) / k_tile,
              (C + c_tile - 1) / c_tile,
              9);
    
    int block_size = 256;
    
    if (K >= 128 && C >= 128) {
        // Large problem - use ultrafast kernel
        conv2d_backward_weight_ultrafast_kernel<<<grid, block_size, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    } else if (K * C <= 4096) {
        // Small problem - use v2 kernel for better parallelism
        dim3 small_grid((K * C + 255) / 256, 9, 1);
        conv2d_backward_weight_v2_kernel<<<small_grid, 256, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    } else {
        // Medium problem - use gemm kernel
        dim3 gemm_grid((K + 31) / 32, (C + 31) / 32, 9);
        conv2d_backward_weight_gemm_kernel<<<gemm_grid, 1024, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    }
    
    // Launch bias gradient kernel if needed
    if (grad_bias) {
        int bias_block_size = 256;
        conv2d_backward_bias_kernel<<<K, bias_block_size, 
            (bias_block_size / WARP_SIZE) * sizeof(float), stream>>>(
            grad_output, grad_bias,
            N, H_out, W_out, K);
    }
}

}  // extern "C" 
