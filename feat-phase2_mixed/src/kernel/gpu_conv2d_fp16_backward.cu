// gpu_conv2d_fp16_backward_v8.cu
// Optimized backward pass for T4 (SM75) with fused SGD update
// Focus: maximize arithmetic intensity for large K, C cases
// 3x3 kernel, padding=1, stride=1, NHWC layout

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>

// ============================================================================
// Device Utilities
// ============================================================================

__device__ __forceinline__ half relu_bwd(half g, half x) {
    return __hgt(x, __float2half(0.0f)) ? g : __float2half(0.0f);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// ============================================================================
// ReLU Backward
// ============================================================================

__global__ void __launch_bounds__(256, 4)
relu_backward_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ conv_out,
    half* __restrict__ grad_in,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    
    if (idx2 + 1 < size) {
        half2 g = *reinterpret_cast<const half2*>(grad_out + idx2);
        half2 c = *reinterpret_cast<const half2*>(conv_out + idx2);
        half2 r;
        r.x = relu_bwd(__low2half(g), __low2half(c));
        r.y = relu_bwd(__high2half(g), __high2half(c));
        *reinterpret_cast<half2*>(grad_in + idx2) = r;
    } else if (idx2 < size) {
        grad_in[idx2] = relu_bwd(grad_out[idx2], conv_out[idx2]);
    }
}

// ============================================================================
// Backward Input - Simple and fast
// Each thread handles 4 consecutive channels at one spatial position
// ============================================================================

__global__ void __launch_bounds__(256, 3)
conv2d_backward_input_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ weight,
    half* __restrict__ grad_in,
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_c_groups = (C + 3) / 4;
    int total = N * H * W * num_c_groups;
    
    if (idx >= total) return;
    
    int c_group = idx % num_c_groups;
    int spatial_idx = idx / num_c_groups;
    int c_base = c_group * 4;
    
    int w = spatial_idx % W;
    int temp = spatial_idx / W;
    int h = temp % H;
    int n = temp / H;
    
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            int oh = h - kh + 1;
            int ow = w - kw + 1;
            
            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                int g_base = n * H_out * W_out * K + oh * W_out * K + ow * K;
                int fkh = 2 - kh, fkw = 2 - kw;
                int w_off = fkh * 3 * C + fkw * C + c_base;
                
                for (int k = 0; k < K; k++) {
                    float gf = __half2float(grad_out[g_base + k]);
                    int w_base = k * 9 * C + w_off;
                    
                    sum0 += gf * __half2float(weight[w_base]);
                    if (c_base + 1 < C) sum1 += gf * __half2float(weight[w_base + 1]);
                    if (c_base + 2 < C) sum2 += gf * __half2float(weight[w_base + 2]);
                    if (c_base + 3 < C) sum3 += gf * __half2float(weight[w_base + 3]);
                }
            }
        }
    }
    
    int out_base = n * H * W * C + h * W * C + w * C + c_base;
    grad_in[out_base] = __float2half(sum0);
    if (c_base + 1 < C) grad_in[out_base + 1] = __float2half(sum1);
    if (c_base + 2 < C) grad_in[out_base + 2] = __float2half(sum2);
    if (c_base + 3 < C) grad_in[out_base + 3] = __float2half(sum3);
}

// ============================================================================
// Backward Weight - LARGE TILE version for K >= 64, C >= 64
// Uses 64x64 output tile with 16 elements per thread
// OPTIMIZATION: half in shared memory, float for compute
// ============================================================================

__global__ void __launch_bounds__(256, 2)
conv2d_backward_weight_large_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ input,
    float* __restrict__ grad_weight,
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    constexpr int K_TILE = 64;
    constexpr int C_TILE = 64;
    constexpr int S_TILE = 64;  // Can be larger now with half precision
    
    // Half precision shared memory - 2x smaller!
    __shared__ half s_grad[S_TILE][K_TILE];
    __shared__ half s_input[S_TILE][C_TILE];
    
    int k_base = blockIdx.x * K_TILE;
    int c_base = blockIdx.y * C_TILE;
    int kh = blockIdx.z / 3;
    int kw = blockIdx.z % 3;
    
    if (k_base >= K || c_base >= C) return;
    
    int tid = threadIdx.x;
    int k_count = min(K_TILE, K - k_base);
    int c_count = min(C_TILE, C - c_base);
    
    // Thread layout: 16x16 threads, each handles 4x4 outputs
    int thread_k = (tid / 16) * 4;  // 0, 4, 8, ..., 60
    int thread_c = (tid % 16) * 4;  // 0, 4, 8, ..., 60
    
    // 16 accumulators per thread (in float for precision)
    float acc[16] = {0};
    
    int spatial_size = N * H_out * W_out;
    
    for (int s_base = 0; s_base < spatial_size; s_base += S_TILE) {
        int s_count = min(S_TILE, spatial_size - s_base);
        
        // Load grad_out directly as half
        for (int i = tid; i < S_TILE * K_TILE; i += 256) {
            int s = i / K_TILE;
            int k = i % K_TILE;
            
            half val = __float2half(0.0f);
            int pos = s_base + s;
            if (pos < spatial_size && k < k_count) {
                int ow = pos % W_out;
                int tmp = pos / W_out;
                int oh = tmp % H_out;
                int n = tmp / H_out;
                val = grad_out[n * H_out * W_out * K + oh * W_out * K + ow * K + k_base + k];
            }
            s_grad[s][k] = val;
        }
        
        // Load input directly as half
        for (int i = tid; i < S_TILE * C_TILE; i += 256) {
            int s = i / C_TILE;
            int c = i % C_TILE;
            
            half val = __float2half(0.0f);
            int pos = s_base + s;
            if (pos < spatial_size && c < c_count) {
                int ow = pos % W_out;
                int tmp = pos / W_out;
                int oh = tmp % H_out;
                int n = tmp / H_out;
                int ih = oh + kh;
                int iw = ow + kw;
                
                if (ih < H && iw < W) {
                    val = input[n * H * W * C + ih * W * C + iw * C + c_base + c];
                }
            }
            s_input[s][c] = val;
        }
        
        __syncthreads();
        
        // Compute 4x4 output tile per thread
        for (int s = 0; s < s_count; s++) {
            // Load from half shared memory, convert to float
            float g0 = __half2float(s_grad[s][thread_k]);
            float g1 = __half2float(s_grad[s][thread_k + 1]);
            float g2 = __half2float(s_grad[s][thread_k + 2]);
            float g3 = __half2float(s_grad[s][thread_k + 3]);
            
            float i0 = __half2float(s_input[s][thread_c]);
            float i1 = __half2float(s_input[s][thread_c + 1]);
            float i2 = __half2float(s_input[s][thread_c + 2]);
            float i3 = __half2float(s_input[s][thread_c + 3]);
            
            // Outer product: 4x4 = 16 MADs
            acc[0]  += g0 * i0;  acc[1]  += g0 * i1;  acc[2]  += g0 * i2;  acc[3]  += g0 * i3;
            acc[4]  += g1 * i0;  acc[5]  += g1 * i1;  acc[6]  += g1 * i2;  acc[7]  += g1 * i3;
            acc[8]  += g2 * i0;  acc[9]  += g2 * i1;  acc[10] += g2 * i2;  acc[11] += g2 * i3;
            acc[12] += g3 * i0;  acc[13] += g3 * i1;  acc[14] += g3 * i2;  acc[15] += g3 * i3;
        }
        
        __syncthreads();
    }
    
    // Write 4x4 output tile
    #pragma unroll
    for (int dk = 0; dk < 4; dk++) {
        #pragma unroll
        for (int dc = 0; dc < 4; dc++) {
            int k = k_base + thread_k + dk;
            int c = c_base + thread_c + dc;
            if (k < K && c < C) {
                grad_weight[k * 9 * C + kh * 3 * C + kw * C + c] = acc[dk * 4 + dc];
            }
        }
    }
}

// ============================================================================
// Backward Weight - MEDIUM TILE version for K >= 32, C >= 32
// Uses 32x32 output tile with 4 elements per thread
// OPTIMIZATION: half in shared memory, float for compute
// ============================================================================

__global__ void __launch_bounds__(256, 2)
conv2d_backward_weight_medium_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ input,
    float* __restrict__ grad_weight,
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    constexpr int K_TILE = 32;
    constexpr int C_TILE = 32;
    constexpr int S_TILE = 128;  // Can be larger with half precision!
    
    // Half precision shared memory
    __shared__ half s_grad[S_TILE][K_TILE];
    __shared__ half s_input[S_TILE][C_TILE];
    
    int k_base = blockIdx.x * K_TILE;
    int c_base = blockIdx.y * C_TILE;
    int kh = blockIdx.z / 3;
    int kw = blockIdx.z % 3;
    
    if (k_base >= K || c_base >= C) return;
    
    int tid = threadIdx.x;
    int k_count = min(K_TILE, K - k_base);
    int c_count = min(C_TILE, C - c_base);
    
    // Each thread handles 4 consecutive elements
    float acc0 = 0.0f, acc1 = 0.0f, acc2 = 0.0f, acc3 = 0.0f;
    
    int elem_base = tid * 4;
    int k0 = elem_base / C_TILE, c0 = elem_base % C_TILE;
    int k1 = (elem_base + 1) / C_TILE, c1 = (elem_base + 1) % C_TILE;
    int k2 = (elem_base + 2) / C_TILE, c2 = (elem_base + 2) % C_TILE;
    int k3 = (elem_base + 3) / C_TILE, c3 = (elem_base + 3) % C_TILE;
    
    bool v0 = (k0 < k_count && c0 < c_count);
    bool v1 = (k1 < k_count && c1 < c_count);
    bool v2 = (k2 < k_count && c2 < c_count);
    bool v3 = (k3 < k_count && c3 < c_count);
    
    int spatial_size = N * H_out * W_out;
    
    for (int s_base = 0; s_base < spatial_size; s_base += S_TILE) {
        int s_count = min(S_TILE, spatial_size - s_base);
        
        // Load grad_out as half
        for (int i = tid; i < S_TILE * K_TILE; i += 256) {
            int s = i / K_TILE;
            int k = i % K_TILE;
            
            half val = __float2half(0.0f);
            int pos = s_base + s;
            if (pos < spatial_size && k < k_count) {
                int ow = pos % W_out;
                int tmp = pos / W_out;
                int oh = tmp % H_out;
                int n = tmp / H_out;
                val = grad_out[n * H_out * W_out * K + oh * W_out * K + ow * K + k_base + k];
            }
            s_grad[s][k] = val;
        }
        
        // Load input as half
        for (int i = tid; i < S_TILE * C_TILE; i += 256) {
            int s = i / C_TILE;
            int c = i % C_TILE;
            
            half val = __float2half(0.0f);
            int pos = s_base + s;
            if (pos < spatial_size && c < c_count) {
                int ow = pos % W_out;
                int tmp = pos / W_out;
                int oh = tmp % H_out;
                int n = tmp / H_out;
                int ih = oh + kh;
                int iw = ow + kw;
                
                if (ih < H && iw < W) {
                    val = input[n * H * W * C + ih * W * C + iw * C + c_base + c];
                }
            }
            s_input[s][c] = val;
        }
        
        __syncthreads();
        
        // Compute - convert half to float on the fly
        for (int s = 0; s < s_count; s++) {
            float g0 = __half2float(s_grad[s][k0]);
            float g1 = __half2float(s_grad[s][k1]);
            float g2 = __half2float(s_grad[s][k2]);
            float g3 = __half2float(s_grad[s][k3]);
            float i0 = __half2float(s_input[s][c0]);
            float i1 = __half2float(s_input[s][c1]);
            float i2 = __half2float(s_input[s][c2]);
            float i3 = __half2float(s_input[s][c3]);
            
            acc0 += g0 * i0;
            acc1 += g1 * i1;
            acc2 += g2 * i2;
            acc3 += g3 * i3;
        }
        
        __syncthreads();
    }
    
    if (v0) grad_weight[(k_base + k0) * 9 * C + kh * 3 * C + kw * C + c_base + c0] = acc0;
    if (v1) grad_weight[(k_base + k1) * 9 * C + kh * 3 * C + kw * C + c_base + c1] = acc1;
    if (v2) grad_weight[(k_base + k2) * 9 * C + kh * 3 * C + kw * C + c_base + c2] = acc2;
    if (v3) grad_weight[(k_base + k3) * 9 * C + kh * 3 * C + kw * C + c_base + c3] = acc3;
}

// ============================================================================
// Backward Weight - SMALL TILE version for small K or C
// ============================================================================

__global__ void __launch_bounds__(256, 2)
conv2d_backward_weight_small_kernel(
    const half* __restrict__ grad_out,
    const half* __restrict__ input,
    float* __restrict__ grad_weight,
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    constexpr int K_TILE = 16;
    constexpr int C_TILE = 16;
    constexpr int S_TILE = 128;
    
    __shared__ float s_grad[S_TILE][K_TILE];
    __shared__ float s_input[S_TILE][C_TILE];
    
    int k_base = blockIdx.x * K_TILE;
    int c_base = blockIdx.y * C_TILE;
    int kh = blockIdx.z / 3;
    int kw = blockIdx.z % 3;
    
    if (k_base >= K || c_base >= C) return;
    
    int tid = threadIdx.x;
    int k_count = min(K_TILE, K - k_base);
    int c_count = min(C_TILE, C - c_base);
    
    int k_local = tid / C_TILE;
    int c_local = tid % C_TILE;
    
    float acc = 0.0f;
    bool valid = (k_local < k_count && c_local < c_count);
    
    int spatial_size = N * H_out * W_out;
    
    for (int s_base = 0; s_base < spatial_size; s_base += S_TILE) {
        int s_count = min(S_TILE, spatial_size - s_base);
        
        for (int i = tid; i < S_TILE * K_TILE; i += 256) {
            int s = i / K_TILE;
            int k = i % K_TILE;
            float val = 0.0f;
            int pos = s_base + s;
            if (pos < spatial_size && k < k_count) {
                int ow = pos % W_out;
                int tmp = pos / W_out;
                int oh = tmp % H_out;
                int n = tmp / H_out;
                val = __half2float(grad_out[n * H_out * W_out * K + oh * W_out * K + ow * K + k_base + k]);
            }
            s_grad[s][k] = val;
        }
        
        for (int i = tid; i < S_TILE * C_TILE; i += 256) {
            int s = i / C_TILE;
            int c = i % C_TILE;
            float val = 0.0f;
            int pos = s_base + s;
            if (pos < spatial_size && c < c_count) {
                int ow = pos % W_out;
                int tmp = pos / W_out;
                int oh = tmp % H_out;
                int n = tmp / H_out;
                int ih = oh + kh;
                int iw = ow + kw;
                if (ih < H && iw < W) {
                    val = __half2float(input[n * H * W * C + ih * W * C + iw * C + c_base + c]);
                }
            }
            s_input[s][c] = val;
        }
        
        __syncthreads();
        
        if (valid) {
            for (int s = 0; s < s_count; s++) {
                acc += s_grad[s][k_local] * s_input[s][c_local];
            }
        }
        
        __syncthreads();
    }
    
    if (valid) {
        grad_weight[(k_base + k_local) * 9 * C + kh * 3 * C + kw * C + c_base + c_local] = acc;
    }
}

// ============================================================================
// Bias Gradient
// ============================================================================

__global__ void __launch_bounds__(256, 4)
conv2d_backward_bias_kernel(
    const half* __restrict__ grad_out,
    float* __restrict__ grad_bias,
    int N, int H_out, int W_out, int K)
{
    __shared__ float s_sum[8];
    
    int k = blockIdx.x;
    int tid = threadIdx.x;
    int spatial = N * H_out * W_out;
    
    float sum = 0.0f;
    for (int pos = tid; pos < spatial; pos += 256) {
        int ow = pos % W_out;
        int tmp = pos / W_out;
        int oh = tmp % H_out;
        int n = tmp / H_out;
        sum += __half2float(grad_out[n * H_out * W_out * K + oh * W_out * K + ow * K + k]);
    }
    
    sum = warp_reduce_sum(sum);
    if (tid % 32 == 0) s_sum[tid / 32] = sum;
    __syncthreads();
    
    if (tid < 8) {
        sum = s_sum[tid];
        sum = warp_reduce_sum(sum);
        if (tid == 0) grad_bias[k] = sum;
    }
}

// ============================================================================
// Fused ReLU + Backward Input
// ============================================================================

__global__ void __launch_bounds__(256, 4)
fused_relu_precompute_kernel(
    const half* __restrict__ upstream,
    const half* __restrict__ conv_out,
    half* __restrict__ grad_relu,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    
    if (idx2 + 1 < size) {
        half2 u = *reinterpret_cast<const half2*>(upstream + idx2);
        half2 c = *reinterpret_cast<const half2*>(conv_out + idx2);
        half2 r;
        r.x = relu_bwd(__low2half(u), __low2half(c));
        r.y = relu_bwd(__high2half(u), __high2half(c));
        *reinterpret_cast<half2*>(grad_relu + idx2) = r;
    } else if (idx2 < size) {
        grad_relu[idx2] = relu_bwd(upstream[idx2], conv_out[idx2]);
    }
}

__global__ void __launch_bounds__(256, 3)
fused_relu_backward_input_kernel(
    const half* __restrict__ upstream,
    const half* __restrict__ conv_out,
    const half* __restrict__ weight,
    half* __restrict__ grad_in,
    int N, int H, int W, int C,
    int H_out, int W_out, int K)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int num_c_groups = (C + 3) / 4;
    int total = N * H * W * num_c_groups;
    
    if (idx >= total) return;
    
    int c_group = idx % num_c_groups;
    int spatial_idx = idx / num_c_groups;
    int c_base = c_group * 4;
    
    int w = spatial_idx % W;
    int temp = spatial_idx / W;
    int h = temp % H;
    int n = temp / H;
    
    float sum0 = 0.0f, sum1 = 0.0f, sum2 = 0.0f, sum3 = 0.0f;
    
    #pragma unroll
    for (int kh = 0; kh < 3; kh++) {
        #pragma unroll
        for (int kw = 0; kw < 3; kw++) {
            int oh = h - kh + 1;
            int ow = w - kw + 1;
            
            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out) {
                int g_base = n * H_out * W_out * K + oh * W_out * K + ow * K;
                int fkh = 2 - kh, fkw = 2 - kw;
                int w_off = fkh * 3 * C + fkw * C + c_base;
                
                for (int k = 0; k < K; k++) {
                    half up = upstream[g_base + k];
                    half co = conv_out[g_base + k];
                    float gf = __half2float(relu_bwd(up, co));
                    int w_base = k * 9 * C + w_off;
                    
                    sum0 += gf * __half2float(weight[w_base]);
                    if (c_base + 1 < C) sum1 += gf * __half2float(weight[w_base + 1]);
                    if (c_base + 2 < C) sum2 += gf * __half2float(weight[w_base + 2]);
                    if (c_base + 3 < C) sum3 += gf * __half2float(weight[w_base + 3]);
                }
            }
        }
    }
    
    int out_base = n * H * W * C + h * W * C + w * C + c_base;
    grad_in[out_base] = __float2half(sum0);
    if (c_base + 1 < C) grad_in[out_base + 1] = __float2half(sum1);
    if (c_base + 2 < C) grad_in[out_base + 2] = __float2half(sum2);
    if (c_base + 3 < C) grad_in[out_base + 3] = __float2half(sum3);
}

// ============================================================================
// Launch Functions
// ============================================================================

extern "C" {

void launch_relu_backward_opt(
    const half* grad_output,
    const half* conv_output,
    half* grad_input,
    int size,
    cudaStream_t stream)
{
    int block = 256;
    int grid = ((size + 1) / 2 + block - 1) / block;
    relu_backward_kernel<<<grid, block, 0, stream>>>(
        grad_output, conv_output, grad_input, size);
}

void launch_conv2d_backward_input_opt(
    const half* grad_output,
    const half* weight,
    half* grad_input,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream)
{
    int num_c_groups = (C + 3) / 4;
    int total = N * H * W * num_c_groups;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    conv2d_backward_input_kernel<<<grid, block, 0, stream>>>(
        grad_output, weight, grad_input,
        N, H, W, C, H_out, W_out, K);
}

void launch_conv2d_backward_weight_opt(
    const half* grad_output,
    const half* input,
    float* grad_weight,
    float* grad_bias,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream)
{
    cudaMemsetAsync(grad_weight, 0, K * 9 * C * sizeof(float), stream);
    cudaMemsetAsync(grad_bias, 0, K * sizeof(float), stream);
    
    // Choose kernel based on K and C sizes
    if (K >= 64 && C >= 64) {
        // Large tile: 64x64 output, 16 elements per thread
        constexpr int K_TILE = 64;
        constexpr int C_TILE = 64;
        dim3 grid((K + K_TILE - 1) / K_TILE,
                  (C + C_TILE - 1) / C_TILE,
                  9);
        conv2d_backward_weight_large_kernel<<<grid, 256, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    } else if (K >= 32 && C >= 32) {
        // Medium tile: 32x32 output, 4 elements per thread
        constexpr int K_TILE = 32;
        constexpr int C_TILE = 32;
        dim3 grid((K + K_TILE - 1) / K_TILE,
                  (C + C_TILE - 1) / C_TILE,
                  9);
        conv2d_backward_weight_medium_kernel<<<grid, 256, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    } else {
        // Small tile: 16x16 output, 1 element per thread
        constexpr int K_TILE = 16;
        constexpr int C_TILE = 16;
        dim3 grid((K + K_TILE - 1) / K_TILE,
                  (C + C_TILE - 1) / C_TILE,
                  9);
        conv2d_backward_weight_small_kernel<<<grid, 256, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    }
    
    conv2d_backward_bias_kernel<<<K, 256, 0, stream>>>(
        grad_output, grad_bias, N, H_out, W_out, K);
}

// ============================================================================
// SGD Update Kernels (for fused backward + SGD)
// ============================================================================

__global__ void __launch_bounds__(256, 4)
sgd_update_weight_kernel(
    float* __restrict__ master_weight,
    half* __restrict__ weight,
    const float* __restrict__ grad,
    float lr,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float w = master_weight[idx];
    float g = grad[idx];
    w -= lr * g;
    master_weight[idx] = w;
    weight[idx] = __float2half(w);
}

__global__ void __launch_bounds__(256, 4)
sgd_update_bias_kernel(
    float* __restrict__ master_bias,
    half* __restrict__ bias,
    const float* __restrict__ grad,
    float lr,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float b = master_bias[idx];
    float g = grad[idx];
    b -= lr * g;
    master_bias[idx] = b;
    bias[idx] = __float2half(b);
}

// ============================================================================
// Fused Backward Weight + SGD Update
// Computes gradients AND applies SGD in one call
// ============================================================================

void launch_conv2d_backward_weight_sgd(
    const half* grad_output,
    const half* input,
    float* grad_weight,          // Temporary buffer for gradients
    float* grad_bias,            // Temporary buffer for bias gradients
    float* master_weight,        // FP32 master weights (updated)
    half* weight,                // FP16 weights (updated)
    float* master_bias,          // FP32 master bias (updated)
    half* bias,                  // FP16 bias (updated)
    float lr,                    // Learning rate
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream)
{
    int weight_size = K * 9 * C;
    int bias_size = K;
    
    // Step 1: Compute gradients (reuse existing optimized kernels)
    cudaMemsetAsync(grad_weight, 0, weight_size * sizeof(float), stream);
    cudaMemsetAsync(grad_bias, 0, bias_size * sizeof(float), stream);
    
    if (K >= 64 && C >= 64) {
        constexpr int K_TILE = 64;
        constexpr int C_TILE = 64;
        dim3 grid((K + K_TILE - 1) / K_TILE,
                  (C + C_TILE - 1) / C_TILE,
                  9);
        conv2d_backward_weight_large_kernel<<<grid, 256, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    } else if (K >= 32 && C >= 32) {
        constexpr int K_TILE = 32;
        constexpr int C_TILE = 32;
        dim3 grid((K + K_TILE - 1) / K_TILE,
                  (C + C_TILE - 1) / C_TILE,
                  9);
        conv2d_backward_weight_medium_kernel<<<grid, 256, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    } else {
        constexpr int K_TILE = 16;
        constexpr int C_TILE = 16;
        dim3 grid((K + K_TILE - 1) / K_TILE,
                  (C + C_TILE - 1) / C_TILE,
                  9);
        conv2d_backward_weight_small_kernel<<<grid, 256, 0, stream>>>(
            grad_output, input, grad_weight,
            N, H, W, C, H_out, W_out, K);
    }
    
    conv2d_backward_bias_kernel<<<K, 256, 0, stream>>>(
        grad_output, grad_bias, N, H_out, W_out, K);
    
    // Step 2: Apply SGD update
    int block = 256;
    int grid_w = (weight_size + block - 1) / block;
    int grid_b = (bias_size + block - 1) / block;
    
    sgd_update_weight_kernel<<<grid_w, block, 0, stream>>>(
        master_weight, weight, grad_weight, lr, weight_size);
    
    sgd_update_bias_kernel<<<grid_b, block, 0, stream>>>(
        master_bias, bias, grad_bias, lr, bias_size);
}

void launch_fused_relu_backward_input_opt(
    const half* upstream_grad,
    const half* conv_output,
    const half* weight,
    half* grad_input,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream)
{
    int num_c_groups = (C + 3) / 4;
    int total = N * H * W * num_c_groups;
    int block = 256;
    int grid = (total + block - 1) / block;
    
    fused_relu_backward_input_kernel<<<grid, block, 0, stream>>>(
        upstream_grad, conv_output, weight, grad_input,
        N, H, W, C, H_out, W_out, K);
}

void launch_fused_relu_backward_input_twopass(
    const half* upstream_grad,
    const half* conv_output,
    const half* weight,
    half* grad_input,
    half* temp_buffer,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream)
{
    int relu_size = N * H_out * W_out * K;
    int block = 256;
    
    int grid1 = ((relu_size + 1) / 2 + block - 1) / block;
    fused_relu_precompute_kernel<<<grid1, block, 0, stream>>>(
        upstream_grad, conv_output, temp_buffer, relu_size);
    
    launch_conv2d_backward_input_opt(temp_buffer, weight, grad_input,
        N, H, W, C, H_out, W_out, K, stream);
}

}  // extern "C"
