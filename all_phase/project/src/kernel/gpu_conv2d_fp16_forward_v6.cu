// gpu_conv2d_fp16_forward_v6.cu
// Optimized FP16 Conv2D forward pass using Tensor Cores (WMMA)
// V6: Optimized memory layout, half in shared memory

#include <cuda_fp16.h>
#include <mma.h>
#include <cuda_runtime.h>
#include <cstdio>

using namespace nvcuda::wmma;

// ============================================================================
// CONFIGURATION - PROPERLY TUNED FOR T4 (48KB shared memory limit)
// ============================================================================
constexpr int WMMA_M = 16;
constexpr int WMMA_N = 16;
constexpr int WMMA_K = 16;

constexpr int WARPS_PER_BLOCK = 4;
constexpr int THREADS_PER_BLOCK = WARPS_PER_BLOCK * 32;  // 128

constexpr int TILE_H = 8;
constexpr int TILE_W = 8;

constexpr int OC_TILES_PER_BLOCK = 2;
constexpr int OC_PER_BLOCK = OC_TILES_PER_BLOCK * WMMA_N;  // 32

constexpr int IC_PER_ITER = WMMA_K;  // 16

constexpr int SMEM_PAD = 8;

constexpr int INPUT_TILE_H = TILE_H + 2;  // 10
constexpr int INPUT_TILE_W = TILE_W + 2;  // 10

// ============================================================================
// V6 OPTIMIZED KERNEL - Fits T4 shared memory, optimized memory layout
// ============================================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4)
conv2d_fp16_wmma_v6_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    const int N, const int H, const int W, const int C, const int K) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y;
    const int oc_tiles_total = (K + OC_PER_BLOCK - 1) / OC_PER_BLOCK;
    const int batch_idx = blockIdx.z / oc_tiles_total;
    const int oc_block_idx = blockIdx.z % oc_tiles_total;

    if (batch_idx >= N) return;

    const int out_h_base = tile_y * TILE_H;
    const int out_w_base = tile_x * TILE_W;
    const int oc_base = oc_block_idx * OC_PER_BLOCK;

    if (out_h_base >= H || out_w_base >= W) return;

    // Shared memory - half precision for efficiency
    __shared__ __align__(16) half s_input[INPUT_TILE_H][INPUT_TILE_W][IC_PER_ITER + SMEM_PAD];
    __shared__ __align__(16) half s_weight[9][OC_TILES_PER_BLOCK][WMMA_N][WMMA_K + SMEM_PAD];
    __shared__ __align__(16) half s_A[WARPS_PER_BLOCK][WMMA_M][WMMA_K + SMEM_PAD];

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[OC_TILES_PER_BLOCK];
    #pragma unroll
    for (int t = 0; t < OC_TILES_PER_BLOCK; ++t) {
        fill_fragment(acc[t], 0.0f);
    }

    const int warp_spatial_offset = warp_id * WMMA_M;

    for (int c_base = 0; c_base < C; c_base += WMMA_K) {
        
        // Load input tile
        {
            const int total_elems = INPUT_TILE_H * INPUT_TILE_W * WMMA_K;
            
            #pragma unroll 4
            for (int idx = tid; idx < total_elems; idx += THREADS_PER_BLOCK) {
                const int ic = idx % WMMA_K;
                const int iw = (idx / WMMA_K) % INPUT_TILE_W;
                const int ih = idx / (WMMA_K * INPUT_TILE_W);

                const int gh = out_h_base + ih - 1;
                const int gw = out_w_base + iw - 1;
                const int gc = c_base + ic;

                half val = __float2half(0.0f);
                if (gh >= 0 && gh < H && gw >= 0 && gw < W && gc < C) {
                    val = input[((batch_idx * H + gh) * W + gw) * C + gc];
                }
                s_input[ih][iw][ic] = val;
            }
        }

        // Load weights
        {
            const int total_elems = 9 * OC_TILES_PER_BLOCK * WMMA_K * WMMA_N;
            
            #pragma unroll 4
            for (int idx = tid; idx < total_elems; idx += THREADS_PER_BLOCK) {
                const int oc = idx % WMMA_N;
                const int ic = (idx / WMMA_N) % WMMA_K;
                const int oc_tile = (idx / (WMMA_N * WMMA_K)) % OC_TILES_PER_BLOCK;
                const int kpos = idx / (WMMA_N * WMMA_K * OC_TILES_PER_BLOCK);

                const int kh = kpos / 3;
                const int kw = kpos % 3;
                const int gc = c_base + ic;
                const int gk = oc_base + oc_tile * WMMA_N + oc;

                half w = __float2half(0.0f);
                if (gc < C && gk < K) {
                    w = weight[((gk * C + gc) * 3 + kh) * 3 + kw];
                }
                s_weight[kpos][oc_tile][oc][ic] = w;
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int kpos = kh * 3 + kw;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int elem = lane_id + i * 32;
                    if (elem >= WMMA_M * WMMA_K) continue;
                    
                    const int m = elem / WMMA_K;
                    const int k = elem % WMMA_K;

                    const int spatial_idx = warp_spatial_offset + m;
                    const int local_h = spatial_idx / TILE_W;
                    const int local_w = spatial_idx % TILE_W;

                    half val = __float2half(0.0f);
                    if (spatial_idx < TILE_H * TILE_W &&
                        out_h_base + local_h < H &&
                        out_w_base + local_w < W) {
                        val = s_input[local_h + kh][local_w + kw][k];
                    }
                    s_A[warp_id][m][k] = val;
                }
                __syncwarp();

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                load_matrix_sync(a_frag, &s_A[warp_id][0][0], WMMA_K + SMEM_PAD);

                #pragma unroll
                for (int t = 0; t < OC_TILES_PER_BLOCK; ++t) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
                    load_matrix_sync(b_frag, &s_weight[kpos][t][0][0], WMMA_K + SMEM_PAD);
                    mma_sync(acc[t], a_frag, b_frag, acc[t]);
                }

                __syncwarp();
            }
        }

        __syncthreads();
    }

    // Store output with bias
    __shared__ float s_out[WARPS_PER_BLOCK][WMMA_M][WMMA_N + 4];

    #pragma unroll
    for (int t = 0; t < OC_TILES_PER_BLOCK; ++t) {
        store_matrix_sync(&s_out[warp_id][0][0], acc[t], WMMA_N + 4, mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int elem = lane_id + i * 32;
            const int m = elem / WMMA_N;
            const int n = elem % WMMA_N;

            if (m >= WMMA_M) continue;
            
            const int spatial_idx = warp_spatial_offset + m;
            if (spatial_idx >= TILE_H * TILE_W) continue;

            const int gh = out_h_base + spatial_idx / TILE_W;
            const int gw = out_w_base + spatial_idx % TILE_W;
            const int gk = oc_base + t * WMMA_N + n;

            if (gh < H && gw < W && gk < K) {
                float val = s_out[warp_id][m][n] + __half2float(bias[gk]);
                output[((batch_idx * H + gh) * W + gw) * K + gk] = __float2half(val);
            }
        }
        __syncwarp();
    }
}

// ============================================================================
// CONV + RELU FUSED KERNEL
// ============================================================================
__global__ void __launch_bounds__(THREADS_PER_BLOCK, 4)
conv2d_relu_fp16_wmma_v6_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    half* __restrict__ conv_output,
    const int N, const int H, const int W, const int C, const int K) {
    
    const int tid = threadIdx.x;
    const int warp_id = tid / 32;
    const int lane_id = tid % 32;

    const int tile_x = blockIdx.x;
    const int tile_y = blockIdx.y;
    const int oc_tiles_total = (K + OC_PER_BLOCK - 1) / OC_PER_BLOCK;
    const int batch_idx = blockIdx.z / oc_tiles_total;
    const int oc_block_idx = blockIdx.z % oc_tiles_total;

    if (batch_idx >= N) return;

    const int out_h_base = tile_y * TILE_H;
    const int out_w_base = tile_x * TILE_W;
    const int oc_base = oc_block_idx * OC_PER_BLOCK;

    if (out_h_base >= H || out_w_base >= W) return;

    __shared__ __align__(16) half s_input[INPUT_TILE_H][INPUT_TILE_W][IC_PER_ITER + SMEM_PAD];
    __shared__ __align__(16) half s_weight[9][OC_TILES_PER_BLOCK][WMMA_N][WMMA_K + SMEM_PAD];
    __shared__ __align__(16) half s_A[WARPS_PER_BLOCK][WMMA_M][WMMA_K + SMEM_PAD];

    fragment<accumulator, WMMA_M, WMMA_N, WMMA_K, float> acc[OC_TILES_PER_BLOCK];
    #pragma unroll
    for (int t = 0; t < OC_TILES_PER_BLOCK; ++t) {
        fill_fragment(acc[t], 0.0f);
    }

    const int warp_spatial_offset = warp_id * WMMA_M;

    for (int c_base = 0; c_base < C; c_base += WMMA_K) {
        
        // Load input tile
        {
            const int total_elems = INPUT_TILE_H * INPUT_TILE_W * WMMA_K;
            #pragma unroll 4
            for (int idx = tid; idx < total_elems; idx += THREADS_PER_BLOCK) {
                const int ic = idx % WMMA_K;
                const int iw = (idx / WMMA_K) % INPUT_TILE_W;
                const int ih = idx / (WMMA_K * INPUT_TILE_W);

                const int gh = out_h_base + ih - 1;
                const int gw = out_w_base + iw - 1;
                const int gc = c_base + ic;

                half val = __float2half(0.0f);
                if (gh >= 0 && gh < H && gw >= 0 && gw < W && gc < C) {
                    val = input[((batch_idx * H + gh) * W + gw) * C + gc];
                }
                s_input[ih][iw][ic] = val;
            }
        }

        // Load weights
        {
            const int total_elems = 9 * OC_TILES_PER_BLOCK * WMMA_K * WMMA_N;
            #pragma unroll 4
            for (int idx = tid; idx < total_elems; idx += THREADS_PER_BLOCK) {
                const int oc = idx % WMMA_N;
                const int ic = (idx / WMMA_N) % WMMA_K;
                const int oc_tile = (idx / (WMMA_N * WMMA_K)) % OC_TILES_PER_BLOCK;
                const int kpos = idx / (WMMA_N * WMMA_K * OC_TILES_PER_BLOCK);

                const int kh = kpos / 3;
                const int kw = kpos % 3;
                const int gc = c_base + ic;
                const int gk = oc_base + oc_tile * WMMA_N + oc;

                half w = __float2half(0.0f);
                if (gc < C && gk < K) {
                    w = weight[((gk * C + gc) * 3 + kh) * 3 + kw];
                }
                s_weight[kpos][oc_tile][oc][ic] = w;
            }
        }

        __syncthreads();

        // Compute
        #pragma unroll
        for (int kh = 0; kh < 3; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < 3; ++kw) {
                const int kpos = kh * 3 + kw;

                #pragma unroll
                for (int i = 0; i < 8; ++i) {
                    const int elem = lane_id + i * 32;
                    if (elem >= WMMA_M * WMMA_K) continue;
                    
                    const int m = elem / WMMA_K;
                    const int k = elem % WMMA_K;

                    const int spatial_idx = warp_spatial_offset + m;
                    const int local_h = spatial_idx / TILE_W;
                    const int local_w = spatial_idx % TILE_W;

                    half val = __float2half(0.0f);
                    if (spatial_idx < TILE_H * TILE_W &&
                        out_h_base + local_h < H &&
                        out_w_base + local_w < W) {
                        val = s_input[local_h + kh][local_w + kw][k];
                    }
                    s_A[warp_id][m][k] = val;
                }
                __syncwarp();

                fragment<matrix_a, WMMA_M, WMMA_N, WMMA_K, half, row_major> a_frag;
                load_matrix_sync(a_frag, &s_A[warp_id][0][0], WMMA_K + SMEM_PAD);

                #pragma unroll
                for (int t = 0; t < OC_TILES_PER_BLOCK; ++t) {
                    fragment<matrix_b, WMMA_M, WMMA_N, WMMA_K, half, col_major> b_frag;
                    load_matrix_sync(b_frag, &s_weight[kpos][t][0][0], WMMA_K + SMEM_PAD);
                    mma_sync(acc[t], a_frag, b_frag, acc[t]);
                }

                __syncwarp();
            }
        }

        __syncthreads();
    }

    // Store output with bias AND ReLU
    __shared__ float s_out[WARPS_PER_BLOCK][WMMA_M][WMMA_N + 4];

    #pragma unroll
    for (int t = 0; t < OC_TILES_PER_BLOCK; ++t) {
        store_matrix_sync(&s_out[warp_id][0][0], acc[t], WMMA_N + 4, mem_row_major);
        __syncwarp();

        #pragma unroll
        for (int i = 0; i < 8; ++i) {
            const int elem = lane_id + i * 32;
            const int m = elem / WMMA_N;
            const int n = elem % WMMA_N;

            if (m >= WMMA_M) continue;
            
            const int spatial_idx = warp_spatial_offset + m;
            if (spatial_idx >= TILE_H * TILE_W) continue;

            const int gh = out_h_base + spatial_idx / TILE_W;
            const int gw = out_w_base + spatial_idx % TILE_W;
            const int gk = oc_base + t * WMMA_N + n;

            if (gh < H && gw < W && gk < K) {
                const size_t out_idx = ((batch_idx * H + gh) * W + gw) * K + gk;
                float conv_val = s_out[warp_id][m][n] + __half2float(bias[gk]);
                
                if (conv_output != nullptr) {
                    conv_output[out_idx] = __float2half(conv_val);
                }
                
                output[out_idx] = __float2half(fmaxf(conv_val, 0.0f));
            }
        }
        __syncwarp();
    }
}

// ============================================================================
// REFERENCE KERNEL (simple, correct)
// ============================================================================
__global__ void conv2d_fp16_reference_kernel(
    const half* __restrict__ input,
    const half* __restrict__ weight,
    const half* __restrict__ bias,
    half* __restrict__ output,
    const int N, const int H, const int W, const int C, const int K){
    const int idx = blockIdx.x * blockDim.x + threadIdx.x;
    const int total = N * H * W * K;
    
    if (idx >= total) return;
    
    const int k = idx % K;
    const int w_out = (idx / K) % W;
    const int h_out = (idx / (K * W)) % H;
    const int n = idx / (K * W * H);
    
    float sum = 0.0f;
    
    for (int c = 0; c < C; ++c) {
        for (int kh = 0; kh < 3; ++kh) {
            for (int kw = 0; kw < 3; ++kw) {
                const int h_in = h_out + kh - 1;
                const int w_in = w_out + kw - 1;
                
                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
                    float in_val = __half2float(input[((n * H + h_in) * W + w_in) * C + c]);
                    float w_val = __half2float(weight[((k * C + c) * 3 + kh) * 3 + kw]);
                    sum += in_val * w_val;
                }
            }
        }
    }
    
    sum += __half2float(bias[k]);
    output[((n * H + h_out) * W + w_out) * K + k] = __float2half(sum);
}

// ============================================================================
// LAUNCHERS
// ============================================================================
extern "C" {

void launch_conv2d_fp16_wmma_optimized(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int N, int H, int W, int C, int K,
    cudaStream_t stream){
    const int tiles_x = (W + TILE_W - 1) / TILE_W;
    const int tiles_y = (H + TILE_H - 1) / TILE_H;
    const int oc_blocks = (K + OC_PER_BLOCK - 1) / OC_PER_BLOCK;
    
    dim3 grid(tiles_x, tiles_y, N * oc_blocks);
    dim3 block(THREADS_PER_BLOCK);
    
    conv2d_fp16_wmma_v6_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, N, H, W, C, K
    );
}

void launch_conv2d_relu_fp16_wmma_optimized(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    half* conv_output,
    int N, int H, int W, int C, int K,
    cudaStream_t stream){
    const int tiles_x = (W + TILE_W - 1) / TILE_W;
    const int tiles_y = (H + TILE_H - 1) / TILE_H;
    const int oc_blocks = (K + OC_PER_BLOCK - 1) / OC_PER_BLOCK;
    
    dim3 grid(tiles_x, tiles_y, N * oc_blocks);
    dim3 block(THREADS_PER_BLOCK);
    
    conv2d_relu_fp16_wmma_v6_kernel<<<grid, block, 0, stream>>>(
        input, weight, bias, output, conv_output, N, H, W, C, K
    );
}

void launch_conv2d_fp16_reference(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int N, int H, int W, int C, int K,
    cudaStream_t stream){
    const int total = N * H * W * K;
    const int threads = 256;
    const int blocks = (total + threads - 1) / threads;
    
    conv2d_fp16_reference_kernel<<<blocks, threads, 0, stream>>>(
        input, weight, bias, output, N, H, W, C, K
    );
}

}  // extern "C"
