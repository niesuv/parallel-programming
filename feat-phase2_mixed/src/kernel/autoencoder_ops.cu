// autoencoder_ops.cu
// Optimized CUDA kernels for Autoencoder operations
// MaxPool2D (2x2), Upsample2D (2x2 nearest neighbor), MSELoss
// FP16 (half precision), NHWC layout, T4 GPU (SM75)

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cfloat>
#include <cstdio>

// ============================================================================
// MaxPool2D Forward (2x2, stride=2)
// Input: [N, H, W, C] -> Output: [N, H/2, W/2, C]
// Also outputs max indices for backward pass
// ============================================================================

__global__ void __launch_bounds__(256, 4)
maxpool2d_forward_kernel(
    const half* __restrict__ input,    // [N, H, W, C]
    half* __restrict__ output,          // [N, H/2, W/2, C]
    int8_t* __restrict__ max_indices,   // [N, H/2, W/2, C] - stores 0,1,2,3 for which of 4 positions was max
    int N, int H, int W, int C)
{
    int H_out = H / 2;
    int W_out = W / 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out * C;
    
    if (idx >= total) return;
    
    // Decode output position
    int c = idx % C;
    int temp = idx / C;
    int ow = temp % W_out;
    temp = temp / W_out;
    int oh = temp % H_out;
    int n = temp / H_out;
    
    // Input position (top-left of 2x2 window)
    int ih = oh * 2;
    int iw = ow * 2;
    
    // Load 4 values from 2x2 window
    int base = n * H * W * C + ih * W * C + iw * C + c;
    half v00 = input[base];                    // (ih, iw)
    half v01 = input[base + C];                // (ih, iw+1)
    half v10 = input[base + W * C];            // (ih+1, iw)
    half v11 = input[base + W * C + C];        // (ih+1, iw+1)
    
    // Find max
    float f00 = __half2float(v00);
    float f01 = __half2float(v01);
    float f10 = __half2float(v10);
    float f11 = __half2float(v11);
    
    float max_val = f00;
    int8_t max_idx = 0;
    
    if (f01 > max_val) { max_val = f01; max_idx = 1; }
    if (f10 > max_val) { max_val = f10; max_idx = 2; }
    if (f11 > max_val) { max_val = f11; max_idx = 3; }
    
    // Write output
    output[idx] = __float2half(max_val);
    max_indices[idx] = max_idx;
}

// Vectorized version - process 2 channels at once
__global__ void __launch_bounds__(256, 4)
maxpool2d_forward_vec_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int8_t* __restrict__ max_indices,
    int N, int H, int W, int C)
{
    int H_out = H / 2;
    int W_out = W / 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out * (C / 2);
    
    if (idx >= total) return;
    
    int c2 = idx % (C / 2);
    int temp = idx / (C / 2);
    int ow = temp % W_out;
    temp = temp / W_out;
    int oh = temp % H_out;
    int n = temp / H_out;
    
    int c = c2 * 2;
    int ih = oh * 2;
    int iw = ow * 2;
    
    int base = n * H * W * C + ih * W * C + iw * C + c;
    
    // Load 4 half2 values
    half2 v00 = *reinterpret_cast<const half2*>(&input[base]);
    half2 v01 = *reinterpret_cast<const half2*>(&input[base + C]);
    half2 v10 = *reinterpret_cast<const half2*>(&input[base + W * C]);
    half2 v11 = *reinterpret_cast<const half2*>(&input[base + W * C + C]);
    
    // Process low half
    float f00_lo = __low2float(v00);
    float f01_lo = __low2float(v01);
    float f10_lo = __low2float(v10);
    float f11_lo = __low2float(v11);
    
    float max_lo = f00_lo;
    int8_t idx_lo = 0;
    if (f01_lo > max_lo) { max_lo = f01_lo; idx_lo = 1; }
    if (f10_lo > max_lo) { max_lo = f10_lo; idx_lo = 2; }
    if (f11_lo > max_lo) { max_lo = f11_lo; idx_lo = 3; }
    
    // Process high half
    float f00_hi = __high2float(v00);
    float f01_hi = __high2float(v01);
    float f10_hi = __high2float(v10);
    float f11_hi = __high2float(v11);
    
    float max_hi = f00_hi;
    int8_t idx_hi = 0;
    if (f01_hi > max_hi) { max_hi = f01_hi; idx_hi = 1; }
    if (f10_hi > max_hi) { max_hi = f10_hi; idx_hi = 2; }
    if (f11_hi > max_hi) { max_hi = f11_hi; idx_hi = 3; }
    
    // Write output
    int out_idx = n * H_out * W_out * C + oh * W_out * C + ow * C + c;
    half2 out_val = __halves2half2(__float2half(max_lo), __float2half(max_hi));
    *reinterpret_cast<half2*>(&output[out_idx]) = out_val;
    max_indices[out_idx] = idx_lo;
    max_indices[out_idx + 1] = idx_hi;
}

// ============================================================================
// MaxPool2D Backward (2x2, stride=2)
// Gradient flows only to the max position
// ============================================================================

__global__ void __launch_bounds__(256, 4)
maxpool2d_backward_kernel(
    const half* __restrict__ grad_output,  // [N, H/2, W/2, C]
    const int8_t* __restrict__ max_indices, // [N, H/2, W/2, C]
    half* __restrict__ grad_input,          // [N, H, W, C]
    int N, int H, int W, int C)
{
    int H_out = H / 2;
    int W_out = W / 2;
    
    // Each thread handles one output gradient element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out * C;
    
    if (idx >= total) return;
    
    int c = idx % C;
    int temp = idx / C;
    int ow = temp % W_out;
    temp = temp / W_out;
    int oh = temp % H_out;
    int n = temp / H_out;
    
    half grad = grad_output[idx];
    int8_t max_idx = max_indices[idx];
    
    int ih = oh * 2;
    int iw = ow * 2;
    int base = n * H * W * C + ih * W * C + iw * C + c;
    
    // Scatter gradient to max position
    // max_idx: 0=(0,0), 1=(0,1), 2=(1,0), 3=(1,1)
    int offset = 0;
    if (max_idx == 1) offset = C;
    else if (max_idx == 2) offset = W * C;
    else if (max_idx == 3) offset = W * C + C;
    
    // Use atomicAdd for safety (multiple outputs might map to same input in edge cases)
    // But for 2x2 non-overlapping, each input has exactly one output, so direct write is fine
    grad_input[base + offset] = grad;
}

// Initialize grad_input to zero, then scatter
__global__ void __launch_bounds__(256, 4)
zero_kernel(half* data, int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    if (idx2 + 1 < size) {
        *reinterpret_cast<half2*>(&data[idx2]) = __half2half2(__float2half(0.0f));
    } else if (idx2 < size) {
        data[idx2] = __float2half(0.0f);
    }
}

// ============================================================================
// Upsample2D Forward (2x2, nearest neighbor)
// Input: [N, H, W, C] -> Output: [N, H*2, W*2, C]
// ============================================================================

__global__ void __launch_bounds__(256, 4)
upsample2d_forward_kernel(
    const half* __restrict__ input,   // [N, H, W, C]
    half* __restrict__ output,         // [N, H*2, W*2, C]
    int N, int H, int W, int C)
{
    int H_out = H * 2;
    int W_out = W * 2;
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H_out * W_out * C;
    
    if (idx >= total) return;
    
    // Decode output position
    int c = idx % C;
    int temp = idx / C;
    int ow = temp % W_out;
    temp = temp / W_out;
    int oh = temp % H_out;
    int n = temp / H_out;
    
    // Map to input position (nearest neighbor = floor division)
    int ih = oh / 2;
    int iw = ow / 2;
    
    // Read input and write to output
    half val = input[n * H * W * C + ih * W * C + iw * C + c];
    output[idx] = val;
}

// Optimized: each thread reads 1 input, writes 4 outputs (the 2x2 replicated block)
__global__ void __launch_bounds__(256, 4)
upsample2d_forward_fast_kernel(
    const half* __restrict__ input,
    half* __restrict__ output,
    int N, int H, int W, int C)
{
    int H_out = H * 2;
    int W_out = W * 2;
    
    // Each thread handles one input element -> writes 4 output elements
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    
    if (idx >= total) return;
    
    int c = idx % C;
    int temp = idx / C;
    int iw = temp % W;
    temp = temp / W;
    int ih = temp % H;
    int n = temp / H;
    
    half val = input[idx];
    
    // Output positions
    int oh = ih * 2;
    int ow = iw * 2;
    int out_base = n * H_out * W_out * C + oh * W_out * C + ow * C + c;
    
    // Write 2x2 block
    output[out_base] = val;                      // (oh, ow)
    output[out_base + C] = val;                  // (oh, ow+1)
    output[out_base + W_out * C] = val;          // (oh+1, ow)
    output[out_base + W_out * C + C] = val;      // (oh+1, ow+1)
}

// ============================================================================
// Upsample2D Backward (2x2, nearest neighbor)
// Sum gradients from 2x2 output region back to input
// ============================================================================

__global__ void __launch_bounds__(256, 4)
upsample2d_backward_kernel(
    const half* __restrict__ grad_output,  // [N, H*2, W*2, C]
    half* __restrict__ grad_input,          // [N, H, W, C]
    int N, int H, int W, int C)
{
    int H_out = H * 2;
    int W_out = W * 2;
    
    // Each thread handles one input gradient element
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * C;
    
    if (idx >= total) return;
    
    int c = idx % C;
    int temp = idx / C;
    int iw = temp % W;
    temp = temp / W;
    int ih = temp % H;
    int n = temp / H;
    
    // Output positions (2x2 region)
    int oh = ih * 2;
    int ow = iw * 2;
    int out_base = n * H_out * W_out * C + oh * W_out * C + ow * C + c;
    
    // Sum 4 gradients
    float sum = __half2float(grad_output[out_base]);
    sum += __half2float(grad_output[out_base + C]);
    sum += __half2float(grad_output[out_base + W_out * C]);
    sum += __half2float(grad_output[out_base + W_out * C + C]);
    
    grad_input[idx] = __float2half(sum);
}

// ============================================================================
// MSE Loss Forward + Backward (Fused)
// Loss = mean((pred - target)^2)
// Grad = 2 * (pred - target) / num_elements
// ============================================================================

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Compute MSE loss (reduction)
__global__ void __launch_bounds__(256, 4)
mse_loss_kernel(
    const half* __restrict__ pred,
    const half* __restrict__ target,
    float* __restrict__ loss,  // Single value output
    int size)
{
    __shared__ float s_sum[8];  // 8 warps per block
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float local_sum = 0.0f;
    
    // Grid-stride loop
    for (int i = idx; i < size; i += blockDim.x * gridDim.x) {
        float p = __half2float(pred[i]);
        float t = __half2float(target[i]);
        float diff = p - t;
        local_sum += diff * diff;
    }
    
    // Warp reduction
    local_sum = warp_reduce_sum(local_sum);
    
    // Store warp results
    if (tid % 32 == 0) {
        s_sum[tid / 32] = local_sum;
    }
    __syncthreads();
    
    // Final reduction by first warp
    if (tid < 8) {
        local_sum = s_sum[tid];
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) {
            atomicAdd(loss, local_sum);
        }
    }
}

// Compute gradient: grad = 2 * (pred - target) / size
__global__ void __launch_bounds__(256, 4)
mse_grad_kernel(
    const half* __restrict__ pred,
    const half* __restrict__ target,
    half* __restrict__ grad,
    int size)
{
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idx2 = idx * 2;
    
    float scale = 2.0f / (float)size;
    
    if (idx2 + 1 < size) {
        half2 p = *reinterpret_cast<const half2*>(&pred[idx2]);
        half2 t = *reinterpret_cast<const half2*>(&target[idx2]);
        
        float g0 = (__half2float(__low2half(p)) - __half2float(__low2half(t))) * scale;
        float g1 = (__half2float(__high2half(p)) - __half2float(__high2half(t))) * scale;
        
        half2 g = __halves2half2(__float2half(g0), __float2half(g1));
        *reinterpret_cast<half2*>(&grad[idx2]) = g;
    } else if (idx2 < size) {
        float p = __half2float(pred[idx2]);
        float t = __half2float(target[idx2]);
        grad[idx2] = __float2half((p - t) * scale);
    }
}

// Fused: compute loss AND gradient in one pass
__global__ void __launch_bounds__(256, 4)
mse_loss_grad_fused_kernel(
    const half* __restrict__ pred,
    const half* __restrict__ target,
    half* __restrict__ grad,
    float* __restrict__ partial_loss,  // Per-block partial sum
    int size)
{
    __shared__ float s_sum[8];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + tid;
    
    float scale = 2.0f / (float)size;
    float local_sum = 0.0f;
    
    // Process elements
    if (idx < size) {
        float p = __half2float(pred[idx]);
        float t = __half2float(target[idx]);
        float diff = p - t;
        
        // Gradient
        grad[idx] = __float2half(diff * scale);
        
        // Loss contribution
        local_sum = diff * diff;
    }
    
    // Reduce for loss
    local_sum = warp_reduce_sum(local_sum);
    
    if (tid % 32 == 0) {
        s_sum[tid / 32] = local_sum;
    }
    __syncthreads();
    
    if (tid < 8) {
        local_sum = s_sum[tid];
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) {
            partial_loss[blockIdx.x] = local_sum;
        }
    }
}

// Final reduction of partial losses
__global__ void reduce_loss_kernel(
    const float* __restrict__ partial,
    float* __restrict__ loss,
    int num_blocks,
    int size)
{
    __shared__ float s_sum[8];
    
    int tid = threadIdx.x;
    float sum = 0.0f;
    
    for (int i = tid; i < num_blocks; i += blockDim.x) {
        sum += partial[i];
    }
    
    sum = warp_reduce_sum(sum);
    
    if (tid % 32 == 0) {
        s_sum[tid / 32] = sum;
    }
    __syncthreads();
    
    if (tid < 8) {
        sum = s_sum[tid];
        sum = warp_reduce_sum(sum);
        if (tid == 0) {
            *loss = sum / (float)size;  // Mean
        }
    }
}

// ============================================================================
// Launch Functions
// ============================================================================

extern "C" {

// MaxPool2D Forward
void launch_maxpool2d_forward(
    const half* input,
    half* output,
    int8_t* max_indices,
    int N, int H, int W, int C,
    cudaStream_t stream)
{
    int H_out = H / 2;
    int W_out = W / 2;
    
    if (C % 2 == 0) {
        // Vectorized version
        int total = N * H_out * W_out * (C / 2);
        int block = 256;
        int grid = (total + block - 1) / block;
        maxpool2d_forward_vec_kernel<<<grid, block, 0, stream>>>(
            input, output, max_indices, N, H, W, C);
    } else {
        int total = N * H_out * W_out * C;
        int block = 256;
        int grid = (total + block - 1) / block;
        maxpool2d_forward_kernel<<<grid, block, 0, stream>>>(
            input, output, max_indices, N, H, W, C);
    }
}

// MaxPool2D Backward
void launch_maxpool2d_backward(
    const half* grad_output,
    const int8_t* max_indices,
    half* grad_input,
    int N, int H, int W, int C,
    cudaStream_t stream)
{
    int H_out = H / 2;
    int W_out = W / 2;
    
    // Zero grad_input first
    int input_size = N * H * W * C;
    int zero_block = 256;
    int zero_grid = ((input_size + 1) / 2 + zero_block - 1) / zero_block;
    zero_kernel<<<zero_grid, zero_block, 0, stream>>>(grad_input, input_size);
    
    // Scatter gradients
    int total = N * H_out * W_out * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    maxpool2d_backward_kernel<<<grid, block, 0, stream>>>(
        grad_output, max_indices, grad_input, N, H, W, C);
}

// Upsample2D Forward
void launch_upsample2d_forward(
    const half* input,
    half* output,
    int N, int H, int W, int C,
    cudaStream_t stream)
{
    // Use fast version (1 input -> 4 outputs per thread)
    int total = N * H * W * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    upsample2d_forward_fast_kernel<<<grid, block, 0, stream>>>(
        input, output, N, H, W, C);
}

// Upsample2D Backward
void launch_upsample2d_backward(
    const half* grad_output,
    half* grad_input,
    int N, int H, int W, int C,
    cudaStream_t stream)
{
    int total = N * H * W * C;
    int block = 256;
    int grid = (total + block - 1) / block;
    upsample2d_backward_kernel<<<grid, block, 0, stream>>>(
        grad_output, grad_input, N, H, W, C);
}

// MSE Loss (returns loss value, writes gradient)
void launch_mse_loss_grad(
    const half* pred,
    const half* target,
    half* grad,
    float* loss,
    float* partial_buffer,  // Temporary buffer of size num_blocks
    int size,
    cudaStream_t stream)
{
    int block = 256;
    int grid = (size + block - 1) / block;
    
    // Fused loss + grad computation
    mse_loss_grad_fused_kernel<<<grid, block, 0, stream>>>(
        pred, target, grad, partial_buffer, size);
    
    // Reduce partial losses
    reduce_loss_kernel<<<1, 256, 0, stream>>>(
        partial_buffer, loss, grid, size);
}

// MSE Loss only (no gradient)
void launch_mse_loss(
    const half* pred,
    const half* target,
    float* loss,
    int size,
    cudaStream_t stream)
{
    cudaMemsetAsync(loss, 0, sizeof(float), stream);
    
    int block = 256;
    int grid = min((size + block - 1) / block, 256);  // Limit blocks for atomic efficiency
    
    mse_loss_kernel<<<grid, block, 0, stream>>>(
        pred, target, loss, size);
    
    // Need to divide by size for mean (do on host or add another kernel)
}

// MSE Gradient only
void launch_mse_grad(
    const half* pred,
    const half* target,
    half* grad,
    int size,
    cudaStream_t stream)
{
    int block = 256;
    int grid = ((size + 1) / 2 + block - 1) / block;
    
    mse_grad_kernel<<<grid, block, 0, stream>>>(
        pred, target, grad, size);
}

}  // extern "C"
