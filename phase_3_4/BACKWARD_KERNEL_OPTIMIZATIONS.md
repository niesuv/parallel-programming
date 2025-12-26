# Conv2D Backward Kernel Optimizations

## Overview

This document describes the optimization techniques used in the FP16 Conv2D backward kernels (`gpu_conv2d_fp16_backward_v8.cu`). The backward pass consists of three components:

1. **Backward Input** (dL/dX) - Gradient with respect to input
2. **Backward Weight** (dL/dW) - Gradient with respect to weights
3. **Backward Bias** (dL/dB) - Gradient with respect to bias

Each has different computational patterns requiring different optimizations.

## Computational Complexity

```
Forward:   O(N × H × W × K × C × 9)
Backward Input:  O(N × H × W × K × C × 9)  - Same as forward
Backward Weight: O(N × H × W × K × C × 9)  - Reduction over N×H×W
Backward Bias:   O(N × H × W × K)          - Reduction over N×H×W
```

---

## Part 1: Backward Input (dL/dX)

### Mathematical Formulation

```
dL/dX[n,h,w,c] = Σ_k Σ_kh Σ_kw dL/dY[n, h-kh+1, w-kw+1, k] × W[k, kh, kw, c]
```

This is essentially a **transposed convolution** - convolve the upstream gradient with flipped weights.

### Optimization 1: Two-Pass Approach for Fused ReLU

**Problem:** ReLU backward requires the pre-activation values, but storing them doubles memory usage.

**Solution:** Use a two-pass approach:

```cpp
// Pass 1: Compute full backward input gradient
launch_conv2d_backward_input_opt(grad_out, weight, temp_grad, ...);

// Pass 2: Apply ReLU mask from saved conv_output
launch_relu_backward_apply(temp_grad, conv_output, grad_input, ...);
```

**Fused Two-Pass Kernel:**
```cpp
void launch_fused_relu_backward_input_twopass(
    const half* upstream_grad,    // dL/dY
    const half* conv_output,      // Pre-ReLU activation (saved from forward)
    const half* weight,           // Convolution weights
    half* grad_input,             // Output: dL/dX
    half* temp_buffer,            // Temporary buffer for intermediate gradient
    ...
) {
    // Pass 1: Backward convolution
    conv2d_backward_input_kernel<<<...>>>(upstream_grad, weight, temp_buffer, ...);
    
    // Pass 2: Apply ReLU mask
    relu_backward_kernel<<<...>>>(temp_buffer, conv_output, grad_input, ...);
}
```

**Benefits:**
- Avoids storing full pre-activation tensor
- Single kernel launch for both operations
- Memory efficient

---

### Optimization 2: Transposed Weight Layout

**Problem:** Backward input requires accessing weights in transposed order compared to forward.

**Forward access pattern:**
```cpp
W[k][kh][kw][c]  // Output channel, kernel position, input channel
```

**Backward input access pattern:**
```cpp
W[k][2-kh][2-kw][c]  // Flipped kernel (180° rotation)
```

**Solution:** Precompute flipped indices or use symmetric access:
```cpp
// Flip kernel indices for transposed convolution
int kh_flip = 2 - kh;
int kw_flip = 2 - kw;
half w_val = weight[k * 9 * C + kh_flip * 3 * C + kw_flip * C + c];
```

---

### Optimization 3: Shared Memory for Weight Reuse

**Observation:** Each weight element is used for many spatial positions.

```cpp
__shared__ half s_weight[K_TILE][9][C_TILE];

// Cooperative loading
int tid = threadIdx.x + threadIdx.y * blockDim.x;
for (int i = tid; i < K_TILE * 9 * C_TILE; i += blockDim.x * blockDim.y) {
    int k = i / (9 * C_TILE);
    int rem = i % (9 * C_TILE);
    int kpos = rem / C_TILE;
    int c = rem % C_TILE;
    s_weight[k][kpos][c] = global_weight[...];
}
__syncthreads();
```

**Benefits:**
- Weights loaded once per tile, used H×W times
- Reduces global memory bandwidth
- Better cache hit rate

---

### Optimization 4: Output-Stationary Dataflow

**What it is:** Each thread is responsible for computing one output gradient element completely.

```cpp
__global__ void conv2d_backward_input_kernel(...) {
    int n = blockIdx.z / C;
    int c = blockIdx.z % C;
    int h = blockIdx.y * blockDim.y + threadIdx.y;
    int w = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (h >= H || w >= W) return;
    
    float acc = 0.0f;  // FP32 accumulator for precision
    
    // Sum over all output channels and kernel positions
    for (int k = 0; k < K; k++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int oh = h - kh + 1;
                int ow = w - kw + 1;
                if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
                    acc += __half2float(grad_out[...]) * __half2float(weight[...]);
                }
            }
        }
    }
    
    grad_input[n * H * W * C + h * W * C + w * C + c] = __float2half(acc);
}
```

**Benefits:**
- No atomic operations needed
- Each output written exactly once
- Natural parallelization over spatial dimensions

---

## Part 2: Backward Weight (dL/dW)

### Mathematical Formulation

```
dL/dW[k,kh,kw,c] = Σ_n Σ_h Σ_w dL/dY[n,h,w,k] × X[n, h+kh-1, w+kw-1, c]
```

This is a **reduction** over the batch and spatial dimensions.

### Challenge: Massive Parallelism vs. Reduction

**Problem:** 
- Weight tensor: K × 9 × C elements (e.g., 256 × 9 × 128 = 294,912)
- Reduction over: N × H × W elements (e.g., 64 × 32 × 32 = 65,536)

Each weight gradient requires summing 65,536 terms!

---

### Optimization 5: Hierarchical Reduction

**Strategy:** Multiple levels of parallel reduction

```
Level 1: Thread-local accumulation (registers)
Level 2: Warp-level reduction (shuffle instructions)
Level 3: Block-level reduction (shared memory)
Level 4: Global reduction (atomic operations)
```

**Implementation:**
```cpp
__global__ void conv2d_backward_weight_kernel(...) {
    // Level 1: Thread accumulates partial sum
    float local_sum = 0.0f;
    
    for (int i = tid; i < N * H * W; i += stride) {
        // Compute contribution from this spatial position
        local_sum += grad_out[...] * input[...];
    }
    
    // Level 2: Warp reduction using shuffle
    for (int offset = 16; offset > 0; offset /= 2) {
        local_sum += __shfl_down_sync(0xffffffff, local_sum, offset);
    }
    
    // Level 3: Block reduction using shared memory
    __shared__ float s_sum[8];  // One per warp
    if (lane_id == 0) {
        s_sum[warp_id] = local_sum;
    }
    __syncthreads();
    
    // Level 4: Final reduction and atomic add
    if (tid == 0) {
        float block_sum = 0;
        for (int i = 0; i < num_warps; i++) {
            block_sum += s_sum[i];
        }
        atomicAdd(&grad_weight[k][kh][kw][c], block_sum);
    }
}
```

---

### Optimization 6: Warp Shuffle for Fast Reduction

**What it is:** Direct register-to-register communication within a warp without shared memory.

```cpp
__device__ float warp_reduce_sum(float val) {
    // All 32 threads in warp participate
    val += __shfl_down_sync(0xffffffff, val, 16);
    val += __shfl_down_sync(0xffffffff, val, 8);
    val += __shfl_down_sync(0xffffffff, val, 4);
    val += __shfl_down_sync(0xffffffff, val, 2);
    val += __shfl_down_sync(0xffffffff, val, 1);
    return val;  // Result in lane 0
}
```

**Shuffle diagram:**
```
Initial: [a0, a1, a2, a3, ..., a31]

After shfl_down 16: [a0+a16, a1+a17, ..., a15+a31, ...]
After shfl_down 8:  [a0+a8+a16+a24, ...]
After shfl_down 4:  [a0+a4+a8+...+a28, ...]
After shfl_down 2:  [sum of even + odd pairs, ...]
After shfl_down 1:  [total sum in lane 0, ...]
```

**Benefits:**
- 5 instructions for 32-element reduction (vs 31 for naive)
- No shared memory needed
- No synchronization barriers
- ~10× faster than shared memory reduction

---

### Optimization 7: Tiled Weight Gradient Computation

**Strategy:** Compute weight gradients in tiles to maximize reuse.

```cpp
// Tile configuration
#define K_TILE 64
#define C_TILE 64

__global__ void conv2d_backward_weight_large_kernel(...) {
    int k_base = blockIdx.x * K_TILE;
    int c_base = blockIdx.y * C_TILE;
    int kh = blockIdx.z / 3;
    int kw = blockIdx.z % 3;
    
    // Each thread handles a 4×4 sub-tile
    int thread_k = (threadIdx.x % 16) * 4;
    int thread_c = (threadIdx.x / 16) * 4;
    
    // Accumulators for 4×4 output tile
    float acc[4][4] = {0};
    
    // Process all spatial positions
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                // Load tiles into shared memory
                __shared__ half s_grad[K_TILE];
                __shared__ half s_input[C_TILE];
                
                // Cooperative loading...
                
                // Outer product accumulation
                for (int dk = 0; dk < 4; dk++) {
                    for (int dc = 0; dc < 4; dc++) {
                        acc[dk][dc] += s_grad[thread_k + dk] * s_input[thread_c + dc];
                    }
                }
            }
        }
    }
    
    // Write accumulated results
    for (int dk = 0; dk < 4; dk++) {
        for (int dc = 0; dc < 4; dc++) {
            grad_weight[...] = acc[dk][dc];
        }
    }
}
```

---

### Optimization 8: Multiple Kernel Variants by Size

Different channel counts require different strategies:

```cpp
void launch_conv2d_backward_weight_sgd(...) {
    if (K >= 64 && C >= 64) {
        // Large: Full 64×64 tiles, maximum parallelism
        conv2d_backward_weight_large_kernel<<<...>>>();
    } else if (K >= 32 && C >= 32) {
        // Medium: 32×32 tiles
        conv2d_backward_weight_medium_kernel<<<...>>>();
    } else {
        // Small: 16×16 tiles or naive
        conv2d_backward_weight_small_kernel<<<...>>>();
    }
}
```

**Large kernel (K≥64, C≥64):**
- 64×64 output tile per block
- 4×4 elements per thread
- Maximum shared memory utilization

**Medium kernel (K≥32, C≥32):**
- 32×32 output tile per block
- Better for moderate channel counts
- Less register pressure

**Small kernel (K<32 or C<32):**
- 16×16 tiles or naive implementation
- For first/last layers with few channels

---

### Optimization 9: FP32 Accumulation with FP16 Storage

**Problem:** FP16 has limited precision (3.3 decimal digits) - accumulating many small values causes precision loss.

**Solution:** Accumulate in FP32, convert to FP16 only for storage/loading:

```cpp
// Load as FP16, accumulate as FP32
float acc = 0.0f;
for (...) {
    half g = grad_out[...];
    half x = input[...];
    acc += __half2float(g) * __half2float(x);  // FP32 multiply-add
}

// Store weight gradient as FP32 (for SGD update)
grad_weight[idx] = acc;
```

**Benefits:**
- Maintains numerical precision during reduction
- Prevents gradient underflow/overflow
- Master weights stay in FP32

---

## Part 3: Backward Bias (dL/dB)

### Mathematical Formulation

```
dL/dB[k] = Σ_n Σ_h Σ_w dL/dY[n,h,w,k]
```

Simple sum reduction over batch and spatial dimensions.

### Optimization 10: Parallel Reduction for Bias

```cpp
__global__ void conv2d_backward_bias_kernel(
    const half* grad_output,
    float* grad_bias,
    int N, int H, int W, int K
) {
    int k = blockIdx.x;  // One block per output channel
    int tid = threadIdx.x;
    int spatial_size = N * H * W;
    
    // Thread-local accumulation
    float local_sum = 0.0f;
    for (int i = tid; i < spatial_size; i += blockDim.x) {
        int n = i / (H * W);
        int rem = i % (H * W);
        int h = rem / W;
        int w = rem % W;
        local_sum += __half2float(grad_output[((n * H + h) * W + w) * K + k]);
    }
    
    // Warp reduction
    local_sum = warp_reduce_sum(local_sum);
    
    // Block reduction
    __shared__ float s_sum[8];
    if (tid % 32 == 0) {
        s_sum[tid / 32] = local_sum;
    }
    __syncthreads();
    
    // Final reduction
    if (tid < 8) {
        local_sum = s_sum[tid];
        local_sum = warp_reduce_sum(local_sum);
        if (tid == 0) {
            grad_bias[k] = local_sum;
        }
    }
}
```

**Grid configuration:**
```cpp
conv2d_backward_bias_kernel<<<K, 256>>>(grad_output, grad_bias, N, H, W, K);
// K blocks, 256 threads each
// Each block computes one bias gradient
```

---

## Part 4: Fused Backward + SGD Update

### Optimization 11: Fused Weight Update

**Problem:** Separate backward and SGD kernels require extra memory traffic.

```cpp
// Separate (inefficient):
backward_weight_kernel<<<...>>>(grad_output, input, grad_weight);  // Write grad
sgd_kernel<<<...>>>(grad_weight, weight, lr);                       // Read grad, update weight
// 2 kernel launches, grad_weight read/written twice
```

**Fused approach:**
```cpp
void launch_conv2d_backward_weight_sgd(
    const half* grad_output,
    const half* input,
    float* grad_weight,        // Temporary buffer
    float* grad_bias,
    float* master_weight,      // FP32 master weights
    half* weight,              // FP16 weights
    float* master_bias,
    half* bias,
    float lr,
    ...
) {
    // Step 1: Compute gradients
    conv2d_backward_weight_kernel<<<...>>>(grad_output, input, grad_weight);
    conv2d_backward_bias_kernel<<<...>>>(grad_output, grad_bias);
    
    // Step 2: Apply SGD update (fused with weight gradient)
    sgd_update_weight_kernel<<<...>>>(master_weight, weight, grad_weight, lr);
    sgd_update_bias_kernel<<<...>>>(master_bias, bias, grad_bias, lr);
}
```

**SGD Update Kernel:**
```cpp
__global__ void sgd_update_weight_kernel(
    float* master_weight,
    half* weight,
    const float* grad,
    float lr,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;
    
    float w = master_weight[idx];
    float g = grad[idx];
    
    // SGD update: w = w - lr * g
    w -= lr * g;
    
    // Store both FP32 master and FP16 working copy
    master_weight[idx] = w;
    weight[idx] = __float2half(w);
}
```

**Benefits:**
- Reduces memory traffic
- Single pass through gradient buffer
- Maintains FP32 master weights for precision

## Memory Access Patterns

### Backward Input
```
Read:  grad_output (sequential), weight (reused)
Write: grad_input (sequential)
Pattern: Output-stationary, good coalescing
```

### Backward Weight
```
Read:  grad_output (strided), input (strided)
Write: grad_weight (random via atomic)
Pattern: Reduction, requires careful tiling
```

### Backward Bias
```
Read:  grad_output (strided by K)
Write: grad_bias (one element per block)
Pattern: Simple reduction, highly parallel
```

---

## Numerical Stability Considerations

### Gradient Clipping (Optional)

For training stability, gradients can be clipped:

```cpp
// In SGD update kernel
float g = grad[idx];

// Skip NaN/Inf
if (isnan(g) || isinf(g)) return;

// Optional: Gradient clipping
// g = fminf(fmaxf(g, -clip_value), clip_value);

w -= lr * g;
```

### Loss Scaling for FP16

When gradients are too small for FP16:

```cpp
// Scale up loss gradient
float loss_scale = 128.0f;
grad = grad * loss_scale;

// In SGD update, scale down
w -= lr * (g / loss_scale);
```

---

## References

1. "Megatron-LM: Training Multi-Billion Parameter Language Models" - Mixed precision training
2. NVIDIA cuDNN Developer Guide - Backward convolution algorithms  
3. "Mixed Precision Training" - Micikevicius et al., ICLR 2018
4. "Optimizing Parallel Reduction in CUDA" - Mark Harris, NVIDIA
