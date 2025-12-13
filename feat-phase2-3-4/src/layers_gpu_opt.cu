#ifdef USE_OPTIMIZED_KERNELS

#include "cuda_utils.h"
#include "gpu_layer.h"

#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <cstdio>

// Declare external kernel from layers_gpu.cu
extern __global__ void relu_forward_kernel(const float *input, float *output,
                                           size_t n);

#define TILE_WIDTH 16
#define TILE_HEIGHT 16

#define WARP_SIZE 32

__constant__ float c_weights_small[8192];
__constant__ float c_bias_small[256];

__global__ void conv2d_forward_tiled_kernel(
    const float *__restrict__ input, const float *__restrict__ weights,
    const float *__restrict__ bias, float *__restrict__ output, int batch_size,
    int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k,
    int stride, int padding) {
  extern __shared__ float shared_mem[];

  int ow = blockIdx.x * TILE_WIDTH + threadIdx.x;
  int oh = blockIdx.y * TILE_HEIGHT + threadIdx.y;
  int oc = blockIdx.z % out_c;
  int n = blockIdx.z / out_c;

  if (ow >= out_w || oh >= out_h || n >= batch_size)
    return;

  float sum = bias[oc];

  const int tile_h = TILE_HEIGHT * stride + k - stride;
  const int tile_w = TILE_WIDTH * stride + k - stride;
  const int tile_size = tile_h * tile_w;
  const int threads_per_block = blockDim.x * blockDim.y;
  const int num_loads = (tile_size + threads_per_block - 1) / threads_per_block;
  const int linear_tid = threadIdx.y * blockDim.x + threadIdx.x;

  float *s_input = shared_mem;

  for (int ic = 0; ic < in_c; ++ic) {
    const int in_start_h = blockIdx.y * TILE_HEIGHT * stride - padding;
    const int in_start_w = blockIdx.x * TILE_WIDTH * stride - padding;
    const size_t in_channel_offset =
        (static_cast<size_t>(n) * in_c + ic) * in_h * in_w;

#pragma unroll 4
    for (int load = 0; load < num_loads; ++load) {
      int linear_idx = load * threads_per_block + linear_tid;
      if (linear_idx < tile_size) {
        int sh = linear_idx / tile_w;
        int sw = linear_idx % tile_w;
        int ih = in_start_h + sh;
        int iw = in_start_w + sw;

        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
          s_input[linear_idx] = input[in_channel_offset + ih * in_w + iw];
        } else {
          s_input[linear_idx] = 0.0f;
        }
      }
    }
    __syncthreads();

    const int local_h = threadIdx.y * stride;
    const int local_w = threadIdx.x * stride;
    const size_t w_ic_offset = (static_cast<size_t>(oc) * in_c + ic) * k * k;

#pragma unroll
    for (int kh = 0; kh < k; ++kh) {
#pragma unroll
      for (int kw = 0; kw < k; ++kw) {
        sum += s_input[(local_h + kh) * tile_w + local_w + kw] *
               weights[w_ic_offset + kh * k + kw];
      }
    }
    __syncthreads();
  }

  size_t out_idx =
      ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
  output[out_idx] = sum;
}

__global__ void conv2d_relu_forward_kernel(
    const float *__restrict__ input, const float *__restrict__ weights,
    const float *__restrict__ bias, float *__restrict__ output, int batch_size,
    int in_c, int in_h, int in_w, int out_c, int out_h, int out_w, int k,
    int stride, int padding) {
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_outputs = batch_size * out_c * out_h * out_w;

  if (idx >= total_outputs)
    return;

  int ow = idx % out_w;
  int temp = idx / out_w;
  int oh = temp % out_h;
  temp = temp / out_h;
  int oc = temp % out_c;
  int n = temp / out_c;

  float sum = bias[oc];

#pragma unroll 4
  for (int ic = 0; ic < in_c; ++ic) {
    for (int kh = 0; kh < k; ++kh) {
      for (int kw = 0; kw < k; ++kw) {
        int ih = oh * stride + kh - padding;
        int iw = ow * stride + kw - padding;

        if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
          size_t in_idx =
              ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
          size_t w_idx =
              ((static_cast<size_t>(oc) * in_c + ic) * k + kh) * k + kw;
          sum += input[in_idx] * weights[w_idx];
        }
      }
    }
  }

  output[idx] = fmaxf(0.0f, sum);
}

__global__ void relu_forward_vectorized_kernel(const float4 *input,
                                               float4 *output, size_t n4) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n4) {
    float4 in = input[idx];
    float4 out;
    out.x = fmaxf(0.0f, in.x);
    out.y = fmaxf(0.0f, in.y);
    out.z = fmaxf(0.0f, in.z);
    out.w = fmaxf(0.0f, in.w);
    output[idx] = out;
  }
}

__global__ void relu_backward_vectorized_kernel(const float4 *input,
                                                const float4 *grad_output,
                                                float4 *grad_input, size_t n4) {
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
  if (idx < n4) {
    float4 in = input[idx];
    float4 go = grad_output[idx];
    float4 gi;
    gi.x = (in.x > 0.0f) ? go.x : 0.0f;
    gi.y = (in.y > 0.0f) ? go.y : 0.0f;
    gi.z = (in.z > 0.0f) ? go.z : 0.0f;
    gi.w = (in.w > 0.0f) ? go.w : 0.0f;
    grad_input[idx] = gi;
  }
}

__global__ void maxpool2d_forward_opt_kernel(const float *__restrict__ input,
                                             float *__restrict__ output,
                                             int batch_size, int channels,
                                             int in_h, int in_w, int out_h,
                                             int out_w, int k, int stride) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z % channels;
  int n = blockIdx.z / channels;

  if (ow >= out_w || oh >= out_h || n >= batch_size)
    return;

  float max_val = -1e30f;

#pragma unroll
  for (int kh = 0; kh < k; ++kh) {
#pragma unroll
    for (int kw = 0; kw < k; ++kw) {
      int ih = oh * stride + kh;
      int iw = ow * stride + kw;

      if (ih < in_h && iw < in_w) {
        size_t in_idx =
            ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
        max_val = fmaxf(max_val, input[in_idx]);
      }
    }
  }

  size_t out_idx =
      ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
  output[out_idx] = max_val;
}

__global__ void upsample2d_forward_opt_kernel(const float *__restrict__ input,
                                              float *__restrict__ output,
                                              int batch_size, int channels,
                                              int in_h, int in_w, int out_h,
                                              int out_w, int scale) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int c = blockIdx.z % channels;
  int n = blockIdx.z / channels;

  if (ow >= out_w || oh >= out_h || n >= batch_size)
    return;

  int ih = oh / scale;
  int iw = ow / scale;

  size_t in_idx =
      ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
  size_t out_idx =
      ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;

  output[out_idx] = input[in_idx];
}

__global__ void mse_loss_grad_fused_kernel(const float *__restrict__ output,
                                           const float *__restrict__ target,
                                           float *__restrict__ grad_output,
                                           float *__restrict__ partial_loss,
                                           float scale, size_t n) {
  extern __shared__ float sdata[];

  size_t tid = threadIdx.x;
  size_t idx = blockIdx.x * blockDim.x + threadIdx.x;

  float local_loss = 0.0f;

  if (idx < n) {
    float diff = output[idx] - target[idx];
    local_loss = diff * diff;
    grad_output[idx] = scale * diff;
  }

  sdata[tid] = local_loss;
  __syncthreads();

  for (unsigned int s = blockDim.x / 2; s > 0; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  if (tid == 0) {
    partial_loss[blockIdx.x] = sdata[0];
  }
}

void gpu_relu_forward_opt(const GPUTensor4D &input, GPUTensor4D &output) {
  size_t n = input.size();
  size_t n4 = n / 4;

  if (output.n != input.n || output.c != input.c || output.h != input.h ||
      output.w != input.w) {
    output.allocate(input.n, input.c, input.h, input.w);
  }

  if (n % 4 == 0) {
    int block_size = 256;
    int grid_size = (n4 + block_size - 1) / block_size;
    relu_forward_vectorized_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const float4 *>(input.d_data),
        reinterpret_cast<float4 *>(output.d_data), n4);
  } else {
    int block_size = 256;
    int grid_size = (n + block_size - 1) / block_size;
    relu_forward_kernel<<<grid_size, block_size>>>(input.d_data, output.d_data,
                                                   n);
  }
  CUDA_CHECK(cudaGetLastError());
}

void gpu_relu_backward_opt(const GPUTensor4D &input,
                           const GPUTensor4D &grad_output,
                           GPUTensor4D &grad_input) {
  size_t n = input.size();
  size_t n4 = n / 4;

  if (grad_input.n != input.n || grad_input.c != input.c ||
      grad_input.h != input.h || grad_input.w != input.w) {
    grad_input.allocate(input.n, input.c, input.h, input.w);
  }

  if (n % 4 == 0) {
    int block_size = 256;
    int grid_size = (n4 + block_size - 1) / block_size;
    relu_backward_vectorized_kernel<<<grid_size, block_size>>>(
        reinterpret_cast<const float4 *>(input.d_data),
        reinterpret_cast<const float4 *>(grad_output.d_data),
        reinterpret_cast<float4 *>(grad_input.d_data), n4);
  }
  CUDA_CHECK(cudaGetLastError());
}

void gpu_maxpool2d_forward_opt(const GPUTensor4D &input, GPUTensor4D &output,
                               int k, int stride) {
  int out_h = (input.h - k) / stride + 1;
  int out_w = (input.w - k) / stride + 1;

  if (output.n != input.n || output.c != input.c || output.h != out_h ||
      output.w != out_w) {
    output.allocate(input.n, input.c, out_h, out_w);
  }

  dim3 block(16, 16);
  dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,
            input.n * input.c);

  maxpool2d_forward_opt_kernel<<<grid, block>>>(
      input.d_data, output.d_data, input.n, input.c, input.h, input.w, out_h,
      out_w, k, stride);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_upsample2d_forward_opt(const GPUTensor4D &input, GPUTensor4D &output,
                                int scale) {
  int out_h = input.h * scale;
  int out_w = input.w * scale;

  if (output.n != input.n || output.c != input.c || output.h != out_h ||
      output.w != out_w) {
    output.allocate(input.n, input.c, out_h, out_w);
  }

  dim3 block(16, 16);
  dim3 grid((out_w + block.x - 1) / block.x, (out_h + block.y - 1) / block.y,
            input.n * input.c);

  upsample2d_forward_opt_kernel<<<grid, block>>>(input.d_data, output.d_data,
                                                 input.n, input.c, input.h,
                                                 input.w, out_h, out_w, scale);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_relu_forward_opt(const GPUTensor4D &input,
                                 const float *d_weights, const float *d_bias,
                                 GPUTensor4D &output, int in_c, int out_c,
                                 int k, int stride, int padding) {
  int out_h = (input.h + 2 * padding - k) / stride + 1;
  int out_w = (input.w + 2 * padding - k) / stride + 1;

  if (output.n != input.n || output.c != out_c || output.h != out_h ||
      output.w != out_w) {
    output.allocate(input.n, out_c, out_h, out_w);
  }

  int total = input.n * out_c * out_h * out_w;
  int block_size = 256;
  int grid_size = (total + block_size - 1) / block_size;

  conv2d_relu_forward_kernel<<<grid_size, block_size>>>(
      input.d_data, d_weights, d_bias, output.d_data, input.n, in_c, input.h,
      input.w, out_c, out_h, out_w, k, stride, padding);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_forward_tiled(const GPUTensor4D &input, const float *d_weights,
                              const float *d_bias, GPUTensor4D &output,
                              int in_c, int out_c, int k, int stride,
                              int padding) {
  int out_h = (input.h + 2 * padding - k) / stride + 1;
  int out_w = (input.w + 2 * padding - k) / stride + 1;

  if (output.n != input.n || output.c != out_c || output.h != out_h ||
      output.w != out_w) {
    output.allocate(input.n, out_c, out_h, out_w);
  }

  dim3 block(TILE_WIDTH, TILE_HEIGHT);
  dim3 grid((out_w + TILE_WIDTH - 1) / TILE_WIDTH,
            (out_h + TILE_HEIGHT - 1) / TILE_HEIGHT, input.n * out_c);

  int tile_h = TILE_HEIGHT * stride + k - stride;
  int tile_w = TILE_WIDTH * stride + k - stride;
  size_t shared_size = tile_h * tile_w * sizeof(float);

  conv2d_forward_tiled_kernel<<<grid, block, shared_size>>>(
      input.d_data, d_weights, d_bias, output.d_data, input.n, in_c, input.h,
      input.w, out_c, out_h, out_w, k, stride, padding);
  CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// OPTIMIZED BACKWARD KERNELS
// ============================================================================

// Optimized conv2d backward data kernel with 2D thread blocks
__global__ void conv2d_backward_data_opt_kernel(
    const float *__restrict__ grad_output, const float *__restrict__ weights,
    float *__restrict__ grad_input, int batch_size, int in_c, int in_h,
    int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
  int iw = blockIdx.x * blockDim.x + threadIdx.x;
  int ih = blockIdx.y * blockDim.y + threadIdx.y;
  int ic_n = blockIdx.z;
  int ic = ic_n % in_c;
  int n = ic_n / in_c;

  if (iw >= in_w || ih >= in_h || n >= batch_size)
    return;

  float sum = 0.0f;

  // For 3x3 kernel with stride 1, we can optimize the loop bounds
  if (k == 3 && stride == 1) {
#pragma unroll
    for (int oc = 0; oc < out_c; ++oc) {
      const size_t go_base =
          (static_cast<size_t>(n) * out_c + oc) * out_h * out_w;
      const size_t w_base = (static_cast<size_t>(oc) * in_c + ic) * 9;

#pragma unroll
      for (int kh = 0; kh < 3; ++kh) {
        int oh = ih + padding - kh;
        if (oh >= 0 && oh < out_h) {
          const size_t go_row = go_base + oh * out_w;
          const size_t w_kh = w_base + kh * 3;

#pragma unroll
          for (int kw = 0; kw < 3; ++kw) {
            int ow = iw + padding - kw;
            if (ow >= 0 && ow < out_w) {
              sum += grad_output[go_row + ow] * weights[w_kh + kw];
            }
          }
        }
      }
    }
  } else {
    for (int oc = 0; oc < out_c; ++oc) {
      for (int kh = 0; kh < k; ++kh) {
        for (int kw = 0; kw < k; ++kw) {
          int oh_check = ih + padding - kh;
          int ow_check = iw + padding - kw;

          if (oh_check % stride == 0 && ow_check % stride == 0) {
            int oh = oh_check / stride;
            int ow = ow_check / stride;

            if (oh >= 0 && oh < out_h && ow >= 0 && ow < out_w) {
              size_t go_idx =
                  ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w +
                  ow;
              size_t w_idx =
                  ((static_cast<size_t>(oc) * in_c + ic) * k + kh) * k + kw;
              sum += grad_output[go_idx] * weights[w_idx];
            }
          }
        }
      }
    }
  }

  size_t gi_idx =
      ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
  grad_input[gi_idx] = sum;
}

// Optimized conv2d backward weights kernel - parallel over weight elements with
// better memory access
__global__ void conv2d_backward_weights_opt_kernel(
    const float *__restrict__ input, const float *__restrict__ grad_output,
    float *__restrict__ grad_weights, int batch_size, int in_c, int in_h,
    int in_w, int out_c, int out_h, int out_w, int k, int stride, int padding) {
  // Each thread handles one weight element
  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  int total_weights = out_c * in_c * k * k;

  if (idx >= total_weights)
    return;

  int kw = idx % k;
  int temp = idx / k;
  int kh = temp % k;
  temp = temp / k;
  int ic = temp % in_c;
  int oc = temp / in_c;

  float sum = 0.0f;

  // Unroll for 3x3 kernel common case
  if (k == 3 && stride == 1) {
    for (int n = 0; n < batch_size; ++n) {
      const size_t in_n_base =
          (static_cast<size_t>(n) * in_c + ic) * in_h * in_w;
      const size_t go_n_base =
          (static_cast<size_t>(n) * out_c + oc) * out_h * out_w;

      for (int oh = 0; oh < out_h; ++oh) {
        int ih = oh + kh - padding;
        if (ih >= 0 && ih < in_h) {
          const size_t in_row = in_n_base + ih * in_w;
          const size_t go_row = go_n_base + oh * out_w;

#pragma unroll 4
          for (int ow = 0; ow < out_w; ++ow) {
            int iw = ow + kw - padding;
            if (iw >= 0 && iw < in_w) {
              sum += input[in_row + iw] * grad_output[go_row + ow];
            }
          }
        }
      }
    }
  } else {
    for (int n = 0; n < batch_size; ++n) {
      for (int oh = 0; oh < out_h; ++oh) {
        for (int ow = 0; ow < out_w; ++ow) {
          int ih = oh * stride + kh - padding;
          int iw = ow * stride + kw - padding;

          if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
            size_t in_idx =
                ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
            size_t go_idx =
                ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w +
                ow;
            sum += input[in_idx] * grad_output[go_idx];
          }
        }
      }
    }
  }

  grad_weights[idx] = sum;
}

// Optimized bias gradient with parallel reduction
__global__ void
conv2d_backward_bias_opt_kernel(const float *__restrict__ grad_output,
                                float *__restrict__ grad_bias, int batch_size,
                                int out_c, int out_h, int out_w) {
  extern __shared__ float sdata[];

  int oc = blockIdx.x;
  int tid = threadIdx.x;
  int total_spatial = batch_size * out_h * out_w;

  float sum = 0.0f;
  for (int i = tid; i < total_spatial; i += blockDim.x) {
    int n = i / (out_h * out_w);
    int spatial = i % (out_h * out_w);
    int oh = spatial / out_w;
    int ow = spatial % out_w;

    size_t idx =
        ((static_cast<size_t>(n) * out_c + oc) * out_h + oh) * out_w + ow;
    sum += grad_output[idx];
  }

  sdata[tid] = sum;
  __syncthreads();

  // Reduction in shared memory
  for (unsigned int s = blockDim.x / 2; s > 32; s >>= 1) {
    if (tid < s) {
      sdata[tid] += sdata[tid + s];
    }
    __syncthreads();
  }

  // Warp-level reduction
  if (tid < 32) {
    volatile float *vsmem = sdata;
    if (blockDim.x >= 64)
      vsmem[tid] += vsmem[tid + 32];
    float val = vsmem[tid];
    for (int offset = 16; offset > 0; offset /= 2) {
      val += __shfl_down_sync(0xffffffff, val, offset);
    }
    if (tid == 0) {
      grad_bias[oc] = val;
    }
  }
}

// Optimized maxpool backward with 2D thread blocks
__global__ void maxpool2d_backward_opt_kernel(
    const float *__restrict__ input, const float *__restrict__ grad_output,
    float *__restrict__ grad_input, int batch_size, int channels, int in_h,
    int in_w, int out_h, int out_w, int k, int stride) {
  int ow = blockIdx.x * blockDim.x + threadIdx.x;
  int oh = blockIdx.y * blockDim.y + threadIdx.y;
  int c_n = blockIdx.z;
  int c = c_n % channels;
  int n = c_n / channels;

  if (ow >= out_w || oh >= out_h || n >= batch_size)
    return;

  const size_t in_base =
      ((static_cast<size_t>(n) * channels + c) * in_h) * in_w;
  const int ih_base = oh * stride;
  const int iw_base = ow * stride;

  float max_val = -1e30f;
  int max_ih = -1, max_iw = -1;

  // Unroll for 2x2 pooling
  if (k == 2 && stride == 2) {
    float v00 = input[in_base + ih_base * in_w + iw_base];
    float v01 = input[in_base + ih_base * in_w + iw_base + 1];
    float v10 = input[in_base + (ih_base + 1) * in_w + iw_base];
    float v11 = input[in_base + (ih_base + 1) * in_w + iw_base + 1];

    max_val = v00;
    max_ih = ih_base;
    max_iw = iw_base;
    if (v01 > max_val) {
      max_val = v01;
      max_ih = ih_base;
      max_iw = iw_base + 1;
    }
    if (v10 > max_val) {
      max_val = v10;
      max_ih = ih_base + 1;
      max_iw = iw_base;
    }
    if (v11 > max_val) {
      max_val = v11;
      max_ih = ih_base + 1;
      max_iw = iw_base + 1;
    }
  } else {
    for (int kh = 0; kh < k; ++kh) {
      for (int kw = 0; kw < k; ++kw) {
        int ih = ih_base + kh;
        int iw = iw_base + kw;
        if (ih < in_h && iw < in_w) {
          float val = input[in_base + ih * in_w + iw];
          if (val > max_val) {
            max_val = val;
            max_ih = ih;
            max_iw = iw;
          }
        }
      }
    }
  }

  if (max_ih >= 0 && max_iw >= 0) {
    size_t go_idx =
        ((static_cast<size_t>(n) * channels + c) * out_h + oh) * out_w + ow;
    size_t gi_idx =
        ((static_cast<size_t>(n) * channels + c) * in_h + max_ih) * in_w +
        max_iw;
    atomicAdd(&grad_input[gi_idx], grad_output[go_idx]);
  }
}

// Optimized upsample backward with 2D blocks
__global__ void
upsample2d_backward_opt_kernel(const float *__restrict__ grad_output,
                               float *__restrict__ grad_input, int batch_size,
                               int channels, int in_h, int in_w, int out_h,
                               int out_w, int scale) {
  int iw = blockIdx.x * blockDim.x + threadIdx.x;
  int ih = blockIdx.y * blockDim.y + threadIdx.y;
  int c_n = blockIdx.z;
  int c = c_n % channels;
  int n = c_n / channels;

  if (iw >= in_w || ih >= in_h || n >= batch_size)
    return;

  float sum = 0.0f;

  // Sum all corresponding output gradients
  int oh_base = ih * scale;
  int ow_base = iw * scale;

  const size_t go_base =
      ((static_cast<size_t>(n) * channels + c) * out_h) * out_w;

#pragma unroll
  for (int sh = 0; sh < scale; ++sh) {
#pragma unroll
    for (int sw = 0; sw < scale; ++sw) {
      int oh = oh_base + sh;
      int ow = ow_base + sw;
      if (oh < out_h && ow < out_w) {
        sum += grad_output[go_base + oh * out_w + ow];
      }
    }
  }

  size_t gi_idx =
      ((static_cast<size_t>(n) * channels + c) * in_h + ih) * in_w + iw;
  grad_input[gi_idx] = sum;
}

// ============================================================================
// BACKWARD WRAPPER FUNCTIONS
// ============================================================================

void gpu_conv2d_backward_data_opt(const GPUTensor4D &grad_output,
                                  const float *d_weights,
                                  GPUTensor4D &grad_input, int batch_size,
                                  int in_c, int in_h, int in_w, int out_c,
                                  int k, int stride, int padding) {
  dim3 block(16, 16);
  dim3 grid((in_w + block.x - 1) / block.x, (in_h + block.y - 1) / block.y,
            batch_size * in_c);

  conv2d_backward_data_opt_kernel<<<grid, block>>>(
      grad_output.d_data, d_weights, grad_input.d_data, batch_size, in_c, in_h,
      in_w, out_c, grad_output.h, grad_output.w, k, stride, padding);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_conv2d_backward_weights_opt(const GPUTensor4D &input,
                                     const GPUTensor4D &grad_output,
                                     float *d_grad_weights, float *d_grad_bias,
                                     int in_c, int out_c, int k, int stride,
                                     int padding) {
  // Weights gradient
  int total_weights = out_c * in_c * k * k;
  int block_size = 256;
  int grid_size = (total_weights + block_size - 1) / block_size;

  conv2d_backward_weights_opt_kernel<<<grid_size, block_size>>>(
      input.d_data, grad_output.d_data, d_grad_weights, input.n, in_c, input.h,
      input.w, out_c, grad_output.h, grad_output.w, k, stride, padding);
  CUDA_CHECK(cudaGetLastError());

  // Bias gradient with parallel reduction
  int bias_block = 256;
  size_t shared_size = bias_block * sizeof(float);
  conv2d_backward_bias_opt_kernel<<<out_c, bias_block, shared_size>>>(
      grad_output.d_data, d_grad_bias, input.n, out_c, grad_output.h,
      grad_output.w);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_maxpool2d_backward_opt(const GPUTensor4D &input,
                                const GPUTensor4D &grad_output,
                                GPUTensor4D &grad_input, int k, int stride) {
  CUDA_CHECK(cudaMemset(grad_input.d_data, 0, grad_input.bytes()));

  dim3 block(16, 16);
  dim3 grid((grad_output.w + block.x - 1) / block.x,
            (grad_output.h + block.y - 1) / block.y, input.n * input.c);

  maxpool2d_backward_opt_kernel<<<grid, block>>>(
      input.d_data, grad_output.d_data, grad_input.d_data, input.n, input.c,
      input.h, input.w, grad_output.h, grad_output.w, k, stride);
  CUDA_CHECK(cudaGetLastError());
}

void gpu_upsample2d_backward_opt(const GPUTensor4D &grad_output,
                                 GPUTensor4D &grad_input, int scale) {
  dim3 block(16, 16);
  dim3 grid((grad_input.w + block.x - 1) / block.x,
            (grad_input.h + block.y - 1) / block.y,
            grad_input.n * grad_input.c);

  upsample2d_backward_opt_kernel<<<grid, block>>>(
      grad_output.d_data, grad_input.d_data, grad_input.n, grad_input.c,
      grad_input.h, grad_input.w, grad_output.h, grad_output.w, scale);
  CUDA_CHECK(cudaGetLastError());
}

#endif // USE_OPTIMIZED_KERNELS