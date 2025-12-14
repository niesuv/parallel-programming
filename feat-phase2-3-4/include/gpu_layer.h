/**
 * @file gpu_layer.h
 * @brief GPU Neural Network Layer Definitions
 *
 * This file defines GPU-accelerated layer classes for the autoencoder.
 * All operations run on NVIDIA CUDA GPUs with optional optimized kernels.
 *
 * Key components:
 * - GPUTensor4D: GPU memory tensor with NCHW layout
 * - GPUConv2DLayer: CUDA convolution with optional fused ReLU
 * - GPUReLULayer: GPU ReLU activation
 * - GPUMaxPool2DLayer: GPU max pooling
 * - GPUUpSample2DLayer: GPU nearest-neighbor upsampling
 *
 * Compile with USE_OPTIMIZED_KERNELS for:
 * - Tiled convolution with shared memory
 * - Warp shuffle reduction
 * - Memory coalescing optimizations
 */

#ifndef GPU_LAYER_H
#define GPU_LAYER_H

#include <cstddef>

// Forward declaration for CUDA stream (defined in cuda_runtime.h)
typedef struct CUstream_st *cudaStream_t;

// =============================================================================
// GPUTensor4D: GPU Memory Tensor
// =============================================================================

/**
 * @brief 4D Tensor stored in GPU device memory
 *
 * Layout: NCHW (Batch, Channels, Height, Width)
 * Memory: Row-major contiguous in device memory (d_data)
 *
 * Non-copyable to prevent accidental GPU memory leaks.
 * Use move semantics for ownership transfer.
 */
struct GPUTensor4D {
  int n = 0;               ///< Batch size
  int c = 0;               ///< Number of channels
  int h = 0;               ///< Height
  int w = 0;               ///< Width
  float *d_data = nullptr; ///< Device memory pointer

  GPUTensor4D() = default;
  GPUTensor4D(int n_, int c_, int h_, int w_);
  ~GPUTensor4D();

  // Non-copyable (prevent double-free)
  GPUTensor4D(const GPUTensor4D &) = delete;
  GPUTensor4D &operator=(const GPUTensor4D &) = delete;

  // Movable
  GPUTensor4D(GPUTensor4D &&other) noexcept;
  GPUTensor4D &operator=(GPUTensor4D &&other) noexcept;

  /// Allocate/reallocate GPU memory
  void allocate(int n_, int c_, int h_, int w_);

  /// Free GPU memory
  void free();

  /// Total number of elements
  size_t size() const { return static_cast<size_t>(n) * c * h * w; }

  /// Total bytes in GPU memory
  size_t bytes() const { return size() * sizeof(float); }

  /// Copy data from CPU (host) to GPU (device)
  void copy_from_host(const float *h_data);

  /// Copy data from GPU (device) to CPU (host)
  void copy_to_host(float *h_data) const;

  /// Async copy from host using CUDA stream
  void copy_from_host_async(const float *h_data, cudaStream_t stream);
};

// =============================================================================
// GPUConv2DLayer: GPU 2D Convolutional Layer
// =============================================================================

/**
 * @brief GPU-accelerated 2D Convolutional Layer
 *
 * Features:
 * - Standard forward/backward passes
 * - Fused ReLU variants for better performance
 * - Supports different kernel sizes, strides, padding
 *
 * Weights stored in GPU memory, initialized to zero.
 * Use copy_weights_from_host() to load pre-trained weights.
 */
class GPUConv2DLayer {
public:
  GPUConv2DLayer(int in_channels, int out_channels, int kernel_size,
                 int stride = 1, int padding = 1);
  ~GPUConv2DLayer();

  /// Standard forward pass
  void forward(const GPUTensor4D &input, GPUTensor4D &output,
               cudaStream_t stream) const;

  /// Forward with fused ReLU activation (faster)
  void forward_fused_relu(const GPUTensor4D &input, GPUTensor4D &output,
                          cudaStream_t stream) const;

  /// Backward pass with gradient descent weight update
  void backward(const GPUTensor4D &input, const GPUTensor4D &grad_output,
                GPUTensor4D &grad_input, float learning_rate,
                cudaStream_t stream);

  /// Backward with fused ReLU gradient
  void backward_fused_relu(const GPUTensor4D &input,
                           const GPUTensor4D &grad_output,
                           GPUTensor4D &grad_input, float learning_rate,
                           cudaStream_t stream);

  /// Load weights from CPU memory
  void copy_weights_from_host(const float *h_weights, const float *h_bias);

  /// Save weights to CPU memory
  void copy_weights_to_host(float *h_weights, float *h_bias) const;

  // Output dimension calculators
  int get_output_h(int input_h) const {
    return (input_h + 2 * padding_ - k_) / stride_ + 1;
  }
  int get_output_w(int input_w) const {
    return (input_w + 2 * padding_ - k_) / stride_ + 1;
  }

  // Getters for layer parameters
  int get_out_channels() const { return out_c_; }
  int get_in_channels() const { return in_c_; }
  int get_kernel_size() const { return k_; }
  int get_stride() const { return stride_; }
  int get_padding() const { return padding_; }

private:
  int in_c_, out_c_, k_, stride_, padding_;
  float *d_weights_ = nullptr;      ///< [out_c, in_c, k, k] in device memory
  float *d_bias_ = nullptr;         ///< [out_c] in device memory
  float *d_grad_weights_ = nullptr; ///< Gradient accumulator
  float *d_grad_bias_ = nullptr;    ///< Gradient accumulator
  size_t weights_size_;
};

// =============================================================================
// GPUReLULayer: GPU ReLU Activation
// =============================================================================

/**
 * @brief GPU ReLU Activation Layer
 *
 * Applies element-wise: output = max(0, input)
 * Stateless - no learnable parameters
 */
class GPUReLULayer {
public:
  void forward(const GPUTensor4D &input, GPUTensor4D &output,
               cudaStream_t stream) const;
  void backward(const GPUTensor4D &input, const GPUTensor4D &grad_output,
                GPUTensor4D &grad_input, cudaStream_t stream) const;
};

// =============================================================================
// GPUMaxPool2DLayer: GPU Max Pooling
// =============================================================================

/**
 * @brief GPU Max Pooling Layer
 *
 * Downsamples by taking maximum in each kernel window.
 * Default: 2x2 pooling, stride 2 (halves spatial dimensions)
 */
class GPUMaxPool2DLayer {
public:
  explicit GPUMaxPool2DLayer(int kernel_size = 2, int stride = 2);

  void forward(const GPUTensor4D &input, GPUTensor4D &output,
               cudaStream_t stream) const;
  void backward(const GPUTensor4D &input, const GPUTensor4D &grad_output,
                GPUTensor4D &grad_input, cudaStream_t stream) const;

  int get_output_h(int input_h) const { return (input_h - k_) / stride_ + 1; }
  int get_output_w(int input_w) const { return (input_w - k_) / stride_ + 1; }

private:
  int k_, stride_;
};

// =============================================================================
// GPUUpSample2DLayer: GPU Upsampling
// =============================================================================

/**
 * @brief GPU Nearest-Neighbor Upsampling Layer
 *
 * Upsamples by replicating pixel values.
 * Default: scale=2 (doubles spatial dimensions)
 */
class GPUUpSample2DLayer {
public:
  explicit GPUUpSample2DLayer(int scale = 2);

  void forward(const GPUTensor4D &input, GPUTensor4D &output,
               cudaStream_t stream) const;
  void backward(const GPUTensor4D &input, const GPUTensor4D &grad_output,
                GPUTensor4D &grad_input, cudaStream_t stream) const;

  int get_output_h(int input_h) const { return input_h * scale_; }
  int get_output_w(int input_w) const { return input_w * scale_; }

private:
  int scale_;
};

// =============================================================================
// GPU Loss Functions
// =============================================================================

/**
 * @brief GPU MSE Loss using parallel reduction
 * @param h_partial_sums Pre-allocated host buffer for reduction
 */
float gpu_mse_loss(const GPUTensor4D &output, const GPUTensor4D &target,
                   float *h_partial_sums, cudaStream_t stream);

/**
 * @brief GPU MSE Loss with gradient computation
 * @param[out] grad_output Gradient tensor (allocated if needed)
 */
float gpu_mse_loss_with_grad(const GPUTensor4D &output,
                             const GPUTensor4D &target,
                             GPUTensor4D &grad_output, float *h_partial_sums,
                             cudaStream_t stream);
                             
void init_epoch_loss_accumulator(float** d_epoch_loss);
void reset_epoch_loss(float* d_epoch_loss, cudaStream_t stream = 0);
float get_epoch_loss(float* d_epoch_loss, size_t total_elements);
void cleanup_epoch_loss_accumulator();
void gpu_mse_loss_accumulate(const GPUTensor4D &output,
                            const GPUTensor4D &target,
                            float* d_epoch_loss,
                            cudaStream_t stream);
void gpu_mse_loss_with_grad_accumulate(const GPUTensor4D &output,
                                      const GPUTensor4D &target,
                                      GPUTensor4D &grad_output,
                                      float* d_epoch_loss,
                                      cudaStream_t stream);

// =============================================================================
// Optimized Kernel Functions (Phase 3)
// =============================================================================

#ifdef USE_OPTIMIZED_KERNELS

// Forward pass optimizations
void gpu_relu_forward_opt(const GPUTensor4D &input, GPUTensor4D &output,
                          cudaStream_t stream);
void gpu_maxpool2d_forward_opt(const GPUTensor4D &input, GPUTensor4D &output,
                               int k, int stride, cudaStream_t stream);
void gpu_upsample2d_forward_opt(const GPUTensor4D &input, GPUTensor4D &output,
                                int scale, cudaStream_t stream);
void gpu_conv2d_forward_tiled(const GPUTensor4D &input, const float *d_weights,
                              const float *d_bias, GPUTensor4D &output,
                              int in_c, int out_c, int k, int stride,
                              int padding, cudaStream_t stream);
void gpu_conv2d_relu_forward_opt(const GPUTensor4D &input,
                                 const float *d_weights, const float *d_bias,
                                 GPUTensor4D &output, int in_c, int out_c,
                                 int k, int stride, int padding,
                                 cudaStream_t stream);

// Backward pass optimizations
void gpu_relu_backward_opt(const GPUTensor4D &input,
                           const GPUTensor4D &grad_output,
                           GPUTensor4D &grad_input, cudaStream_t stream);
void gpu_conv2d_backward_data_opt(const GPUTensor4D &grad_output,
                                  const float *d_weights,
                                  GPUTensor4D &grad_input, int batch_size,
                                  int in_c, int in_h, int in_w, int out_c,
                                  int k, int stride, int padding,
                                  cudaStream_t stream);
void gpu_conv2d_backward_weights_opt(const GPUTensor4D &input,
                                     const GPUTensor4D &grad_output,
                                     float *d_grad_weights, float *d_grad_bias,
                                     int in_c, int out_c, int k, int stride,
                                     int padding, cudaStream_t stream);
void gpu_maxpool2d_backward_opt(const GPUTensor4D &input,
                                const GPUTensor4D &grad_output,
                                GPUTensor4D &grad_input, int k, int stride,
                                cudaStream_t stream);
void gpu_upsample2d_backward_opt(const GPUTensor4D &grad_output,
                                 GPUTensor4D &grad_input, int scale,
                                 cudaStream_t stream);
void gpu_conv2d_relu_backward_data_opt(const GPUTensor4D &grad_output,
                                       const float *d_weights,
                                       const GPUTensor4D &input,
                                       GPUTensor4D &grad_input, int batch_size,
                                       int in_c, int in_h, int in_w, int out_c,
                                       int k, int stride, int padding,
                                       cudaStream_t stream);

#endif // USE_OPTIMIZED_KERNELS

#endif // GPU_LAYER_H
