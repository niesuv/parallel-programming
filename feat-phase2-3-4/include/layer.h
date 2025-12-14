/**
 * @file layer.h
 * @brief CPU Neural Network Layer Definitions
 *
 * This file defines the fundamental data structures and layer classes
 * for the CPU implementation of the autoencoder neural network.
 *
 * Key components:
 * - Tensor4D: 4D tensor (N, C, H, W) for image batch storage
 * - Conv2DLayer: Convolutional layer with forward/backward pass
 * - ReLULayer: ReLU activation function
 * - MaxPool2DLayer: Max pooling for downsampling
 * - UpSample2DLayer: Nearest-neighbor upsampling for decoder
 */

#ifndef LAYER_H
#define LAYER_H

#include <algorithm>
#include <cstddef>
#include <iosfwd>
#include <vector>

// =============================================================================
// Tensor4D: 4D Tensor Data Structure
// =============================================================================

/**
 * @brief 4D Tensor for storing batched image data
 *
 * Layout: NCHW (Batch, Channels, Height, Width)
 * Memory: Row-major contiguous storage
 */
struct Tensor4D {
  int n = 0;               ///< Batch size
  int c = 0;               ///< Number of channels
  int h = 0;               ///< Height
  int w = 0;               ///< Width
  std::vector<float> data; ///< Contiguous data storage

  Tensor4D() = default;

  /**
   * @brief Construct tensor with given dimensions, initialized to zero
   */
  Tensor4D(int n_, int c_, int h_, int w_)
      : n(n_), c(c_), h(h_), w(w_),
        data(static_cast<std::size_t>(n_) * c_ * h_ * w_, 0.0f) {}

  /**
   * @brief Access element at position [batch, channel, row, col]
   */
  inline float &at(int ni, int ci, int hi, int wi) {
    std::size_t idx =
        (((static_cast<std::size_t>(ni) * c + ci) * h + hi) * w + wi);
    return data[idx];
  }

  inline const float &at(int ni, int ci, int hi, int wi) const {
    std::size_t idx =
        (((static_cast<std::size_t>(ni) * c + ci) * h + hi) * w + wi);
    return data[idx];
  }
};

// =============================================================================
// Conv2DLayer: 2D Convolutional Layer
// =============================================================================

/**
 * @brief 2D Convolutional Layer
 *
 * Implements convolution with learnable weights and biases.
 * Output size: (H + 2*padding - kernel_size) / stride + 1
 */
class Conv2DLayer {
public:
  Conv2DLayer(int in_channels, int out_channels, int kernel_size,
              int stride = 1, int padding = 1);

  /// Forward pass: input -> output
  Tensor4D forward(const Tensor4D &input) const;

  /// Backward pass: compute gradients and update weights
  Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output,
                    float learning_rate);

  void save(std::ostream &os) const;
  void load(std::istream &is);

private:
  int in_c_;                   ///< Input channels
  int out_c_;                  ///< Output channels
  int k_;                      ///< Kernel size
  int stride_;                 ///< Stride
  int padding_;                ///< Padding
  std::vector<float> weights_; ///< Shape: [out_c, in_c, k, k]
  std::vector<float> bias_;    ///< Shape: [out_c]
};

// =============================================================================
// ReLULayer: Rectified Linear Unit Activation
// =============================================================================

/**
 * @brief ReLU Activation Layer
 *
 * Applies element-wise: output = max(0, input)
 */
class ReLULayer {
public:
  Tensor4D forward(const Tensor4D &input) const;
  Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output) const;
};

// =============================================================================
// MaxPool2DLayer: Max Pooling Layer
// =============================================================================

/**
 * @brief Max Pooling Layer
 *
 * Downsamples input by taking max value in each kernel window.
 * Default: 2x2 pooling with stride 2 (halves spatial dimensions)
 */
class MaxPool2DLayer {
public:
  explicit MaxPool2DLayer(int kernel_size = 2, int stride = 2);

  Tensor4D forward(const Tensor4D &input) const;
  Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output) const;

private:
  int k_;      ///< Kernel size
  int stride_; ///< Stride
};

// =============================================================================
// UpSample2DLayer: Upsampling Layer
// =============================================================================

/**
 * @brief Nearest-Neighbor Upsampling Layer
 *
 * Upsamples input by replicating values.
 * Default: scale=2 (doubles spatial dimensions)
 */
class UpSample2DLayer {
public:
  explicit UpSample2DLayer(int scale = 2);

  Tensor4D forward(const Tensor4D &input) const;
  Tensor4D backward(const Tensor4D &input, const Tensor4D &grad_output) const;

private:
  int scale_; ///< Scale factor
};

// =============================================================================
// Loss Functions
// =============================================================================

/**
 * @brief Mean Squared Error Loss
 * @return MSE = mean((output - target)^2)
 */
float mse_loss(const Tensor4D &output, const Tensor4D &target);

/**
 * @brief MSE Loss with Gradient Computation
 * @param[out] grad_output Gradient with respect to output
 * @return MSE loss value
 */
float mse_loss_with_grad(const Tensor4D &output, const Tensor4D &target,
                         Tensor4D &grad_output);

#endif // LAYER_H
