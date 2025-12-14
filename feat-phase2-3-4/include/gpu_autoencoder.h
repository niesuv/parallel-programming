/**
 * @file gpu_autoencoder.h
 * @brief GPU-Accelerated Autoencoder Neural Network
 *
 * CUDA implementation of the convolutional autoencoder.
 * All operations run on GPU with optional optimized kernels.
 *
 * Input:  [N, 3, 32, 32]  - RGB images (CIFAR-10)
 * Latent: [N, 128, 8, 8]  - Compressed representation (8192 features)
 * Output: [N, 3, 32, 32]  - Reconstructed images
 */

#ifndef GPU_AUTOENCODER_H
#define GPU_AUTOENCODER_H

#include "gpu_layer.h"
#include "layer.h"

#include <string>
#include <vector>

/**
 * @brief GPU Convolutional Autoencoder
 *
 * CUDA-accelerated version with:
 * - Async CUDA streams for parallel execution
 * - Fused Conv+ReLU kernels for performance
 * - Binary weight file I/O (compatible with CPU version)
 *
 * Architecture (same as CPU):
 * ┌─────────── ENCODER ───────────┐
 * │ Conv1(3→256) → ReLU → Pool    │  32×32 → 16×16
 * │ Conv2(256→128) → ReLU → Pool  │  16×16 → 8×8
 * └───────────────────────────────┘
 * ┌─────────── DECODER ───────────┐
 * │ Conv3(128→128) → ReLU → Up    │  8×8 → 16×16
 * │ Conv4(128→256) → ReLU → Up    │  16×16 → 32×32
 * │ Conv5(256→3)                  │  Output layer
 * └───────────────────────────────┘
 */
class GPUAutoencoder {
public:
  GPUAutoencoder();
  ~GPUAutoencoder();

  /// Load weights from CPU autoencoder (deprecated)
  void load_weights_from_cpu(const class Autoencoder &cpu_ae);

  /// Full forward pass (encoder + decoder)
  void forward(const GPUTensor4D &input, GPUTensor4D &output,
               cudaStream_t stream);

  /// Training step with gradient descent
  float train_step(const GPUTensor4D &input, const GPUTensor4D &target,
                   float learning_rate, cudaStream_t stream,
                   float *h_partial_sums);

  /// Encoder only: extract 8192-dim feature vector for SVM
  void encode(const GPUTensor4D &input, GPUTensor4D &latent,
              cudaStream_t stream);

  /// Save weights to binary file
  bool save_weights(const std::string &path) const;

  /// Load weights from binary file
  bool load_weights(const std::string &path);

  /// Wait for all GPU operations to complete
  void synchronize();

private:
  // ===== ENCODER LAYERS =====
  GPUConv2DLayer conv1_; ///< 3 → 256 channels
  GPUReLULayer relu1_;
  GPUMaxPool2DLayer pool1_; ///< 32×32 → 16×16

  GPUConv2DLayer conv2_; ///< 256 → 128 channels
  GPUReLULayer relu2_;
  GPUMaxPool2DLayer pool2_; ///< 16×16 → 8×8 (latent)

  // ===== DECODER LAYERS =====
  GPUConv2DLayer conv3_; ///< 128 → 128 channels
  GPUReLULayer relu3_;
  GPUUpSample2DLayer up1_; ///< 8×8 → 16×16

  GPUConv2DLayer conv4_; ///< 128 → 256 channels
  GPUReLULayer relu4_;
  GPUUpSample2DLayer up2_; ///< 16×16 → 32×32

  GPUConv2DLayer conv5_; ///< 256 → 3 channels (output)

  // ===== INTERMEDIATE ACTIVATIONS =====
  GPUTensor4D x0_, x1_, x2_, x3_, x4_, x5_, x6_;
  GPUTensor4D x7_, x8_, x9_, x10_, x11_, x12_, x13_;

  // ===== GRADIENTS FOR BACKPROP =====
  GPUTensor4D g0_, g1_, g2_, g3_, g4_, g5_, g6_;
  GPUTensor4D g7_, g8_, g9_, g10_, g11_, g12_, g13_;

  void copy_input(const GPUTensor4D &input);
};

// =============================================================================
// Utility Functions
// =============================================================================

/// Copy CPU tensor to GPU tensor
void tensor_cpu_to_gpu(const Tensor4D &cpu_tensor, GPUTensor4D &gpu_tensor);

/// Copy GPU tensor to CPU tensor
void tensor_gpu_to_cpu(const GPUTensor4D &gpu_tensor, Tensor4D &cpu_tensor);

/// Copy raw CPU data to GPU tensor
void batch_cpu_to_gpu(const float *cpu_data, int n, int c, int h, int w,
                      GPUTensor4D &gpu_tensor);

#endif // GPU_AUTOENCODER_H
