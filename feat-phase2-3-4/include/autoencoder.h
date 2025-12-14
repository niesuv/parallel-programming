/**
 * @file autoencoder.h
 * @brief CPU Autoencoder Neural Network
 *
 * Convolutional autoencoder for image reconstruction.
 * Architecture: Encoder (Conv→ReLU→Pool) × 2 + Decoder (Conv→ReLU→Up) × 2 +
 * Conv
 *
 * Input:  [N, 3, 32, 32]  - RGB images (CIFAR-10)
 * Latent: [N, 128, 8, 8]  - Compressed representation (8192 features)
 * Output: [N, 3, 32, 32]  - Reconstructed images
 */

#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "layer.h"
#include <string>

/**
 * @brief CPU Convolutional Autoencoder
 *
 * Architecture:
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
class Autoencoder {
public:
  Autoencoder();

  /// Full forward pass (encoder + decoder)
  Tensor4D forward(const Tensor4D &input) const;

  /// Training step: forward + MSE loss + backward + weight update
  float train_step(const Tensor4D &input, const Tensor4D &target,
                   float learning_rate);

  /// Encoder only: extract 8192-dim feature vector
  Tensor4D encode(const Tensor4D &input) const;

  /// Save weights to binary file
  bool save_weights(const std::string &path) const;

  /// Load weights from binary file
  bool load_weights(const std::string &path);

private:
  // ===== ENCODER LAYERS =====
  Conv2DLayer conv1_; ///< 3 → 256 channels
  ReLULayer relu1_;
  MaxPool2DLayer pool1_; ///< 32×32 → 16×16

  Conv2DLayer conv2_; ///< 256 → 128 channels
  ReLULayer relu2_;
  MaxPool2DLayer pool2_; ///< 16×16 → 8×8 (latent space)

  // ===== DECODER LAYERS =====
  Conv2DLayer conv3_; ///< 128 → 128 channels
  ReLULayer relu3_;
  UpSample2DLayer up1_; ///< 8×8 → 16×16

  Conv2DLayer conv4_; ///< 128 → 256 channels
  ReLULayer relu4_;
  UpSample2DLayer up2_; ///< 16×16 → 32×32

  Conv2DLayer conv5_; ///< 256 → 3 channels (output)
};

#endif // AUTOENCODER_H
