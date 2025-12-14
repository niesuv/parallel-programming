/**
 * @file autoencoder.cpp
 * @brief CPU Autoencoder Implementation
 *
 * Implements forward pass, backward pass (training step), and feature
 * extraction. Uses 5 convolutional layers with ReLU activations, max pooling,
 * and upsampling.
 */

#include "autoencoder.h"

#include <fstream>
#include <stdexcept>

// =============================================================================
// Constructor
// =============================================================================

Autoencoder::Autoencoder()
    // Encoder layers
    : conv1_(3, 256, 3, 1, 1),   // Input → 256 channels
      pool1_(2, 2),              // 32×32 → 16×16
      conv2_(256, 128, 3, 1, 1), // 256 → 128 channels
      pool2_(2, 2),              // 16×16 → 8×8 (latent)
                                 // Decoder layers
      conv3_(128, 128, 3, 1, 1), // 128 → 128 channels
      up1_(2),                   // 8×8 → 16×16
      conv4_(128, 256, 3, 1, 1), // 128 → 256 channels
      up2_(2),                   // 16×16 → 32×32
      conv5_(256, 3, 3, 1, 1)    // 256 → 3 channels (output)
{}

// =============================================================================
// Forward Pass
// =============================================================================

/**
 * @brief Full forward pass (encoder + decoder)
 * @param input [N, 3, 32, 32] batch of images
 * @return [N, 3, 32, 32] reconstructed images
 */
Tensor4D Autoencoder::forward(const Tensor4D &input) const {
  // Encoder
  Tensor4D x = conv1_.forward(input);
  x = relu1_.forward(x);
  x = pool1_.forward(x); // → [N, 256, 16, 16]

  x = conv2_.forward(x);
  x = relu2_.forward(x);
  x = pool2_.forward(x); // → [N, 128, 8, 8] (latent)

  // Decoder
  x = conv3_.forward(x);
  x = relu3_.forward(x);
  x = up1_.forward(x); // → [N, 128, 16, 16]

  x = conv4_.forward(x);
  x = relu4_.forward(x);
  x = up2_.forward(x); // → [N, 256, 32, 32]

  x = conv5_.forward(x); // → [N, 3, 32, 32]
  return x;
}

// =============================================================================
// Feature Extraction (Encoder Only)
// =============================================================================

/**
 * @brief Extract latent features (encoder only)
 * @param input [N, 3, 32, 32] batch of images
 * @return [N, 128, 8, 8] latent representation (8192 features per image)
 */
Tensor4D Autoencoder::encode(const Tensor4D &input) const {
  Tensor4D x = conv1_.forward(input);
  x = relu1_.forward(x);
  x = pool1_.forward(x);

  x = conv2_.forward(x);
  x = relu2_.forward(x);
  x = pool2_.forward(x);

  return x; // Latent: [N, 128, 8, 8]
}

// =============================================================================
// Training Step
// =============================================================================

/**
 * @brief Single training step: forward + MSE loss + backward + weight update
 * @param input Input images
 * @param target Target images (same as input for autoencoder)
 * @param learning_rate Learning rate for gradient descent
 * @return MSE loss value
 */
float Autoencoder::train_step(const Tensor4D &input, const Tensor4D &target,
                              float learning_rate) {
  // === Forward Pass (save all intermediate activations) ===
  Tensor4D x0 = input;
  Tensor4D x1 = conv1_.forward(x0);
  Tensor4D x2 = relu1_.forward(x1);
  Tensor4D x3 = pool1_.forward(x2);

  Tensor4D x4 = conv2_.forward(x3);
  Tensor4D x5 = relu2_.forward(x4);
  Tensor4D x6 = pool2_.forward(x5);

  Tensor4D x7 = conv3_.forward(x6);
  Tensor4D x8 = relu3_.forward(x7);
  Tensor4D x9 = up1_.forward(x8);

  Tensor4D x10 = conv4_.forward(x9);
  Tensor4D x11 = relu4_.forward(x10);
  Tensor4D x12 = up2_.forward(x11);

  Tensor4D x13 = conv5_.forward(x12);

  // === Compute Loss ===
  Tensor4D grad_x13;
  float loss = mse_loss_with_grad(x13, target, grad_x13);

  // === Backward Pass (gradient descent) ===
  Tensor4D g12 = conv5_.backward(x12, grad_x13, learning_rate);
  Tensor4D g11 = up2_.backward(x11, g12);
  Tensor4D g10 = relu4_.backward(x10, g11);
  Tensor4D g9 = conv4_.backward(x9, g10, learning_rate);
  Tensor4D g8 = up1_.backward(x8, g9);
  Tensor4D g7 = relu3_.backward(x7, g8);
  Tensor4D g6 = conv3_.backward(x6, g7, learning_rate);
  Tensor4D g5 = pool2_.backward(x5, g6);
  Tensor4D g4 = relu2_.backward(x4, g5);
  Tensor4D g3 = conv2_.backward(x3, g4, learning_rate);
  Tensor4D g2 = pool1_.backward(x2, g3);
  Tensor4D g1 = relu1_.backward(x1, g2);
  (void)conv1_.backward(x0, g1, learning_rate);

  return loss;
}

// =============================================================================
// Weight Save/Load
// =============================================================================

bool Autoencoder::save_weights(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;

  conv1_.save(out);
  conv2_.save(out);
  conv3_.save(out);
  conv4_.save(out);
  conv5_.save(out);

  return static_cast<bool>(out);
}

bool Autoencoder::load_weights(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;

  try {
    conv1_.load(in);
    conv2_.load(in);
    conv3_.load(in);
    conv4_.load(in);
    conv5_.load(in);
  } catch (const std::exception &) {
    return false;
  }

  return static_cast<bool>(in);
}
