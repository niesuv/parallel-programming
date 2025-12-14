/**
 * @file gpu_autoencoder.cu
 * @brief GPU Autoencoder Implementation
 *
 * CUDA implementation of the convolutional autoencoder.
 * Uses fused Conv+ReLU kernels for optimal GPU performance.
 *
 * Features:
 * - Async CUDA stream operations
 * - Fused forward pass kernels
 * - Binary weight file I/O with magic number validation
 */

#include "autoencoder.h"
#include "cuda_utils.h"
#include "gpu_autoencoder.h"

#include <cuda_runtime.h>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <vector>

// =============================================================================
// Constructor / Destructor
// =============================================================================

GPUAutoencoder::GPUAutoencoder()
    // Encoder layers
    : conv1_(3, 256, 3, 1, 1),   // 3→256 channels
      pool1_(2, 2),              // 32×32 → 16×16
      conv2_(256, 128, 3, 1, 1), // 256→128 channels
      pool2_(2, 2),              // 16×16 → 8×8 (latent)
                                 // Decoder layers
      conv3_(128, 128, 3, 1, 1), // 128→128 channels
      up1_(2),                   // 8×8 → 16×16
      conv4_(128, 256, 3, 1, 1), // 128→256 channels
      up2_(2),                   // 16×16 → 32×32
      conv5_(256, 3, 3, 1, 1)    // 256→3 channels (output)
{}

GPUAutoencoder::~GPUAutoencoder() {}

void GPUAutoencoder::synchronize() { CUDA_CHECK(cudaDeviceSynchronize()); }

// =============================================================================
// Forward Pass
// =============================================================================

/**
 * @brief Full forward pass (encoder + decoder)
 * Uses fused Conv+ReLU kernels for performance
 */
void GPUAutoencoder::forward(const GPUTensor4D &input, GPUTensor4D &output,
                             cudaStream_t stream) {
  // Encoder
  conv1_.forward_fused_relu(input, x2_, stream);
  pool1_.forward(x2_, x3_, stream); // → [N, 256, 16, 16]

  conv2_.forward_fused_relu(x3_, x5_, stream);
  pool2_.forward(x5_, x6_, stream); // → [N, 128, 8, 8]

  // Decoder
  conv3_.forward_fused_relu(x6_, x8_, stream);
  up1_.forward(x8_, x9_, stream); // → [N, 128, 16, 16]

  conv4_.forward_fused_relu(x9_, x11_, stream);
  up2_.forward(x11_, x12_, stream); // → [N, 256, 32, 32]

  conv5_.forward(x12_, output, stream); // → [N, 3, 32, 32]
}

// =============================================================================
// Feature Extraction (Encoder Only)
// =============================================================================

/**
 * @brief Extract latent features for SVM classification
 * @param latent Output tensor [N, 128, 8, 8] (8192 features per image)
 */
void GPUAutoencoder::encode(const GPUTensor4D &input, GPUTensor4D &latent,
                            cudaStream_t stream) {
  conv1_.forward_fused_relu(input, x2_, stream);
  pool1_.forward(x2_, x3_, stream);

  conv2_.forward_fused_relu(x3_, x5_, stream);
  pool2_.forward(x5_, latent, stream); // Latent: [N, 128, 8, 8]
}

// =============================================================================
// Training Step
// =============================================================================

/**
 * @brief Single training step with gradient descent
 * @param h_partial_sums Pre-allocated host buffer for loss reduction
 * @return MSE loss value
 */
float GPUAutoencoder::train_step(const GPUTensor4D &input,
                                 const GPUTensor4D &target, float learning_rate,
                                 cudaStream_t stream, float *h_partial_sums) {
  // === Forward Pass ===
  conv1_.forward_fused_relu(input, x2_, stream);
  pool1_.forward(x2_, x3_, stream);

  conv2_.forward_fused_relu(x3_, x5_, stream);
  pool2_.forward(x5_, x6_, stream);

  conv3_.forward_fused_relu(x6_, x8_, stream);
  up1_.forward(x8_, x9_, stream);

  conv4_.forward_fused_relu(x9_, x11_, stream);
  up2_.forward(x11_, x12_, stream);

  conv5_.forward(x12_, x13_, stream);

  // === Compute Loss ===
  float loss =
      gpu_mse_loss_with_grad(x13_, target, g13_, h_partial_sums, stream);

  // === Backward Pass ===
  conv5_.backward(x12_, g13_, g12_, learning_rate, stream);
  up2_.backward(x11_, g12_, g11_, stream);
  conv4_.backward_fused_relu(x9_, g10_, g9_, learning_rate, stream);
  up1_.backward(x8_, g9_, g8_, stream);
  conv3_.backward_fused_relu(x6_, g7_, g6_, learning_rate, stream);
  pool2_.backward(x5_, g6_, g5_, stream);
  conv2_.backward_fused_relu(x3_, g4_, g3_, learning_rate, stream);
  pool1_.backward(x2_, g3_, g2_, stream);
  conv1_.backward_fused_relu(input, g1_, g0_, learning_rate, stream);

  return loss;
}

// =============================================================================
// Weight Save/Load
// =============================================================================

// Weight file format:
// [4 bytes] Magic: 0x48414557 ("WEAH")
// [4 bytes] Version: 1
// [4 bytes] Num layers: 5
// For each layer: [in_c][out_c][k][w_size][weights...][b_size][biases...]

bool GPUAutoencoder::save_weights(const std::string &path) const {
  std::ofstream out(path, std::ios::binary);
  if (!out)
    return false;

  // Write header
  const uint32_t MAGIC = 0x48414557;
  const uint32_t VERSION = 1;
  const uint32_t NUM_LAYERS = 5;
  out.write(reinterpret_cast<const char *>(&MAGIC), sizeof(uint32_t));
  out.write(reinterpret_cast<const char *>(&VERSION), sizeof(uint32_t));
  out.write(reinterpret_cast<const char *>(&NUM_LAYERS), sizeof(uint32_t));

  // Lambda to save a conv layer
  auto save_conv = [&out](const GPUConv2DLayer &layer) {
    int in_c = layer.get_in_channels();
    int out_c = layer.get_out_channels();
    int k = layer.get_kernel_size();
    out.write(reinterpret_cast<const char *>(&in_c), sizeof(int));
    out.write(reinterpret_cast<const char *>(&out_c), sizeof(int));
    out.write(reinterpret_cast<const char *>(&k), sizeof(int));

    size_t w_size = static_cast<size_t>(out_c) * in_c * k * k;
    int b_size = out_c;

    // Copy weights from GPU to host
    std::vector<float> h_weights(w_size);
    std::vector<float> h_bias(b_size);
    layer.copy_weights_to_host(h_weights.data(), h_bias.data());

    // Write weights and biases
    int w_size_int = static_cast<int>(w_size);
    out.write(reinterpret_cast<const char *>(&w_size_int), sizeof(int));
    out.write(reinterpret_cast<const char *>(h_weights.data()),
              static_cast<std::streamsize>(w_size) * sizeof(float));
    out.write(reinterpret_cast<const char *>(&b_size), sizeof(int));
    out.write(reinterpret_cast<const char *>(h_bias.data()),
              static_cast<std::streamsize>(b_size) * sizeof(float));
  };

  save_conv(conv1_);
  save_conv(conv2_);
  save_conv(conv3_);
  save_conv(conv4_);
  save_conv(conv5_);

  return static_cast<bool>(out);
}

bool GPUAutoencoder::load_weights(const std::string &path) {
  std::ifstream in(path, std::ios::binary);
  if (!in)
    return false;

  // Read and validate header
  const uint32_t EXPECTED_MAGIC = 0x48414557;
  uint32_t magic = 0, version = 0, num_layers = 0;
  in.read(reinterpret_cast<char *>(&magic), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&version), sizeof(uint32_t));
  in.read(reinterpret_cast<char *>(&num_layers), sizeof(uint32_t));

  if (magic != EXPECTED_MAGIC) {
    std::cerr << "Invalid weight file format (bad magic number)" << std::endl;
    return false;
  }
  if (version != 1) {
    std::cerr << "Unsupported weight file version: " << version << std::endl;
    return false;
  }
  if (num_layers != 5) {
    std::cerr << "Layer count mismatch: expected 5, got " << num_layers
              << std::endl;
    return false;
  }

  // Lambda to load a conv layer
  auto load_conv = [&in](GPUConv2DLayer &layer) -> bool {
    int in_c = 0, out_c = 0, k = 0;
    in.read(reinterpret_cast<char *>(&in_c), sizeof(int));
    in.read(reinterpret_cast<char *>(&out_c), sizeof(int));
    in.read(reinterpret_cast<char *>(&k), sizeof(int));

    // Validate layer shape
    if (in_c != layer.get_in_channels() || out_c != layer.get_out_channels() ||
        k != layer.get_kernel_size()) {
      std::cerr << "Layer shape mismatch" << std::endl;
      return false;
    }

    size_t expected_w = static_cast<size_t>(out_c) * in_c * k * k;
    int expected_b = out_c;

    // Read weights
    int w_size = 0;
    in.read(reinterpret_cast<char *>(&w_size), sizeof(int));
    if (static_cast<size_t>(w_size) != expected_w)
      return false;

    std::vector<float> h_weights(w_size);
    in.read(reinterpret_cast<char *>(h_weights.data()),
            static_cast<std::streamsize>(w_size) * sizeof(float));

    // Read biases
    int b_size = 0;
    in.read(reinterpret_cast<char *>(&b_size), sizeof(int));
    if (b_size != expected_b)
      return false;

    std::vector<float> h_bias(b_size);
    in.read(reinterpret_cast<char *>(h_bias.data()),
            static_cast<std::streamsize>(b_size) * sizeof(float));

    // Copy to GPU
    layer.copy_weights_from_host(h_weights.data(), h_bias.data());
    return true;
  };

  try {
    if (!load_conv(conv1_))
      return false;
    if (!load_conv(conv2_))
      return false;
    if (!load_conv(conv3_))
      return false;
    if (!load_conv(conv4_))
      return false;
    if (!load_conv(conv5_))
      return false;
  } catch (...) {
    return false;
  }

  return static_cast<bool>(in);
}

void GPUAutoencoder::load_weights_from_cpu(const Autoencoder & /* cpu_ae */) {
  std::cerr << "Warning: load_weights_from_cpu not implemented. "
            << "Use save/load_weights instead." << std::endl;
}

// =============================================================================
// Utility Functions
// =============================================================================

void tensor_cpu_to_gpu(const Tensor4D &cpu_tensor, GPUTensor4D &gpu_tensor) {
  if (gpu_tensor.n != cpu_tensor.n || gpu_tensor.c != cpu_tensor.c ||
      gpu_tensor.h != cpu_tensor.h || gpu_tensor.w != cpu_tensor.w) {
    gpu_tensor.allocate(cpu_tensor.n, cpu_tensor.c, cpu_tensor.h, cpu_tensor.w);
  }
  gpu_tensor.copy_from_host(cpu_tensor.data.data());
}

void tensor_gpu_to_cpu(const GPUTensor4D &gpu_tensor, Tensor4D &cpu_tensor) {
  cpu_tensor = Tensor4D(gpu_tensor.n, gpu_tensor.c, gpu_tensor.h, gpu_tensor.w);
  gpu_tensor.copy_to_host(cpu_tensor.data.data());
}

void batch_cpu_to_gpu(const float *cpu_data, int n, int c, int h, int w,
                      GPUTensor4D &gpu_tensor) {
  if (gpu_tensor.n != n || gpu_tensor.c != c || gpu_tensor.h != h ||
      gpu_tensor.w != w) {
    gpu_tensor.allocate(n, c, h, w);
  }
  gpu_tensor.copy_from_host(cpu_data);
}
