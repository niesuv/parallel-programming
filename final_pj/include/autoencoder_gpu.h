#ifndef AUTOENCODER_GPU_H
#define AUTOENCODER_GPU_H

#include "layers.h"
#include "config.h"

// GPU Autoencoder structure with device memory pointers
typedef struct {
    // Device pointers for layer weights
    float* d_enc1_weights;
    float* d_enc1_bias;
    float* d_enc2_weights;
    float* d_enc2_bias;
    float* d_dec1_weights;
    float* d_dec1_bias;
    float* d_dec2_weights;
    float* d_dec2_bias;
    float* d_dec3_weights;
    float* d_dec3_bias;

    // Device pointers for weight gradients
    float* d_enc1_d_weights;
    float* d_enc1_d_bias;
    float* d_enc2_d_weights;
    float* d_enc2_d_bias;
    float* d_dec1_d_weights;
    float* d_dec1_d_bias;
    float* d_dec2_d_weights;
    float* d_dec2_d_bias;
    float* d_dec3_d_weights;
    float* d_dec3_d_bias;

    // Device pointers for activations (forward pass)
    float* d_input;           // (batch, 3, 32, 32)
    float* d_enc1_out;        // (batch, 256, 32, 32)
    float* d_pool1_out;       // (batch, 256, 16, 16)
    float* d_enc2_out;        // (batch, 128, 16, 16)
    float* d_latent;          // (batch, 128, 8, 8) - encoded representation
    float* d_dec1_out;        // (batch, 128, 8, 8)
    float* d_up1_out;         // (batch, 128, 16, 16)
    float* d_dec2_out;        // (batch, 256, 16, 16)
    float* d_up2_out;         // (batch, 256, 32, 32)
    float* d_output;          // (batch, 3, 32, 32)

    // Device pointers for gradients (backward pass)
    float* d_grad_output;     // Gradient w.r.t output
    float* d_grad_up2_out;
    float* d_grad_dec2_out;
    float* d_grad_up1_out;
    float* d_grad_dec1_out;
    float* d_grad_latent;
    float* d_grad_enc2_out;
    float* d_grad_pool1_out;
    float* d_grad_enc1_out;

    // Device pointer for target (for loss computation)
    float* d_target;

    // MaxPool index buffers (for backward pass)
    int* d_pool1_indices;
    int* d_pool2_indices;

    // Training parameters
    float learning_rate;
    int batch_size;
    int num_epochs;

    // Layer dimensions (stored for convenience)
    int enc1_out_size;
    int pool1_out_size;
    int enc2_out_size;
    int latent_size;
    int dec1_out_size;
    int up1_out_size;
    int dec2_out_size;
    int up2_out_size;
    int output_size;
} GPUAutoencoder;

// 2.1 GPU Memory Management Functions

// Create GPU autoencoder and allocate all device memory
GPUAutoencoder* gpu_autoencoder_create(float learning_rate, int batch_size, int num_epochs);

// Free all device memory
void gpu_autoencoder_free(GPUAutoencoder* gpu_ae);

// Copy weights from CPU autoencoder to GPU
void gpu_autoencoder_copy_weights_to_device(GPUAutoencoder* gpu_ae,
                                            Conv2DLayer* enc1, Conv2DLayer* enc2,
                                            Conv2DLayer* dec1, Conv2DLayer* dec2, Conv2DLayer* dec3);

// Copy weights from GPU back to CPU
void gpu_autoencoder_copy_weights_to_host(GPUAutoencoder* gpu_ae,
                                          Conv2DLayer* enc1, Conv2DLayer* enc2,
                                          Conv2DLayer* dec1, Conv2DLayer* dec2, Conv2DLayer* dec3);

// 2.2 Naive GPU Kernel Declarations (implemented in .cu file)

// Forward pass on GPU
void gpu_autoencoder_forward(GPUAutoencoder* gpu_ae, const float* h_input, int batch_size);

// Backward pass on GPU
void gpu_autoencoder_backward(GPUAutoencoder* gpu_ae, const float* h_target, int batch_size);

// Update weights using SGD
void gpu_autoencoder_update_weights(GPUAutoencoder* gpu_ae);

// Compute MSE loss
float gpu_autoencoder_compute_loss(GPUAutoencoder* gpu_ae, const float* h_target, int batch_size);

// 2.3 Training function
float gpu_autoencoder_train_epoch(GPUAutoencoder* gpu_ae, float* train_data, int num_samples, int verbose);

#endif // AUTOENCODER_GPU_H
