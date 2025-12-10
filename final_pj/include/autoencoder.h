#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include "layers.h"
#include "config.h"
#include "device.h"

// Autoencoder architecture
typedef struct {
    // Encoder layers
    Conv2DLayer* enc_conv1;     // 3 -> 256, (32,32,3) -> (32,32,256)
    MaxPool2DLayer* enc_pool1;  // -> (16,16,256)
    Conv2DLayer* enc_conv2;     // 256 -> 128, -> (16,16,128)
    MaxPool2DLayer* enc_pool2;  // -> (8,8,128) LATENT

    // Decoder layers
    Conv2DLayer* dec_conv1;     // 128 -> 128, (8,8,128) -> (8,8,128)
    UpSample2DLayer* dec_up1;   // -> (16,16,128)
    Conv2DLayer* dec_conv2;     // 128 -> 256, -> (16,16,256)
    UpSample2DLayer* dec_up2;   // -> (32,32,256)
    Conv2DLayer* dec_conv3;     // 256 -> 3, -> (32,32,3)

    // Training parameters
    float learning_rate;
    int batch_size;
    int num_epochs;

    // Activations for forward pass
    float* enc1_out;      // (batch, 256, 32, 32)
    float* pool1_out;     // (batch, 256, 16, 16)
    float* enc2_out;      // (batch, 128, 16, 16)
    float* latent;        // (batch, 128, 8, 8) - encoded representation
    float* dec1_out;      // (batch, 128, 8, 8)
    float* up1_out;       // (batch, 128, 16, 16)
    float* dec2_out;      // (batch, 256, 16, 16)
    float* up2_out;       // (batch, 256, 32, 32)
    float* output;        // (batch, 3, 32, 32)

    // Gradients for backward pass
    float* d_output;
    float* d_up2_out;
    float* d_dec2_out;
    float* d_up1_out;
    float* d_dec1_out;
    float* d_latent;
    float* d_enc2_out;
    float* d_pool1_out;
    float* d_enc1_out;

    DeviceType device;
} Autoencoder;

// Create and initialize autoencoder
Autoencoder* autoencoder_create(float learning_rate, int batch_size, int num_epochs, DeviceType device);
void autoencoder_free(Autoencoder* ae);

// Forward pass: input (batch, 3, 32, 32) -> output (batch, 3, 32, 32)
void autoencoder_forward(Autoencoder* ae, const float* input, int batch_size);

// Backward pass: compute gradients
void autoencoder_backward(Autoencoder* ae, const float* input, const float* target, int batch_size);

// Update weights using computed gradients
void autoencoder_update_weights(Autoencoder* ae);

// Extract latent representation (encoder only)
void autoencoder_encode(Autoencoder* ae, const float* input, float* latent_out, int batch_size);

// Training function
float autoencoder_train_epoch(Autoencoder* ae, float* train_data, int num_samples, int verbose);

// Save/Load model
int autoencoder_save_weights(Autoencoder* ae, const char* filename);
int autoencoder_load_weights(Autoencoder* ae, const char* filename);

// Print model summary
void autoencoder_print_summary(Autoencoder* ae);

#endif // AUTOENCODER_H
