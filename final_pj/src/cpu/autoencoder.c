#include "autoencoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Create autoencoder
Autoencoder* autoencoder_create(float learning_rate, int batch_size, int num_epochs, DeviceType device) {
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║     Creating Autoencoder Architecture         ║\n");
    printf("╚════════════════════════════════════════════════╝\n\n");

    Autoencoder* ae = (Autoencoder*)malloc(sizeof(Autoencoder));
    if (!ae) return NULL;

    ae->learning_rate = learning_rate;
    ae->batch_size = batch_size;
    ae->num_epochs = num_epochs;
    ae->device = device;

    // Create encoder layers
    printf("Creating Encoder layers...\n");
    ae->enc_conv1 = conv2d_create(3, 256, 3, 1, 1, 32, 32, device);     // (32,32,3) -> (32,32,256)
    ae->enc_pool1 = maxpool2d_create(2, 2, 256, 32, 32, device);        // -> (16,16,256)
    ae->enc_conv2 = conv2d_create(256, 128, 3, 1, 1, 16, 16, device);   // -> (16,16,128)
    ae->enc_pool2 = maxpool2d_create(2, 2, 128, 16, 16, device);        // -> (8,8,128)
    printf("  ✓ Conv2D(3->256) + MaxPool -> (16,16,256)\n");
    printf("  ✓ Conv2D(256->128) + MaxPool -> (8,8,128) [LATENT]\n");

    // Create decoder layers
    printf("\nCreating Decoder layers...\n");
    ae->dec_conv1 = conv2d_create(128, 128, 3, 1, 1, 8, 8, device);     // (8,8,128) -> (8,8,128)
    ae->dec_up1 = upsample2d_create(2, 128, 8, 8, device);              // -> (16,16,128)
    ae->dec_conv2 = conv2d_create(128, 256, 3, 1, 1, 16, 16, device);   // -> (16,16,256)
    ae->dec_up2 = upsample2d_create(2, 256, 16, 16, device);            // -> (32,32,256)
    ae->dec_conv3 = conv2d_create(256, 3, 3, 1, 1, 32, 32, device);     // -> (32,32,3)
    printf("  ✓ Conv2D(128->128) + UpSample -> (16,16,128)\n");
    printf("  ✓ Conv2D(128->256) + UpSample -> (32,32,256)\n");
    printf("  ✓ Conv2D(256->3) -> (32,32,3) [OUTPUT]\n");

    // Initialize weights
    printf("\nInitializing weights...\n");
    srand(time(NULL));
    conv2d_init_weights(ae->enc_conv1);
    conv2d_init_weights(ae->enc_conv2);
    conv2d_init_weights(ae->dec_conv1);
    conv2d_init_weights(ae->dec_conv2);
    conv2d_init_weights(ae->dec_conv3);
    printf("  ✓ Weights initialized using He initialization\n");

    // Allocate activation memory
    printf("\nAllocating activation memory for batch_size=%d...\n", batch_size);
    ae->enc1_out = (float*)malloc(batch_size * 256 * 32 * 32 * sizeof(float));
    ae->pool1_out = (float*)malloc(batch_size * 256 * 16 * 16 * sizeof(float));
    ae->enc2_out = (float*)malloc(batch_size * 128 * 16 * 16 * sizeof(float));
    ae->latent = (float*)malloc(batch_size * 128 * 8 * 8 * sizeof(float));
    ae->dec1_out = (float*)malloc(batch_size * 128 * 8 * 8 * sizeof(float));
    ae->up1_out = (float*)malloc(batch_size * 128 * 16 * 16 * sizeof(float));
    ae->dec2_out = (float*)malloc(batch_size * 256 * 16 * 16 * sizeof(float));
    ae->up2_out = (float*)malloc(batch_size * 256 * 32 * 32 * sizeof(float));
    ae->output = (float*)malloc(batch_size * 3 * 32 * 32 * sizeof(float));

    // Allocate gradient memory
    ae->d_output = (float*)malloc(batch_size * 3 * 32 * 32 * sizeof(float));
    ae->d_up2_out = (float*)malloc(batch_size * 256 * 32 * 32 * sizeof(float));
    ae->d_dec2_out = (float*)malloc(batch_size * 256 * 16 * 16 * sizeof(float));
    ae->d_up1_out = (float*)malloc(batch_size * 128 * 16 * 16 * sizeof(float));
    ae->d_dec1_out = (float*)malloc(batch_size * 128 * 8 * 8 * sizeof(float));
    ae->d_latent = (float*)malloc(batch_size * 128 * 8 * 8 * sizeof(float));
    ae->d_enc2_out = (float*)malloc(batch_size * 128 * 16 * 16 * sizeof(float));
    ae->d_pool1_out = (float*)malloc(batch_size * 256 * 16 * 16 * sizeof(float));
    ae->d_enc1_out = (float*)malloc(batch_size * 256 * 32 * 32 * sizeof(float));

    printf("  ✓ Memory allocated successfully\n");

    return ae;
}

// Free autoencoder
void autoencoder_free(Autoencoder* ae) {
    if (!ae) return;

    // Free layers
    conv2d_free(ae->enc_conv1);
    maxpool2d_free(ae->enc_pool1);
    conv2d_free(ae->enc_conv2);
    maxpool2d_free(ae->enc_pool2);
    conv2d_free(ae->dec_conv1);
    upsample2d_free(ae->dec_up1);
    conv2d_free(ae->dec_conv2);
    upsample2d_free(ae->dec_up2);
    conv2d_free(ae->dec_conv3);

    // Free activations
    free(ae->enc1_out);
    free(ae->pool1_out);
    free(ae->enc2_out);
    free(ae->latent);
    free(ae->dec1_out);
    free(ae->up1_out);
    free(ae->dec2_out);
    free(ae->up2_out);
    free(ae->output);

    // Free gradients
    free(ae->d_output);
    free(ae->d_up2_out);
    free(ae->d_dec2_out);
    free(ae->d_up1_out);
    free(ae->d_dec1_out);
    free(ae->d_latent);
    free(ae->d_enc2_out);
    free(ae->d_pool1_out);
    free(ae->d_enc1_out);

    free(ae);
}

// Forward pass
void autoencoder_forward(Autoencoder* ae, const float* input, int batch_size) {
    // ENCODER
    // Conv1 + ReLU: (batch, 3, 32, 32) -> (batch, 256, 32, 32)
    conv2d_forward_cpu(ae->enc_conv1, input, ae->enc1_out, batch_size);
    relu_forward_cpu(ae->enc1_out, batch_size * 256 * 32 * 32);

    // MaxPool1: -> (batch, 256, 16, 16)
    maxpool2d_forward_cpu(ae->enc_pool1, ae->enc1_out, ae->pool1_out, batch_size);

    // Conv2 + ReLU: -> (batch, 128, 16, 16)
    conv2d_forward_cpu(ae->enc_conv2, ae->pool1_out, ae->enc2_out, batch_size);
    relu_forward_cpu(ae->enc2_out, batch_size * 128 * 16 * 16);

    // MaxPool2: -> (batch, 128, 8, 8) [LATENT]
    maxpool2d_forward_cpu(ae->enc_pool2, ae->enc2_out, ae->latent, batch_size);

    // DECODER
    // Conv1 + ReLU: (batch, 128, 8, 8) -> (batch, 128, 8, 8)
    conv2d_forward_cpu(ae->dec_conv1, ae->latent, ae->dec1_out, batch_size);
    relu_forward_cpu(ae->dec1_out, batch_size * 128 * 8 * 8);

    // UpSample1: -> (batch, 128, 16, 16)
    upsample2d_forward_cpu(ae->dec_up1, ae->dec1_out, ae->up1_out, batch_size);

    // Conv2 + ReLU: -> (batch, 256, 16, 16)
    conv2d_forward_cpu(ae->dec_conv2, ae->up1_out, ae->dec2_out, batch_size);
    relu_forward_cpu(ae->dec2_out, batch_size * 256 * 16 * 16);

    // UpSample2: -> (batch, 256, 32, 32)
    upsample2d_forward_cpu(ae->dec_up2, ae->dec2_out, ae->up2_out, batch_size);

    // Conv3 (no activation): -> (batch, 3, 32, 32)
    conv2d_forward_cpu(ae->dec_conv3, ae->up2_out, ae->output, batch_size);
}

// Backward pass
void autoencoder_backward(Autoencoder* ae, const float* input, const float* target, int batch_size) {
    int output_size = batch_size * 3 * 32 * 32;

    // Compute loss gradient
    mse_backward_cpu(ae->output, target, ae->d_output, output_size);

    // DECODER BACKWARD
    // Conv3 backward
    conv2d_backward_cpu(ae->dec_conv3, ae->d_output, ae->d_up2_out, batch_size);

    // UpSample2 backward
    upsample2d_backward_cpu(ae->dec_up2, ae->d_up2_out, ae->d_dec2_out, batch_size);

    // ReLU + Conv2 backward
    relu_backward_cpu(ae->dec2_out, ae->d_dec2_out, ae->d_dec2_out, batch_size * 256 * 16 * 16);
    conv2d_backward_cpu(ae->dec_conv2, ae->d_dec2_out, ae->d_up1_out, batch_size);

    // UpSample1 backward
    upsample2d_backward_cpu(ae->dec_up1, ae->d_up1_out, ae->d_dec1_out, batch_size);

    // ReLU + Conv1 backward
    relu_backward_cpu(ae->dec1_out, ae->d_dec1_out, ae->d_dec1_out, batch_size * 128 * 8 * 8);
    conv2d_backward_cpu(ae->dec_conv1, ae->d_dec1_out, ae->d_latent, batch_size);

    // ENCODER BACKWARD
    // MaxPool2 backward
    maxpool2d_backward_cpu(ae->enc_pool2, ae->d_latent, ae->d_enc2_out, batch_size);

    // ReLU + Conv2 backward
    relu_backward_cpu(ae->enc2_out, ae->d_enc2_out, ae->d_enc2_out, batch_size * 128 * 16 * 16);
    conv2d_backward_cpu(ae->enc_conv2, ae->d_enc2_out, ae->d_pool1_out, batch_size);

    // MaxPool1 backward
    maxpool2d_backward_cpu(ae->enc_pool1, ae->d_pool1_out, ae->d_enc1_out, batch_size);

    // ReLU + Conv1 backward
    relu_backward_cpu(ae->enc1_out, ae->d_enc1_out, ae->d_enc1_out, batch_size * 256 * 32 * 32);
    float* d_input_dummy = (float*)malloc(batch_size * 3 * 32 * 32 * sizeof(float));
    conv2d_backward_cpu(ae->enc_conv1, ae->d_enc1_out, d_input_dummy, batch_size);
    free(d_input_dummy);
}

// Helper function to update conv layer weights
static void update_conv_weights(Conv2DLayer* layer, float lr) {
    int weight_size = layer->out_channels * layer->in_channels *
                     layer->kernel_size * layer->kernel_size;
    for (int i = 0; i < weight_size; i++) {
        layer->weights[i] -= lr * layer->d_weights[i];
        layer->d_weights[i] = 0.0f;  // Reset gradient
    }
    for (int i = 0; i < layer->out_channels; i++) {
        layer->bias[i] -= lr * layer->d_bias[i];
        layer->d_bias[i] = 0.0f;  // Reset gradient
    }
}

// Update weights using gradients
void autoencoder_update_weights(Autoencoder* ae) {
    float lr = ae->learning_rate;

    update_conv_weights(ae->enc_conv1, lr);
    update_conv_weights(ae->enc_conv2, lr);
    update_conv_weights(ae->dec_conv1, lr);
    update_conv_weights(ae->dec_conv2, lr);
    update_conv_weights(ae->dec_conv3, lr);
}

// Train for one epoch
float autoencoder_train_epoch(Autoencoder* ae, float* train_data, int num_samples, int verbose) {
    int batch_size = ae->batch_size;
    int num_batches = (num_samples + batch_size - 1) / batch_size;
    float total_loss = 0.0f;

    for (int batch_idx = 0; batch_idx < num_batches; batch_idx++) {
        int start_idx = batch_idx * batch_size;
        int end_idx = start_idx + batch_size;
        if (end_idx > num_samples) end_idx = num_samples;
        int actual_batch_size = end_idx - start_idx;

        // Get batch data
        float* batch_data = &train_data[start_idx * 3 * 32 * 32];

        // Forward pass
        autoencoder_forward(ae, batch_data, actual_batch_size);

        // Compute loss
        float batch_loss = mse_loss_cpu(ae->output, batch_data, actual_batch_size * 3 * 32 * 32);
        total_loss += batch_loss;

        // Backward pass
        autoencoder_backward(ae, batch_data, batch_data, actual_batch_size);

        // Update weights
        autoencoder_update_weights(ae);

        if (verbose >= 2 && batch_idx % 100 == 0) {
            printf("    Batch %d/%d, Loss: %.6f\r", batch_idx + 1, num_batches, batch_loss);
            fflush(stdout);
        }
    }

    return total_loss / num_batches;
}

// Extract latent representation
void autoencoder_encode(Autoencoder* ae, const float* input, float* latent_out, int batch_size) {
    // Run encoder only
    conv2d_forward_cpu(ae->enc_conv1, input, ae->enc1_out, batch_size);
    relu_forward_cpu(ae->enc1_out, batch_size * 256 * 32 * 32);
    maxpool2d_forward_cpu(ae->enc_pool1, ae->enc1_out, ae->pool1_out, batch_size);
    conv2d_forward_cpu(ae->enc_conv2, ae->pool1_out, ae->enc2_out, batch_size);
    relu_forward_cpu(ae->enc2_out, batch_size * 128 * 16 * 16);
    maxpool2d_forward_cpu(ae->enc_pool2, ae->enc2_out, latent_out, batch_size);
}

// Print model summary
void autoencoder_print_summary(Autoencoder* ae) {
    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║          Autoencoder Model Summary            ║\n");
    printf("╚════════════════════════════════════════════════╝\n\n");

    int total_params = 0;

    printf("ENCODER:\n");
    printf("  Conv2D(3->256, 3x3):   %d params\n", 3*256*3*3 + 256);
    total_params += 3*256*3*3 + 256;
    printf("  MaxPool2D(2x2):        0 params\n");
    printf("  Conv2D(256->128, 3x3): %d params\n", 256*128*3*3 + 128);
    total_params += 256*128*3*3 + 128;
    printf("  MaxPool2D(2x2):        0 params\n");
    printf("  → Latent: (batch, 128, 8, 8) = 8192 features\n");

    printf("\nDECODER:\n");
    printf("  Conv2D(128->128, 3x3): %d params\n", 128*128*3*3 + 128);
    total_params += 128*128*3*3 + 128;
    printf("  UpSample2D(2x):        0 params\n");
    printf("  Conv2D(128->256, 3x3): %d params\n", 128*256*3*3 + 256);
    total_params += 128*256*3*3 + 256;
    printf("  UpSample2D(2x):        0 params\n");
    printf("  Conv2D(256->3, 3x3):   %d params\n", 256*3*3*3 + 3);
    total_params += 256*3*3*3 + 3;

    printf("\n────────────────────────────────────────────────\n");
    printf("Total parameters: %d\n", total_params);
    printf("Training config:\n");
    printf("  Batch size: %d\n", ae->batch_size);
    printf("  Learning rate: %.6f\n", ae->learning_rate);
    printf("  Epochs: %d\n", ae->num_epochs);
    printf("  Device: %s\n", device_type_to_string(ae->device));
    printf("════════════════════════════════════════════════\n\n");
}

// Helper to save conv layer weights
static void save_conv_layer(FILE* f, Conv2DLayer* layer) {
    int weight_size = layer->out_channels * layer->in_channels *
                     layer->kernel_size * layer->kernel_size;
    fwrite(layer->weights, sizeof(float), weight_size, f);
    fwrite(layer->bias, sizeof(float), layer->out_channels, f);
}

// Helper to load conv layer weights
static void load_conv_layer(FILE* f, Conv2DLayer* layer) {
    int weight_size = layer->out_channels * layer->in_channels *
                     layer->kernel_size * layer->kernel_size;
    fread(layer->weights, sizeof(float), weight_size, f);
    fread(layer->bias, sizeof(float), layer->out_channels, f);
}

// Save model weights
int autoencoder_save_weights(Autoencoder* ae, const char* filename) {
    printf("Saving model weights to %s...\n", filename);
    FILE* f = fopen(filename, "wb");
    if (!f) return -1;

    save_conv_layer(f, ae->enc_conv1);
    save_conv_layer(f, ae->enc_conv2);
    save_conv_layer(f, ae->dec_conv1);
    save_conv_layer(f, ae->dec_conv2);
    save_conv_layer(f, ae->dec_conv3);

    fclose(f);
    printf("✓ Weights saved successfully\n");
    return 0;
}

// Load model weights
int autoencoder_load_weights(Autoencoder* ae, const char* filename) {
    printf("Loading model weights from %s...\n", filename);
    FILE* f = fopen(filename, "rb");
    if (!f) return -1;

    load_conv_layer(f, ae->enc_conv1);
    load_conv_layer(f, ae->enc_conv2);
    load_conv_layer(f, ae->dec_conv1);
    load_conv_layer(f, ae->dec_conv2);
    load_conv_layer(f, ae->dec_conv3);

    fclose(f);
    printf("✓ Weights loaded successfully\n");
    return 0;
}
