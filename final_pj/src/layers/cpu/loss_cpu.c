#include "layers.h"
#include <stdio.h>

// MSE loss (Mean Squared Error)
float mse_loss_cpu(const float* predictions, const float* targets, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = predictions[i] - targets[i];
        sum += diff * diff;
    }
    return sum / size;
}

// MSE backward (gradient)
void mse_backward_cpu(const float* predictions, const float* targets, float* d_output, int size) {
    for (int i = 0; i < size; i++) {
        d_output[i] = 2.0f * (predictions[i] - targets[i]) / size;
    }
}

// Helper function to print layer info
void print_layer_info(const char* name, int batch, int channels, int height, int width) {
    printf("  %s: (%d, %d, %d, %d)\n", name, batch, channels, height, width);
}
