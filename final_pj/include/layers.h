#ifndef LAYERS_H
#define LAYERS_H

#include "config.h"
#include "device.h"
#include <stddef.h>

// Conv2D layer structure
typedef struct {
    int in_channels;
    int out_channels;
    int kernel_size;
    int stride;
    int padding;
    int input_h, input_w;
    int output_h, output_w;

    // Weights: (out_channels, in_channels, kernel_size, kernel_size)
    float* weights;
    float* bias;

    // Gradients
    float* d_weights;
    float* d_bias;

    // Cache for backward pass
    float* input_cache;

    DeviceType device;
} Conv2DLayer;

// MaxPool2D layer structure
typedef struct {
    int kernel_size;
    int stride;
    int input_h, input_w;
    int output_h, output_w;
    int channels;

    // Cache for backward pass (indices of max values)
    int* max_indices;

    DeviceType device;
} MaxPool2DLayer;

// UpSample2D layer structure
typedef struct {
    int scale_factor;
    int input_h, input_w;
    int output_h, output_w;
    int channels;

    DeviceType device;
} UpSample2DLayer;

// ===== Conv2D operations =====
Conv2DLayer* conv2d_create(int in_channels, int out_channels, int kernel_size,
                           int stride, int padding, int input_h, int input_w,
                           DeviceType device);
void conv2d_free(Conv2DLayer* layer);
void conv2d_init_weights(Conv2DLayer* layer);

// Forward: input (batch, in_channels, H, W) -> output (batch, out_channels, H', W')
void conv2d_forward_cpu(Conv2DLayer* layer, const float* input, float* output, int batch_size);
void conv2d_forward_cuda(Conv2DLayer* layer, const float* input, float* output, int batch_size);

// Backward: d_output (batch, out_channels, H', W') -> d_input (batch, in_channels, H, W)
void conv2d_backward_cpu(Conv2DLayer* layer, const float* d_output, float* d_input, int batch_size);
void conv2d_backward_cuda(Conv2DLayer* layer, const float* d_output, float* d_input, int batch_size);

// ===== ReLU operations =====
// Forward: inplace operation
void relu_forward_cpu(float* data, int size);
void relu_forward_cuda(float* data, int size);

// Backward: d_output * (input > 0)
void relu_backward_cpu(const float* input, const float* d_output, float* d_input, int size);
void relu_backward_cuda(const float* input, const float* d_output, float* d_input, int size);

// ===== MaxPool2D operations =====
MaxPool2DLayer* maxpool2d_create(int kernel_size, int stride, int channels,
                                  int input_h, int input_w, DeviceType device);
void maxpool2d_free(MaxPool2DLayer* layer);

// Forward: input (batch, channels, H, W) -> output (batch, channels, H/2, W/2)
void maxpool2d_forward_cpu(MaxPool2DLayer* layer, const float* input, float* output, int batch_size);
void maxpool2d_forward_cuda(MaxPool2DLayer* layer, const float* input, float* output, int batch_size);

// Backward
void maxpool2d_backward_cpu(MaxPool2DLayer* layer, const float* d_output, float* d_input, int batch_size);
void maxpool2d_backward_cuda(MaxPool2DLayer* layer, const float* d_output, float* d_input, int batch_size);

// ===== UpSample2D operations =====
UpSample2DLayer* upsample2d_create(int scale_factor, int channels,
                                    int input_h, int input_w, DeviceType device);
void upsample2d_free(UpSample2DLayer* layer);

// Forward: input (batch, channels, H, W) -> output (batch, channels, H*2, W*2)
void upsample2d_forward_cpu(UpSample2DLayer* layer, const float* input, float* output, int batch_size);
void upsample2d_forward_cuda(UpSample2DLayer* layer, const float* input, float* output, int batch_size);

// Backward
void upsample2d_backward_cpu(UpSample2DLayer* layer, const float* d_output, float* d_input, int batch_size);
void upsample2d_backward_cuda(UpSample2DLayer* layer, const float* d_output, float* d_input, int batch_size);

// ===== Loss functions =====
// MSE loss for reconstruction
float mse_loss_cpu(const float* predictions, const float* targets, int size);
float mse_loss_forward_cuda(const float* predictions, const float* targets, int size);

// MSE gradient
void mse_backward_cpu(const float* predictions, const float* targets, float* d_output, int size);
void mse_loss_backward_cuda(const float* predictions, const float* targets, float* d_output, int size);

// ===== Helper functions =====
void print_layer_info(const char* name, int batch, int channels, int height, int width);

#endif // LAYERS_H
