#include "layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define _USE_MATH_DEFINES
#include <math.h>

#ifndef M_PI
#define M_PI 3.14159265358979323846
#endif

// Create Conv2D layer
Conv2DLayer* conv2d_create(int in_channels, int out_channels, int kernel_size,
                           int stride, int padding, int input_h, int input_w,
                           DeviceType device) {
    Conv2DLayer* layer = (Conv2DLayer*)malloc(sizeof(Conv2DLayer));
    if (!layer) return NULL;

    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->padding = padding;
    layer->input_h = input_h;
    layer->input_w = input_w;
    layer->device = device;

    // Calculate output dimensions
    layer->output_h = (input_h + 2 * padding - kernel_size) / stride + 1;
    layer->output_w = (input_w + 2 * padding - kernel_size) / stride + 1;

    // Allocate weights: (out_channels, in_channels, kernel_size, kernel_size)
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    layer->weights = (float*)malloc(weight_size * sizeof(float));
    layer->bias = (float*)malloc(out_channels * sizeof(float));
    layer->d_weights = (float*)calloc(weight_size, sizeof(float));
    layer->d_bias = (float*)calloc(out_channels, sizeof(float));

    // Allocate input cache for backward pass
    // Will be allocated during forward pass based on batch size
    layer->input_cache = NULL;

    if (!layer->weights || !layer->bias || !layer->d_weights || !layer->d_bias) {
        conv2d_free(layer);
        return NULL;
    }

    return layer;
}

// Free Conv2D layer
void conv2d_free(Conv2DLayer* layer) {
    if (layer) {
        if (layer->weights) free(layer->weights);
        if (layer->bias) free(layer->bias);
        if (layer->d_weights) free(layer->d_weights);
        if (layer->d_bias) free(layer->d_bias);
        if (layer->input_cache) free(layer->input_cache);
        free(layer);
    }
}

// Initialize weights using He initialization
void conv2d_init_weights(Conv2DLayer* layer) {
    int weight_size = layer->out_channels * layer->in_channels *
                      layer->kernel_size * layer->kernel_size;

    // He initialization: std = sqrt(2.0 / (in_channels * kernel_size^2))
    float std = sqrtf(2.0f / (layer->in_channels * layer->kernel_size * layer->kernel_size));

    for (int i = 0; i < weight_size; i++) {
        // Box-Muller transform for normal distribution
        float u1 = (float)rand() / RAND_MAX;
        float u2 = (float)rand() / RAND_MAX;
        layer->weights[i] = std * sqrtf(-2.0f * logf(u1)) * cosf(2.0f * M_PI * u2);
    }

    // Initialize bias to zero
    for (int i = 0; i < layer->out_channels; i++) {
        layer->bias[i] = 0.0f;
    }
}

// Conv2D forward pass (CPU implementation)
void conv2d_forward_cpu(Conv2DLayer* layer, const float* input, float* output, int batch_size) {
    int in_c = layer->in_channels;
    int out_c = layer->out_channels;
    int k = layer->kernel_size;
    int s = layer->stride;
    int p = layer->padding;
    int in_h = layer->input_h;
    int in_w = layer->input_w;
    int out_h = layer->output_h;
    int out_w = layer->output_w;

    // Cache input for backward pass
    int input_size = batch_size * in_c * in_h * in_w;
    if (layer->input_cache) free(layer->input_cache);
    layer->input_cache = (float*)malloc(input_size * sizeof(float));
    memcpy(layer->input_cache, input, input_size * sizeof(float));

    // Zero output
    int output_size = batch_size * out_c * out_h * out_w;
    memset(output, 0, output_size * sizeof(float));

    // Perform convolution
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float sum = layer->bias[oc];

                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = oh * s + kh - p;
                                int iw = ow * s + kw - p;

                                // Check bounds (padding)
                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                                    int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                                    sum += input[input_idx] * layer->weights[weight_idx];
                                }
                            }
                        }
                    }

                    int output_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

// Conv2D backward pass (CPU implementation)
void conv2d_backward_cpu(Conv2DLayer* layer, const float* d_output, float* d_input, int batch_size) {
    int in_c = layer->in_channels;
    int out_c = layer->out_channels;
    int k = layer->kernel_size;
    int s = layer->stride;
    int p = layer->padding;
    int in_h = layer->input_h;
    int in_w = layer->input_w;
    int out_h = layer->output_h;
    int out_w = layer->output_w;

    // Zero gradients
    memset(d_input, 0, batch_size * in_c * in_h * in_w * sizeof(float));

    // Compute weight gradients and input gradients
    for (int b = 0; b < batch_size; b++) {
        for (int oc = 0; oc < out_c; oc++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int output_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                    float d_out_val = d_output[output_idx];

                    // Bias gradient
                    layer->d_bias[oc] += d_out_val;

                    for (int ic = 0; ic < in_c; ic++) {
                        for (int kh = 0; kh < k; kh++) {
                            for (int kw = 0; kw < k; kw++) {
                                int ih = oh * s + kh - p;
                                int iw = ow * s + kw - p;

                                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                                    int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                                    int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;

                                    // Weight gradient
                                    layer->d_weights[weight_idx] += layer->input_cache[input_idx] * d_out_val;

                                    // Input gradient
                                    d_input[input_idx] += layer->weights[weight_idx] * d_out_val;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
