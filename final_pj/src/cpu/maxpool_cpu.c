#include "layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <float.h>

// Create MaxPool2D layer
MaxPool2DLayer* maxpool2d_create(int kernel_size, int stride, int channels,
                                  int input_h, int input_w, DeviceType device) {
    MaxPool2DLayer* layer = (MaxPool2DLayer*)malloc(sizeof(MaxPool2DLayer));
    if (!layer) return NULL;

    layer->kernel_size = kernel_size;
    layer->stride = stride;
    layer->channels = channels;
    layer->input_h = input_h;
    layer->input_w = input_w;
    layer->device = device;

    // Calculate output dimensions
    layer->output_h = (input_h - kernel_size) / stride + 1;
    layer->output_w = (input_w - kernel_size) / stride + 1;

    // Allocate cache for max indices (needed for backward pass)
    // Will be allocated during forward pass based on batch size
    layer->max_indices = NULL;

    return layer;
}

// Free MaxPool2D layer
void maxpool2d_free(MaxPool2DLayer* layer) {
    if (layer) {
        if (layer->max_indices) free(layer->max_indices);
        free(layer);
    }
}

// MaxPool2D forward pass (CPU implementation)
void maxpool2d_forward_cpu(MaxPool2DLayer* layer, const float* input, float* output, int batch_size) {
    int c = layer->channels;
    int in_h = layer->input_h;
    int in_w = layer->input_w;
    int out_h = layer->output_h;
    int out_w = layer->output_w;
    int k = layer->kernel_size;
    int s = layer->stride;

    // Allocate/reallocate indices cache
    int cache_size = batch_size * c * out_h * out_w;
    if (layer->max_indices) free(layer->max_indices);
    layer->max_indices = (int*)malloc(cache_size * sizeof(int));

    // Perform max pooling
    for (int b = 0; b < batch_size; b++) {
        for (int ch = 0; ch < c; ch++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    float max_val = -FLT_MAX;
                    int max_idx = 0;

                    // Find max in kernel window
                    for (int kh = 0; kh < k; kh++) {
                        for (int kw = 0; kw < k; kw++) {
                            int ih = oh * s + kh;
                            int iw = ow * s + kw;

                            if (ih < in_h && iw < in_w) {
                                int input_idx = ((b * c + ch) * in_h + ih) * in_w + iw;
                                if (input[input_idx] > max_val) {
                                    max_val = input[input_idx];
                                    max_idx = input_idx;
                                }
                            }
                        }
                    }

                    int output_idx = ((b * c + ch) * out_h + oh) * out_w + ow;
                    output[output_idx] = max_val;
                    layer->max_indices[output_idx] = max_idx;
                }
            }
        }
    }
}

// MaxPool2D backward pass (CPU implementation)
void maxpool2d_backward_cpu(MaxPool2DLayer* layer, const float* d_output, float* d_input, int batch_size) {
    int c = layer->channels;
    int in_h = layer->input_h;
    int in_w = layer->input_w;
    int out_h = layer->output_h;
    int out_w = layer->output_w;

    // Zero input gradients
    memset(d_input, 0, batch_size * c * in_h * in_w * sizeof(float));

    // Route gradients to max positions
    int output_size = batch_size * c * out_h * out_w;
    for (int i = 0; i < output_size; i++) {
        int max_idx = layer->max_indices[i];
        d_input[max_idx] += d_output[i];
    }
}
