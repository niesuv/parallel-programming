#include "layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Create UpSample2D layer
UpSample2DLayer* upsample2d_create(int scale_factor, int channels,
                                    int input_h, int input_w, DeviceType device) {
    UpSample2DLayer* layer = (UpSample2DLayer*)malloc(sizeof(UpSample2DLayer));
    if (!layer) return NULL;

    layer->scale_factor = scale_factor;
    layer->channels = channels;
    layer->input_h = input_h;
    layer->input_w = input_w;
    layer->device = device;

    // Calculate output dimensions
    layer->output_h = input_h * scale_factor;
    layer->output_w = input_w * scale_factor;

    return layer;
}

// Free UpSample2D layer
void upsample2d_free(UpSample2DLayer* layer) {
    if (layer) {
        free(layer);
    }
}

// UpSample2D forward pass (CPU implementation) - Nearest neighbor
void upsample2d_forward_cpu(UpSample2DLayer* layer, const float* input, float* output, int batch_size) {
    int c = layer->channels;
    int in_h = layer->input_h;
    int in_w = layer->input_w;
    int out_h = layer->output_h;
    int out_w = layer->output_w;
    int scale = layer->scale_factor;

    // Nearest neighbor upsampling
    for (int b = 0; b < batch_size; b++) {
        for (int ch = 0; ch < c; ch++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    // Map output pixel to input pixel (nearest neighbor)
                    int ih = oh / scale;
                    int iw = ow / scale;

                    int input_idx = ((b * c + ch) * in_h + ih) * in_w + iw;
                    int output_idx = ((b * c + ch) * out_h + oh) * out_w + ow;

                    output[output_idx] = input[input_idx];
                }
            }
        }
    }
}

// UpSample2D backward pass (CPU implementation)
void upsample2d_backward_cpu(UpSample2DLayer* layer, const float* d_output, float* d_input, int batch_size) {
    int c = layer->channels;
    int in_h = layer->input_h;
    int in_w = layer->input_w;
    int out_h = layer->output_h;
    int out_w = layer->output_w;
    int scale = layer->scale_factor;

    // Zero input gradients
    memset(d_input, 0, batch_size * c * in_h * in_w * sizeof(float));

    // Accumulate gradients from all output pixels that map to same input pixel
    for (int b = 0; b < batch_size; b++) {
        for (int ch = 0; ch < c; ch++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int ih = oh / scale;
                    int iw = ow / scale;

                    int input_idx = ((b * c + ch) * in_h + ih) * in_w + iw;
                    int output_idx = ((b * c + ch) * out_h + oh) * out_w + ow;

                    d_input[input_idx] += d_output[output_idx];
                }
            }
        }
    }
}
