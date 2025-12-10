#include "layers.h"
#include <cuda_runtime.h>
#include <stdio.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Upsample2D forward kernel (nearest neighbor)
__global__ void upsample2d_forward_kernel(
    const float* input, float* output,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int scale_factor) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode to (batch, channel, out_h, out_w)
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int b = idx / (out_w * out_h * channels);

    // Nearest neighbor: map output position to input position
    int ih = oh / scale_factor;
    int iw = ow / scale_factor;

    int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
    output[idx] = input[input_idx];
}

// Upsample2D backward kernel
__global__ void upsample2d_backward_kernel(
    const float* d_output, float* d_input,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int scale_factor) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * channels * in_h * in_w;
    if (idx >= total_inputs) return;

    // Decode to (batch, channel, in_h, in_w)
    int iw = idx % in_w;
    int ih = (idx / in_w) % in_h;
    int c = (idx / (in_w * in_h)) % channels;
    int b = idx / (in_w * in_h * channels);

    float grad = 0.0f;

    // Accumulate gradients from all output positions that used this input
    for (int oh = ih * scale_factor; oh < (ih + 1) * scale_factor && oh < out_h; oh++) {
        for (int ow = iw * scale_factor; ow < (iw + 1) * scale_factor && ow < out_w; ow++) {
            int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
            grad += d_output[output_idx];
        }
    }

    d_input[idx] = grad;
}

// Upsample2D forward on GPU
void upsample2d_forward_cuda(UpSample2DLayer* layer, const float* input,
                             float* output, int batch_size) {
    int total_outputs = batch_size * layer->channels * layer->output_h * layer->output_w;

    const int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;

    upsample2d_forward_kernel<<<blocks, threads>>>(
        input, output,
        batch_size, layer->channels, layer->input_h, layer->input_w,
        layer->output_h, layer->output_w, layer->scale_factor
    );

    CUDA_CHECK(cudaGetLastError());
}

// Upsample2D backward on GPU
void upsample2d_backward_cuda(UpSample2DLayer* layer, const float* d_output,
                              float* d_input, int batch_size) {
    int total_inputs = batch_size * layer->channels * layer->input_h * layer->input_w;

    const int threads = 256;
    int blocks = (total_inputs + threads - 1) / threads;

    upsample2d_backward_kernel<<<blocks, threads>>>(
        d_output, d_input,
        batch_size, layer->channels, layer->input_h, layer->input_w,
        layer->output_h, layer->output_w, layer->scale_factor
    );

    CUDA_CHECK(cudaGetLastError());
}
