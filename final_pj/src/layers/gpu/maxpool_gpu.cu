#include "layers.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <float.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// MaxPool2D forward kernel
__global__ void maxpool2d_forward_kernel(
    const float* input, float* output, int* indices,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int kernel_size, int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * channels * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode to (batch, channel, out_h, out_w)
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % channels;
    int b = idx / (out_w * out_h * channels);

    float max_val = -FLT_MAX;
    int max_idx = 0;

    // Find max in kernel_size x kernel_size window
    for (int kh = 0; kh < kernel_size; kh++) {
        for (int kw = 0; kw < kernel_size; kw++) {
            int ih = oh * stride + kh;
            int iw = ow * stride + kw;

            if (ih < in_h && iw < in_w) {
                int input_idx = ((b * channels + c) * in_h + ih) * in_w + iw;
                float val = input[input_idx];
                if (val > max_val) {
                    max_val = val;
                    max_idx = kh * kernel_size + kw;
                }
            }
        }
    }

    output[idx] = max_val;
    indices[idx] = max_idx;
}

// MaxPool2D backward kernel
__global__ void maxpool2d_backward_kernel(
    const float* d_output, float* d_input, const int* indices,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int kernel_size, int stride) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * channels * in_h * in_w;
    if (idx >= total_inputs) return;

    // Decode to (batch, channel, in_h, in_w)
    int iw = idx % in_w;
    int ih = (idx / in_w) % in_h;
    int c = (idx / (in_w * in_h)) % channels;
    int b = idx / (in_w * in_h * channels);

    float grad = 0.0f;

    // Check all output positions that could have used this input
    for (int oh = 0; oh < out_h; oh++) {
        for (int ow = 0; ow < out_w; ow++) {
            int start_h = oh * stride;
            int start_w = ow * stride;

            // Check if this input position is in the pooling window
            if (ih >= start_h && ih < start_h + kernel_size &&
                iw >= start_w && iw < start_w + kernel_size) {

                int output_idx = ((b * channels + c) * out_h + oh) * out_w + ow;
                int local_idx = (ih - start_h) * kernel_size + (iw - start_w);

                // If this input was the max, accumulate gradient
                if (indices[output_idx] == local_idx) {
                    grad += d_output[output_idx];
                }
            }
        }
    }

    d_input[idx] = grad;
}

// MaxPool2D forward on GPU
void maxpool2d_forward_cuda(MaxPool2DLayer* layer, const float* input,
                            float* output, int batch_size) {
    int total_outputs = batch_size * layer->channels * layer->output_h * layer->output_w;

    const int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;

    maxpool2d_forward_kernel<<<blocks, threads>>>(
        input, output, layer->max_indices,
        batch_size, layer->channels, layer->input_h, layer->input_w,
        layer->output_h, layer->output_w, layer->kernel_size, layer->stride
    );

    CUDA_CHECK(cudaGetLastError());
}

// MaxPool2D backward on GPU
void maxpool2d_backward_cuda(MaxPool2DLayer* layer, const float* d_output,
                             float* d_input, int batch_size) {
    int total_inputs = batch_size * layer->channels * layer->input_h * layer->input_w;

    const int threads = 256;
    int blocks = (total_inputs + threads - 1) / threads;

    maxpool2d_backward_kernel<<<blocks, threads>>>(
        d_output, d_input, layer->max_indices,
        batch_size, layer->channels, layer->input_h, layer->input_w,
        layer->output_h, layer->output_w, layer->kernel_size, layer->stride
    );

    CUDA_CHECK(cudaGetLastError());
}
