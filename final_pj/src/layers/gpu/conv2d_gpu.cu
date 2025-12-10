#include "layers.h"
#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>
#include <math.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Naive Conv2D forward kernel
__global__ void conv2d_forward_kernel(
    const float* input, const float* weights, const float* bias, float* output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w, int k, int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_outputs = batch_size * out_c * out_h * out_w;
    if (idx >= total_outputs) return;

    // Decode index to (batch, out_channel, out_h, out_w)
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_c;
    int b = idx / (out_w * out_h * out_c);

    float sum = bias[oc];

    // Convolve over input channels and kernel
    for (int ic = 0; ic < in_c; ic++) {
        for (int kh = 0; kh < k; kh++) {
            for (int kw = 0; kw < k; kw++) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;

                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                    int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                    sum += input[input_idx] * weights[weight_idx];
                }
            }
        }
    }

    output[idx] = sum;
}

// Naive Conv2D backward input kernel
__global__ void conv2d_backward_input_kernel(
    const float* d_output, const float* weights, float* d_input,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w, int k, int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_inputs = batch_size * in_c * in_h * in_w;
    if (idx >= total_inputs) return;

    // Decode to (batch, in_channel, in_h, in_w)
    int iw = idx % in_w;
    int ih = (idx / in_w) % in_h;
    int ic = (idx / (in_w * in_h)) % in_c;
    int b = idx / (in_w * in_h * in_c);

    float sum = 0.0f;

    // Accumulate gradients from all output positions this input contributes to
    for (int oc = 0; oc < out_c; oc++) {
        for (int kh = 0; kh < k; kh++) {
            for (int kw = 0; kw < k; kw++) {
                // Find output positions affected by this input
                int oh_start = (ih + padding - kh) / stride;
                int ow_start = (iw + padding - kw) / stride;

                if (oh_start >= 0 && oh_start < out_h && ow_start >= 0 && ow_start < out_w) {
                    // Check if this input position exactly contributes to this output
                    if ((ih + padding - kh) % stride == 0 && (iw + padding - kw) % stride == 0) {
                        int d_output_idx = ((b * out_c + oc) * out_h + oh_start) * out_w + ow_start;
                        int weight_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                        sum += d_output[d_output_idx] * weights[weight_idx];
                    }
                }
            }
        }
    }

    d_input[idx] = sum;
}

// Naive Conv2D backward weights kernel
__global__ void conv2d_backward_weights_kernel(
    const float* input, const float* d_output, float* d_weights, float* d_bias,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w, int k, int stride, int padding) {

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_weights = out_c * in_c * k * k;

    if (idx < total_weights) {
        // Decode to (out_channel, in_channel, kh, kw)
        int kw = idx % k;
        int kh = (idx / k) % k;
        int ic = (idx / (k * k)) % in_c;
        int oc = idx / (k * k * in_c);

        float sum = 0.0f;

        for (int b = 0; b < batch_size; b++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;

                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        int input_idx = ((b * in_c + ic) * in_h + ih) * in_w + iw;
                        int d_output_idx = ((b * out_c + oc) * out_h + oh) * out_w + ow;
                        sum += input[input_idx] * d_output[d_output_idx];
                    }
                }
            }
        }

        d_weights[idx] = sum;
    }

    // Compute bias gradient
    if (idx < out_c) {
        float bias_sum = 0.0f;
        for (int b = 0; b < batch_size; b++) {
            for (int oh = 0; oh < out_h; oh++) {
                for (int ow = 0; ow < out_w; ow++) {
                    int d_output_idx = ((b * out_c + idx) * out_h + oh) * out_w + ow;
                    bias_sum += d_output[d_output_idx];
                }
            }
        }
        d_bias[idx] = bias_sum;
    }
}

// Conv2D forward on GPU
void conv2d_forward_cuda(Conv2DLayer* layer, const float* input, float* output, int batch_size) {
    int total_outputs = batch_size * layer->out_channels * layer->output_h * layer->output_w;

    const int threads = 256;
    int blocks = (total_outputs + threads - 1) / threads;

    conv2d_forward_kernel<<<blocks, threads>>>(
        input, layer->weights, layer->bias, output,
        batch_size, layer->in_channels, layer->input_h, layer->input_w,
        layer->out_channels, layer->output_h, layer->output_w,
        layer->kernel_size, layer->stride, layer->padding
    );

    CUDA_CHECK(cudaGetLastError());
}

// Conv2D backward on GPU
void conv2d_backward_cuda(Conv2DLayer* layer, const float* d_output, float* d_input, int batch_size) {
    const int threads = 256;

    // Backward through input
    int total_inputs = batch_size * layer->in_channels * layer->input_h * layer->input_w;
    int blocks_input = (total_inputs + threads - 1) / threads;

    conv2d_backward_input_kernel<<<blocks_input, threads>>>(
        d_output, layer->weights, d_input,
        batch_size, layer->in_channels, layer->input_h, layer->input_w,
        layer->out_channels, layer->output_h, layer->output_w,
        layer->kernel_size, layer->stride, layer->padding
    );

    // Backward through weights and bias
    int total_weights = layer->out_channels * layer->in_channels *
                        layer->kernel_size * layer->kernel_size;
    int blocks_weights = (total_weights + threads - 1) / threads;

    conv2d_backward_weights_kernel<<<blocks_weights, threads>>>(
        layer->input_cache, d_output, layer->d_weights, layer->d_bias,
        batch_size, layer->in_channels, layer->input_h, layer->input_w,
        layer->out_channels, layer->output_h, layer->output_w,
        layer->kernel_size, layer->stride, layer->padding
    );

    CUDA_CHECK(cudaGetLastError());
}
