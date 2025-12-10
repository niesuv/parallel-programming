#include "gpu_layer.h"
#include <stdio.h>
#include <stdlib.h>

#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// Conv2D specific data
typedef struct {
    int kernel_size;
    int stride;
    int padding;
    int weight_size;
    int bias_size;

    // Cache input for backward pass
    float* d_input_cache;
    int input_cache_size;
} GPUConv2DData;

// Forward declarations of kernels (from autoencoder_gpu.cu)
extern "C" {
__global__ void naive_conv2d_forward_kernel(
    const float* input, const float* weights, const float* bias, float* output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w, int k, int stride, int padding);

__global__ void naive_conv2d_backward_input_kernel(
    const float* d_output, const float* weights, float* d_input,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w, int k, int stride, int padding);

__global__ void naive_conv2d_backward_weights_kernel(
    const float* input, const float* d_output, float* d_weights, float* d_bias,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w, int k, int stride, int padding);

__global__ void naive_conv2d_backward_bias_kernel(
    const float* d_output, float* d_bias,
    int batch_size, int out_c, int out_h, int out_w);
}

// Conv2D Forward
static void conv2d_forward(GPULayer* layer, const float* input, float* output, int batch_size) {
    GPUConv2DData* data = (GPUConv2DData*)layer->layer_data;

    // Cache input for backward pass
    int input_size = batch_size * layer->in_channels * layer->in_h * layer->in_w;
    CUDA_CHECK(cudaMemcpy(data->d_input_cache, input, input_size * sizeof(float),
                          cudaMemcpyDeviceToDevice));

    // Launch kernel
    const int threads = 256;
    int total_outputs = batch_size * layer->out_channels * layer->out_h * layer->out_w;
    int blocks = (total_outputs + threads - 1) / threads;

    naive_conv2d_forward_kernel<<<blocks, threads>>>(
        input,
        (float*)layer->d_weights,
        (float*)layer->d_bias,
        output,
        batch_size,
        layer->in_channels, layer->in_h, layer->in_w,
        layer->out_channels, layer->out_h, layer->out_w,
        data->kernel_size, data->stride, data->padding
    );
    CUDA_CHECK(cudaGetLastError());
}

// Conv2D Backward
static void conv2d_backward(GPULayer* layer, const float* d_output, float* d_input, int batch_size) {
    GPUConv2DData* data = (GPUConv2DData*)layer->layer_data;

    const int threads = 256;

    // Compute gradient w.r.t input
    int input_size = batch_size * layer->in_channels * layer->in_h * layer->in_w;
    int blocks = (input_size + threads - 1) / threads;
    naive_conv2d_backward_input_kernel<<<blocks, threads>>>(
        d_output, (float*)layer->d_weights, d_input,
        batch_size,
        layer->in_channels, layer->in_h, layer->in_w,
        layer->out_channels, layer->out_h, layer->out_w,
        data->kernel_size, data->stride, data->padding
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute gradient w.r.t weights
    blocks = (data->weight_size + threads - 1) / threads;
    naive_conv2d_backward_weights_kernel<<<blocks, threads>>>(
        data->d_input_cache, d_output,
        (float*)layer->d_weights_grad,
        (float*)layer->d_bias_grad,
        batch_size,
        layer->in_channels, layer->in_h, layer->in_w,
        layer->out_channels, layer->out_h, layer->out_w,
        data->kernel_size, data->stride, data->padding
    );
    CUDA_CHECK(cudaGetLastError());

    // Compute gradient w.r.t bias
    blocks = (layer->out_channels + threads - 1) / threads;
    naive_conv2d_backward_bias_kernel<<<blocks, threads>>>(
        d_output, (float*)layer->d_bias_grad,
        batch_size, layer->out_channels, layer->out_h, layer->out_w
    );
    CUDA_CHECK(cudaGetLastError());
}

// Conv2D Update Weights (SGD)
__global__ void sgd_update_kernel(float* weights, const float* gradients, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * gradients[idx];
    }
}

static void conv2d_update_weights(GPULayer* layer, float learning_rate) {
    GPUConv2DData* data = (GPUConv2DData*)layer->layer_data;
    const int threads = 256;

    // Update weights
    int blocks = (data->weight_size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        (float*)layer->d_weights,
        (float*)layer->d_weights_grad,
        learning_rate,
        data->weight_size
    );

    // Update bias
    blocks = (data->bias_size + threads - 1) / threads;
    sgd_update_kernel<<<blocks, threads>>>(
        (float*)layer->d_bias,
        (float*)layer->d_bias_grad,
        learning_rate,
        data->bias_size
    );
    CUDA_CHECK(cudaGetLastError());
}

// Copy weights to device
static void conv2d_copy_weights_to_device(GPULayer* layer, const void* h_weights, const void* h_bias) {
    GPUConv2DData* data = (GPUConv2DData*)layer->layer_data;

    CUDA_CHECK(cudaMemcpy(layer->d_weights, h_weights,
                          data->weight_size * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(layer->d_bias, h_bias,
                          data->bias_size * sizeof(float), cudaMemcpyHostToDevice));
}

// Copy weights to host
static void conv2d_copy_weights_to_host(GPULayer* layer, void* h_weights, void* h_bias) {
    GPUConv2DData* data = (GPUConv2DData*)layer->layer_data;

    CUDA_CHECK(cudaMemcpy(h_weights, layer->d_weights,
                          data->weight_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bias, layer->d_bias,
                          data->bias_size * sizeof(float), cudaMemcpyDeviceToHost));
}

// Print layer info
static void conv2d_print_info(const GPULayer* layer) {
    GPUConv2DData* data = (GPUConv2DData*)layer->layer_data;
    printf("  [Conv2D] %d->%d, kernel=%dx%d, stride=%d, padding=%d\n",
           layer->in_channels, layer->out_channels,
           data->kernel_size, data->kernel_size,
           data->stride, data->padding);
    printf("           Input: (%d,%d,%d) -> Output: (%d,%d,%d)\n",
           layer->in_channels, layer->in_h, layer->in_w,
           layer->out_channels, layer->out_h, layer->out_w);
}

// Free layer
static void conv2d_free(GPULayer* layer) {
    if (!layer) return;

    GPUConv2DData* data = (GPUConv2DData*)layer->layer_data;

    if (layer->d_weights) CUDA_CHECK(cudaFree(layer->d_weights));
    if (layer->d_bias) CUDA_CHECK(cudaFree(layer->d_bias));
    if (layer->d_weights_grad) CUDA_CHECK(cudaFree(layer->d_weights_grad));
    if (layer->d_bias_grad) CUDA_CHECK(cudaFree(layer->d_bias_grad));
    if (data && data->d_input_cache) CUDA_CHECK(cudaFree(data->d_input_cache));

    free(data);
    free(layer);
}

// VTable for Conv2D
static const GPULayerVTable conv2d_vtable = {
    .forward = conv2d_forward,
    .backward = conv2d_backward,
    .update_weights = conv2d_update_weights,
    .copy_weights_to_device = conv2d_copy_weights_to_device,
    .copy_weights_to_host = conv2d_copy_weights_to_host,
    .print_info = conv2d_print_info,
    .free = conv2d_free
};

// Create Conv2D layer
GPULayer* gpu_layer_conv2d_create(int in_channels, int out_channels, int kernel_size,
                                  int stride, int padding, int in_h, int in_w, int batch_size) {
    GPULayer* layer = (GPULayer*)malloc(sizeof(GPULayer));
    if (!layer) return NULL;

    GPUConv2DData* data = (GPUConv2DData*)malloc(sizeof(GPUConv2DData));
    if (!data) {
        free(layer);
        return NULL;
    }

    // Initialize layer properties
    layer->type = GPU_LAYER_CONV2D;
    layer->vtable = &conv2d_vtable;
    layer->batch_size = batch_size;
    layer->in_channels = in_channels;
    layer->out_channels = out_channels;
    layer->in_h = in_h;
    layer->in_w = in_w;

    // Calculate output dimensions
    layer->out_h = (in_h + 2 * padding - kernel_size) / stride + 1;
    layer->out_w = (in_w + 2 * padding - kernel_size) / stride + 1;

    // Initialize Conv2D specific data
    data->kernel_size = kernel_size;
    data->stride = stride;
    data->padding = padding;
    data->weight_size = out_channels * in_channels * kernel_size * kernel_size;
    data->bias_size = out_channels;
    data->input_cache_size = batch_size * in_channels * in_h * in_w;

    layer->layer_data = data;

    // Allocate device memory
    CUDA_CHECK(cudaMalloc(&layer->d_weights, data->weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias, data->bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_weights_grad, data->weight_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&layer->d_bias_grad, data->bias_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&data->d_input_cache, data->input_cache_size * sizeof(float)));

    printf("✅ Created Conv2D layer: %d->%d, %dx%d -> %dx%d\n",
           in_channels, out_channels, in_h, in_w, layer->out_h, layer->out_w);

    return layer;
}
