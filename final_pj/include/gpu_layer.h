#ifndef GPU_LAYER_H
#define GPU_LAYER_H

#include <cuda_runtime.h>

// Forward declarations
typedef struct GPULayer GPULayer;
typedef struct GPULayerVTable GPULayerVTable;

// Layer types for identification
typedef enum {
    GPU_LAYER_CONV2D,
    GPU_LAYER_RELU,
    GPU_LAYER_MAXPOOL2D,
    GPU_LAYER_UPSAMPLE2D,
    GPU_LAYER_MSE_LOSS
} GPULayerType;

// Virtual table for polymorphism (OOP in C)
struct GPULayerVTable {
    // Forward pass
    void (*forward)(GPULayer* layer, const float* input, float* output, int batch_size);

    // Backward pass (compute gradients)
    void (*backward)(GPULayer* layer, const float* d_output, float* d_input, int batch_size);

    // Update weights (for layers with parameters)
    void (*update_weights)(GPULayer* layer, float learning_rate);

    // Copy weights host <-> device
    void (*copy_weights_to_device)(GPULayer* layer, const void* h_weights, const void* h_bias);
    void (*copy_weights_to_host)(GPULayer* layer, void* h_weights, void* h_bias);

    // Get layer info
    void (*print_info)(const GPULayer* layer);

    // Cleanup
    void (*free)(GPULayer* layer);
};

// Base GPU Layer structure
struct GPULayer {
    GPULayerType type;
    const GPULayerVTable* vtable;

    // Common properties
    int batch_size;
    int in_channels;
    int out_channels;
    int in_h, in_w;
    int out_h, out_w;

    // Device memory pointers (layer-specific)
    void* d_weights;
    void* d_bias;
    void* d_weights_grad;
    void* d_bias_grad;
    void* d_cache;  // For storing intermediate values (e.g., max indices, input cache)

    // Layer-specific data
    void* layer_data;  // Points to specific layer implementation
};

// Layer creation functions
GPULayer* gpu_layer_conv2d_create(int in_channels, int out_channels, int kernel_size,
                                  int stride, int padding, int in_h, int in_w, int batch_size);

GPULayer* gpu_layer_relu_create(int channels, int h, int w, int batch_size);

GPULayer* gpu_layer_maxpool2d_create(int channels, int in_h, int in_w,
                                     int pool_size, int stride, int batch_size);

GPULayer* gpu_layer_upsample2d_create(int channels, int in_h, int in_w,
                                      int scale_factor, int batch_size);

GPULayer* gpu_layer_mse_loss_create(int size, int batch_size);

// Generic layer interface (calls vtable functions)
static inline void gpu_layer_forward(GPULayer* layer, const float* input,
                                     float* output, int batch_size) {
    if (layer && layer->vtable && layer->vtable->forward) {
        layer->vtable->forward(layer, input, output, batch_size);
    }
}

static inline void gpu_layer_backward(GPULayer* layer, const float* d_output,
                                     float* d_input, int batch_size) {
    if (layer && layer->vtable && layer->vtable->backward) {
        layer->vtable->backward(layer, d_output, d_input, batch_size);
    }
}

static inline void gpu_layer_update_weights(GPULayer* layer, float learning_rate) {
    if (layer && layer->vtable && layer->vtable->update_weights) {
        layer->vtable->update_weights(layer, learning_rate);
    }
}

static inline void gpu_layer_copy_weights_to_device(GPULayer* layer,
                                                    const void* h_weights, const void* h_bias) {
    if (layer && layer->vtable && layer->vtable->copy_weights_to_device) {
        layer->vtable->copy_weights_to_device(layer, h_weights, h_bias);
    }
}

static inline void gpu_layer_copy_weights_to_host(GPULayer* layer,
                                                  void* h_weights, void* h_bias) {
    if (layer && layer->vtable && layer->vtable->copy_weights_to_host) {
        layer->vtable->copy_weights_to_host(layer, h_weights, h_bias);
    }
}

static inline void gpu_layer_print_info(const GPULayer* layer) {
    if (layer && layer->vtable && layer->vtable->print_info) {
        layer->vtable->print_info(layer);
    }
}

static inline void gpu_layer_free(GPULayer* layer) {
    if (layer && layer->vtable && layer->vtable->free) {
        layer->vtable->free(layer);
    }
}

#endif // GPU_LAYER_H
