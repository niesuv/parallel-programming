#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>

GPUMaxPool2DLayer::GPUMaxPool2DLayer(int kernel_size, int stride)
    : k_(kernel_size), stride_(stride) {}

void GPUMaxPool2DLayer::forward_fp16(
    const GPUTensor4D &input_fp16,
    GPUTensor4D &output_fp16) const {

#ifdef USE_OPTIMIZED_KERNELS
    gpu_maxpool2d_forward_fp16_opt(
        input_fp16,
        output_fp16,
        k_, stride_);
#else
    throw std::runtime_error("MaxPool FP16 kernel not available");
#endif
}

void GPUMaxPool2DLayer::backward_fp16(
    const GPUTensor4D &input_fp16,
    const GPUTensor4D &grad_output_fp16,
    GPUTensor4D &grad_input_fp16) const {

#ifdef USE_OPTIMIZED_KERNELS
    gpu_maxpool2d_backward_fp16_opt(
        input_fp16,
        grad_output_fp16,
        grad_input_fp16,
        k_, stride_);
#else
    throw std::runtime_error("MaxPool backward kernel not available");
#endif
}