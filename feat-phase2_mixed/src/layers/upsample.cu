#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>

GPUUpSample2DLayer::GPUUpSample2DLayer(int scale)
    : scale_(scale) {}

void GPUUpSample2DLayer::forward_fp16(
    const GPUTensor4D &input_fp16,
    GPUTensor4D &output_fp16) const {

#ifdef USE_OPTIMIZED_KERNELS
    gpu_upsample2d_forward_fp16_opt(
        input_fp16,
        output_fp16,
        scale_);
#else
    throw std::runtime_error("Upsample FP16 kernel not available");
#endif
}

void GPUUpSample2DLayer::backward_fp16(
    const GPUTensor4D &grad_output_fp16,
    GPUTensor4D &grad_input_fp16) const {

#ifdef USE_OPTIMIZED_KERNELS
    gpu_upsample2d_backward_fp16_opt(
        grad_output_fp16,
        grad_input_fp16,
        scale_);
#else
    throw std::runtime_error("Upsample backward kernel not available");
#endif
}