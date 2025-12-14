#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>

void GPUReLULayer::forward_fp16(
    const GPUTensor4D &input_fp16,
    GPUTensor4D &output_fp16) const {

#ifdef USE_OPTIMIZED_KERNELS
    gpu_relu_forward_fp16_opt(input_fp16, output_fp16);
#else
    throw std::runtime_error("ReLU FP16 kernel not available");
#endif
}

void GPUReLULayer::backward_fp16(
    const GPUTensor4D &input_fp16,
    const GPUTensor4D &grad_output_fp16,
    GPUTensor4D &grad_input_fp16) const {

#ifdef USE_OPTIMIZED_KERNELS
    gpu_relu_backward_fp16_opt(
        input_fp16,
        grad_output_fp16,
        grad_input_fp16);
#else
    throw std::runtime_error("ReLU backward kernel not available");
#endif
}