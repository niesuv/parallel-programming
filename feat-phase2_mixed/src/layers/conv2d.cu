#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>

GPUConv2DLayer::GPUConv2DLayer(
    int in_channels,
    int out_channels,
    int kernel_size,
    int stride,
    int padding)
    :in_c_(in_channels),
    out_c_(out_channels),
    k_(kernel_size),
    stride_(stride),
    padding_(padding) {
        weights_size_ = static_cast<size_t>(
        out_c_ * in_c_ * k_ * k_);

    // FP16 forward weights
    CUDA_CHECK(cudaMalloc(&d_weights_fp16_,
                            weights_size_ * sizeof(__half)));
    CUDA_CHECK(cudaMalloc(&d_bias_fp16_,
                            out_c_ * sizeof(__half)));

    // FP32 master weights
    CUDA_CHECK(cudaMalloc(&d_weights_fp32_,
                            weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_bias_fp32_,
                            out_c_ * sizeof(float)));

    // FP32 gradients
    CUDA_CHECK(cudaMalloc(&d_grad_weights_fp32_,
                            weights_size_ * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_grad_bias_fp32_,
                            out_c_ * sizeof(float)));
}

GPUConv2DLayer::~GPUConv2DLayer() {
    cudaFree(d_weights_fp16_);
    cudaFree(d_bias_fp16_);
    cudaFree(d_weights_fp32_);
    cudaFree(d_bias_fp32_);
    cudaFree(d_grad_weights_fp32_);
    cudaFree(d_grad_bias_fp32_);
}

/* ---------- Forward (FP16) ---------- */

void GPUConv2DLayer::forward_fp16(
    const GPUTensor4D &input_fp16,
    GPUTensor4D &output_fp16) const {

#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_forward_fp16(
        input_fp16,
        d_weights_fp16_,
        d_bias_fp16_,
        output_fp16,
        in_c_, out_c_,
        k_, stride_, padding_);
#else
    throw std::runtime_error("Conv2D FP16 kernel not available");
#endif
}

/* ---------- Backward (Mixed Precision) ---------- */

void GPUConv2DLayer::backward_fp16(
    const GPUTensor4D &input_fp16,
    const GPUTensor4D &grad_output_fp16,
    GPUTensor4D &grad_input_fp16,
    float learning_rate,
    GPULossScaler &scaler) {

  // dX (FP16)
#ifdef USE_OPTIMIZED_KERNELS
    gpu_conv2d_backward_data_fp16(
        grad_output_fp16,
        d_weights_fp16_,
        grad_input_fp16,
        in_c_, out_c_,
        k_, stride_, padding_);

  // dW, db (FP32, unscaled inside)
    gpu_conv2d_backward_weights_fp16(
        input_fp16,
        grad_output_fp16,
        d_grad_weights_fp32_,
        d_grad_bias_fp32_,
        scaler.scale,
        in_c_, out_c_,
        k_, stride_, padding_);
#else
    throw std::runtime_error("Conv2D backward kernel not available");
#endif

    // SGD update (FP32 master weights)
    size_t threads = 256;
    size_t blocks_w =
        (weights_size_ + threads - 1) / threads;
    size_t blocks_b =
        (out_c_ + threads - 1) / threads;

    sgd_update_fp32<<<blocks_w, threads>>>(
        d_weights_fp32_,
        d_grad_weights_fp32_,
        learning_rate,
        weights_size_);

    sgd_update_fp32<<<blocks_b, threads>>>(
        d_bias_fp32_,
        d_grad_bias_fp32_,
        learning_rate,
        out_c_);

    // FP32 → FP16 copy
    gpu_fp32_to_fp16(
        d_weights_fp32_,
        d_weights_fp16_,
        weights_size_);

    gpu_fp32_to_fp16(
        d_bias_fp32_,
        d_bias_fp16_,
        out_c_);
}

/* ---------- Weight I/O ---------- */

void GPUConv2DLayer::copy_weights_from_host_fp32(
    const float *h_weights,
    const float *h_bias) {

    CUDA_CHECK(cudaMemcpy(
        d_weights_fp32_,
        h_weights,
        weights_size_ * sizeof(float),
        cudaMemcpyHostToDevice));

    CUDA_CHECK(cudaMemcpy(
        d_bias_fp32_,
        h_bias,
        out_c_ * sizeof(float),
        cudaMemcpyHostToDevice));

    gpu_fp32_to_fp16(
        d_weights_fp32_,
        d_weights_fp16_,
        weights_size_);

    gpu_fp32_to_fp16(
        d_bias_fp32_,
        d_bias_fp16_,
        out_c_);
}

void GPUConv2DLayer::copy_weights_to_host_fp32(
    float *h_weights,
    float *h_bias) const {

    CUDA_CHECK(cudaMemcpy(
        h_weights,
        d_weights_fp32_,
        weights_size_ * sizeof(float),
        cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaMemcpy(
        h_bias,
        d_bias_fp32_,
        out_c_ * sizeof(float),
        cudaMemcpyDeviceToHost));
}