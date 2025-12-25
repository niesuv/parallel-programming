// gpu_conv2d_fp16.cu
// Implementation of GPUConv2DLayerFP16 and GPUConvReLULayerFP16 classes

#include "gpu_conv2d_fp16.h"
#include "cuda_utils.h"
#include <cmath>
#include <random>

// ============================================================================
// GPUConv2DLayerFP16 Implementation
// ============================================================================

GPUConv2DLayerFP16::GPUConv2DLayerFP16(
    int in_channels, 
    int out_channels, 
    int kernel_size,
    int stride, 
    int padding)
    : in_c_(in_channels)
    , out_c_(out_channels)
    , k_(kernel_size)
    , stride_(stride)
    , padding_(padding)
{
    // Allocate weight tensors [K, 3, 3, C] for NHWC layout
    weights_fp32_.allocate_nchw(out_channels, kernel_size * kernel_size, in_channels, 1);
    weights_fp16_.allocate_nchw(out_channels, kernel_size * kernel_size, in_channels, 1);
    
    // Allocate bias tensors [K]
    bias_fp32_.allocate_nchw(1, out_channels, 1, 1);
    bias_fp16_.allocate_nchw(1, out_channels, 1, 1);
    
    // Allocate gradient tensors
    grad_weights_.allocate_nchw(out_channels, kernel_size * kernel_size, in_channels, 1);
    grad_bias_.allocate_nchw(1, out_channels, 1, 1);
    
    // Initialize weights
    initialize_weights();
}

void GPUConv2DLayerFP16::initialize_weights()
{
    // He initialization
    float std_dev = std::sqrt(2.0f / (in_c_ * k_ * k_));
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<float> dist(0.0f, std_dev);
    
    size_t weight_size = weights_fp32_.size();
    std::vector<float> h_weights(weight_size);
    for (size_t i = 0; i < weight_size; i++) {
        h_weights[i] = dist(gen);
    }
    weights_fp32_.copy_from_host(h_weights.data());
    
    // Zero bias
    std::vector<float> h_bias(out_c_, 0.0f);
    bias_fp32_.copy_from_host(h_bias.data());
    
    // Sync to FP16
    sync_weights_to_fp16();
}

void GPUConv2DLayerFP16::load_weights_from_fp32(const float* h_weight, const float* h_bias)
{
    weights_fp32_.copy_from_host(h_weight);
    bias_fp32_.copy_from_host(h_bias);
    sync_weights_to_fp16();
}

void GPUConv2DLayerFP16::sync_weights_to_fp16(cudaStream_t stream)
{
    weights_fp16_.convert_from(weights_fp32_, stream);
    bias_fp16_.convert_from(bias_fp32_, stream);
}

void GPUConv2DLayerFP16::forward(
    const GPUTensor4D<half>& input,
    GPUTensor4D<half>& output,
    cudaStream_t stream)
{
    // Cache input for backward pass
    cached_input_.allocate_nhwc(input.batch(), input.height(), input.width(), input.channels());
    cudaMemcpyAsync(cached_input_.data(), input.data(), 
                    input.bytes(), cudaMemcpyDeviceToDevice, stream);
    
    // Compute output dimensions
    int H_out = (input.height() + 2 * padding_ - k_) / stride_ + 1;
    int W_out = (input.width() + 2 * padding_ - k_) / stride_ + 1;
    
    // Allocate output if needed
    if (output.batch() != input.batch() || 
        output.height() != static_cast<size_t>(H_out) ||
        output.width() != static_cast<size_t>(W_out) ||
        output.channels() != static_cast<size_t>(out_c_)) {
        output.allocate_nhwc(input.batch(), H_out, W_out, out_c_);
    }
    
    // Launch forward kernel
    launch_conv2d_fp16_wmma_v6(
        input.data(),
        weights_fp16_.data(),
        bias_fp16_.data(),
        output.data(),
        input.batch(), input.height(), input.width(), input.channels(), out_c_,
        stream);
}

void GPUConv2DLayerFP16::backward(
    const GPUTensor4D<half>& grad_output,
    GPUTensor4D<half>& grad_input,
    cudaStream_t stream)
{
    int N = grad_output.batch();
    int H_out = grad_output.height();
    int W_out = grad_output.width();
    int K = out_c_;
    int H = cached_input_.height();
    int W = cached_input_.width();
    int C = in_c_;
    
    // Allocate grad_input if needed
    if (grad_input.batch() != static_cast<size_t>(N) ||
        grad_input.height() != static_cast<size_t>(H) ||
        grad_input.width() != static_cast<size_t>(W) ||
        grad_input.channels() != static_cast<size_t>(C)) {
        grad_input.allocate_nhwc(N, H, W, C);
    }
    
    // Compute gradient w.r.t. input
    launch_conv2d_backward_input(
        grad_output.data(),
        weights_fp16_.data(),
        grad_input.data(),
        N, H, W, C,
        H_out, W_out, K,
        stream);
    
    // Compute gradient w.r.t. weights and bias
    launch_conv2d_backward_weight(
        grad_output.data(),
        cached_input_.data(),
        grad_weights_.data(),
        grad_bias_.data(),
        N, H, W, C,
        H_out, W_out, K,
        stream);
}

void GPUConv2DLayerFP16::zero_grad(cudaStream_t stream)
{
    grad_weights_.zero(stream);
    grad_bias_.zero(stream);
}

// ============================================================================
// GPUConvReLULayerFP16 Implementation
// ============================================================================

GPUConvReLULayerFP16::GPUConvReLULayerFP16(
    int in_channels, 
    int out_channels, 
    int kernel_size,
    int stride, 
    int padding)
    : GPUConv2DLayerFP16(in_channels, out_channels, kernel_size, stride, padding)
{
}

void GPUConvReLULayerFP16::forward(
    const GPUTensor4D<half>& input,
    GPUTensor4D<half>& output,
    cudaStream_t stream)
{
    // Cache input for backward pass
    cached_input_.allocate_nhwc(input.batch(), input.height(), input.width(), input.channels());
    cudaMemcpyAsync(cached_input_.data(), input.data(), 
                    input.bytes(), cudaMemcpyDeviceToDevice, stream);
    
    // Compute output dimensions
    int H_out = (input.height() + 2 * padding_ - k_) / stride_ + 1;
    int W_out = (input.width() + 2 * padding_ - k_) / stride_ + 1;
    
    // Allocate outputs if needed
    if (output.batch() != input.batch() || 
        output.height() != static_cast<size_t>(H_out) ||
        output.width() != static_cast<size_t>(W_out) ||
        output.channels() != static_cast<size_t>(out_c_)) {
        output.allocate_nhwc(input.batch(), H_out, W_out, out_c_);
        conv_output_.allocate_nhwc(input.batch(), H_out, W_out, out_c_);
    }
    
    // Launch fused conv+relu forward kernel
    launch_conv2d_relu_fp16_wmma_v6(
        input.data(),
        weights_fp16_.data(),
        bias_fp16_.data(),
        output.data(),
        conv_output_.data(),  // Store pre-ReLU output for backward
        input.batch(), input.height(), input.width(), input.channels(), out_c_,
        stream);
}

void GPUConvReLULayerFP16::backward(
    const GPUTensor4D<half>& grad_output,
    GPUTensor4D<half>& grad_input,
    cudaStream_t stream)
{
    int N = grad_output.batch();
    int H_out = grad_output.height();
    int W_out = grad_output.width();
    int K = out_c_;
    int H = cached_input_.height();
    int W = cached_input_.width();
    int C = in_c_;
    
    // Allocate grad_input if needed
    if (grad_input.batch() != static_cast<size_t>(N) ||
        grad_input.height() != static_cast<size_t>(H) ||
        grad_input.width() != static_cast<size_t>(W) ||
        grad_input.channels() != static_cast<size_t>(C)) {
        grad_input.allocate_nhwc(N, H, W, C);
    }
    
    // Allocate temporary for grad through ReLU
    if (grad_conv_output_.batch() != static_cast<size_t>(N)) {
        grad_conv_output_.allocate_nhwc(N, H_out, W_out, K);
    }
    
    // Option 1: Separate ReLU backward + conv backward
    // ReLU backward
    // launch_relu_backward_fp16(
    //     grad_output.data(),
    //     conv_output_.data(),
    //     grad_conv_output_.data(),
    //     N * H_out * W_out * K,
    //     stream);
    
    // // Conv backward input
    // launch_conv2d_backward_input(
    //     grad_conv_output_.data(),
    //     weights_fp16_.data(),
    //     grad_input.data(),
    //     N, H, W, C,
    //     H_out, W_out, K,
    //     stream);
    
    // // Conv backward weight
    // launch_conv2d_backward_weight(
    //     grad_conv_output_.data(),
    //     cached_input_.data(),
    //     grad_weights_.data(),
    //     grad_bias_.data(),
    //     N, H, W, C,
    //     H_out, W_out, K,
    //     stream);
    
    // Alternative Option 2: Fused kernel (can uncomment to use)
    
    launch_fused_relu_conv2d_backward_input(
        grad_output.data(),
        conv_output_.data(),
        weights_fp16_.data(),
        grad_input.data(),
        N, H, W, C,
        H_out, W_out, K,
        stream);
    
    // Still need separate weight backward with ReLU gradient
    launch_relu_backward_fp16(
        grad_output.data(),
        conv_output_.data(),
        grad_conv_output_.data(),
        N * H_out * W_out * K,
        stream);
    
    launch_conv2d_backward_weight(
        grad_conv_output_.data(),
        cached_input_.data(),
        grad_weights_.data(),
        grad_bias_.data(),
        N, H, W, C,
        H_out, W_out, K,
        stream);
    
}
