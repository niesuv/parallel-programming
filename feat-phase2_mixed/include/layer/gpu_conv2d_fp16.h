// gpu_conv2d_fp16.h 

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include "gpu_tensor.h"


extern "C" {

// Forward kernels
void launch_conv2d_fp16_wmma_v6(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    int N, int H, int W, int C, int K,
    cudaStream_t stream);

void launch_conv2d_relu_fp16_wmma_v6(
    const half* input,
    const half* weight,
    const half* bias,
    half* output,
    half* conv_output,
    int N, int H, int W, int C, int K,
    cudaStream_t stream);

// Backward kernels
void launch_relu_backward_fp16(
    const half* grad_output,
    const half* conv_output,
    half* grad_input,
    size_t size,
    cudaStream_t stream);

void launch_conv2d_backward_input(
    const half* grad_output,
    const half* weight,
    half* grad_input,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream);

void launch_conv2d_backward_weight(
    const half* grad_output,
    const half* input,
    float* grad_weight,
    float* grad_bias,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream);

void launch_fused_relu_conv2d_backward_input(
    const half* upstream_grad,
    const half* conv_output,
    const half* weight,
    half* grad_input,
    int N, int H, int W, int C,
    int H_out, int W_out, int K,
    cudaStream_t stream);

}

class GPUConv2DLayerFP16
{
public:
    GPUConv2DLayerFP16(
        int in_channels, 
        int out_channels, 
        int kernel_size,
        int stride = 1, 
        int padding = 1);
    
    virtual ~GPUConv2DLayerFP16() = default;
    
    void load_weights_from_fp32(const float* h_weight, const float* h_bias);
    void sync_weights_to_fp16(cudaStream_t stream = 0);
    
    virtual void forward(
        const GPUTensor4D<half>& input,
        GPUTensor4D<half>& output,
        cudaStream_t stream = 0);
    
    virtual void backward(
        const GPUTensor4D<half>& grad_output,
        GPUTensor4D<half>& grad_input,
        cudaStream_t stream = 0);
    
    void zero_grad(cudaStream_t stream = 0);
    
    // Accessors
    int in_channels() const { return in_c_; }
    int out_channels() const { return out_c_; }
    int kernel_size() const { return k_; }
    int stride() const { return stride_; }
    int padding() const { return padding_; }
    
    const GPUTensor4D<half>& weights_fp16() const { return weights_fp16_; }
    const GPUTensor4D<half>& bias_fp16() const { return bias_fp16_; }
    
    GPUTensor4D<float>& weights_fp32() { return weights_fp32_; }
    GPUTensor4D<float>& bias_fp32() { return bias_fp32_; }
    const GPUTensor4D<float>& weights_fp32() const { return weights_fp32_; }
    const GPUTensor4D<float>& bias_fp32() const { return bias_fp32_; }
    
    GPUTensor4D<float>& grad_weights() { return grad_weights_; }
    GPUTensor4D<float>& grad_bias() { return grad_bias_; }
    const GPUTensor4D<float>& grad_weights() const { return grad_weights_; }
    const GPUTensor4D<float>& grad_bias() const { return grad_bias_; }

protected:
    int in_c_;
    int out_c_;
    int k_;
    int stride_;
    int padding_;
    
    GPUTensor4D<float> weights_fp32_;
    GPUTensor4D<float> bias_fp32_;
    GPUTensor4D<half> weights_fp16_;
    GPUTensor4D<half> bias_fp16_;
    GPUTensor4D<float> grad_weights_;
    GPUTensor4D<float> grad_bias_;
    GPUTensor4D<half> cached_input_;
    
    void initialize_weights();
};

// ============================================================================
// GPU CONVOLUTION + RELU LAYER - FP16
// ============================================================================

class GPUConvReLULayerFP16 : public GPUConv2DLayerFP16
{
public:
    GPUConvReLULayerFP16(
        int in_channels, 
        int out_channels, 
        int kernel_size,
        int stride = 1, 
        int padding = 1);
    
    ~GPUConvReLULayerFP16() override = default;
    
    void forward(
        const GPUTensor4D<half>& input,
        GPUTensor4D<half>& output,
        cudaStream_t stream = 0) override;
    
    void backward(
        const GPUTensor4D<half>& grad_output,
        GPUTensor4D<half>& grad_input,
        cudaStream_t stream = 0) override;
    
    const GPUTensor4D<half>& conv_output() const { return conv_output_; }

protected:
    GPUTensor4D<half> conv_output_;
    GPUTensor4D<half> grad_conv_output_;
};