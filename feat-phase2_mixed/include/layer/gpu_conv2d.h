#pragma once

#include "gpu_tensor.h"
#include <cuda_runtime.h>
#include <vector>
#include <string>
#include <iosfwd>

class GPUConv2DLayer {
public:
    GPUConv2DLayer(int in_channels, int out_channels, int kernel_size,
                    int stride = 1, int padding = 1);
    virtual ~GPUConv2DLayer() = default;
    
    GPUConv2DLayer(const GPUConv2DLayer&) = delete;
    GPUConv2DLayer& operator=(const GPUConv2DLayer&) = delete;
    GPUConv2DLayer(GPUConv2DLayer&&) = default;
    GPUConv2DLayer& operator=(GPUConv2DLayer&&) = default;
    
    virtual void forward(const GPUTensor4D<float>& input, 
                        GPUTensor4D<float>& output,
                        cudaStream_t stream = 0);
    
    virtual void backward(const GPUTensor4D<float>& input,
                        const GPUTensor4D<float>& grad_output,
                        GPUTensor4D<float>& grad_input,
                        float learning_rate,
                        cudaStream_t stream = 0);
    
    void save_weights(std::ostream& os) const;
    void load_weights(std::istream& is);
    
    int in_channels() const { return in_c_; }
    int out_channels() const { return out_c_; }
    int kernel_size() const { return k_; }
    int stride() const { return stride_; }
    int padding() const { return padding_; }
    size_t num_parameters() const;

    GPUTensor4D<float>& weights() { return weights_; }
    const GPUTensor4D<float>& weights() const { return weights_; }
    GPUTensor4D<float>& bias() { return bias_; }
    const GPUTensor4D<float>& bias() const { return bias_; }
    GPUTensor4D<float>& grad_weights() { return grad_weights_; }
    GPUTensor4D<float>& grad_bias() { return grad_bias_; }

    
protected:
    int in_c_, out_c_, k_, stride_, padding_;
    
    GPUTensor4D<float> weights_;
    GPUTensor4D<float> bias_;
    GPUTensor4D<float> grad_weights_;
    GPUTensor4D<float> grad_bias_;
    
    void initialize_weights();
    void update_weights(float learning_rate, cudaStream_t stream);
};

class GPUConvReLULayer : public GPUConv2DLayer {
public:
    GPUConvReLULayer(int in_channels, int out_channels, int kernel_size,
                    int stride = 1, int padding = 1);
    ~GPUConvReLULayer() override = default;
    
    void forward(const GPUTensor4D<float>& input, 
                GPUTensor4D<float>& output,
                cudaStream_t stream = 0) override;
    
    void backward(const GPUTensor4D<float>& input,
                const GPUTensor4D<float>& grad_output,
                GPUTensor4D<float>& grad_input,
                float learning_rate,
                cudaStream_t stream = 0) override;
    GPUTensor4D<float>& conv_output() { return conv_output_; }
    const GPUTensor4D<float>& conv_output() const { return conv_output_; }
    
private:
    GPUTensor4D<float> conv_output_;
};

// Forward kernels
void launch_conv2d_forward(
    const float* input, const float* weight, const float* bias,
    float* output,
    int batch, int in_c, int out_c,
    int in_h, int in_w, int out_h, int out_w,
    int k, int stride, int padding, cudaStream_t stream);

void launch_conv2d_relu_forward_fused(
    const float* input, const float* weight, const float* bias,
    float* output, float* conv_output,
    int batch, int in_c, int out_c,
    int in_h, int in_w, int out_h, int out_w,
    int k, int stride, int padding, cudaStream_t stream);

// Backward data kernels
void launch_conv2d_backward_data(
    const float* grad_output, const float* weight,
    float* grad_input,
    int batch, int in_c, int out_c,
    int in_h, int in_w, int out_h, int out_w,
    int k, int stride, int padding, cudaStream_t stream);

void launch_conv2d_relu_backward_data(
    const float* conv_output, const float* grad_output, const float* weight,
    float* grad_input,
    int batch, int in_c, int out_c,
    int in_h, int in_w, int out_h, int out_w,
    int k, int stride, int padding, cudaStream_t stream);

// Backward filter kernels
void launch_conv2d_backward_filter(
    const float* input, const float* grad_output,
    float* grad_weight, float* grad_bias,
    int batch, int in_c, int out_c,
    int in_h, int in_w, int out_h, int out_w,
    int k, int stride, int padding, cudaStream_t stream);

void launch_conv2d_relu_backward_filter(
    const float* input, const float* conv_output, const float* grad_output,
    float* grad_weight, float* grad_bias,
    int batch, int in_c, int out_c,
    int in_h, int in_w, int out_h, int out_w,
    int k, int stride, int padding, cudaStream_t stream);

// Utility kernels
void launch_sgd_update(
    float* param, const float* grad, float lr, int size, cudaStream_t stream);


