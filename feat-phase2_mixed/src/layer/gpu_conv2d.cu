#pragma once
#include "gpu_tensor.h"
#include <cuda_runtime.h>
#include <cuda_fp16.h>  // ← ADD THIS
#include <iostream>

// ============================================================================
// BASE LAYER (Stays FP32 for backwards compatibility)
// ============================================================================

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

// ============================================================================
// FP32 CONV+RELU (Your existing class)
// ============================================================================

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

// ============================================================================
// NEW: FP16 LAYERS WITH TENSOR CORES
// ============================================================================

enum class DataLayout {
    NCHW,  // Channels first (your current format)
    NHWC   // Channels last (required for Tensor Cores)
};

class GPUConv2DLayerFP16 {
public:
    GPUConv2DLayerFP16(int in_channels, int out_channels, int kernel_size,
                       int stride = 1, int padding = 1,
                       DataLayout layout = DataLayout::NHWC);  // Default NHWC for Tensor Cores
    virtual ~GPUConv2DLayerFP16() = default;

    GPUConv2DLayerFP16(const GPUConv2DLayerFP16&) = delete;
    GPUConv2DLayerFP16& operator=(const GPUConv2DLayerFP16&) = delete;

    virtual void forward(const GPUTensor4D<half>& input, 
                        GPUTensor4D<half>& output,
                        cudaStream_t stream = 0);
                        
    virtual void backward(const GPUTensor4D<half>& input,
                         const GPUTensor4D<half>& grad_output,
                         GPUTensor4D<half>& grad_input,
                         float learning_rate,
                         cudaStream_t stream = 0);

    // ========================================
    // NEW: Mixed precision support
    // ========================================
    void load_from_fp32(const GPUConv2DLayer& fp32_layer, cudaStream_t stream = 0);
    void save_to_fp32(GPUConv2DLayer& fp32_layer, cudaStream_t stream = 0) const;

    // Accessors
    int in_channels() const { return in_c_; }
    int out_channels() const { return out_c_; }
    DataLayout layout() const { return layout_; }

    GPUTensor4D<half>& weights() { return weights_fp16_; }
    GPUTensor4D<half>& bias() { return bias_fp16_; }

protected:
    int in_c_, out_c_, k_, stride_, padding_;
    DataLayout layout_;
    
    // FP16 weights for fast compute
    GPUTensor4D<half> weights_fp16_;
    GPUTensor4D<half> bias_fp16_;
    GPUTensor4D<half> grad_weights_fp16_;
    GPUTensor4D<half> grad_bias_fp16_;
    
    // Optional: FP32 master weights for stable training
    GPUTensor4D<float> master_weights_;
    GPUTensor4D<float> master_bias_;

    void initialize_weights();
    void update_weights(float learning_rate, cudaStream_t stream);
};

// ============================================================================
// NEW: FP16 CONV+RELU WITH TENSOR CORES
// ============================================================================

class GPUConvReLULayerFP16 : public GPUConv2DLayerFP16 {
public:
    GPUConvReLULayerFP16(int in_channels, int out_channels, int kernel_size,
                         int stride = 1, int padding = 1,
                         DataLayout layout = DataLayout::NHWC);
    ~GPUConvReLULayerFP16() override = default;

    void forward(const GPUTensor4D<half>& input, 
                GPUTensor4D<half>& output,
                cudaStream_t stream = 0) override;
                
    void backward(const GPUTensor4D<half>& input,
                 const GPUTensor4D<half>& grad_output,
                 GPUTensor4D<half>& grad_input,
                 float learning_rate,
                 cudaStream_t stream = 0) override;

    GPUTensor4D<half>& conv_output() { return conv_output_; }

private:
    GPUTensor4D<half> conv_output_;  // Pre-ReLU activations
};