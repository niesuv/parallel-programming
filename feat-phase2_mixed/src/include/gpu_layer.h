#ifndef GPU_LAYER_H
#define GPU_LAYER_H

#include <cstddef>
#include <cuda_fp16.h>

// Loss scaler

enum class GPUDtype {
    FP16,
    FP32
};

struct GPULossScaler {
    float scale = 1024.0f;
    float min_scale = 1.0f;
    float max_scale = 65536.0f;

    void increase() {
        scale = (scale < max_scale) ? scale * 2.0f : scale;
    }

    void decrease() {
        scale = (scale > min_scale) ? scale * 0.5f : scale;
    }
};

struct GPUTensor4D {
    int n = 0;
    int c = 0;
    int h = 0;
    int w = 0;

    void *d_data = nullptr;
    GPUDtype dtype = GPUDtype::FP16;

    GPUTensor4D() = default;
    GPUTensor4D(int n_, int c_, int h_, int w_,
                GPUDtype dtype_ = GPUDtype::FP16);

    ~GPUTensor4D();

    GPUTensor4D(const GPUTensor4D &) = delete;
    GPUTensor4D &operator=(const GPUTensor4D &) = delete;

    GPUTensor4D(GPUTensor4D &&other) noexcept;
    GPUTensor4D &operator=(GPUTensor4D &&other) noexcept;

    void allocate(int n_, int c_, int h_, int w_,
                    GPUDtype dtype_ = GPUDtype::FP16);
    void free();

    size_t size() const {
        return static_cast<size_t>(n) * c * h * w;
    }

    size_t element_size() const {
        return (dtype == GPUDtype::FP16) ? sizeof(__half) : sizeof(float);
    }

    size_t bytes() const {
        return size() * element_size();
    }

    /* Host <-> Device copy
        Host data is always FP32 */
    void copy_from_host_fp32(const float *h_data);
    void copy_to_host_fp32(float *h_data) const;
};


class GPUConv2DLayer {
public:
    GPUConv2DLayer(int in_channels,
                    int out_channels,
                    int kernel_size,
                    int stride = 1,
                    int padding = 1);

    ~GPUConv2DLayer();

    /* Forward: FP16 */
    void forward_fp16(const GPUTensor4D &input_fp16,
                        GPUTensor4D &output_fp16) const;

    /* Backward:
        - grad_output is FP16 (scaled)
        - grad_input is FP16 (scaled)
        - weights update via FP32 master
    */
    void backward_fp16(const GPUTensor4D &input_fp16,
                        const GPUTensor4D &grad_output_fp16,
                        GPUTensor4D &grad_input_fp16,
                        float learning_rate,
                        GPULossScaler &scaler);

    /* Weight I/O (FP32 on host) */
    void copy_weights_from_host_fp32(const float *h_weights,
                                    const float *h_bias);

    void copy_weights_to_host_fp32(float *h_weights,
                                    float *h_bias) const;

    int get_output_h(int input_h) const {
        return (input_h + 2 * padding_ - k_) / stride_ + 1;
    }

    int get_output_w(int input_w) const {
        return (input_w + 2 * padding_ - k_) / stride_ + 1;
    }

    int get_out_channels() const { return out_c_; }
    int get_in_channels() const { return in_c_; }
    int get_kernel_size() const { return k_; }
    int get_stride() const { return stride_; }
    int get_padding() const { return padding_; }

private:
    int in_c_, out_c_, k_, stride_, padding_;
    size_t weights_size_;

    /* FP16 weights for forward */
    __half *d_weights_fp16_ = nullptr;
    __half *d_bias_fp16_ = nullptr;

    /* FP32 master weights */
    float *d_weights_fp32_ = nullptr;
    float *d_bias_fp32_ = nullptr;

    /* FP32 gradients */
    float *d_grad_weights_fp32_ = nullptr;
    float *d_grad_bias_fp32_ = nullptr;
};


class GPUReLULayer {
public:
    void forward_fp16(const GPUTensor4D &input_fp16,
                    GPUTensor4D &output_fp16) const;

    void backward_fp16(const GPUTensor4D &input_fp16,
                    const GPUTensor4D &grad_output_fp16,
                    GPUTensor4D &grad_input_fp16) const;
};


class GPUMaxPool2DLayer {
public:
    explicit GPUMaxPool2DLayer(int kernel_size = 2, int stride = 2);

    void forward_fp16(const GPUTensor4D &input_fp16,
                        GPUTensor4D &output_fp16) const;

    void backward_fp16(const GPUTensor4D &input_fp16,
                        const GPUTensor4D &grad_output_fp16,
                        GPUTensor4D &grad_input_fp16) const;

    int get_output_h(int input_h) const {
        return (input_h - k_) / stride_ + 1;
    }

    int get_output_w(int input_w) const {
        return (input_w - k_) / stride_ + 1;
    }

private:
    int k_, stride_;
};

class GPUUpSample2DLayer {
public:
    explicit GPUUpSample2DLayer(int scale = 2);

    void forward_fp16(const GPUTensor4D &input_fp16,
                        GPUTensor4D &output_fp16) const;

    void backward_fp16(const GPUTensor4D &grad_output_fp16,
                        GPUTensor4D &grad_input_fp16) const;

    int get_output_h(int input_h) const {
        return input_h * scale_;
    }

    int get_output_w(int input_w) const {
        return input_w * scale_;
    }

private:
    int scale_;
};

/* FP32 loss */
float gpu_mse_loss_fp32(const GPUTensor4D &output_fp16,
                        const GPUTensor4D &target_fp16);

/* Backward (FP16, scaled) */
void gpu_mse_loss_backward_fp16_scaled(
    const GPUTensor4D &output_fp16,
    const GPUTensor4D &target_fp16,
    GPUTensor4D &grad_output_fp16,
    float loss_scale);


void gpu_fp32_to_fp16(const float *src,
                    __half *dst,
                    size_t n);

void gpu_fp16_to_fp32(const __half *src,
                    float *dst,
                    size_t n);

void gpu_conv2d_forward_fp16(
    const GPUTensor4D &input_fp16,
    const __half *d_weights_fp16,
    const __half *d_bias_fp16,
    GPUTensor4D &output_fp16,
    int in_c, int out_c,
    int k, int stride, int padding);

void gpu_conv2d_backward_data_fp16(
    const GPUTensor4D &grad_output_fp16,
    const __half *d_weights_fp16,
    GPUTensor4D &grad_input_fp16,
    int in_c, int out_c,
    int k, int stride, int padding);

void gpu_conv2d_backward_weights_fp16(
    const GPUTensor4D &input_fp16,
    const GPUTensor4D &grad_output_fp16,
    float *d_grad_weights_fp32,
    float *d_grad_bias_fp32,
    float loss_scale,
    int in_c, int out_c,
    int k, int stride, int padding);
//
// OPTIMIZED KERNELS

#ifdef USE_OPTIMIZED_KERNELS

void gpu_relu_forward_fp16_opt(const GPUTensor4D &input_fp16,
                                GPUTensor4D &output_fp16);

void gpu_relu_backward_fp16_opt(const GPUTensor4D &input_fp16,
                                const GPUTensor4D &grad_output_fp16,
                                GPUTensor4D &grad_input_fp16);

void gpu_maxpool2d_forward_fp16_opt(const GPUTensor4D &input_fp16,
                                    GPUTensor4D &output_fp16,
                                    int k, int stride);

void gpu_maxpool2d_backward_fp16_opt(const GPUTensor4D &input_fp16,
                                const GPUTensor4D &grad_output_fp16,
                                GPUTensor4D &grad_input_fp16,
                                int k, int stride);

void gpu_upsample2d_forward_fp16_opt(const GPUTensor4D &input_fp16,
                                    GPUTensor4D &output_fp16,
                                    int scale);

void gpu_upsample2d_backward_fp16_opt(const GPUTensor4D &grad_output_fp16,
                                    GPUTensor4D &grad_input_fp16,
                                    int scale);


#endif // USE_OPTIMIZED_KERNELS

#endif // GPU_LAYER_H
