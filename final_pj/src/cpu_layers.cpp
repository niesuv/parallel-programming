#include "cpu_layers.h"
#include <iostream>
#include <random>
#include <cmath>

// Conv2D Implementation
Conv2D::Conv2D(int in_ch, int out_ch, int k, int pad, int str)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k),
      padding(pad), stride(str), height(0), width(0)
{

    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    weights = new float[weight_size];
    weight_grad = new float[weight_size];
    bias = new float[out_channels];
    bias_grad = new float[out_channels];
    input_cache = nullptr;

    initializeWeights();
}

Conv2D::~Conv2D()
{
    delete[] weights;
    delete[] weight_grad;
    delete[] bias;
    delete[] bias_grad;
    if (input_cache)
        delete[] input_cache;
}

void Conv2D::initializeWeights()
{
    // He initialization for ReLU: N(0, sqrt(2/n_in))
    float fan_in = in_channels * kernel_size * kernel_size;
    float std = std::sqrt(2.0f / fan_in);

    std::mt19937 rng(42);
    std::normal_distribution<float> dist(0.0f, std);

    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    for (int i = 0; i < weight_size; i++)
    {
        weights[i] = dist(rng);
    }

    // Initialize bias to zero
    for (int i = 0; i < out_channels; i++)
    {
        bias[i] = 0.0f;
    }
}

void Conv2D::forward(const float *input, float *output, int batch_size)
{
    if (!height || !width)
    {
        // Infer dimensions from context - this would be set by the caller
        height = 32;
        width = 32;
    }

    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;
    int out_size = batch_size * out_channels * out_h * out_w;
    int in_size = batch_size * in_channels * height * width;

    // Cache input for backward pass
    if (input_cache)
        delete[] input_cache;
    input_cache = new float[in_size];
    std::memcpy(input_cache, input, in_size * sizeof(float));

    // For each output element
    for (int b = 0; b < batch_size; b++)
    {
        for (int c_out = 0; c_out < out_channels; c_out++)
        {
            for (int h_out = 0; h_out < out_h; h_out++)
            {
                for (int w_out = 0; w_out < out_w; w_out++)
                {
                    float sum = bias[c_out];

                    // Convolve
                    for (int c_in = 0; c_in < in_channels; c_in++)
                    {
                        for (int kh = 0; kh < kernel_size; kh++)
                        {
                            for (int kw = 0; kw < kernel_size; kw++)
                            {
                                int h_in = h_out * stride + kh - padding;
                                int w_in = w_out * stride + kw - padding;

                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width)
                                {
                                    int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;
                                    sum += input[input_idx] * weights[weight_idx];
                                }
                            }
                        }
                    }

                    int output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                    output[output_idx] = sum;
                }
            }
        }
    }
}

void Conv2D::backward(const float *input, const float *output_grad,
                      float *input_grad, int batch_size)
{
    int out_h = (height + 2 * padding - kernel_size) / stride + 1;
    int out_w = (width + 2 * padding - kernel_size) / stride + 1;

    // Zero out gradients
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    std::fill(weight_grad, weight_grad + weight_size, 0.0f);
    std::fill(bias_grad, bias_grad + out_channels, 0.0f);
    std::fill(input_grad, input_grad + batch_size * in_channels * height * width, 0.0f);

    // Compute gradients
    for (int b = 0; b < batch_size; b++)
    {
        for (int c_out = 0; c_out < out_channels; c_out++)
        {
            for (int h_out = 0; h_out < out_h; h_out++)
            {
                for (int w_out = 0; w_out < out_w; w_out++)
                {
                    int output_idx = ((b * out_channels + c_out) * out_h + h_out) * out_w + w_out;
                    float grad_out = output_grad[output_idx];

                    // Accumulate bias gradient
                    bias_grad[c_out] += grad_out;

                    // Compute weight gradient and input gradient
                    for (int c_in = 0; c_in < in_channels; c_in++)
                    {
                        for (int kh = 0; kh < kernel_size; kh++)
                        {
                            for (int kw = 0; kw < kernel_size; kw++)
                            {
                                int h_in = h_out * stride + kh - padding;
                                int w_in = w_out * stride + kw - padding;

                                if (h_in >= 0 && h_in < height && w_in >= 0 && w_in < width)
                                {
                                    int input_idx = ((b * in_channels + c_in) * height + h_in) * width + w_in;
                                    int weight_idx = ((c_out * in_channels + c_in) * kernel_size + kh) * kernel_size + kw;

                                    weight_grad[weight_idx] += grad_out * input[input_idx];
                                    input_grad[input_idx] += grad_out * weights[weight_idx];
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Conv2D::updateWeights(float learning_rate)
{
    int weight_size = out_channels * in_channels * kernel_size * kernel_size;
    for (int i = 0; i < weight_size; i++)
    {
        weights[i] -= learning_rate * weight_grad[i];
    }

    for (int i = 0; i < out_channels; i++)
    {
        bias[i] -= learning_rate * bias_grad[i];
    }
}

// ReLU Implementation
void ReLU::forward(const float *input, float *output, int size)
{
    for (int i = 0; i < size; i++)
    {
        output[i] = std::max(0.0f, input[i]);
    }
}

void ReLU::backward(const float *input, const float *output_grad,
                    float *input_grad, int size)
{
    for (int i = 0; i < size; i++)
    {
        input_grad[i] = (input[i] > 0) ? output_grad[i] : 0.0f;
    }
}

// MaxPool2D Implementation
MaxPool2D::MaxPool2D(int pool_sz, int str)
    : pool_size(pool_sz), stride(str), max_indices(nullptr), allocated_size(0) {}

MaxPool2D::~MaxPool2D()
{
    if (max_indices)
        delete[] max_indices;
}

void MaxPool2D::forward(const float *input, float *output, int batch_size,
                        int channels, int height, int width)
{
    int out_h = height / stride;
    int out_w = width / stride;
    int total_outputs = batch_size * channels * out_h * out_w;

    // Allocate or reallocate max_indices
    if (allocated_size < total_outputs)
    {
        if (max_indices)
            delete[] max_indices;
        max_indices = new int[total_outputs];
        allocated_size = total_outputs;
    }

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h_out = 0; h_out < out_h; h_out++)
            {
                for (int w_out = 0; w_out < out_w; w_out++)
                {
                    float max_val = -1e9f;
                    int max_idx = -1;

                    for (int ph = 0; ph < pool_size; ph++)
                    {
                        for (int pw = 0; pw < pool_size; pw++)
                        {
                            int h_in = h_out * stride + ph;
                            int w_in = w_out * stride + pw;
                            int input_idx = ((b * channels + c) * height + h_in) * width + w_in;

                            if (input[input_idx] > max_val)
                            {
                                max_val = input[input_idx];
                                max_idx = input_idx;
                            }
                        }
                    }

                    int output_idx = ((b * channels + c) * out_h + h_out) * out_w + w_out;
                    output[output_idx] = max_val;
                    max_indices[output_idx] = max_idx;
                }
            }
        }
    }
}

void MaxPool2D::backward(const float *output_grad, float *input_grad,
                         int batch_size, int channels, int out_h, int out_w)
{
    int total_outputs = batch_size * channels * out_h * out_w;

    for (int i = 0; i < total_outputs; i++)
    {
        int input_idx = max_indices[i];
        input_grad[input_idx] += output_grad[i];
    }
}

// UpSample2D Implementation
UpSample2D::UpSample2D(int scale, Method m)
    : scale_factor(scale), method(m) {}

UpSample2D::~UpSample2D() = default;

void UpSample2D::forward(const float *input, float *output, int batch_size,
                         int channels, int height, int width)
{
    int out_h = height * scale_factor;
    int out_w = width * scale_factor;

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float val = input[((b * channels + c) * height + h) * width + w];

                    // Nearest neighbor: replicate to scale_factor × scale_factor block
                    for (int sh = 0; sh < scale_factor; sh++)
                    {
                        for (int sw = 0; sw < scale_factor; sw++)
                        {
                            int out_h_idx = h * scale_factor + sh;
                            int out_w_idx = w * scale_factor + sw;
                            int output_idx = ((b * channels + c) * out_h + out_h_idx) * out_w + out_w_idx;
                            output[output_idx] = val;
                        }
                    }
                }
            }
        }
    }
}

void UpSample2D::backward(const float *output_grad, float *input_grad,
                          int batch_size, int channels, int height, int width)
{
    int out_h = height * scale_factor;
    int out_w = width * scale_factor;

    for (int b = 0; b < batch_size; b++)
    {
        for (int c = 0; c < channels; c++)
        {
            for (int h = 0; h < height; h++)
            {
                for (int w = 0; w < width; w++)
                {
                    float grad_sum = 0.0f;

                    // Sum gradients from all scale_factor × scale_factor blocks
                    for (int sh = 0; sh < scale_factor; sh++)
                    {
                        for (int sw = 0; sw < scale_factor; sw++)
                        {
                            int out_h_idx = h * scale_factor + sh;
                            int out_w_idx = w * scale_factor + sw;
                            int output_idx = ((b * channels + c) * out_h + out_h_idx) * out_w + out_w_idx;
                            grad_sum += output_grad[output_idx];
                        }
                    }

                    int input_idx = ((b * channels + c) * height + h) * width + w;
                    input_grad[input_idx] = grad_sum;
                }
            }
        }
    }
}

// MSELoss Implementation
float MSELoss::computeLoss(const float *predicted, const float *target,
                           float *grad_output, int size)
{
    float loss = 0.0f;

    for (int i = 0; i < size; i++)
    {
        float diff = predicted[i] - target[i];
        loss += diff * diff;
        grad_output[i] = 2.0f * diff / size;
    }

    return loss / size;
}
