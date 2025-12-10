#include "../include/cpu_layers.h"
#include <cmath>
#include <cstring>
#include <random>
#include <algorithm>

// Conv2D implementation (naive)
Conv2D::Conv2D(int in_ch, int out_ch, int k, int pad, int str, int h, int w)
    : in_channels(in_ch), out_channels(out_ch), kernel_size(k), padding(pad), stride(str), in_h(h), in_w(w)
{
    weights.resize(out_channels * in_channels * kernel_size * kernel_size);
    bias.resize(out_channels);
    weight_grad.resize(weights.size());
    bias_grad.resize(bias.size());
    initializeWeights();
}

void Conv2D::initializeWeights()
{
    // He init
    std::mt19937 gen(42);
    float stddev = std::sqrt(2.0f / (in_channels * kernel_size * kernel_size));
    std::normal_distribution<float> d(0.0f, stddev);
    for (auto &w : weights)
        w = d(gen);
    for (auto &b : bias)
        b = 0.0f;
}

void Conv2D::forward(const float *input, float *output, int batch_size, int H, int W)
{
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    // For each element in output
    int out_spatial = H_out * W_out;
    for (int n = 0; n < batch_size; ++n)
    {
        for (int oc = 0; oc < out_channels; ++oc)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    float sum = bias[oc];
                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        for (int kh = 0; kh < kernel_size; ++kh)
                        {
                            for (int kw = 0; kw < kernel_size; ++kw)
                            {
                                int h_in = h * stride + kh - padding;
                                int w_in = w * stride + kw - padding;
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                                {
                                    int in_idx = ((n * in_channels + ic) * H + h_in) * W + w_in;
                                    int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                    sum += input[in_idx] * weights[w_idx];
                                }
                            }
                        }
                    }
                    int out_idx = ((n * out_channels + oc) * H_out + h) * W_out + w;
                    output[out_idx] = sum;
                }
            }
        }
    }
}

void Conv2D::backward(const float *input, const float *output_grad, float *input_grad,
                      int batch_size, int H, int W)
{
    int H_out = (H + 2 * padding - kernel_size) / stride + 1;
    int W_out = (W + 2 * padding - kernel_size) / stride + 1;

    // zero gradients
    std::fill(weight_grad.begin(), weight_grad.end(), 0.0f);
    std::fill(bias_grad.begin(), bias_grad.end(), 0.0f);
    int input_size = batch_size * in_channels * H * W;
    std::fill(input_grad, input_grad + input_size, 0.0f);

    for (int n = 0; n < batch_size; ++n)
    {
        for (int oc = 0; oc < out_channels; ++oc)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int out_idx = ((n * out_channels + oc) * H_out + h) * W_out + w;
                    float go = output_grad[out_idx];
                    bias_grad[oc] += go;
                    for (int ic = 0; ic < in_channels; ++ic)
                    {
                        for (int kh = 0; kh < kernel_size; ++kh)
                        {
                            for (int kw = 0; kw < kernel_size; ++kw)
                            {
                                int h_in = h * stride + kh - padding;
                                int w_in = w * stride + kw - padding;
                                if (h_in >= 0 && h_in < H && w_in >= 0 && w_in < W)
                                {
                                    int in_idx = ((n * in_channels + ic) * H + h_in) * W + w_in;
                                    int w_idx = ((oc * in_channels + ic) * kernel_size + kh) * kernel_size + kw;
                                    weight_grad[w_idx] += input[in_idx] * go;
                                    input_grad[in_idx] += weights[w_idx] * go;
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}

void Conv2D::updateWeights(float lr)
{
    for (size_t i = 0; i < weights.size(); ++i)
        weights[i] -= lr * weight_grad[i];
    for (size_t i = 0; i < bias.size(); ++i)
        bias[i] -= lr * bias_grad[i];
}

// ReLU
void ReLU::forward(const float *input, float *output, int size)
{
    for (int i = 0; i < size; ++i)
        output[i] = input[i] > 0.0f ? input[i] : 0.0f;
}
void ReLU::backward(const float *input, const float *output_grad, float *input_grad, int size)
{
    for (int i = 0; i < size; ++i)
        input_grad[i] = (input[i] > 0.0f) ? output_grad[i] : 0.0f;
}

// MaxPool2D
MaxPool2D::MaxPool2D(int pool_sz, int str) : pool_size(pool_sz), stride(str) {}

void MaxPool2D::forward(const float *input, float *output, int *max_indices,
                        int batch_size, int channels, int height, int width)
{
    int out_h = height / stride;
    int out_w = width / stride;
    for (int n = 0; n < batch_size; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int oh = 0; oh < out_h; ++oh)
            {
                for (int ow = 0; ow < out_w; ++ow)
                {
                    int out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
                    float maxv = -INFINITY;
                    int maxidx = -1;
                    for (int ph = 0; ph < pool_size; ++ph)
                    {
                        for (int pw = 0; pw < pool_size; ++pw)
                        {
                            int ih = oh * stride + ph;
                            int iw = ow * stride + pw;
                            int in_idx = ((n * channels + c) * height + ih) * width + iw;
                            float v = input[in_idx];
                            if (v > maxv)
                            {
                                maxv = v;
                                maxidx = in_idx;
                            }
                        }
                    }
                    output[out_idx] = maxv;
                    max_indices[out_idx] = maxidx;
                }
            }
        }
    }
}

void MaxPool2D::backward(const float *output_grad, float *input_grad, const int *max_indices,
                         int batch_size, int channels, int out_h, int out_w, int height, int width)
{
    int input_size = batch_size * channels * height * width;
    std::fill(input_grad, input_grad + input_size, 0.0f);
    for (int idx = 0; idx < batch_size * channels * out_h * out_w; ++idx)
    {
        int in_idx = max_indices[idx];
        if (in_idx >= 0)
            input_grad[in_idx] += output_grad[idx];
    }
}

// UpSample2D (nearest neighbor)
UpSample2D::UpSample2D(int scale_factor) : scale(scale_factor) {}

void UpSample2D::forward(const float *input, float *output, int batch_size, int channels, int height, int width)
{
    int H_out = height * scale;
    int W_out = width * scale;
    for (int n = 0; n < batch_size; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int hin = h / scale;
                    int win = w / scale;
                    int in_idx = ((n * channels + c) * height + hin) * width + win;
                    int out_idx = ((n * channels + c) * H_out + h) * W_out + w;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

void UpSample2D::backward(const float *output_grad, float *input_grad, int batch_size, int channels, int height, int width)
{
    int H_out = height * scale;
    int W_out = width * scale;
    int input_size = batch_size * channels * height * width;
    std::fill(input_grad, input_grad + input_size, 0.0f);
    for (int n = 0; n < batch_size; ++n)
    {
        for (int c = 0; c < channels; ++c)
        {
            for (int h = 0; h < H_out; ++h)
            {
                for (int w = 0; w < W_out; ++w)
                {
                    int hin = h / scale;
                    int win = w / scale;
                    int in_idx = ((n * channels + c) * height + hin) * width + win;
                    int out_idx = ((n * channels + c) * H_out + h) * W_out + w;
                    input_grad[in_idx] += output_grad[out_idx];
                }
            }
        }
    }
}

// MSELoss
float MSELoss::computeLoss(const float *predicted, const float *target, float *grad_output, int size)
{
    double loss = 0.0;
    for (int i = 0; i < size; ++i)
    {
        float diff = predicted[i] - target[i];
        loss += diff * diff;
        grad_output[i] = 2.0f * diff / size;
    }
    return static_cast<float>(loss / size);
}
