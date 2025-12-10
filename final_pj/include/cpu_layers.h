#pragma once

#include <cstddef>
#include <vector>

// Naive CPU implementations of layers needed for the autoencoder.

class Conv2D
{
public:
    int in_channels, out_channels, kernel_size;
    int padding, stride;
    int in_h, in_w;             // dimensions of input feature maps
    std::vector<float> weights; // [out_ch, in_ch, K, K]
    std::vector<float> bias;    // [out_ch]

    // gradients
    std::vector<float> weight_grad;
    std::vector<float> bias_grad;

    Conv2D(int in_ch, int out_ch, int k, int pad = 1, int str = 1, int h = 0, int w = 0);
    void initializeWeights();

    // Forward: input [N, C_in, H, W] -> output [N, C_out, H', W']
    void forward(const float *input, float *output, int batch_size, int H, int W);

    // Backward: given output_grad [N, C_out, H', W'] compute input_grad and accumulate weight/bias grads
    void backward(const float *input, const float *output_grad, float *input_grad,
                  int batch_size, int H, int W);

    void updateWeights(float lr);
};

class ReLU
{
public:
    void forward(const float *input, float *output, int size);
    void backward(const float *input, const float *output_grad, float *input_grad, int size);
};

class MaxPool2D
{
public:
    int pool_size, stride;
    // store indices of max per forward call externally (caller provides buffer)
    MaxPool2D(int pool_sz = 2, int str = 2);
    void forward(const float *input, float *output, int *max_indices,
                 int batch_size, int channels, int height, int width);
    void backward(const float *output_grad, float *input_grad, const int *max_indices,
                  int batch_size, int channels, int out_h, int out_w, int height, int width);
};

class UpSample2D
{
public:
    int scale;
    UpSample2D(int scale_factor = 2);
    void forward(const float *input, float *output, int batch_size, int channels, int height, int width);
    void backward(const float *output_grad, float *input_grad, int batch_size, int channels, int height, int width);
};

class MSELoss
{
public:
    // computes loss and fills grad_output (size = N*C*H*W)
    float computeLoss(const float *predicted, const float *target, float *grad_output, int size);
};
