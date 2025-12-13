#pragma once

#include <cmath>
#include <algorithm>
#include <cstring>

class Conv2D
{
private:
    int in_channels, out_channels, kernel_size;
    int height, width;
    int padding, stride;
    float *weights; // [out_ch, in_ch, K, K]
    float *bias;    // [out_ch]
    float *weight_grad;
    float *bias_grad;
    float *input_cache; // For backward pass

public:
    Conv2D(int in_ch, int out_ch, int k, int pad, int str);
    ~Conv2D();

    void forward(const float *input, float *output, int batch_size);
    void backward(const float *input, const float *output_grad,
                  float *input_grad, int batch_size);
    void updateWeights(float learning_rate);
    void initializeWeights();

    // Getters
    int getOutChannels() const { return out_channels; }
    int getOutHeight(int h) const { return (h + 2 * padding - kernel_size) / stride + 1; }
    int getOutWidth(int w) const { return (w + 2 * padding - kernel_size) / stride + 1; }
    float *getWeights() { return weights; }
    float *getBias() { return bias; }
};

class ReLU
{
public:
    ReLU() = default;
    ~ReLU() = default;

    void forward(const float *input, float *output, int size);
    void backward(const float *input, const float *output_grad,
                  float *input_grad, int size);
};

class MaxPool2D
{
private:
    int pool_size, stride;
    int *max_indices;
    int allocated_size;

public:
    MaxPool2D(int pool_sz, int str);
    ~MaxPool2D();

    void forward(const float *input, float *output, int batch_size,
                 int channels, int height, int width);
    void backward(const float *output_grad, float *input_grad,
                  int batch_size, int channels, int out_h, int out_w);
};

class UpSample2D
{
private:
    int scale_factor;

public:
    enum Method
    {
        NEAREST,
        BILINEAR
    };

    UpSample2D(int scale, Method m = NEAREST);
    ~UpSample2D();

    void forward(const float *input, float *output, int batch_size,
                 int channels, int height, int width);
    void backward(const float *output_grad, float *input_grad,
                  int batch_size, int channels, int height, int width);

private:
    Method method;
};

class MSELoss
{
public:
    MSELoss() = default;
    ~MSELoss() = default;

    float computeLoss(const float *predicted, const float *target,
                      float *grad_output, int size);
};
