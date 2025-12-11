#pragma once

#include "cpu_layers.h"
#include <string>

class Autoencoder
{
private:
    // Encoder layers - now public
    ReLU *enc_relu1;
    MaxPool2D *enc_pool1;
    ReLU *enc_relu2;
    MaxPool2D *enc_pool2;

    // Decoder layers - now public
    ReLU *dec_relu1;
    UpSample2D *dec_up1;
    ReLU *dec_relu2;
    UpSample2D *dec_up2;

    // Loss
    MSELoss *loss;

    // Activation buffers
    float *enc_conv1_out;
    float *enc_relu1_out;
    float *enc_pool1_out;
    float *enc_conv2_out;
    float *enc_relu2_out;
    float *latent;
    float *dec_conv1_out;
    float *dec_relu1_out;
    float *dec_up1_out;
    float *dec_conv2_out;
    float *dec_relu2_out;
    float *dec_up2_out;
    float *reconstruction;

    // Gradient buffers
    float *recon_grad;
    float *dec_up2_grad;
    float *dec_relu2_grad;
    float *dec_conv2_grad;
    float *dec_up1_grad;
    float *dec_relu1_grad;
    float *dec_conv1_grad;
    float *latent_grad;
    float *enc_pool2_grad;
    float *enc_relu2_grad;
    float *enc_conv2_grad;
    float *enc_pool1_grad;
    float *enc_relu1_grad;
    float *enc_conv1_grad;

    int max_batch_size;

public:
    Autoencoder(int max_batch = 64);
    ~Autoencoder();

    float forward(const float *input, int batch_size);
    void backward(const float *input, int batch_size);
    void updateWeights(float learning_rate);

    void extractFeatures(const float *input, float *features, int num_images, int batch_size = 64);

    void saveWeights(const std::string &filename);
    void loadWeights(const std::string &filename);

    // Public access to layers for GPU training
    Conv2D *enc_conv1;
    Conv2D *enc_conv2;
    Conv2D *dec_conv1;
    Conv2D *dec_conv2;
    Conv2D *dec_conv3;
};
