#pragma once

#include "cpu_layers.h"
#include <string>
#include <vector>

class Autoencoder
{
public:
    // encoder
    Conv2D *enc_conv1; // 3 -> 256, 3x3
    ReLU *enc_relu1;
    MaxPool2D *enc_pool1;
    Conv2D *enc_conv2; // 256 -> 128
    ReLU *enc_relu2;
    MaxPool2D *enc_pool2;

    // decoder
    Conv2D *dec_conv1; // 128 -> 128
    ReLU *dec_relu1;
    UpSample2D *dec_up1;
    Conv2D *dec_conv2; // 128 -> 256
    ReLU *dec_relu2;
    UpSample2D *dec_up2;
    Conv2D *dec_conv3; // 256 -> 3

    MSELoss *loss;

    // buffers allocated per max batch
    int max_batch_size;
    std::vector<float> buffers; // large contiguous buffer for activations
    std::vector<int> pool1_indices;
    std::vector<int> pool2_indices;

    Autoencoder(int max_batch = 64);
    ~Autoencoder();

    // Forward: returns loss
    float forward(const float *input, int batch_size);
    void backward(const float *input, int batch_size);
    void updateWeights(float learning_rate);

    // Extract features (encoder only) - features in row-major per image, size 8192
    void extractFeatures(const float *input, float *features, int num_images);

    // Save/load weights
    bool saveWeights(const std::string &filename);
    bool loadWeights(const std::string &filename);
};
