#pragma once

#include <cuda_runtime.h>
#include <string>

class GPUAutoencoder
{
private:
    // Device weight pointers
    float *d_enc_conv1_weights, *d_enc_conv1_bias;
    float *d_enc_conv2_weights, *d_enc_conv2_bias;
    float *d_dec_conv1_weights, *d_dec_conv1_bias;
    float *d_dec_conv2_weights, *d_dec_conv2_bias;
    float *d_dec_conv3_weights, *d_dec_conv3_bias;

    // Device activation buffers
    float *d_input;
    float *d_enc_conv1_out;
    float *d_enc_relu1_out;
    float *d_enc_pool1_out;
    float *d_enc_conv2_out;
    float *d_enc_relu2_out;
    float *d_latent;
    float *d_dec_conv1_out;
    float *d_dec_relu1_out;
    float *d_dec_up1_out;
    float *d_dec_conv2_out;
    float *d_dec_relu2_out;
    float *d_dec_up2_out;
    float *d_reconstruction;

    // Device gradient buffers
    float *d_recon_grad;
    float *d_dec_up2_grad;
    float *d_dec_relu2_grad;
    float *d_dec_conv2_grad;
    float *d_dec_up1_grad;
    float *d_dec_relu1_grad;
    float *d_dec_conv1_grad;
    float *d_latent_grad;
    float *d_enc_pool2_grad;
    float *d_enc_relu2_grad;
    float *d_enc_conv2_grad;
    float *d_enc_pool1_grad;
    float *d_enc_relu1_grad;
    float *d_enc_conv1_grad;

    // Device gradient accumulators for weights
    float *d_enc_conv1_weight_grad, *d_enc_conv1_bias_grad;
    float *d_enc_conv2_weight_grad, *d_enc_conv2_bias_grad;
    float *d_dec_conv1_weight_grad, *d_dec_conv1_bias_grad;
    float *d_dec_conv2_weight_grad, *d_dec_conv2_bias_grad;
    float *d_dec_conv3_weight_grad, *d_dec_conv3_bias_grad;

    // Max pool indices
    int *d_pool1_indices, *d_pool2_indices;

    int max_batch_size;

public:
    GPUAutoencoder(int max_batch = 64);
    ~GPUAutoencoder();

    void allocateMemory();
    void freeMemory();

    void copyWeightsToDevice(float *enc_conv1_w, float *enc_conv1_b,
                             float *enc_conv2_w, float *enc_conv2_b,
                             float *dec_conv1_w, float *dec_conv1_b,
                             float *dec_conv2_w, float *dec_conv2_b,
                             float *dec_conv3_w, float *dec_conv3_b);

    void copyWeightsToHost(float *enc_conv1_w, float *enc_conv1_b,
                           float *enc_conv2_w, float *enc_conv2_b,
                           float *dec_conv1_w, float *dec_conv1_b,
                           float *dec_conv2_w, float *dec_conv2_b,
                           float *dec_conv3_w, float *dec_conv3_b);

    float forward(const float *h_input, int batch_size);
    void backward(const float *h_input, int batch_size);
    void updateWeights(float learning_rate);

    void extractFeatures(const float *h_input, float *h_features, int num_images, int batch_size = 64);

    void saveWeights(const std::string &filename);
    void loadWeights(const std::string &filename);
};
