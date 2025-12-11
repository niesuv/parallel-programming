#include "autoencoder.h"
#include <iostream>
#include <cstring>
#include <fstream>

Autoencoder::Autoencoder(int max_batch) : max_batch_size(max_batch)
{
    // Encoder
    enc_conv1 = new Conv2D(3, 256, 3, 1, 1); // 32x32 -> 32x32
    enc_relu1 = new ReLU();
    enc_pool1 = new MaxPool2D(2, 2);           // 32x32 -> 16x16
    enc_conv2 = new Conv2D(256, 128, 3, 1, 1); // 16x16 -> 16x16
    enc_relu2 = new ReLU();
    enc_pool2 = new MaxPool2D(2, 2); // 16x16 -> 8x8

    // Decoder
    dec_conv1 = new Conv2D(128, 128, 3, 1, 1); // 8x8 -> 8x8
    dec_relu1 = new ReLU();
    dec_up1 = new UpSample2D(2, UpSample2D::NEAREST); // 8x8 -> 16x16
    dec_conv2 = new Conv2D(128, 256, 3, 1, 1);        // 16x16 -> 16x16
    dec_relu2 = new ReLU();
    dec_up2 = new UpSample2D(2, UpSample2D::NEAREST); // 16x16 -> 32x32
    dec_conv3 = new Conv2D(256, 3, 3, 1, 1);          // 32x32 -> 32x32

    loss = new MSELoss();

    // Allocate activation buffers
    enc_conv1_out = new float[max_batch * 256 * 32 * 32];
    enc_relu1_out = new float[max_batch * 256 * 32 * 32];
    enc_pool1_out = new float[max_batch * 256 * 16 * 16];
    enc_conv2_out = new float[max_batch * 128 * 16 * 16];
    enc_relu2_out = new float[max_batch * 128 * 16 * 16];
    latent = new float[max_batch * 128 * 8 * 8];
    dec_conv1_out = new float[max_batch * 128 * 8 * 8];
    dec_relu1_out = new float[max_batch * 128 * 8 * 8];
    dec_up1_out = new float[max_batch * 128 * 16 * 16];
    dec_conv2_out = new float[max_batch * 256 * 16 * 16];
    dec_relu2_out = new float[max_batch * 256 * 16 * 16];
    dec_up2_out = new float[max_batch * 256 * 32 * 32];
    reconstruction = new float[max_batch * 3 * 32 * 32];

    // Allocate gradient buffers
    recon_grad = new float[max_batch * 3 * 32 * 32];
    dec_up2_grad = new float[max_batch * 256 * 32 * 32];
    dec_relu2_grad = new float[max_batch * 256 * 16 * 16];
    dec_conv2_grad = new float[max_batch * 128 * 16 * 16];
    dec_up1_grad = new float[max_batch * 128 * 16 * 16];
    dec_relu1_grad = new float[max_batch * 128 * 8 * 8];
    dec_conv1_grad = new float[max_batch * 128 * 8 * 8];
    latent_grad = new float[max_batch * 128 * 8 * 8];
    enc_pool2_grad = new float[max_batch * 128 * 16 * 16];
    enc_relu2_grad = new float[max_batch * 128 * 16 * 16];
    enc_conv2_grad = new float[max_batch * 256 * 16 * 16];
    enc_pool1_grad = new float[max_batch * 256 * 16 * 16];
    enc_relu1_grad = new float[max_batch * 256 * 32 * 32];
    enc_conv1_grad = new float[max_batch * 3 * 32 * 32];
}

Autoencoder::~Autoencoder()
{
    delete enc_conv1;
    delete enc_relu1;
    delete enc_pool1;
    delete enc_conv2;
    delete enc_relu2;
    delete enc_pool2;
    delete dec_conv1;
    delete dec_relu1;
    delete dec_up1;
    delete dec_conv2;
    delete dec_relu2;
    delete dec_up2;
    delete dec_conv3;
    delete loss;

    delete[] enc_conv1_out;
    delete[] enc_relu1_out;
    delete[] enc_pool1_out;
    delete[] enc_conv2_out;
    delete[] enc_relu2_out;
    delete[] latent;
    delete[] dec_conv1_out;
    delete[] dec_relu1_out;
    delete[] dec_up1_out;
    delete[] dec_conv2_out;
    delete[] dec_relu2_out;
    delete[] dec_up2_out;
    delete[] reconstruction;

    delete[] recon_grad;
    delete[] dec_up2_grad;
    delete[] dec_relu2_grad;
    delete[] dec_conv2_grad;
    delete[] dec_up1_grad;
    delete[] dec_relu1_grad;
    delete[] dec_conv1_grad;
    delete[] latent_grad;
    delete[] enc_pool2_grad;
    delete[] enc_relu2_grad;
    delete[] enc_conv2_grad;
    delete[] enc_pool1_grad;
    delete[] enc_relu1_grad;
    delete[] enc_conv1_grad;
}

float Autoencoder::forward(const float *input, int batch_size)
{
    // Encoder
    enc_conv1->forward(input, enc_conv1_out, batch_size);
    enc_relu1->forward(enc_conv1_out, enc_relu1_out, batch_size * 256 * 32 * 32);
    enc_pool1->forward(enc_relu1_out, enc_pool1_out, batch_size, 256, 32, 32);

    enc_conv2->forward(enc_pool1_out, enc_conv2_out, batch_size);
    enc_relu2->forward(enc_conv2_out, enc_relu2_out, batch_size * 128 * 16 * 16);
    enc_pool2->forward(enc_relu2_out, latent, batch_size, 128, 16, 16);

    // Decoder
    dec_conv1->forward(latent, dec_conv1_out, batch_size);
    dec_relu1->forward(dec_conv1_out, dec_relu1_out, batch_size * 128 * 8 * 8);
    dec_up1->forward(dec_relu1_out, dec_up1_out, batch_size, 128, 8, 8);

    dec_conv2->forward(dec_up1_out, dec_conv2_out, batch_size);
    dec_relu2->forward(dec_conv2_out, dec_relu2_out, batch_size * 256 * 16 * 16);
    dec_up2->forward(dec_relu2_out, dec_up2_out, batch_size, 256, 16, 16);

    dec_conv3->forward(dec_up2_out, reconstruction, batch_size);

    // Compute loss
    float loss_val = loss->computeLoss(reconstruction, input, recon_grad,
                                       batch_size * 3 * 32 * 32);

    return loss_val;
}

void Autoencoder::backward(const float *input, int batch_size)
{
    // Backward through decoder
    std::memset(dec_up2_grad, 0, batch_size * 256 * 32 * 32 * sizeof(float));
    dec_conv3->backward(dec_up2_out, recon_grad, dec_up2_grad, batch_size);

    std::memset(dec_relu2_grad, 0, batch_size * 256 * 16 * 16 * sizeof(float));
    dec_up2->backward(dec_up2_grad, dec_relu2_grad, batch_size, 256, 16, 16);

    std::memset(dec_conv2_grad, 0, batch_size * 128 * 16 * 16 * sizeof(float));
    dec_relu2->backward(dec_conv2_out, dec_relu2_grad, dec_conv2_grad, batch_size * 256 * 16 * 16);

    std::memset(dec_up1_grad, 0, batch_size * 128 * 16 * 16 * sizeof(float));
    dec_conv2->backward(dec_up1_out, dec_conv2_grad, dec_up1_grad, batch_size);

    std::memset(dec_relu1_grad, 0, batch_size * 128 * 8 * 8 * sizeof(float));
    dec_up1->backward(dec_up1_grad, dec_relu1_grad, batch_size, 128, 8, 8);

    std::memset(dec_conv1_grad, 0, batch_size * 128 * 8 * 8 * sizeof(float));
    dec_relu1->backward(dec_conv1_out, dec_relu1_grad, dec_conv1_grad, batch_size * 128 * 8 * 8);

    std::memset(latent_grad, 0, batch_size * 128 * 8 * 8 * sizeof(float));
    dec_conv1->backward(latent, dec_conv1_grad, latent_grad, batch_size);

    // Backward through encoder
    std::memset(enc_pool2_grad, 0, batch_size * 128 * 16 * 16 * sizeof(float));
    enc_pool2->backward(latent_grad, enc_pool2_grad, batch_size, 128, 8, 8);

    std::memset(enc_relu2_grad, 0, batch_size * 128 * 16 * 16 * sizeof(float));
    enc_relu2->backward(enc_conv2_out, enc_pool2_grad, enc_relu2_grad, batch_size * 128 * 16 * 16);

    std::memset(enc_conv2_grad, 0, batch_size * 256 * 16 * 16 * sizeof(float));
    enc_conv2->backward(enc_pool1_out, enc_relu2_grad, enc_conv2_grad, batch_size);

    std::memset(enc_pool1_grad, 0, batch_size * 256 * 16 * 16 * sizeof(float));
    enc_pool1->backward(enc_conv2_grad, enc_pool1_grad, batch_size, 256, 16, 16);

    std::memset(enc_relu1_grad, 0, batch_size * 256 * 32 * 32 * sizeof(float));
    enc_relu1->backward(enc_conv1_out, enc_pool1_grad, enc_relu1_grad, batch_size * 256 * 32 * 32);

    std::memset(enc_conv1_grad, 0, batch_size * 3 * 32 * 32 * sizeof(float));
    enc_conv1->backward(input, enc_relu1_grad, enc_conv1_grad, batch_size);
}

void Autoencoder::updateWeights(float learning_rate)
{
    enc_conv1->updateWeights(learning_rate);
    enc_conv2->updateWeights(learning_rate);
    dec_conv1->updateWeights(learning_rate);
    dec_conv2->updateWeights(learning_rate);
    dec_conv3->updateWeights(learning_rate);
}

void Autoencoder::extractFeatures(const float *input, float *features, int num_images, int batch_size)
{
    int num_batches = (num_images + batch_size - 1) / batch_size;

    for (int b = 0; b < num_batches; b++)
    {
        int current_batch_size = std::min(batch_size, num_images - b * batch_size);

        // Forward through encoder only
        enc_conv1->forward(input + b * batch_size * 3 * 32 * 32, enc_conv1_out, current_batch_size);
        enc_relu1->forward(enc_conv1_out, enc_relu1_out, current_batch_size * 256 * 32 * 32);
        enc_pool1->forward(enc_relu1_out, enc_pool1_out, current_batch_size, 256, 32, 32);

        enc_conv2->forward(enc_pool1_out, enc_conv2_out, current_batch_size);
        enc_relu2->forward(enc_conv2_out, enc_relu2_out, current_batch_size * 128 * 16 * 16);
        enc_pool2->forward(enc_relu2_out, latent, current_batch_size, 128, 16, 16);

        // Copy latent features
        std::memcpy(features + b * batch_size * 8192, latent,
                    current_batch_size * 8192 * sizeof(float));
    }
}

void Autoencoder::saveWeights(const std::string &filename)
{
    std::ofstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file for saving: " << filename << std::endl;
        return;
    }

    // Save encoder weights
    int enc_conv1_size = 256 * 3 * 3 * 3;
    file.write(reinterpret_cast<char *>(enc_conv1->getWeights()), enc_conv1_size * sizeof(float));
    file.write(reinterpret_cast<char *>(enc_conv1->getBias()), 256 * sizeof(float));

    int enc_conv2_size = 128 * 256 * 3 * 3;
    file.write(reinterpret_cast<char *>(enc_conv2->getWeights()), enc_conv2_size * sizeof(float));
    file.write(reinterpret_cast<char *>(enc_conv2->getBias()), 128 * sizeof(float));

    // Save decoder weights
    int dec_conv1_size = 128 * 128 * 3 * 3;
    file.write(reinterpret_cast<char *>(dec_conv1->getWeights()), dec_conv1_size * sizeof(float));
    file.write(reinterpret_cast<char *>(dec_conv1->getBias()), 128 * sizeof(float));

    int dec_conv2_size = 256 * 128 * 3 * 3;
    file.write(reinterpret_cast<char *>(dec_conv2->getWeights()), dec_conv2_size * sizeof(float));
    file.write(reinterpret_cast<char *>(dec_conv2->getBias()), 256 * sizeof(float));

    int dec_conv3_size = 3 * 256 * 3 * 3;
    file.write(reinterpret_cast<char *>(dec_conv3->getWeights()), dec_conv3_size * sizeof(float));
    file.write(reinterpret_cast<char *>(dec_conv3->getBias()), 3 * sizeof(float));

    file.close();
    std::cout << "Weights saved to " << filename << std::endl;
}

void Autoencoder::loadWeights(const std::string &filename)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file for loading: " << filename << std::endl;
        return;
    }

    // Load encoder weights
    int enc_conv1_size = 256 * 3 * 3 * 3;
    file.read(reinterpret_cast<char *>(enc_conv1->getWeights()), enc_conv1_size * sizeof(float));
    file.read(reinterpret_cast<char *>(enc_conv1->getBias()), 256 * sizeof(float));

    int enc_conv2_size = 128 * 256 * 3 * 3;
    file.read(reinterpret_cast<char *>(enc_conv2->getWeights()), enc_conv2_size * sizeof(float));
    file.read(reinterpret_cast<char *>(enc_conv2->getBias()), 128 * sizeof(float));

    // Load decoder weights
    int dec_conv1_size = 128 * 128 * 3 * 3;
    file.read(reinterpret_cast<char *>(dec_conv1->getWeights()), dec_conv1_size * sizeof(float));
    file.read(reinterpret_cast<char *>(dec_conv1->getBias()), 128 * sizeof(float));

    int dec_conv2_size = 256 * 128 * 3 * 3;
    file.read(reinterpret_cast<char *>(dec_conv2->getWeights()), dec_conv2_size * sizeof(float));
    file.read(reinterpret_cast<char *>(dec_conv2->getBias()), 256 * sizeof(float));

    int dec_conv3_size = 3 * 256 * 3 * 3;
    file.read(reinterpret_cast<char *>(dec_conv3->getWeights()), dec_conv3_size * sizeof(float));
    file.read(reinterpret_cast<char *>(dec_conv3->getBias()), 3 * sizeof(float));

    file.close();
    std::cout << "Weights loaded from " << filename << std::endl;
}
