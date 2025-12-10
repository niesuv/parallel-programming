#include "../include/autoencoder.h"
#include <cstring>
#include <fstream>
#include <iostream>

Autoencoder::Autoencoder(int max_batch) : max_batch_size(max_batch)
{
    // Create layers per architecture
    enc_conv1 = new Conv2D(3, 256, 3, 1, 1, 32, 32);
    enc_relu1 = new ReLU();
    enc_pool1 = new MaxPool2D(2, 2);
    enc_conv2 = new Conv2D(256, 128, 3, 1, 1, 16, 16);
    enc_relu2 = new ReLU();
    enc_pool2 = new MaxPool2D(2, 2);

    dec_conv1 = new Conv2D(128, 128, 3, 1, 1, 8, 8);
    dec_relu1 = new ReLU();
    dec_up1 = new UpSample2D(2);
    dec_conv2 = new Conv2D(128, 256, 3, 1, 1, 16, 16);
    dec_relu2 = new ReLU();
    dec_up2 = new UpSample2D(2);
    dec_conv3 = new Conv2D(256, 3, 3, 1, 1, 32, 32);

    loss = new MSELoss();

    // allocate activation buffers for maximum batch
    int max_input = max_batch * 3 * 32 * 32;
    int max_enc1 = max_batch * 256 * 32 * 32;
    int max_pool1 = max_batch * 256 * 16 * 16;
    int max_enc2 = max_batch * 128 * 16 * 16;
    int max_latent = max_batch * 128 * 8 * 8;
    int max_dec1 = max_latent;
    int max_up1 = max_batch * 128 * 16 * 16;
    int max_dec2 = max_up1 * 1; // 256 channels after conv
    int max_up2 = max_batch * 256 * 32 * 32;
    int max_recon = max_batch * 3 * 32 * 32;

    size_t total = max_input + max_enc1 + max_pool1 + max_enc2 + max_latent + max_dec1 + max_up1 + max_dec2 + max_up2 + max_recon;
    buffers.resize(total);

    pool1_indices.resize(max_pool1);
    pool2_indices.resize(max_latent);
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
}

float Autoencoder::forward(const float *input, int batch_size)
{
    // Layout pointers into buffers
    size_t offset = 0;
    float *in_buf = buffers.data() + offset;
    offset += batch_size * 3 * 32 * 32;
    float *enc1 = buffers.data() + offset;
    offset += batch_size * 256 * 32 * 32;
    float *enc1_relu = enc1; // reuse
    float *pool1 = buffers.data() + offset;
    offset += batch_size * 256 * 16 * 16;
    float *enc2 = buffers.data() + offset;
    offset += batch_size * 128 * 16 * 16;
    float *enc2_relu = enc2;
    float *latent = buffers.data() + offset;
    offset += batch_size * 128 * 8 * 8;
    float *dec1 = buffers.data() + offset;
    offset += batch_size * 128 * 8 * 8;
    float *up1 = buffers.data() + offset;
    offset += batch_size * 128 * 16 * 16;
    float *dec2 = buffers.data() + offset;
    offset += batch_size * 256 * 16 * 16;
    float *up2 = buffers.data() + offset;
    offset += batch_size * 256 * 32 * 32;
    float *recon = buffers.data() + offset;
    offset += batch_size * 3 * 32 * 32;

    // copy input into in_buf
    memcpy(in_buf, input, sizeof(float) * batch_size * 3 * 32 * 32);

    // Encoder
    enc_conv1->forward(in_buf, enc1, batch_size, 32, 32);
    enc_relu1->forward(enc1, enc1_relu, batch_size * 256 * 32 * 32);
    enc_pool1->forward(enc1_relu, pool1, pool1_indices.data(), batch_size, 256, 32, 32);

    enc_conv2->forward(pool1, enc2, batch_size, 16, 16);
    enc_relu2->forward(enc2, enc2_relu, batch_size * 128 * 16 * 16);
    enc_pool2->forward(enc2_relu, latent, pool2_indices.data(), batch_size, 128, 16, 16);

    // Decoder
    dec_conv1->forward(latent, dec1, batch_size, 8, 8);
    dec_relu1->forward(dec1, dec1, batch_size * 128 * 8 * 8);
    dec_up1->forward(dec1, up1, batch_size, 128, 8, 8);

    dec_conv2->forward(up1, dec2, batch_size, 16, 16);
    dec_relu2->forward(dec2, dec2, batch_size * 256 * 16 * 16);
    dec_up2->forward(dec2, up2, batch_size, 256, 16, 16);

    dec_conv3->forward(up2, recon, batch_size, 32, 32);

    // loss
    int size = batch_size * 3 * 32 * 32;
    std::vector<float> grad(size);
    float loss_val = loss->computeLoss(recon, in_buf, grad.data(), size);

    // store gradient in place of recon's grad buffer for backward
    // For simplicity, keep recon grad in buffers after recon (reuse part of buffers)
    // We'll copy grad to the end of buffers (space exists)
    // Here simply copy to recon (not safe if overwritten), so allocate temp on stack above

    // Save gradient in a member temp? For simplicity, allocate temporary and store in a hidden vector
    // Implemented in backward call by recomputing grad via MSELoss again to avoid storing now

    return loss_val;
}

void Autoencoder::backward(const float *input, int batch_size)
{
    // For simplicity, implement a basic backward propagation using stored buffers by re-running forward
    // then computing gradients and calling layer backward functions.
    // Layout pointers into buffers (same as forward)
    size_t offset = 0;
    float *in_buf = buffers.data() + offset;
    offset += batch_size * 3 * 32 * 32;
    float *enc1 = buffers.data() + offset;
    offset += batch_size * 256 * 32 * 32;
    float *pool1 = buffers.data() + offset;
    offset += batch_size * 256 * 16 * 16;
    float *enc2 = buffers.data() + offset;
    offset += batch_size * 128 * 16 * 16;
    float *latent = buffers.data() + offset;
    offset += batch_size * 128 * 8 * 8;
    float *dec1 = buffers.data() + offset;
    offset += batch_size * 128 * 8 * 8;
    float *up1 = buffers.data() + offset;
    offset += batch_size * 128 * 16 * 16;
    float *dec2 = buffers.data() + offset;
    offset += batch_size * 256 * 16 * 16;
    float *up2 = buffers.data() + offset;
    offset += batch_size * 256 * 32 * 32;
    float *recon = buffers.data() + offset;
    offset += batch_size * 3 * 32 * 32;

    // recompute recon and intermediate activations (call forward again but avoid double-copying input)
    // In this simple implementation we'll call forward to ensure buffers filled
    forward(input, batch_size);

    int size = batch_size * 3 * 32 * 32;
    std::vector<float> recon_grad(size);
    loss->computeLoss(recon, in_buf, recon_grad.data(), size);

    // Backprop through decoder
    // Allocate gradient buffers per layer
    std::vector<float> grad_up2(batch_size * 256 * 32 * 32);
    std::vector<float> grad_dec2(batch_size * 256 * 16 * 16);
    std::vector<float> grad_up1(batch_size * 128 * 16 * 16);
    std::vector<float> grad_dec1(batch_size * 128 * 8 * 8);
    std::vector<float> grad_latent(batch_size * 128 * 8 * 8);
    std::vector<float> grad_enc2(batch_size * 128 * 16 * 16);
    std::vector<float> grad_pool1(batch_size * 256 * 16 * 16);
    std::vector<float> grad_enc1(batch_size * 256 * 32 * 32);
    std::vector<float> grad_input(batch_size * 3 * 32 * 32);

    // dec_conv3 backward: output is recon, output_grad is recon_grad
    dec_conv3->backward(up2, recon_grad.data(), grad_up2.data(), batch_size, 32, 32);

    // dec_up2 backward: grad_up2 -> grad_dec2
    dec_up2->backward(grad_up2.data(), grad_dec2.data(), batch_size, 256, 16, 16);

    // dec_conv2 backward
    dec_conv2->backward(up1, grad_dec2.data(), grad_up1.data(), batch_size, 16, 16);

    // dec_up1 backward
    dec_up1->backward(grad_up1.data(), grad_dec1.data(), batch_size, 128, 8, 8);

    // dec_conv1 backward
    dec_conv1->backward(latent, grad_dec1.data(), grad_latent.data(), batch_size, 8, 8);

    // Encoder backwards: pool2 backward (maxpool)
    enc_pool2->backward(grad_latent.data(), grad_enc2.data(), pool2_indices.data(), batch_size, 128, 8, 8, 16, 16);

    enc_relu2->backward(enc2, grad_enc2.data(), grad_enc2.data(), batch_size * 128 * 16 * 16);
    enc_conv2->backward(pool1, grad_enc2.data(), grad_pool1.data(), batch_size, 16, 16);

    enc_pool1->backward(grad_pool1.data(), grad_enc1.data(), pool1_indices.data(), batch_size, 256, 16, 16, 32, 32);
    enc_relu1->backward(enc1, grad_enc1.data(), grad_enc1.data(), batch_size * 256 * 32 * 32);
    enc_conv1->backward(in_buf, grad_enc1.data(), grad_input.data(), batch_size, 32, 32);

    // Finally, grad_input holds gradients wrt input (unused). Weight grads are accumulated in conv layers.
}

void Autoencoder::updateWeights(float learning_rate)
{
    enc_conv1->updateWeights(learning_rate);
    enc_conv2->updateWeights(learning_rate);
    dec_conv1->updateWeights(learning_rate);
    dec_conv2->updateWeights(learning_rate);
    dec_conv3->updateWeights(learning_rate);
}

void Autoencoder::extractFeatures(const float *input, float *features, int num_images)
{
    // Process in batches of max_batch_size
    int batch = max_batch_size;
    int total = num_images;
    int processed = 0;
    while (processed < total)
    {
        int cur = std::min(batch, total - processed);
        forward(input + processed * 3 * 32 * 32, cur);
        // latent is located in buffers at known offset
        size_t offset = 0;
        offset += cur * 3 * 32 * 32;
        offset += cur * 256 * 32 * 32;
        offset += cur * 256 * 16 * 16;
        offset += 0;                                                   // enc2
        float *latent = buffers.data() + offset + cur * 128 * 16 * 16; // ensure correct offset calculation
        // For simplicity compute features by copying latent (N,128,8,8) contiguous
        int feat_size = 128 * 8 * 8;
        for (int i = 0; i < cur; ++i)
        {
            memcpy(features + (processed + i) * feat_size, latent + i * feat_size, sizeof(float) * feat_size);
        }
        processed += cur;
    }
}

bool Autoencoder::saveWeights(const std::string &filename)
{
    std::ofstream ofs(filename, std::ios::binary);
    if (!ofs)
        return false;
    auto writeVec = [&](const std::vector<float> &v)
    { ofs.write(reinterpret_cast<const char *>(v.data()), v.size() * sizeof(float)); };
    writeVec(enc_conv1->weights);
    writeVec(enc_conv1->bias);
    writeVec(enc_conv2->weights);
    writeVec(enc_conv2->bias);
    writeVec(dec_conv1->weights);
    writeVec(dec_conv1->bias);
    writeVec(dec_conv2->weights);
    writeVec(dec_conv2->bias);
    writeVec(dec_conv3->weights);
    writeVec(dec_conv3->bias);
    return true;
}

bool Autoencoder::loadWeights(const std::string &filename)
{
    std::ifstream ifs(filename, std::ios::binary);
    if (!ifs)
        return false;
    auto readVec = [&](std::vector<float> &v)
    { ifs.read(reinterpret_cast<char *>(v.data()), v.size() * sizeof(float)); };
    readVec(enc_conv1->weights);
    readVec(enc_conv1->bias);
    readVec(enc_conv2->weights);
    readVec(enc_conv2->bias);
    readVec(dec_conv1->weights);
    readVec(dec_conv1->bias);
    readVec(dec_conv2->weights);
    readVec(dec_conv2->bias);
    readVec(dec_conv3->weights);
    readVec(dec_conv3->bias);
    return true;
}
