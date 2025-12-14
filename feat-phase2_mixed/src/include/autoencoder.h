#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <string>

#include "gpu_layer.h"

/*
    Autoencoder runs on GPU
    - Activations: FP16
    - Weights: FP16 forward + FP32 master
    - Loss: FP32
*/

class Autoencoder {
public:
    Autoencoder();

    // FORWARD (Inference)
    GPUTensor4D forward_fp16(const GPUTensor4D &input_fp16) const;

    // TRAIN STEP (Mixed Precision)
    float train_step_fp16(const GPUTensor4D &input_fp16,
                            const GPUTensor4D &target_fp16,
                            float learning_rate);

    // ENCODER ONLY
    GPUTensor4D encode_fp16(const GPUTensor4D &input_fp16) const;

    // WEIGHT I/O
    bool save_weights(const std::string &path) const;
    bool load_weights(const std::string &path);

private:
    // ENCODER
    GPUConv2DLayer conv1_;
    GPUReLULayer   relu1_;
    GPUMaxPool2DLayer pool1_;

    GPUConv2DLayer conv2_;
    GPUReLULayer   relu2_;
    GPUMaxPool2DLayer pool2_;

    // BOTTLENECK
    GPUConv2DLayer conv3_;
    GPUReLULayer   relu3_;
    GPUUpSample2DLayer up1_;

    // DECODER
    GPUConv2DLayer conv4_;
    GPUReLULayer   relu4_;
    GPUUpSample2DLayer up2_;

    GPUConv2DLayer conv5_;

    // LOSS SCALER (AMP)
    mutable GPULossScaler loss_scaler_;
};

#endif // AUTOENCODER_H
