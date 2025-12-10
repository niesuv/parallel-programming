#include "layers.h"
#include <stdio.h>

// ReLU forward pass (in-place)
void relu_forward_cpu(float* data, int size) {
    for (int i = 0; i < size; i++) {
        if (data[i] < 0.0f) {
            data[i] = 0.0f;
        }
    }
}

// ReLU backward pass
void relu_backward_cpu(const float* input, const float* d_output, float* d_input, int size) {
    for (int i = 0; i < size; i++) {
        d_input[i] = (input[i] > 0.0f) ? d_output[i] : 0.0f;
    }
}
