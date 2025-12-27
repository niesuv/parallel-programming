/**
 * Naive CPU Autoencoder - Simple as possible, pure C++
 * No CUDA, no optimizations, just straightforward loops
 * 
 * Build:
 *   g++ -O2 -o naive_cpu_autoencoder naive_cpu_autoencoder.cpp -lm
 * 
 * Run:
 *   ./naive_cpu_autoencoder --data data --epochs 10 --lr 0.001 --batch 32
 */

#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <string>
#include <chrono>
#include <algorithm>

// ============================================================================
// Tensor Class - Simple dynamic array
// ============================================================================

class Tensor {
public:
    std::vector<float> data;
    std::vector<int> shape;
    int size;
    
    Tensor() : size(0) {}
    
    Tensor(std::vector<int> dims) : shape(dims) {
        size = 1;
        for (int d : dims) size *= d;
        data.resize(size, 0.0f);
    }
    
    void resize(std::vector<int> dims) {
        shape = dims;
        size = 1;
        for (int d : dims) size *= d;
        data.resize(size, 0.0f);
    }
    
    void zero() {
        std::fill(data.begin(), data.end(), 0.0f);
    }
    
    void random_init(float scale = 0.1f) {
        for (int i = 0; i < size; i++) {
            data[i] = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        }
    }
    
    float& operator[](int idx) { return data[idx]; }
    const float& operator[](int idx) const { return data[idx]; }
    
    float* ptr() { return data.data(); }
    const float* ptr() const { return data.data(); }
};

// ============================================================================
// Naive Operations - Simple loops
// ============================================================================

// ----- Conv2D Forward (3x3, stride=1, padding=1) -----
void naive_conv2d_forward(
    const Tensor& input,   // [N, H, W, C]
    const Tensor& weight,  // [K, 3, 3, C]
    const Tensor& bias,    // [K]
    Tensor& output,        // [N, H, W, K]
    int N, int H, int W, int C, int K
) {
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int k = 0; k < K; k++) {
                    float sum = 0.0f;
                    
                    for (int c = 0; c < C; c++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int ih = h + kh - 1;
                                int iw = w + kw - 1;
                                
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int in_idx = ((n * H + ih) * W + iw) * C + c;
                                    int w_idx = ((k * 3 + kh) * 3 + kw) * C + c;
                                    sum += input[in_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    
                    int out_idx = ((n * H + h) * W + w) * K + k;
                    output[out_idx] = sum + bias[k];
                }
            }
        }
    }
}

// ----- Conv2D Backward Input -----
void naive_conv2d_backward_input(
    const Tensor& grad_output,  // [N, H, W, K]
    const Tensor& weight,       // [K, 3, 3, C]
    Tensor& grad_input,         // [N, H, W, C]
    int N, int H, int W, int C, int K
) {
    grad_input.zero();
    
    for (int n = 0; n < N; n++) {
        for (int h = 0; h < H; h++) {
            for (int w = 0; w < W; w++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    
                    for (int k = 0; k < K; k++) {
                        for (int kh = 0; kh < 3; kh++) {
                            for (int kw = 0; kw < 3; kw++) {
                                int oh = h - kh + 1;
                                int ow = w - kw + 1;
                                
                                if (oh >= 0 && oh < H && ow >= 0 && ow < W) {
                                    int go_idx = ((n * H + oh) * W + ow) * K + k;
                                    int w_idx = ((k * 3 + kh) * 3 + kw) * C + c;
                                    sum += grad_output[go_idx] * weight[w_idx];
                                }
                            }
                        }
                    }
                    
                    int gi_idx = ((n * H + h) * W + w) * C + c;
                    grad_input[gi_idx] = sum;
                }
            }
        }
    }
}

// ----- Conv2D Backward Weight -----
void naive_conv2d_backward_weight(
    const Tensor& input,        // [N, H, W, C]
    const Tensor& grad_output,  // [N, H, W, K]
    Tensor& grad_weight,        // [K, 3, 3, C]
    int N, int H, int W, int C, int K
) {
    grad_weight.zero();
    
    for (int k = 0; k < K; k++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    
                    for (int n = 0; n < N; n++) {
                        for (int h = 0; h < H; h++) {
                            for (int w = 0; w < W; w++) {
                                int ih = h + kh - 1;
                                int iw = w + kw - 1;
                                
                                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                                    int in_idx = ((n * H + ih) * W + iw) * C + c;
                                    int go_idx = ((n * H + h) * W + w) * K + k;
                                    sum += input[in_idx] * grad_output[go_idx];
                                }
                            }
                        }
                    }
                    
                    int gw_idx = ((k * 3 + kh) * 3 + kw) * C + c;
                    grad_weight[gw_idx] = sum;
                }
            }
        }
    }
}

// ----- Conv2D Backward Bias -----
void naive_conv2d_backward_bias(
    const Tensor& grad_output,  // [N, H, W, K]
    Tensor& grad_bias,          // [K]
    int N, int H, int W, int K
) {
    grad_bias.zero();
    
    for (int k = 0; k < K; k++) {
        float sum = 0.0f;
        for (int n = 0; n < N; n++) {
            for (int h = 0; h < H; h++) {
                for (int w = 0; w < W; w++) {
                    sum += grad_output[((n * H + h) * W + w) * K + k];
                }
            }
        }
        grad_bias[k] = sum;
    }
}

// ----- ReLU Forward -----
void naive_relu_forward(const Tensor& input, Tensor& output) {
    for (int i = 0; i < input.size; i++) {
        output[i] = std::max(0.0f, input[i]);
    }
}

// ----- ReLU Backward -----
void naive_relu_backward(const Tensor& grad_output, const Tensor& input, Tensor& grad_input) {
    for (int i = 0; i < input.size; i++) {
        grad_input[i] = input[i] > 0.0f ? grad_output[i] : 0.0f;
    }
}

// ----- MaxPool2D Forward (2x2, stride=2) -----
void naive_maxpool_forward(
    const Tensor& input,   // [N, H, W, C]
    Tensor& output,        // [N, H/2, W/2, C]
    std::vector<int>& indices,
    int N, int H, int W, int C
) {
    int H_out = H / 2;
    int W_out = W / 2;
    
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                for (int c = 0; c < C; c++) {
                    int h_in = h_out * 2;
                    int w_in = w_out * 2;
                    
                    float max_val = -1e10f;
                    int max_idx = 0;
                    
                    for (int dh = 0; dh < 2; dh++) {
                        for (int dw = 0; dw < 2; dw++) {
                            int in_idx = ((n * H + h_in + dh) * W + w_in + dw) * C + c;
                            if (input[in_idx] > max_val) {
                                max_val = input[in_idx];
                                max_idx = in_idx;
                            }
                        }
                    }
                    
                    int out_idx = ((n * H_out + h_out) * W_out + w_out) * C + c;
                    output[out_idx] = max_val;
                    indices[out_idx] = max_idx;
                }
            }
        }
    }
}

// ----- MaxPool2D Backward -----
void naive_maxpool_backward(
    const Tensor& grad_output,
    const std::vector<int>& indices,
    Tensor& grad_input,
    int N, int H, int W, int C
) {
    grad_input.zero();
    
    int H_out = H / 2;
    int W_out = W / 2;
    
    for (int i = 0; i < N * H_out * W_out * C; i++) {
        grad_input[indices[i]] += grad_output[i];
    }
}

// ----- Upsample2D Forward (2x, nearest neighbor) -----
void naive_upsample_forward(
    const Tensor& input,   // [N, H, W, C]
    Tensor& output,        // [N, H*2, W*2, C]
    int N, int H, int W, int C
) {
    int H_out = H * 2;
    int W_out = W * 2;
    
    for (int n = 0; n < N; n++) {
        for (int h_out = 0; h_out < H_out; h_out++) {
            for (int w_out = 0; w_out < W_out; w_out++) {
                for (int c = 0; c < C; c++) {
                    int h_in = h_out / 2;
                    int w_in = w_out / 2;
                    
                    int in_idx = ((n * H + h_in) * W + w_in) * C + c;
                    int out_idx = ((n * H_out + h_out) * W_out + w_out) * C + c;
                    output[out_idx] = input[in_idx];
                }
            }
        }
    }
}

// ----- Upsample2D Backward -----
void naive_upsample_backward(
    const Tensor& grad_output,  // [N, H*2, W*2, C]
    Tensor& grad_input,         // [N, H, W, C]
    int N, int H, int W, int C
) {
    grad_input.zero();
    
    int H_out = H * 2;
    int W_out = W * 2;
    
    for (int n = 0; n < N; n++) {
        for (int h_in = 0; h_in < H; h_in++) {
            for (int w_in = 0; w_in < W; w_in++) {
                for (int c = 0; c < C; c++) {
                    float sum = 0.0f;
                    for (int dh = 0; dh < 2; dh++) {
                        for (int dw = 0; dw < 2; dw++) {
                            int h_out = h_in * 2 + dh;
                            int w_out = w_in * 2 + dw;
                            int out_idx = ((n * H_out + h_out) * W_out + w_out) * C + c;
                            sum += grad_output[out_idx];
                        }
                    }
                    int in_idx = ((n * H + h_in) * W + w_in) * C + c;
                    grad_input[in_idx] = sum;
                }
            }
        }
    }
}

// ----- MSE Loss -----
float naive_mse_loss(const Tensor& pred, const Tensor& target) {
    float sum = 0.0f;
    for (int i = 0; i < pred.size; i++) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }
    return sum / pred.size;
}

// ----- MSE Loss Backward -----
void naive_mse_backward(const Tensor& pred, const Tensor& target, Tensor& grad) {
    for (int i = 0; i < pred.size; i++) {
        grad[i] = 2.0f * (pred[i] - target[i]) / pred.size;
    }
}

// ----- SGD Update -----
void naive_sgd_update(Tensor& weight, const Tensor& grad, float lr) {
    for (int i = 0; i < weight.size; i++) {
        weight[i] -= lr * grad[i];
    }
}

// ============================================================================
// Naive CPU Autoencoder Class
// ============================================================================

class NaiveCPUAutoencoder {
public:
    int batch_size;
    float learning_rate;
    
    // Weights and biases
    Tensor w1, b1;  // Conv1: 3 -> 256
    Tensor w2, b2;  // Conv2: 256 -> 128
    Tensor w3, b3;  // Conv3: 128 -> 128
    Tensor w4, b4;  // Conv4: 128 -> 256
    Tensor w5, b5;  // Conv5: 256 -> 3
    
    // Gradients
    Tensor dw1, db1, dw2, db2, dw3, db3, dw4, db4, dw5, db5;
    
    // Activations
    Tensor a0;  // Input
    Tensor z1, a1, p1;
    Tensor z2, a2, p2;
    Tensor z3, a3;
    Tensor u1, z4, a4;
    Tensor u2, z5;
    
    // Pool indices
    std::vector<int> idx1, idx2;
    
    // Activation gradients
    Tensor dz5, du2, da4, dz4, du1;
    Tensor da3, dz3, dp2, da2, dz2;
    Tensor dp1, da1, dz1;
    
    NaiveCPUAutoencoder(int batch, float lr) : batch_size(batch), learning_rate(lr) {
        printf("Creating Naive CPU Autoencoder...\n");
        printf("  Batch size: %d\n", batch_size);
        printf("  Learning rate: %f\n", learning_rate);
        
        int N = batch_size;
        
        // Initialize weights
        w1.resize({256, 3, 3, 3}); w1.random_init(sqrtf(2.0f / 27));
        w2.resize({128, 3, 3, 256}); w2.random_init(sqrtf(2.0f / 2304));
        w3.resize({128, 3, 3, 128}); w3.random_init(sqrtf(2.0f / 1152));
        w4.resize({256, 3, 3, 128}); w4.random_init(sqrtf(2.0f / 1152));
        w5.resize({3, 3, 3, 256}); w5.random_init(sqrtf(2.0f / 2304));
        
        b1.resize({256}); b2.resize({128}); b3.resize({128});
        b4.resize({256}); b5.resize({3});
        
        // Gradients
        dw1.resize({256, 3, 3, 3}); dw2.resize({128, 3, 3, 256});
        dw3.resize({128, 3, 3, 128}); dw4.resize({256, 3, 3, 128});
        dw5.resize({3, 3, 3, 256});
        db1.resize({256}); db2.resize({128}); db3.resize({128});
        db4.resize({256}); db5.resize({3});
        
        // Activations
        a0.resize({N, 32, 32, 3});
        z1.resize({N, 32, 32, 256}); a1.resize({N, 32, 32, 256});
        p1.resize({N, 16, 16, 256}); idx1.resize(N * 16 * 16 * 256);
        z2.resize({N, 16, 16, 128}); a2.resize({N, 16, 16, 128});
        p2.resize({N, 8, 8, 128}); idx2.resize(N * 8 * 8 * 128);
        z3.resize({N, 8, 8, 128}); a3.resize({N, 8, 8, 128});
        u1.resize({N, 16, 16, 128});
        z4.resize({N, 16, 16, 256}); a4.resize({N, 16, 16, 256});
        u2.resize({N, 32, 32, 256});
        z5.resize({N, 32, 32, 3});
        
        // Activation gradients
        dz5.resize({N, 32, 32, 3});
        du2.resize({N, 32, 32, 256});
        da4.resize({N, 16, 16, 256}); dz4.resize({N, 16, 16, 256});
        du1.resize({N, 16, 16, 128});
        da3.resize({N, 8, 8, 128}); dz3.resize({N, 8, 8, 128});
        dp2.resize({N, 8, 8, 128});
        da2.resize({N, 16, 16, 128}); dz2.resize({N, 16, 16, 128});
        dp1.resize({N, 16, 16, 256});
        da1.resize({N, 32, 32, 256}); dz1.resize({N, 32, 32, 256});
        
        printf("  Model created.\n\n");
    }
    
    float forward(const float* input) {
        int N = batch_size;
        
        // Copy input
        memcpy(a0.ptr(), input, N * 32 * 32 * 3 * sizeof(float));
        
        // Encoder
        naive_conv2d_forward(a0, w1, b1, z1, N, 32, 32, 3, 256);
        naive_relu_forward(z1, a1);
        naive_maxpool_forward(a1, p1, idx1, N, 32, 32, 256);
        
        naive_conv2d_forward(p1, w2, b2, z2, N, 16, 16, 256, 128);
        naive_relu_forward(z2, a2);
        naive_maxpool_forward(a2, p2, idx2, N, 16, 16, 128);
        
        naive_conv2d_forward(p2, w3, b3, z3, N, 8, 8, 128, 128);
        naive_relu_forward(z3, a3);
        
        // Decoder
        naive_upsample_forward(a3, u1, N, 8, 8, 128);
        naive_conv2d_forward(u1, w4, b4, z4, N, 16, 16, 128, 256);
        naive_relu_forward(z4, a4);
        
        naive_upsample_forward(a4, u2, N, 16, 16, 256);
        naive_conv2d_forward(u2, w5, b5, z5, N, 32, 32, 256, 3);
        
        return naive_mse_loss(z5, a0);
    }
    
    void backward() {
        int N = batch_size;
        
        // MSE gradient
        naive_mse_backward(z5, a0, dz5);
        
        // Conv5 backward
        naive_conv2d_backward_weight(u2, dz5, dw5, N, 32, 32, 256, 3);
        naive_conv2d_backward_bias(dz5, db5, N, 32, 32, 3);
        naive_conv2d_backward_input(dz5, w5, du2, N, 32, 32, 256, 3);
        
        // Upsample2 backward
        naive_upsample_backward(du2, da4, N, 16, 16, 256);
        
        // ReLU4 + Conv4 backward
        naive_relu_backward(da4, z4, dz4);
        naive_conv2d_backward_weight(u1, dz4, dw4, N, 16, 16, 128, 256);
        naive_conv2d_backward_bias(dz4, db4, N, 16, 16, 256);
        naive_conv2d_backward_input(dz4, w4, du1, N, 16, 16, 128, 256);
        
        // Upsample1 backward
        naive_upsample_backward(du1, da3, N, 8, 8, 128);
        
        // ReLU3 + Conv3 backward
        naive_relu_backward(da3, z3, dz3);
        naive_conv2d_backward_weight(p2, dz3, dw3, N, 8, 8, 128, 128);
        naive_conv2d_backward_bias(dz3, db3, N, 8, 8, 128);
        naive_conv2d_backward_input(dz3, w3, dp2, N, 8, 8, 128, 128);
        
        // Pool2 backward
        naive_maxpool_backward(dp2, idx2, da2, N, 16, 16, 128);
        
        // ReLU2 + Conv2 backward
        naive_relu_backward(da2, z2, dz2);
        naive_conv2d_backward_weight(p1, dz2, dw2, N, 16, 16, 256, 128);
        naive_conv2d_backward_bias(dz2, db2, N, 16, 16, 128);
        naive_conv2d_backward_input(dz2, w2, dp1, N, 16, 16, 256, 128);
        
        // Pool1 backward
        naive_maxpool_backward(dp1, idx1, da1, N, 32, 32, 256);
        
        // ReLU1 + Conv1 backward
        naive_relu_backward(da1, z1, dz1);
        naive_conv2d_backward_weight(a0, dz1, dw1, N, 32, 32, 3, 256);
        naive_conv2d_backward_bias(dz1, db1, N, 32, 32, 256);
    }
    
    void update() {
        naive_sgd_update(w1, dw1, learning_rate);
        naive_sgd_update(b1, db1, learning_rate);
        naive_sgd_update(w2, dw2, learning_rate);
        naive_sgd_update(b2, db2, learning_rate);
        naive_sgd_update(w3, dw3, learning_rate);
        naive_sgd_update(b3, db3, learning_rate);
        naive_sgd_update(w4, dw4, learning_rate);
        naive_sgd_update(b4, db4, learning_rate);
        naive_sgd_update(w5, dw5, learning_rate);
        naive_sgd_update(b5, db5, learning_rate);
    }
    
    float train_step(const float* input) {
        float loss = forward(input);
        backward();
        update();
        return loss;
    }
};

// ============================================================================
// CIFAR-10 Data Loader
// ============================================================================

class CIFAR10Loader {
public:
    std::vector<float> images;
    std::vector<int> labels;
    int num_samples;
    int current_idx;
    
    CIFAR10Loader(const std::string& data_dir) : current_idx(0), num_samples(0) {
        printf("Loading CIFAR-10 from %s...\n", data_dir.c_str());
        
        for (int batch = 1; batch <= 5; batch++) {
            char filename[256];
            snprintf(filename, sizeof(filename), "%s/data_batch_%d.bin", data_dir.c_str(), batch);
            
            FILE* f = fopen(filename, "rb");
            if (!f) {
                printf("Warning: Could not open %s\n", filename);
                continue;
            }
            
            for (int i = 0; i < 10000; i++) {
                unsigned char label;
                unsigned char pixels[3072];
                
                if (fread(&label, 1, 1, f) != 1) break;
                if (fread(pixels, 1, 3072, f) != 3072) break;
                
                labels.push_back(label);
                
                // CHW to HWC, normalize to [0,1]
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        for (int c = 0; c < 3; c++) {
                            float val = pixels[c * 1024 + h * 32 + w] / 255.0f;
                            images.push_back(val);
                        }
                    }
                }
                num_samples++;
            }
            fclose(f);
        }
        
        printf("  Loaded %d samples\n\n", num_samples);
    }
    
    void get_batch(float* batch, int batch_size) {
        for (int i = 0; i < batch_size; i++) {
            int idx = (current_idx + i) % num_samples;
            memcpy(batch + i * 3072, images.data() + idx * 3072, 3072 * sizeof(float));
        }
        current_idx = (current_idx + batch_size) % num_samples;
    }
    
    void shuffle() {
        current_idx = rand() % num_samples;
    }
};

// ============================================================================
// Main
// ============================================================================

int main(int argc, char** argv) {
    printf("=== Naive CPU Autoencoder Training ===\n\n");
    
    int batch_size = 16;  // Smaller default for CPU
    float learning_rate = 0.001f;
    int num_epochs = 5;
    int max_train = -1;  // -1 means use all
    std::string data_dir = "data";
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--max_train") == 0 && i + 1 < argc) {
            max_train = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--help") == 0 || strcmp(argv[i], "-h") == 0) {
            printf("Usage: %s [options]\n", argv[0]);
            printf("Options:\n");
            printf("  --batch N      Batch size (default: 16)\n");
            printf("  --lr F         Learning rate (default: 0.001)\n");
            printf("  --epochs N     Number of epochs (default: 5)\n");
            printf("  --data DIR     Data directory (default: data)\n");
            printf("  --max_train N  Max training samples, -1 for all (default: -1)\n");
            printf("  --help         Show this help\n");
            return 0;
        }
    }
    
    printf("Hyperparameters:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Learning rate: %f\n", learning_rate);
    printf("  Epochs: %d\n", num_epochs);
    printf("  Data dir: %s\n", data_dir.c_str());
    printf("  Max train: %d\n\n", max_train);
    
    CIFAR10Loader loader(data_dir);
    if (loader.num_samples == 0) {
        printf("Error: No data loaded!\n");
        return 1;
    }
    
    // Limit training samples if requested
    if (max_train > 0 && max_train < loader.num_samples) {
        loader.num_samples = max_train;
        printf("Limited to %d training samples\n\n", max_train);
    }
    
    NaiveCPUAutoencoder model(batch_size, learning_rate);
    
    std::vector<float> h_batch(batch_size * 32 * 32 * 3);
    int num_batches = loader.num_samples / batch_size;
    
    printf("Training: %d samples, %d batches per epoch\n", loader.num_samples, num_batches);
    printf("WARNING: CPU training is VERY slow. Consider using GPU version.\n\n");
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        loader.shuffle();
        float epoch_loss = 0.0f;
        
        auto start = std::chrono::high_resolution_clock::now();
        
        for (int batch = 0; batch < num_batches; batch++) {
            loader.get_batch(h_batch.data(), batch_size);
            
            float loss = model.train_step(h_batch.data());
            epoch_loss += loss;
            
            if ((batch + 1) % 10 == 0 || batch == num_batches - 1) {
                printf("  Epoch %d [%4d/%4d] loss: %.6f\n", 
                       epoch + 1, batch + 1, num_batches, loss);
            }
        }
        
        auto end = std::chrono::high_resolution_clock::now();
        double epoch_sec = std::chrono::duration<double>(end - start).count();
        
        float avg_loss = epoch_loss / num_batches;
        float throughput = (float)(num_batches * batch_size) / epoch_sec;
        
        printf("Epoch %d complete: avg_loss=%.6f, time=%.1fs, throughput=%.1f img/s\n\n",
               epoch + 1, avg_loss, epoch_sec, throughput);
    }
    
    printf("=== Training Complete ===\n");
    
    return 0;
}
