#ifndef AUTOENCODER_H
#define AUTOENCODER_H

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <string>

// Layer Structures
struct Conv2DLayer {
    int C_in, C_out, H, W;
    half* weight;
    half* bias;
    float* master_weight;
    float* master_bias;
    float* grad_weight;
    float* grad_bias;
    half* input_saved;
    half* conv_output;
    half* output;
    half* grad_input;
    half* temp_buffer;  // For two-pass backward
    bool has_relu;
    
    int weight_size() const { return C_out * 9 * C_in; }
    int bias_size() const { return C_out; }
};

struct MaxPool2DLayer {
    int N, H, W, C;
    int8_t* max_indices;
    half* output;
    half* grad_input;
};

struct Upsample2DLayer {
    int N, H, W, C;
    half* output;
    half* grad_input;
};

// ============================================================================
// Autoencoder Model
// ============================================================================

class Autoencoder {
public:
    int batch_size;
    float learning_rate;
    
    // Encoder layers
    Conv2DLayer conv1;      // (N, 32, 32, 3) -> (N, 32, 32, 256) + ReLU
    MaxPool2DLayer pool1;   // (N, 32, 32, 256) -> (N, 16, 16, 256)
    Conv2DLayer conv2;      // (N, 16, 16, 256) -> (N, 16, 16, 128) + ReLU
    MaxPool2DLayer pool2;   // (N, 16, 16, 128) -> (N, 8, 8, 128)
    Conv2DLayer conv3;      // (N, 8, 8, 128) -> (N, 8, 8, 128) + ReLU [latent]
    
    // Decoder layers
    Upsample2DLayer up1;    // (N, 8, 8, 128) -> (N, 16, 16, 128)
    Conv2DLayer conv4;      // (N, 16, 16, 128) -> (N, 16, 16, 256) + ReLU
    Upsample2DLayer up2;    // (N, 16, 16, 256) -> (N, 32, 32, 256)
    Conv2DLayer conv5;      // (N, 32, 32, 256) -> (N, 32, 32, 3) [no activation]
    
    // I/O buffers
    half* input;            // Input images (N, 32, 32, 3)
    half* output;           // Reconstructed images (N, 32, 32, 3)
    half* target;           // Target images for loss
    half* latent;           // Latent features (N, 8, 8, 128) - for encode()
    
    // Loss computation
    float* loss;
    float* loss_partial;
    half* grad_output;
    
    cudaStream_t stream;
    
    // Constructors / Destructor
    // Create new model (optionally load weights)
    Autoencoder(int batch, float lr = 0.01f, const std::string& weights_path = "");
    
    ~Autoencoder();
    
    // Core Methods
    // Forward pass (full autoencoder)
    void forward();
    
    // Backward pass + SGD update, returns loss
    float backward();
    
    // Encode only (get latent features)
    // If save_features=true, copies latent to CPU and returns pointer (caller must free)
    // If save_features=false, returns nullptr (latent stays on GPU in this->latent)
    half* encode(bool save_features = false);
    
    // Weight I/O
    // Save weights to binary file
    void save_weights(const std::string& filename);
    
    // Load weights from binary file
    bool load_weights(const std::string& filename);

    // Utility
    // Get latent size per sample
    int latent_size() const { return 8 * 8 * 128; }
    
    // Get total latent buffer size
    int latent_buffer_size() const { return batch_size * latent_size(); }
    
private:
    void allocate_layers();
    void free_layers();
    void init_weights();
};

#endif // AUTOENCODER_H
