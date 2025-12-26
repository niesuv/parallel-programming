// train_AE.cu
// Training script for CIFAR-10 autoencoder
//
// Dataset Organization:
//   - Uses all 50,000 CIFAR-10 training images
//   - Labels are IGNORED (unsupervised reconstruction task)
//   - Target = Input (autoencoder learns to reconstruct images)
//
// The trained encoder will later be used to extract features for SVM classification.

#include "autoencoder.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>
#include <algorithm>
#include <random>

// ============================================================================
// CIFAR-10 Data Loading (Images only, labels ignored for autoencoder)
// ============================================================================

struct CIFAR10Dataset {
    std::vector<uint8_t> images;  // [N, 32, 32, 3] in HWC format
    std::vector<uint8_t> labels;  // [N] - loaded but NOT used for training
    int num_samples;
    
    bool load(const std::string& data_dir, bool train = true) {
        std::vector<std::string> files;
        if (train) {
            files = {
                data_dir + "/data_batch_1.bin",
                data_dir + "/data_batch_2.bin",
                data_dir + "/data_batch_3.bin",
                data_dir + "/data_batch_4.bin",
                data_dir + "/data_batch_5.bin"
            };
        } else {
            files = {data_dir + "/test_batch.bin"};
        }
        
        images.clear();
        labels.clear();
        
        for (const auto& filename : files) {
            FILE* f = fopen(filename.c_str(), "rb");
            if (!f) {
                printf("Failed to open: %s\n", filename.c_str());
                return false;
            }
            
            // Each record: 1 byte label + 3072 bytes image (CHW format in file)
            uint8_t record[3073];
            while (fread(record, 1, 3073, f) == 3073) {
                labels.push_back(record[0]);
                
                // Convert CHW to HWC
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        int pixel_idx = h * 32 + w;
                        images.push_back(record[1 + pixel_idx]);           // R
                        images.push_back(record[1 + 1024 + pixel_idx]);    // G
                        images.push_back(record[1 + 2048 + pixel_idx]);    // B
                    }
                }
            }
            fclose(f);
        }
        
        num_samples = labels.size();
        printf("Loaded %d samples from %s\n", num_samples, train ? "training set" : "test set");
        return num_samples > 0;
    }
    
    // Get a batch of images as FP16, normalized to [0, 1]
    void get_batch(half* d_batch, const std::vector<int>& indices, cudaStream_t stream) {
        int batch_size = indices.size();
        std::vector<half> h_batch(batch_size * 32 * 32 * 3);
        
        for (int i = 0; i < batch_size; i++) {
            int idx = indices[i];
            for (int j = 0; j < 32 * 32 * 3; j++) {
                h_batch[i * 32 * 32 * 3 + j] = __float2half(images[idx * 32 * 32 * 3 + j] / 255.0f);
            }
        }
        
        cudaMemcpyAsync(d_batch, h_batch.data(), batch_size * 32 * 32 * 3 * sizeof(half),
                        cudaMemcpyHostToDevice, stream);
    }
};

// ============================================================================
// Main Training Loop
// ============================================================================

int main(int argc, char** argv) {
    printf("=== CIFAR-10 Autoencoder Training ===\n\n");
    
    // Hyperparameters
    int batch_size = 64;
    float learning_rate = 0.003f;
    int num_epochs = 30;
    std::string data_dir = "data";
    std::string weights_dir = "weights";
    std::string load_weights_path = "";
    
    // Parse command line args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--weights_dir") == 0 && i + 1 < argc) {
            weights_dir = argv[++i];
        } else if (strcmp(argv[i], "--load") == 0 && i + 1 < argc) {
            load_weights_path = argv[++i];
        }
    }
    
    // Build weight file paths
    std::string best_weights_path = weights_dir + "/autoencoder_best.bin";
    std::string final_weights_path = weights_dir + "/autoencoder_final.bin";
    
    printf("Hyperparameters:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Learning rate: %f\n", learning_rate);
    printf("  Epochs: %d\n", num_epochs);
    printf("  Data dir: %s\n", data_dir.c_str());
    printf("  Weights dir: %s\n", weights_dir.c_str());
    if (!load_weights_path.empty()) {
        printf("  Load weights: %s\n", load_weights_path.c_str());
    }
    printf("\n");
    
    // Load CIFAR-10
    CIFAR10Dataset train_data;
    if (!train_data.load(data_dir, true)) {
        printf("Failed to load training data from %s\n", data_dir.c_str());
        return 1;
    }
    
    // Create model (optionally load weights)
    printf("\nCreating model...\n");
    Autoencoder model(batch_size, learning_rate, load_weights_path);
    printf("Model created.\n\n");
    
    // Training
    int num_batches = train_data.num_samples / batch_size;
    printf("Training: %d samples, %d batches per epoch\n\n", train_data.num_samples, num_batches);
    
    // Create index array for shuffling
    std::vector<int> indices(train_data.num_samples);
    for (int i = 0; i < train_data.num_samples; i++) indices[i] = i;
    
    std::mt19937 rng(42);
    
    cudaEvent_t epoch_start, epoch_end;
    cudaEventCreate(&epoch_start);
    cudaEventCreate(&epoch_end);
    
    float best_loss = 1e10f;
    int epochs_without_improvement = 0;
    float current_lr = learning_rate;
    
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        // Update model's learning rate
        model.learning_rate = current_lr;
        
        // Shuffle indices
        std::shuffle(indices.begin(), indices.end(), rng);
        
        float epoch_loss = 0.0f;
        int num_batches_processed = 0;
        
        cudaEventRecord(epoch_start);
        
        for (int batch = 0; batch < num_batches; batch++) {
            // Get batch indices
            std::vector<int> batch_indices(indices.begin() + batch * batch_size,
                                           indices.begin() + (batch + 1) * batch_size);
            
            // Load batch to GPU
            train_data.get_batch(model.input, batch_indices, model.stream);
            
            // For autoencoder: target = input
            cudaMemcpyAsync(model.target, model.input, 
                           batch_size * 32 * 32 * 3 * sizeof(half),
                           cudaMemcpyDeviceToDevice, model.stream);
            
            // Forward + Backward
            model.forward();
            float loss = model.backward();
            
            epoch_loss += loss;
            num_batches_processed++;
            
            // Print progress
            if ((batch + 1) % 100 == 0 || batch == num_batches - 1) {
                printf("  Epoch %d [%4d/%4d] loss: %.6f\n", 
                       epoch + 1, batch + 1, num_batches, loss);
            }
        }
        
        cudaEventRecord(epoch_end);
        cudaEventSynchronize(epoch_end);
        
        float epoch_ms;
        cudaEventElapsedTime(&epoch_ms, epoch_start, epoch_end);
        
        float avg_loss = epoch_loss / num_batches_processed;
        float throughput = (float)(num_batches_processed * batch_size) / (epoch_ms / 1000.0f);
        
        printf("Epoch %d complete: avg_loss=%.6f, lr=%.6f, time=%.2fs, throughput=%.0f img/s",
               epoch + 1, avg_loss, current_lr, epoch_ms / 1000.0f, throughput);
        
        // Save best weights and check for improvement
        if (avg_loss < best_loss) {
            best_loss = avg_loss;
            epochs_without_improvement = 0;
            model.save_weights(best_weights_path);
            printf(" [NEW BEST - saved]");
        } else {
            epochs_without_improvement++;
            // Reduce LR if no improvement for 3 epochs
            if (epochs_without_improvement >= 3) {
                current_lr *= 0.5f;
                epochs_without_improvement = 0;
                printf(" [LR decay -> %.6f]", current_lr);
                
                // Stop if LR too small
                if (current_lr < 1e-6f) {
                    printf("\nLR too small, stopping early.\n");
                    break;
                }
            }
        }
        printf("\n\n");
    }
    
    cudaEventDestroy(epoch_start);
    cudaEventDestroy(epoch_end);
    
    // Save final weights
    printf("Saving final weights to '%s'...\n", final_weights_path.c_str());
    model.save_weights(final_weights_path);
    printf("Best loss achieved: %.6f (saved to '%s')\n", best_loss, best_weights_path.c_str());
    printf("Done!\n");
    
    // Check for CUDA errors
    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        printf("\nCUDA Error: %s\n", cudaGetErrorString(err));
        return 1;
    }
    
    printf("\n=== Training Complete ===\n");
    return 0;
}
