// extract_features.cu
// Extract latent features from CIFAR-10 using trained autoencoder
//
// Dataset Organization:
//   - Training set: 50,000 images -> extract features + save labels for SVM training
//   - Test set: 10,000 images -> extract features + save labels for SVM evaluation
//
// Output files:
//   - {prefix}_train_features.bin: 50,000 x 8192 float features
//   - {prefix}_train_labels.bin: 50,000 int labels (0-9)
//   - {prefix}_test_features.bin: 10,000 x 8192 float features  
//   - {prefix}_test_labels.bin: 10,000 int labels (0-9)

#include "autoencoder.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <string>

// CIFAR-10 Data Loading


struct CIFAR10Dataset {
    std::vector<uint8_t> images;  // [N, 32, 32, 3] in HWC format
    std::vector<uint8_t> labels;  // [N]
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
            
            uint8_t record[3073];
            while (fread(record, 1, 3073, f) == 3073) {
                labels.push_back(record[0]);
                
                // Convert CHW to HWC
                for (int h = 0; h < 32; h++) {
                    for (int w = 0; w < 32; w++) {
                        int pixel_idx = h * 32 + w;
                        images.push_back(record[1 + pixel_idx]);
                        images.push_back(record[1 + 1024 + pixel_idx]);
                        images.push_back(record[1 + 2048 + pixel_idx]);
                    }
                }
            }
            fclose(f);
        }
        
        num_samples = labels.size();
        printf("Loaded %d samples from %s\n", num_samples, train ? "training set" : "test set");
        return num_samples > 0;
    }
    
    void get_batch(half* d_batch, int start_idx, int batch_size, cudaStream_t stream) {
        std::vector<half> h_batch(batch_size * 32 * 32 * 3);
        
        for (int i = 0; i < batch_size; i++) {
            int idx = start_idx + i;
            if (idx >= num_samples) idx = num_samples - 1;  // Pad with last sample
            
            for (int j = 0; j < 32 * 32 * 3; j++) {
                h_batch[i * 32 * 32 * 3 + j] = __float2half(images[idx * 32 * 32 * 3 + j] / 255.0f);
            }
        }
        
        cudaMemcpyAsync(d_batch, h_batch.data(), batch_size * 32 * 32 * 3 * sizeof(half),
                        cudaMemcpyHostToDevice, stream);
    }
};

// Main
int main(int argc, char** argv) {
    printf("=== CIFAR-10 Feature Extraction ===\n\n");
    
    // Parameters
    int batch_size = 64;
    std::string data_dir = "data";
    std::string weights_path = "autoencoder_best.bin";
    std::string output_prefix = "cifar10_features";
    
    // Parse args
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--data") == 0 && i + 1 < argc) {
            data_dir = argv[++i];
        } else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc) {
            weights_path = argv[++i];
        } else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc) {
            output_prefix = argv[++i];
        } else if (strcmp(argv[i], "--batch") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        }
    }
    
    printf("Parameters:\n");
    printf("  Batch size: %d\n", batch_size);
    printf("  Data dir: %s\n", data_dir.c_str());
    printf("  Weights: %s\n", weights_path.c_str());
    printf("  Output prefix: %s\n\n", output_prefix.c_str());
    
    // Load model
    printf("Loading model...\n");
    Autoencoder model(batch_size, 0.0f, weights_path);
    
    int latent_dim = model.latent_size();  // 8*8*128 = 8192
    printf("Latent dimension: %d\n\n", latent_dim);
    
    // Process both train and test sets
    const char* split_names[] = {"train", "test"};
    bool is_train[] = {true, false};
    
    for (int split = 0; split < 2; split++) {
        printf("Processing %s set...\n", split_names[split]);
        
        CIFAR10Dataset dataset;
        if (!dataset.load(data_dir, is_train[split])) {
            printf("Failed to load %s data\n", split_names[split]);
            return 1;
        }
        
        int num_samples = dataset.num_samples;
        int num_batches = (num_samples + batch_size - 1) / batch_size;
        
        // Allocate output buffers
        std::vector<float> all_features(num_samples * latent_dim);
        std::vector<int> all_labels(num_samples);
        
        // Copy labels
        for (int i = 0; i < num_samples; i++) {
            all_labels[i] = dataset.labels[i];
        }
        
        // Extract features batch by batch
        printf("Extracting features...\n");
        
        cudaEvent_t start, stop;
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
        cudaEventRecord(start);
        
        for (int b = 0; b < num_batches; b++) {
            int start_idx = b * batch_size;
            int actual_batch = std::min(batch_size, num_samples - start_idx);
            
            // Load batch
            dataset.get_batch(model.input, start_idx, batch_size, model.stream);
            
            // Encode
            model.encode(false);  // Keep on GPU
            
            // Copy latent features to host
            std::vector<half> h_latent(batch_size * latent_dim);
            cudaMemcpy(h_latent.data(), model.latent, batch_size * latent_dim * sizeof(half),
                      cudaMemcpyDeviceToHost);
            
            // Convert to float and store
            for (int i = 0; i < actual_batch; i++) {
                int sample_idx = start_idx + i;
                for (int j = 0; j < latent_dim; j++) {
                    all_features[sample_idx * latent_dim + j] = __half2float(h_latent[i * latent_dim + j]);
                }
            }
            
            if ((b + 1) % 50 == 0 || b == num_batches - 1) {
                printf("  [%4d/%4d] batches processed\n", b + 1, num_batches);
            }
        }
        
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        float ms;
        cudaEventElapsedTime(&ms, start, stop);
        printf("Extraction time: %.2f ms (%.0f samples/sec)\n", ms, num_samples / (ms / 1000.0f));
        
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
        
        // Save features to binary file
        std::string features_file = output_prefix + "_" + split_names[split] + "_features.bin";
        std::string labels_file = output_prefix + "_" + split_names[split] + "_labels.bin";
        
        printf("Saving to %s ...\n", features_file.c_str());
        FILE* f_features = fopen(features_file.c_str(), "wb");
        if (f_features) {
            // Header: num_samples, latent_dim
            fwrite(&num_samples, sizeof(int), 1, f_features);
            fwrite(&latent_dim, sizeof(int), 1, f_features);
            // Data
            fwrite(all_features.data(), sizeof(float), num_samples * latent_dim, f_features);
            fclose(f_features);
        }
        
        printf("Saving to %s ...\n", labels_file.c_str());
        FILE* f_labels = fopen(labels_file.c_str(), "wb");
        if (f_labels) {
            fwrite(&num_samples, sizeof(int), 1, f_labels);
            fwrite(all_labels.data(), sizeof(int), num_samples, f_labels);
            fclose(f_labels);
        }
        
        printf("%s set: %d samples, %d features each\n\n", split_names[split], num_samples, latent_dim);
    }
    
    printf("=== Feature Extraction Complete ===\n");
    printf("Files created:\n");
    printf("  %s_train_features.bin\n", output_prefix.c_str());
    printf("  %s_train_labels.bin\n", output_prefix.c_str());
    printf("  %s_test_features.bin\n", output_prefix.c_str());
    printf("  %s_test_labels.bin\n", output_prefix.c_str());
    
    return 0;
}
