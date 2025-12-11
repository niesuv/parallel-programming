#include "gpu_autoencoder.h"
#include "data_loader.h"
#include "autoencoder.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <fstream>
#include <iomanip>

void extractFeaturesGPU(GPUAutoencoder *gpu_model, Autoencoder *cpu_model,
                        CIFAR10Dataset *dataset, const std::string &output_file)
{
    int train_size = dataset->getTrainSize();
    int test_size = dataset->getTestSize();
    int total_size = train_size + test_size;
    int batch_size = 128;

    std::cout << "\n=== Feature Extraction ===" << std::endl;
    std::cout << "Total images: " << total_size << std::endl;
    std::cout << "Batch size: " << batch_size << std::endl;
    std::cout << "Output file: " << output_file << std::endl;
    std::cout << "===========================\n"
              << std::endl;

    // Copy weights to GPU
    std::cout << "Copying weights to GPU..." << std::endl;
    gpu_model->copyWeightsToDevice(
        cpu_model->enc_conv1->getWeights(), cpu_model->enc_conv1->getBias(),
        cpu_model->enc_conv2->getWeights(), cpu_model->enc_conv2->getBias(),
        cpu_model->dec_conv1->getWeights(), cpu_model->dec_conv1->getBias(),
        cpu_model->dec_conv2->getWeights(), cpu_model->dec_conv2->getBias(),
        cpu_model->dec_conv3->getWeights(), cpu_model->dec_conv3->getBias());

    // Allocate feature buffer
    float *all_features = new float[total_size * 8192];

    auto start = std::chrono::high_resolution_clock::now();

    // Extract training features
    std::cout << "Extracting training features..." << std::endl;
    gpu_model->extractFeatures(dataset->getTrainImages(), all_features, train_size, batch_size);

    // Extract test features
    std::cout << "Extracting test features..." << std::endl;
    gpu_model->extractFeatures(dataset->getTestImages(), all_features + train_size * 8192, test_size, batch_size);

    auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end - start).count();

    std::cout << "Feature extraction completed in " << duration / 1000.0f << "s" << std::endl;
    std::cout << "Speed: " << total_size / (duration / 1000.0f) << " images/sec" << std::endl;

    // Save features to file
    std::cout << "Saving features to file..." << std::endl;
    std::ofstream file(output_file, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file for saving: " << output_file << std::endl;
        delete[] all_features;
        return;
    }

    file.write(reinterpret_cast<char *>(all_features), total_size * 8192 * sizeof(float));
    file.close();

    std::cout << "Features saved successfully" << std::endl;
    std::cout << "File size: " << total_size * 8192 * sizeof(float) / (1024.0f * 1024.0f * 1024.0f) << " GB\n"
              << std::endl;

    delete[] all_features;
}

int main(int argc, char **argv)
{
    // Default values
    std::string data_dir = "./data/cifar-10-batches-bin";
    std::string output_file = "../build/cifar10_features.bin";
    std::string weights_file = "../build/autoencoder_gpu.weights";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
        {
            data_dir = argv[++i];
        }
        else if (strcmp(argv[i], "--output") == 0 && i + 1 < argc)
        {
            output_file = argv[++i];
        }
        else if (strcmp(argv[i], "--weights") == 0 && i + 1 < argc)
        {
            weights_file = argv[++i];
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --data-dir PATH       Path to CIFAR-10 data directory" << std::endl;
            std::cout << "  --output FILE         Output features file (default: ../build/cifar10_features.bin)" << std::endl;
            std::cout << "  --weights FILE        Pre-trained weights file" << std::endl;
            std::cout << "  --help                Show this help message" << std::endl;
            return 0;
        }
    }

    // Load dataset
    CIFAR10Dataset dataset;

    if (!dataset.loadTrainingData(data_dir))
    {
        std::cerr << "Failed to load training data from " << data_dir << std::endl;
        return 1;
    }

    if (!dataset.loadTestData(data_dir))
    {
        std::cerr << "Failed to load test data from " << data_dir << std::endl;
        return 1;
    }

    if (!dataset.loadClassNames(data_dir + "/batches.meta.txt"))
    {
        std::cerr << "Failed to load class names" << std::endl;
        return 1;
    }

    dataset.normalize();

    // Create CPU autoencoder
    Autoencoder cpu_model(128);

    // Load pre-trained weights if available
    try
    {
        cpu_model.loadWeights(weights_file);
    }
    catch (...)
    {
        std::cout << "Note: Could not load pre-trained weights from " << weights_file << std::endl;
    }

    // Create GPU autoencoder
    GPUAutoencoder gpu_model(128);

    // Extract features
    extractFeaturesGPU(&gpu_model, &cpu_model, &dataset, output_file);

    return 0;
}
