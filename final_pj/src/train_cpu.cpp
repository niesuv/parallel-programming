#include "autoencoder.h"
#include "data_loader.h"
#include <iostream>
#include <chrono>
#include <cstring>
#include <iomanip>

void trainAutoencoder(Autoencoder *model, CIFAR10Dataset *dataset,
                      int epochs, int batch_size, float learning_rate,
                      int num_train_samples = 50000)
{
    int effective_train_size = std::min(num_train_samples, 50000);
    int num_batches = effective_train_size / batch_size;

    std::cout << "\n=== CPU Autoencoder Training ===" << std::endl;
    std::cout << "Epochs: " << epochs << std::endl;
    std::cout << "Batch Size: " << batch_size << std::endl;
    std::cout << "Learning Rate: " << learning_rate << std::endl;
    std::cout << "Training Samples: " << effective_train_size << std::endl;
    std::cout << "Number of Batches: " << num_batches << std::endl;
    std::cout << "==============================\n"
              << std::endl;

    auto training_start = std::chrono::high_resolution_clock::now();

    for (int epoch = 0; epoch < epochs; epoch++)
    {
        dataset->shuffle();
        float epoch_loss = 0.0f;

        auto epoch_start = std::chrono::high_resolution_clock::now();

        for (int batch = 0; batch < num_batches; batch++)
        {
            // Allocate batch memory
            float *batch_images = new float[batch_size * 32 * 32 * 3];
            uint8_t *batch_labels = new uint8_t[batch_size];

            // Get batch
            dataset->getBatch(batch, batch_size, batch_images, batch_labels);

            // Forward pass
            float loss = model->forward(batch_images, batch_size);
            epoch_loss += loss;

            // Backward pass
            model->backward(batch_images, batch_size);

            // Update weights
            model->updateWeights(learning_rate);

            delete[] batch_images;
            delete[] batch_labels;

            // Progress output
            if (batch % 50 == 0)
            {
                std::cout << "Epoch " << std::setw(2) << epoch << " | Batch " << std::setw(4) << batch
                          << "/" << std::setw(4) << num_batches << " | Loss: " << std::fixed << std::setprecision(6)
                          << loss << std::endl;
            }
        }

        auto epoch_end = std::chrono::high_resolution_clock::now();
        auto duration = std::chrono::duration_cast<std::chrono::seconds>(
                            epoch_end - epoch_start)
                            .count();

        float avg_loss = epoch_loss / num_batches;
        std::cout << "Epoch " << std::setw(2) << epoch << " completed in " << std::setw(4) << duration
                  << "s | Avg Loss: " << std::fixed << std::setprecision(6) << avg_loss << std::endl;
    }

    auto training_end = std::chrono::high_resolution_clock::now();
    auto total_time = std::chrono::duration_cast<std::chrono::seconds>(
                          training_end - training_start)
                          .count();

    std::cout << "\n=== Training Complete ===" << std::endl;
    std::cout << "Total Time: " << total_time << "s" << std::endl;
    std::cout << "Average Time/Epoch: " << total_time / epochs << "s" << std::endl;
    std::cout << "=========================\n"
              << std::endl;

    // Save weights
    model->saveWeights("../build/autoencoder_cpu.weights");
}

int main(int argc, char **argv)
{
    // Default values
    int epochs = 20;
    int batch_size = 32;
    float learning_rate = 0.001f;
    int num_train_samples = 50000;
    std::string data_dir = "./data/cifar-10-batches-bin";

    // Parse command-line arguments
    for (int i = 1; i < argc; i++)
    {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
        {
            epochs = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
        {
            batch_size = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc)
        {
            learning_rate = std::atof(argv[++i]);
        }
        else if (strcmp(argv[i], "--num-samples") == 0 && i + 1 < argc)
        {
            num_train_samples = std::atoi(argv[++i]);
        }
        else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
        {
            data_dir = argv[++i];
        }
        else if (strcmp(argv[i], "--help") == 0)
        {
            std::cout << "Usage: " << argv[0] << " [options]" << std::endl;
            std::cout << "  --epochs N            Number of training epochs (default: 20)" << std::endl;
            std::cout << "  --batch-size N        Batch size (default: 32)" << std::endl;
            std::cout << "  --lr LR               Learning rate (default: 0.001)" << std::endl;
            std::cout << "  --num-samples N       Number of training samples (default: 50000)" << std::endl;
            std::cout << "  --data-dir PATH       Path to CIFAR-10 data directory" << std::endl;
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

    if (!dataset.loadClassNames(data_dir + "/batches.meta.txt"))
    {
        std::cerr << "Failed to load class names" << std::endl;
        return 1;
    }

    dataset.normalize();

    // Create autoencoder
    Autoencoder model(batch_size);

    // Train
    trainAutoencoder(&model, &dataset, epochs, batch_size, learning_rate, num_train_samples);

    return 0;
}
