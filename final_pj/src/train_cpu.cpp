#include <iostream>
#include <cstring>
#include <chrono>
#include "../include/data_loader.h"
#include "../include/autoencoder.h"

int main(int argc, char **argv)
{
    int epochs = 2;
    int batch_size = 32;
    float learning_rate = 0.001f;
    int num_train_samples = 50000;
    std::string data_dir = "./data/cifar-10-batches-bin";

    for (int i = 1; i < argc; ++i)
    {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc)
            epochs = atoi(argv[++i]);
        else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc)
            batch_size = atoi(argv[++i]);
        else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc)
            learning_rate = atof(argv[++i]);
        else if (strcmp(argv[i], "--num-samples") == 0 && i + 1 < argc)
            num_train_samples = atoi(argv[++i]);
        else if (strcmp(argv[i], "--data-dir") == 0 && i + 1 < argc)
            data_dir = argv[++i];
    }

    CIFAR10Dataset ds;
    if (!ds.loadClassNames(data_dir + "/batches.meta.txt"))
    {
        std::cerr << "Warning: could not load class names." << std::endl;
    }
    if (!ds.loadTrainingData(data_dir))
    {
        std::cerr << "Failed to load training data." << std::endl;
        return 1;
    }

    if (!ds.loadTestData(data_dir))
    {
        std::cerr << "Failed to load test data." << std::endl;
        return 1;
    }

    std::cout << "Train size: " << ds.getTrainSize() << ", Test size: " << ds.getTestSize() << std::endl;

    Autoencoder model(64);

    int num_batches = num_train_samples / batch_size;
    for (int epoch = 0; epoch < epochs; ++epoch)
    {
        ds.shuffle();
        float epoch_loss = 0.0f;
        auto start = std::chrono::high_resolution_clock::now();
        for (int b = 0; b < num_batches; ++b)
        {
            float *batch_images = new float[batch_size * 3 * 32 * 32];
            uint8_t *batch_labels = new uint8_t[batch_size];
            ds.getBatch(b, batch_size, batch_images, batch_labels);

            float loss = model.forward(batch_images, batch_size);
            epoch_loss += loss;
            model.backward(batch_images, batch_size);
            model.updateWeights(learning_rate);

            delete[] batch_images;
            delete[] batch_labels;
            if (b % 100 == 0)
                std::cout << "Epoch " << epoch << " Batch " << b << " Loss " << loss << std::endl;
        }
        auto end = std::chrono::high_resolution_clock::now();
        auto dur = std::chrono::duration_cast<std::chrono::seconds>(end - start).count();
        std::cout << "Epoch " << epoch << " done in " << dur << "s, Avg Loss " << epoch_loss / num_batches << std::endl;
    }

    model.saveWeights("autoencoder_cpu.weights");
    std::cout << "Training complete. Weights saved to autoencoder_cpu.weights" << std::endl;
    return 0;
}
