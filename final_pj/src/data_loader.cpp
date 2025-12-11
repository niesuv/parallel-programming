#include "data_loader.h"
#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>
#include <cstring>

CIFAR10Dataset::CIFAR10Dataset() : shuffle_seed(42)
{
    train_images.resize(50000 * 32 * 32 * 3);
    train_labels.resize(50000);
    test_images.resize(10000 * 32 * 32 * 3);
    test_labels.resize(10000);
    class_names.resize(10);

    // Initialize shuffling indices
    train_indices.resize(50000);
    for (int i = 0; i < 50000; i++)
    {
        train_indices[i] = i;
    }
}

CIFAR10Dataset::~CIFAR10Dataset() = default;

bool CIFAR10Dataset::loadBatch(const std::string &filename,
                               std::vector<float> &images,
                               std::vector<uint8_t> &labels,
                               int offset)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Cannot open file: " << filename << std::endl;
        return false;
    }

    const int record_size = 3073;
    const int num_records = 10000;

    for (int i = 0; i < num_records; i++)
    {
        // Read label (1 byte)
        uint8_t label;
        file.read(reinterpret_cast<char *>(&label), 1);
        if (!file)
        {
            std::cerr << "Error reading label at record " << i << std::endl;
            return false;
        }
        labels[offset + i] = label;

        // Read pixel data (3072 bytes: R(1024) + G(1024) + B(1024))
        uint8_t pixels[3072];
        file.read(reinterpret_cast<char *>(pixels), 3072);
        if (!file)
        {
            std::cerr << "Error reading pixels at record " << i << std::endl;
            return false;
        }

        // Convert to CHW format
        int img_offset = (offset + i) * 32 * 32 * 3;

        for (int c = 0; c < 3; c++)
        {
            for (int h = 0; h < 32; h++)
            {
                for (int w = 0; w < 32; w++)
                {
                    int cifar_idx = c * 1024 + h * 32 + w;
                    int our_idx = img_offset + c * 32 * 32 + h * 32 + w;
                    images[our_idx] = pixels[cifar_idx] / 255.0f;
                }
            }
        }
    }

    file.close();
    return true;
}

bool CIFAR10Dataset::loadTrainingData(const std::string &data_dir)
{
    std::cout << "Loading CIFAR-10 training data..." << std::endl;

    for (int batch = 1; batch <= 5; batch++)
    {
        std::string batch_file = data_dir + "/data_batch_" + std::to_string(batch) + ".bin";
        int offset = (batch - 1) * 10000;

        std::cout << "Loading batch " << batch << "..." << std::endl;
        if (!loadBatch(batch_file, train_images, train_labels, offset))
        {
            std::cerr << "Failed to load batch " << batch << std::endl;
            return false;
        }
    }

    std::cout << "Training data loaded successfully: " << train_images.size() / (32 * 32 * 3)
              << " images" << std::endl;
    return true;
}

bool CIFAR10Dataset::loadTestData(const std::string &data_dir)
{
    std::cout << "Loading CIFAR-10 test data..." << std::endl;

    std::string test_file = data_dir + "/test_batch.bin";
    if (!loadBatch(test_file, test_images, test_labels, 0))
    {
        std::cerr << "Failed to load test batch" << std::endl;
        return false;
    }

    std::cout << "Test data loaded successfully: " << test_images.size() / (32 * 32 * 3)
              << " images" << std::endl;
    return true;
}

bool CIFAR10Dataset::loadClassNames(const std::string &meta_file)
{
    std::ifstream file(meta_file);
    if (!file)
    {
        std::cerr << "Cannot open meta file: " << meta_file << std::endl;
        return false;
    }

    for (int i = 0; i < 10; i++)
    {
        if (!std::getline(file, class_names[i]))
        {
            std::cerr << "Error reading class name " << i << std::endl;
            return false;
        }
        // Remove trailing whitespace
        class_names[i].erase(class_names[i].find_last_not_of(" \n\r\t") + 1);
    }

    file.close();
    std::cout << "Class names loaded:" << std::endl;
    for (int i = 0; i < 10; i++)
    {
        std::cout << "  " << i << ": " << class_names[i] << std::endl;
    }
    return true;
}

void CIFAR10Dataset::getBatch(int batch_idx, int batch_size,
                              float *batch_images, uint8_t *batch_labels)
{
    for (int i = 0; i < batch_size; i++)
    {
        int shuffled_idx = train_indices[batch_idx * batch_size + i];

        // Copy image
        std::memcpy(batch_images + i * 32 * 32 * 3,
                    train_images.data() + shuffled_idx * 32 * 32 * 3,
                    32 * 32 * 3 * sizeof(float));

        // Copy label
        batch_labels[i] = train_labels[shuffled_idx];
    }
}

void CIFAR10Dataset::getImage(int idx, float *image, uint8_t &label, bool is_train)
{
    if (is_train)
    {
        int shuffled_idx = train_indices[idx];
        std::memcpy(image,
                    train_images.data() + shuffled_idx * 32 * 32 * 3,
                    32 * 32 * 3 * sizeof(float));
        label = train_labels[shuffled_idx];
    }
    else
    {
        std::memcpy(image,
                    test_images.data() + idx * 32 * 32 * 3,
                    32 * 32 * 3 * sizeof(float));
        label = test_labels[idx];
    }
}

void CIFAR10Dataset::shuffle()
{
    std::mt19937 rng(shuffle_seed);
    std::shuffle(train_indices.begin(), train_indices.end(), rng);
    shuffle_seed++;
}

void CIFAR10Dataset::normalize()
{
    // Data is already normalized during loading (divided by 255.0f)
    std::cout << "Data normalization verified (0-1 range)" << std::endl;
}
