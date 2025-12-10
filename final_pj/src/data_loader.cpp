#include "../include/data_loader.h"

#include <fstream>
#include <iostream>
#include <algorithm>
#include <random>

CIFAR10Dataset::CIFAR10Dataset() {}

static bool loadBatch(const std::string &filename,
                      std::vector<float> &images,
                      std::vector<uint8_t> &labels,
                      int offset)
{
    std::ifstream file(filename, std::ios::binary);
    if (!file)
    {
        std::cerr << "Failed to open " << filename << std::endl;
        return false;
    }

    const int record_size = 3073; // 1 label + 3072 pixels
    const int num_records = 10000;

    std::vector<uint8_t> buffer(record_size);

    for (int i = 0; i < num_records; ++i)
    {
        file.read(reinterpret_cast<char *>(buffer.data()), record_size);
        if (file.gcount() != record_size)
        {
            std::cerr << "Unexpected EOF while reading " << filename << " at record " << i << std::endl;
            return false;
        }

        int img_idx = offset + i;
        // label
        labels[img_idx] = buffer[0];

        // pixels: R(1024), G(1024), B(1024)
        int img_offset = img_idx * 3 * 32 * 32; // CHW
        for (int c = 0; c < 3; ++c)
        {
            for (int h = 0; h < 32; ++h)
            {
                for (int w = 0; w < 32; ++w)
                {
                    int cifar_idx = 1 + c * 1024 + h * 32 + w;
                    int our_idx = img_offset + c * 32 * 32 + h * 32 + w;
                    images[our_idx] = static_cast<float>(buffer[cifar_idx]) / 255.0f;
                }
            }
        }
    }

    return true;
}

bool CIFAR10Dataset::loadTrainingData(const std::string &data_dir)
{
    // allocate
    const int num_train = 50000;
    train_images.assign(num_train * 3 * 32 * 32, 0.0f);
    train_labels.assign(num_train, 0);
    train_indices.resize(num_train);
    for (int i = 0; i < num_train; ++i)
        train_indices[i] = i;

    for (int b = 1; b <= 5; ++b)
    {
        std::string filename = data_dir + "/data_batch_" + std::to_string(b) + ".bin";
        int offset = (b - 1) * 10000;
        if (!loadBatch(filename, train_images, train_labels, offset))
        {
            std::cerr << "Failed to load training batch: " << filename << std::endl;
            return false;
        }
    }

    return true;
}

bool CIFAR10Dataset::loadTestData(const std::string &data_dir)
{
    const int num_test = 10000;
    test_images.assign(num_test * 3 * 32 * 32, 0.0f);
    test_labels.assign(num_test, 0);

    std::string filename = data_dir + "/test_batch.bin";
    if (!loadBatch(filename, test_images, test_labels, 0))
    {
        std::cerr << "Failed to load test batch: " << filename << std::endl;
        return false;
    }

    return true;
}

bool CIFAR10Dataset::loadClassNames(const std::string &meta_file)
{
    std::ifstream file(meta_file);
    if (!file)
    {
        std::cerr << "Failed to open meta file: " << meta_file << std::endl;
        return false;
    }

    class_names.clear();
    std::string line;
    while (std::getline(file, line))
    {
        if (!line.empty())
            class_names.push_back(line);
    }

    if (class_names.size() < 10)
    {
        std::cerr << "Warning: expected 10 class names, got " << class_names.size() << std::endl;
    }
    return true;
}

void CIFAR10Dataset::getBatch(int batch_idx, int batch_size, float *batch_images, uint8_t *batch_labels)
{
    int start = batch_idx * batch_size;
    int train_size = getTrainSize();
    if (start >= train_size)
    {
        std::cerr << "getBatch: batch_idx out of range" << std::endl;
        return;
    }

    int actual = std::min(batch_size, train_size - start);
    int single_size = 3 * 32 * 32;
    for (int i = 0; i < actual; ++i)
    {
        int idx = train_indices[start + i];
        // copy image
        std::copy_n(train_images.data() + idx * single_size, single_size, batch_images + i * single_size);
        batch_labels[i] = train_labels[idx];
    }
    // If requested batch_size > remaining, zero-pad remaining images
    if (actual < batch_size)
    {
        int remaining = batch_size - actual;
        std::fill(batch_images + actual * single_size, batch_images + batch_size * single_size, 0.0f);
        std::fill(batch_labels + actual, batch_labels + batch_size, 0);
    }
}

void CIFAR10Dataset::shuffle()
{
    std::mt19937 rng(rng_seed);
    std::shuffle(train_indices.begin(), train_indices.end(), rng);
    // change seed for next shuffle
    rng_seed += 1;
}

void CIFAR10Dataset::normalize()
{
    // Already normalized on load, but provide a re-normalize in case of changes
    int total = getTrainSize() * 3 * 32 * 32;
    for (int i = 0; i < total; ++i)
    {
        if (train_images[i] > 1.0f)
            train_images[i] = train_images[i] / 255.0f;
    }

    total = getTestSize() * 3 * 32 * 32;
    for (int i = 0; i < total; ++i)
    {
        if (test_images[i] > 1.0f)
            test_images[i] = test_images[i] / 255.0f;
    }
}
