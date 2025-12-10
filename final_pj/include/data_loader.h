#pragma once

#include <string>
#include <vector>
#include <cstdint>

class CIFAR10Dataset
{
private:
    std::vector<float> train_images;      // CHW: 50000 x 3 x 32 x 32
    std::vector<uint8_t> train_labels;    // 50000
    std::vector<float> test_images;       // 10000 x 3 x 32 x 32
    std::vector<uint8_t> test_labels;     // 10000
    std::vector<std::string> class_names; // 10 classes

    std::vector<int> train_indices; // for shuffling
    unsigned int rng_seed = 12345;

public:
    CIFAR10Dataset();

    // Load all 5 training batches
    bool loadTrainingData(const std::string &data_dir);

    // Load test batch
    bool loadTestData(const std::string &data_dir);

    // Load class names from batches.meta.txt
    bool loadClassNames(const std::string &meta_file);

    // Get batch of images (with shuffling support)
    // batch_images must be preallocated size: batch_size * 3 * 32 * 32
    void getBatch(int batch_idx, int batch_size, float *batch_images, uint8_t *batch_labels);

    // Shuffle training data
    void shuffle();

    // Normalize: uint8 [0,255] → float [0,1] (already done while loading, but exposed)
    void normalize();

    // Getters
    int getTrainSize() const { return static_cast<int>(train_labels.size()); }
    int getTestSize() const { return static_cast<int>(test_labels.size()); }
    const std::string &getClassName(int idx) const { return class_names.at(idx); }
    uint8_t getTrainLabel(int idx) const { return train_labels.at(idx); }
    uint8_t getTestLabel(int idx) const { return test_labels.at(idx); }
};
