#pragma once

#include <vector>
#include <string>
#include <cstdint>

class CIFAR10Dataset
{
private:
    std::vector<float> train_images;      // 50000 × 32 × 32 × 3
    std::vector<uint8_t> train_labels;    // 50000
    std::vector<float> test_images;       // 10000 × 32 × 32 × 3
    std::vector<uint8_t> test_labels;     // 10000
    std::vector<std::string> class_names; // 10 classes

    // Shuffling indices
    std::vector<int> train_indices;
    int shuffle_seed;

    bool loadBatch(const std::string &filename,
                   std::vector<float> &images,
                   std::vector<uint8_t> &labels,
                   int offset);

public:
    CIFAR10Dataset();
    ~CIFAR10Dataset();

    // Load all training batches
    bool loadTrainingData(const std::string &data_dir);

    // Load test batch
    bool loadTestData(const std::string &data_dir);

    // Load class names from batches.meta.txt
    bool loadClassNames(const std::string &meta_file);

    // Get batch of images (with shuffling support)
    void getBatch(int batch_idx, int batch_size,
                  float *batch_images, uint8_t *batch_labels);

    // Get single image
    void getImage(int idx, float *image, uint8_t &label, bool is_train = true);

    // Shuffle training data
    void shuffle();

    // Normalize: uint8 [0,255] → float [0,1]
    void normalize();

    // Getters
    int getTrainSize() const { return 50000; }
    int getTestSize() const { return 10000; }
    int getNumClasses() const { return 10; }
    std::string getClassName(int idx) const
    {
        return (idx >= 0 && idx < 10) ? class_names[idx] : "Unknown";
    }

    // Direct access
    const float *getTrainImages() const { return train_images.data(); }
    const uint8_t *getTrainLabels() const { return train_labels.data(); }
    const float *getTestImages() const { return test_images.data(); }
    const uint8_t *getTestLabels() const { return test_labels.data(); }

    uint8_t getTrainLabel(int idx) const { return train_labels[idx]; }
    uint8_t getTestLabel(int idx) const { return test_labels[idx]; }
};
