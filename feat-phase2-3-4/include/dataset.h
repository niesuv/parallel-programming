/**
 * @file dataset.h
 * @brief CIFAR-10 Dataset Loader
 *
 * Loads CIFAR-10 dataset from binary files.
 * Dataset specifications:
 * - 60,000 32×32 RGB images (50K train + 10K test)
 * - 10 classes: airplane, automobile, bird, cat, deer, dog, frog, horse, ship,
 * truck
 * - Binary format: 1 byte label + 3072 bytes pixels (R, G, B planes)
 */

#ifndef DATASET_H
#define DATASET_H

#include <string>
#include <vector>

/**
 * @brief Container for a batch of CIFAR-10 images
 */
struct CifarBatch {
  std::vector<float> images; ///< Normalized pixels [0,1], shape: [N, 3, 32, 32]
  std::vector<int> labels;   ///< Class labels [0-9]
  int num_images = 0;        ///< Number of images in batch
};

/**
 * @brief CIFAR-10 Dataset Loader
 *
 * Loads training (50K) and test (10K) images from binary files.
 * Images are normalized to [0, 1] range during loading.
 *
 * Expected directory structure:
 *   {data_dir}/data_batch_1.bin ... data_batch_5.bin (training)
 *   {data_dir}/test_batch.bin (test)
 */
class CIFAR10Dataset {
public:
  // Image dimensions
  static constexpr int IMAGE_HEIGHT = 32;
  static constexpr int IMAGE_WIDTH = 32;
  static constexpr int IMAGE_CHANNELS = 3;
  static constexpr int IMAGE_SIZE = IMAGE_CHANNELS * IMAGE_HEIGHT * IMAGE_WIDTH;

  /// Load dataset from directory
  explicit CIFAR10Dataset(const std::string &data_dir);

  /// Access training data
  const CifarBatch &train() const { return train_; }

  /// Access test data
  const CifarBatch &test() const { return test_; }

private:
  CifarBatch train_; ///< 50,000 training images
  CifarBatch test_;  ///< 10,000 test images

  /// Load single batch file
  static CifarBatch load_batch(const std::string &file_path);
};

#endif // DATASET_H
