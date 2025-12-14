/**
 * @file dataset.cpp
 * @brief CIFAR-10 Dataset Loader Implementation
 *
 * Loads CIFAR-10 binary files into memory.
 * Binary format per record: [1 byte label][3072 bytes pixels (RGB planes)]
 */

#include "dataset.h"

#include <fstream>
#include <stdexcept>

namespace {

// Each record: 1 byte label + 3072 bytes image data
constexpr int RECORD_BYTES = 1 + CIFAR10Dataset::IMAGE_SIZE;

/**
 * @brief Load a single CIFAR-10 batch file
 * @param file_path Path to .bin file (data_batch_*.bin or test_batch.bin)
 * @return CifarBatch with images normalized to [0, 1]
 */
CifarBatch load_cifar10_batch_impl(const std::string &file_path) {
  // Open file and get size
  std::ifstream file(file_path, std::ios::binary | std::ios::ate);
  if (!file) {
    throw std::runtime_error("Failed to open CIFAR-10 batch: " + file_path);
  }

  std::streamsize file_size = file.tellg();
  if (file_size % RECORD_BYTES != 0) {
    throw std::runtime_error("Invalid CIFAR-10 batch file size: " + file_path);
  }

  int num_records = static_cast<int>(file_size / RECORD_BYTES);
  std::vector<unsigned char> buffer(static_cast<size_t>(file_size));

  // Read entire file into buffer
  file.seekg(0, std::ios::beg);
  if (!file.read(reinterpret_cast<char *>(buffer.data()), file_size)) {
    throw std::runtime_error("Failed to read CIFAR-10 batch: " + file_path);
  }

  // Parse records into batch
  CifarBatch batch;
  batch.num_images = num_records;
  batch.labels.resize(num_records);
  batch.images.resize(static_cast<size_t>(num_records) *
                      CIFAR10Dataset::IMAGE_SIZE);

  for (int i = 0; i < num_records; ++i) {
    const unsigned char *record = buffer.data() + i * RECORD_BYTES;

    // First byte is label (0-9)
    batch.labels[i] = static_cast<int>(record[0]);

    // Remaining 3072 bytes are RGB pixel values
    const unsigned char *pixels = record + 1;
    for (int j = 0; j < CIFAR10Dataset::IMAGE_SIZE; ++j) {
      // Normalize to [0, 1]
      batch.images[static_cast<size_t>(i) * CIFAR10Dataset::IMAGE_SIZE + j] =
          static_cast<float>(pixels[j]) / 255.0f;
    }
  }

  return batch;
}

/**
 * @brief Append source batch to destination batch
 */
void append_batch(CifarBatch &dst, const CifarBatch &src) {
  if (src.num_images == 0)
    return;
  if (dst.num_images == 0) {
    dst = src;
    return;
  }

  dst.images.reserve(dst.images.size() + src.images.size());
  dst.labels.reserve(dst.labels.size() + src.labels.size());

  dst.images.insert(dst.images.end(), src.images.begin(), src.images.end());
  dst.labels.insert(dst.labels.end(), src.labels.begin(), src.labels.end());

  dst.num_images = static_cast<int>(dst.labels.size());
}

} // namespace

// =============================================================================
// CIFAR10Dataset Implementation
// =============================================================================

CifarBatch CIFAR10Dataset::load_batch(const std::string &file_path) {
  return load_cifar10_batch_impl(file_path);
}

CIFAR10Dataset::CIFAR10Dataset(const std::string &data_dir) {
  // Load 5 training batches (50,000 images total)
  for (int i = 1; i <= 5; ++i) {
    std::string path = data_dir + "/data_batch_" + std::to_string(i) + ".bin";
    CifarBatch batch = load_cifar10_batch_impl(path);
    append_batch(train_, batch);
  }

  // Load test batch (10,000 images)
  std::string test_path = data_dir + "/test_batch.bin";
  test_ = load_cifar10_batch_impl(test_path);
}
