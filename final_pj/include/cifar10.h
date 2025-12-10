#ifndef CIFAR10_H
#define CIFAR10_H

#include <stdint.h>
#include <stdlib.h>
#include "config.h"
#include "device.h"

// CIFAR-10 constants
#define CIFAR10_IMAGE_SIZE 3072      // 32x32x3
#define CIFAR10_IMAGE_WIDTH 32
#define CIFAR10_IMAGE_HEIGHT 32
#define CIFAR10_NUM_CHANNELS 3
#define CIFAR10_NUM_CLASSES 10
#define CIFAR10_IMAGES_PER_BATCH 10000
#define CIFAR10_TRAIN_BATCHES 5
#define CIFAR10_TRAIN_SIZE 50000     // 5 batches * 10000
#define CIFAR10_TEST_SIZE 10000
#define CIFAR10_RECORD_SIZE 3073     // 1 label + 3072 pixels

// Data structures
typedef struct {
    float* data;        // Normalized pixel values [0, 1], shape: (num_images, 3072)
    uint8_t* labels;    // Labels [0-9], shape: (num_images,)
    int num_images;
    int image_size;
    DeviceType device;  // Where the data is stored (CPU or CUDA)
    DeviceMemory* device_data;    // Device memory for data (NULL if CPU)
    DeviceMemory* device_labels;  // Device memory for labels (NULL if CPU)
} CIFAR10Dataset;

typedef struct {
    float* data;        // Batch of images, shape: (batch_size, 3072)
    uint8_t* labels;    // Batch of labels, shape: (batch_size,)
    int batch_size;
    int current_batch;
    int num_batches;
    int* indices;       // For shuffling
    DeviceType device;  // Where the batch data is stored
} CIFAR10Batch;

// Function prototypes
CIFAR10Dataset* cifar10_create_dataset(int num_images);
void cifar10_free_dataset(CIFAR10Dataset* dataset);

int cifar10_load_batch(const char* filename, CIFAR10Dataset* dataset, int offset);
CIFAR10Dataset* cifar10_load_train_data(const char* data_dir);
CIFAR10Dataset* cifar10_load_test_data(const char* data_dir);

CIFAR10Batch* cifar10_create_batch_iterator(CIFAR10Dataset* dataset, int batch_size, int shuffle);
void cifar10_free_batch_iterator(CIFAR10Batch* batch);
int cifar10_next_batch(CIFAR10Batch* batch, CIFAR10Dataset* dataset);
void cifar10_reset_batch_iterator(CIFAR10Batch* batch);

void cifar10_shuffle_indices(int* indices, int size);
void cifar10_print_dataset_info(CIFAR10Dataset* dataset);

// Device transfer functions
int cifar10_transfer_to_device(CIFAR10Dataset* dataset, DeviceType device);
int cifar10_transfer_to_host(CIFAR10Dataset* dataset);

#endif // CIFAR10_H
