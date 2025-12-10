#include "cifar10.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

// Create a new CIFAR-10 dataset
CIFAR10Dataset* cifar10_create_dataset(int num_images) {
    CIFAR10Dataset* dataset = (CIFAR10Dataset*)malloc(sizeof(CIFAR10Dataset));
    if (!dataset) {
        fprintf(stderr, "Failed to allocate memory for dataset\n");
        return NULL;
    }

    dataset->num_images = num_images;
    dataset->image_size = CIFAR10_IMAGE_SIZE;
    dataset->device = DEVICE_CPU;  // Default to CPU
    dataset->device_data = NULL;
    dataset->device_labels = NULL;

    // Allocate memory for images (float for normalized values)
    dataset->data = (float*)malloc(num_images * CIFAR10_IMAGE_SIZE * sizeof(float));
    if (!dataset->data) {
        fprintf(stderr, "Failed to allocate memory for image data\n");
        free(dataset);
        return NULL;
    }

    // Allocate memory for labels
    dataset->labels = (uint8_t*)malloc(num_images * sizeof(uint8_t));
    if (!dataset->labels) {
        fprintf(stderr, "Failed to allocate memory for labels\n");
        free(dataset->data);
        free(dataset);
        return NULL;
    }

    return dataset;
}

// Free dataset memory
void cifar10_free_dataset(CIFAR10Dataset* dataset) {
    if (dataset) {
        if (dataset->data) free(dataset->data);
        if (dataset->labels) free(dataset->labels);
        if (dataset->device_data) device_free(dataset->device_data);
        if (dataset->device_labels) device_free(dataset->device_labels);
        free(dataset);
    }
}

// Load a single batch file
int cifar10_load_batch(const char* filename, CIFAR10Dataset* dataset, int offset) {
    FILE* file = fopen(filename, "rb");
    if (!file) {
        fprintf(stderr, "Failed to open file: %s\n", filename);
        return -1;
    }

    printf("Loading batch from: %s\n", filename);

    uint8_t buffer[CIFAR10_RECORD_SIZE];

    for (int i = 0; i < CIFAR10_IMAGES_PER_BATCH; i++) {
        size_t bytes_read = fread(buffer, 1, CIFAR10_RECORD_SIZE, file);
        if (bytes_read != CIFAR10_RECORD_SIZE) {
            fprintf(stderr, "Failed to read image %d from %s\n", i, filename);
            fclose(file);
            return -1;
        }

        int idx = offset + i;

        // Read label (first byte)
        dataset->labels[idx] = buffer[0];

        // Read and normalize pixel data (next 3072 bytes)
        // Convert from uint8 [0, 255] to float [0, 1]
        for (int j = 0; j < CIFAR10_IMAGE_SIZE; j++) {
            dataset->data[idx * CIFAR10_IMAGE_SIZE + j] = buffer[j + 1] / 255.0f;
        }
    }

    fclose(file);
    return 0;
}

// Load all training data (5 batches)
CIFAR10Dataset* cifar10_load_train_data(const char* data_dir) {
    CIFAR10Dataset* dataset = cifar10_create_dataset(CIFAR10_TRAIN_SIZE);
    if (!dataset) {
        return NULL;
    }

    char filepath[512];

    // Load all 5 training batches
    for (int batch = 1; batch <= CIFAR10_TRAIN_BATCHES; batch++) {
        snprintf(filepath, sizeof(filepath), "%s/data_batch_%d.bin", data_dir, batch);
        int offset = (batch - 1) * CIFAR10_IMAGES_PER_BATCH;

        if (cifar10_load_batch(filepath, dataset, offset) != 0) {
            fprintf(stderr, "Failed to load training batch %d\n", batch);
            cifar10_free_dataset(dataset);
            return NULL;
        }
    }

    printf("Successfully loaded %d training images\n", dataset->num_images);
    return dataset;
}

// Load test data (1 batch)
CIFAR10Dataset* cifar10_load_test_data(const char* data_dir) {
    CIFAR10Dataset* dataset = cifar10_create_dataset(CIFAR10_TEST_SIZE);
    if (!dataset) {
        return NULL;
    }

    char filepath[512];
    snprintf(filepath, sizeof(filepath), "%s/test_batch.bin", data_dir);

    if (cifar10_load_batch(filepath, dataset, 0) != 0) {
        fprintf(stderr, "Failed to load test batch\n");
        cifar10_free_dataset(dataset);
        return NULL;
    }

    printf("Successfully loaded %d test images\n", dataset->num_images);
    return dataset;
}

// Create batch iterator
CIFAR10Batch* cifar10_create_batch_iterator(CIFAR10Dataset* dataset, int batch_size, int shuffle) {
    CIFAR10Batch* batch = (CIFAR10Batch*)malloc(sizeof(CIFAR10Batch));
    if (!batch) {
        fprintf(stderr, "Failed to allocate memory for batch iterator\n");
        return NULL;
    }

    batch->batch_size = batch_size;
    batch->current_batch = 0;
    batch->num_batches = (dataset->num_images + batch_size - 1) / batch_size;
    batch->device = DEVICE_CPU;  // Batches are always on CPU for now

    // Allocate memory for batch data
    batch->data = (float*)malloc(batch_size * CIFAR10_IMAGE_SIZE * sizeof(float));
    batch->labels = (uint8_t*)malloc(batch_size * sizeof(uint8_t));

    if (!batch->data || !batch->labels) {
        fprintf(stderr, "Failed to allocate memory for batch data\n");
        if (batch->data) free(batch->data);
        if (batch->labels) free(batch->labels);
        free(batch);
        return NULL;
    }

    // Create indices array
    batch->indices = (int*)malloc(dataset->num_images * sizeof(int));
    if (!batch->indices) {
        fprintf(stderr, "Failed to allocate memory for indices\n");
        free(batch->data);
        free(batch->labels);
        free(batch);
        return NULL;
    }

    // Initialize indices
    for (int i = 0; i < dataset->num_images; i++) {
        batch->indices[i] = i;
    }

    // Shuffle if requested
    if (shuffle) {
        cifar10_shuffle_indices(batch->indices, dataset->num_images);
    }

    return batch;
}

// Free batch iterator
void cifar10_free_batch_iterator(CIFAR10Batch* batch) {
    if (batch) {
        if (batch->data) free(batch->data);
        if (batch->labels) free(batch->labels);
        if (batch->indices) free(batch->indices);
        free(batch);
    }
}

// Get next batch
int cifar10_next_batch(CIFAR10Batch* batch, CIFAR10Dataset* dataset) {
    if (batch->current_batch >= batch->num_batches) {
        return 0; // No more batches
    }

    int start_idx = batch->current_batch * batch->batch_size;
    int end_idx = start_idx + batch->batch_size;
    if (end_idx > dataset->num_images) {
        end_idx = dataset->num_images;
    }

    int actual_batch_size = end_idx - start_idx;

    // Copy data for this batch
    for (int i = 0; i < actual_batch_size; i++) {
        int data_idx = batch->indices[start_idx + i];

        // Copy image data
        memcpy(&batch->data[i * CIFAR10_IMAGE_SIZE],
               &dataset->data[data_idx * CIFAR10_IMAGE_SIZE],
               CIFAR10_IMAGE_SIZE * sizeof(float));

        // Copy label
        batch->labels[i] = dataset->labels[data_idx];
    }

    batch->current_batch++;
    return actual_batch_size;
}

// Reset batch iterator
void cifar10_reset_batch_iterator(CIFAR10Batch* batch) {
    batch->current_batch = 0;
}

// Shuffle indices using Fisher-Yates algorithm
void cifar10_shuffle_indices(int* indices, int size) {
    srand(time(NULL));
    for (int i = size - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

// Print dataset information
void cifar10_print_dataset_info(CIFAR10Dataset* dataset) {
    printf("\n=== CIFAR-10 Dataset Info ===\n");
    printf("Number of images: %d\n", dataset->num_images);
    printf("Image size: %d (32x32x3)\n", dataset->image_size);

    // Count labels
    int label_counts[CIFAR10_NUM_CLASSES] = {0};
    for (int i = 0; i < dataset->num_images; i++) {
        label_counts[dataset->labels[i]]++;
    }

    printf("\nLabel distribution:\n");
    const char* class_names[] = {
        "airplane", "automobile", "bird", "cat", "deer",
        "dog", "frog", "horse", "ship", "truck"
    };

    for (int i = 0; i < CIFAR10_NUM_CLASSES; i++) {
        printf("  Class %d (%s): %d images\n", i, class_names[i], label_counts[i]);
    }

    // Print sample pixel statistics
    float min_val = 1.0f, max_val = 0.0f, sum = 0.0f;
    for (int i = 0; i < CIFAR10_IMAGE_SIZE; i++) {
        float val = dataset->data[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }

    printf("\nFirst image statistics:\n");
    printf("  Min pixel value: %.4f\n", min_val);
    printf("  Max pixel value: %.4f\n", max_val);
    printf("  Mean pixel value: %.4f\n", sum / CIFAR10_IMAGE_SIZE);
    printf("  Label: %d (%s)\n", dataset->labels[0], class_names[dataset->labels[0]]);
    printf("  Device: %s\n", device_type_to_string(dataset->device));
    printf("============================\n\n");
}

// Transfer dataset to device
int cifar10_transfer_to_device(CIFAR10Dataset* dataset, DeviceType device) {
    if (!dataset) return -1;

    if (dataset->device == device) {
        printf("Dataset already on %s\n", device_type_to_string(device));
        return 0;
    }

    // Free existing device memory if any
    if (dataset->device_data) device_free(dataset->device_data);
    if (dataset->device_labels) device_free(dataset->device_labels);

    if (device == DEVICE_CPU) {
        // Data is already on CPU, just update the flag
        dataset->device = DEVICE_CPU;
        dataset->device_data = NULL;
        dataset->device_labels = NULL;
        return 0;
    }
    else if (device == DEVICE_CUDA) {
        printf("Transferring dataset to CUDA...\n");

        // Allocate device memory
        size_t data_size = dataset->num_images * CIFAR10_IMAGE_SIZE * sizeof(float);
        size_t labels_size = dataset->num_images * sizeof(uint8_t);

        dataset->device_data = device_malloc(data_size, DEVICE_CUDA);
        dataset->device_labels = device_malloc(labels_size, DEVICE_CUDA);

        if (!dataset->device_data || !dataset->device_labels) {
            fprintf(stderr, "Failed to allocate device memory\n");
            if (dataset->device_data) device_free(dataset->device_data);
            if (dataset->device_labels) device_free(dataset->device_labels);
            return -1;
        }

        // Copy data to device
        if (device_memcpy_host_to_device(dataset->device_data, dataset->data, data_size) != 0 ||
            device_memcpy_host_to_device(dataset->device_labels, dataset->labels, labels_size) != 0) {
            fprintf(stderr, "Failed to transfer data to device\n");
            device_free(dataset->device_data);
            device_free(dataset->device_labels);
            return -1;
        }

        dataset->device = DEVICE_CUDA;
        printf("Dataset transferred to CUDA successfully\n");
        return 0;
    }

    return -1;
}

// Transfer dataset back to host
int cifar10_transfer_to_host(CIFAR10Dataset* dataset) {
    if (!dataset) return -1;

    if (dataset->device == DEVICE_CPU) {
        return 0;  // Already on host
    }

    printf("Transferring dataset from %s to CPU...\n", device_type_to_string(dataset->device));

    // Copy data back to host
    size_t data_size = dataset->num_images * CIFAR10_IMAGE_SIZE * sizeof(float);
    size_t labels_size = dataset->num_images * sizeof(uint8_t);

    if (device_memcpy_device_to_host(dataset->data, dataset->device_data, data_size) != 0 ||
        device_memcpy_device_to_host(dataset->labels, dataset->device_labels, labels_size) != 0) {
        fprintf(stderr, "Failed to transfer data to host\n");
        return -1;
    }

    // Free device memory
    device_free(dataset->device_data);
    device_free(dataset->device_labels);
    dataset->device_data = NULL;
    dataset->device_labels = NULL;
    dataset->device = DEVICE_CPU;

    printf("Dataset transferred to CPU successfully\n");
    return 0;
}
