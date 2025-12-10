#include "cifar10.h"
#include <stdio.h>
#include <stdlib.h>

int main(int argc, char** argv) {
    if (argc != 2) {
        fprintf(stderr, "Usage: %s <path_to_cifar10_data_directory>\n", argv[0]);
        fprintf(stderr, "Example: %s ./cifar-10-batches-bin\n", argv[0]);
        return 1;
    }

    const char* data_dir = argv[1];

    printf("=================================================\n");
    printf("CIFAR-10 Data Loading Test\n");
    printf("=================================================\n\n");

    // Test 1: Load training data
    printf("TEST 1: Loading training data...\n");
    CIFAR10Dataset* train_data = cifar10_load_train_data(data_dir);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }
    cifar10_print_dataset_info(train_data);

    // Test 2: Load test data
    printf("TEST 2: Loading test data...\n");
    CIFAR10Dataset* test_data = cifar10_load_test_data(data_dir);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\n");
        cifar10_free_dataset(train_data);
        return 1;
    }
    cifar10_print_dataset_info(test_data);

    // Test 3: Batch iteration without shuffling
    printf("TEST 3: Batch iteration (batch_size=100, no shuffle)...\n");
    int batch_size = 100;
    CIFAR10Batch* batch_iter = cifar10_create_batch_iterator(train_data, batch_size, 0);
    if (!batch_iter) {
        fprintf(stderr, "Failed to create batch iterator\n");
        cifar10_free_dataset(train_data);
        cifar10_free_dataset(test_data);
        return 1;
    }

    printf("Total batches: %d\n", batch_iter->num_batches);

    int batch_count = 0;
    int actual_size;
    while ((actual_size = cifar10_next_batch(batch_iter, train_data)) > 0) {
        batch_count++;
        if (batch_count <= 3 || batch_count == batch_iter->num_batches) {
            printf("  Batch %d: size=%d, first_label=%d, last_label=%d\n",
                   batch_count, actual_size, batch_iter->labels[0],
                   batch_iter->labels[actual_size - 1]);
        } else if (batch_count == 4) {
            printf("  ...\n");
        }
    }
    printf("Successfully iterated through %d batches\n\n", batch_count);

    // Test 4: Batch iteration with shuffling
    printf("TEST 4: Batch iteration (batch_size=100, with shuffle)...\n");
    cifar10_free_batch_iterator(batch_iter);
    batch_iter = cifar10_create_batch_iterator(train_data, batch_size, 1);
    if (!batch_iter) {
        fprintf(stderr, "Failed to create batch iterator\n");
        cifar10_free_dataset(train_data);
        cifar10_free_dataset(test_data);
        return 1;
    }

    batch_count = 0;
    while ((actual_size = cifar10_next_batch(batch_iter, train_data)) > 0) {
        batch_count++;
        if (batch_count <= 3) {
            printf("  Batch %d: size=%d, first_label=%d, last_label=%d\n",
                   batch_count, actual_size, batch_iter->labels[0],
                   batch_iter->labels[actual_size - 1]);
        }
    }
    printf("Successfully iterated through %d shuffled batches\n\n", batch_count);

    // Test 5: Reset and iterate again
    printf("TEST 5: Reset iterator and iterate again...\n");
    cifar10_reset_batch_iterator(batch_iter);
    batch_count = 0;
    while ((actual_size = cifar10_next_batch(batch_iter, train_data)) > 0) {
        batch_count++;
    }
    printf("Successfully iterated through %d batches after reset\n\n", batch_count);

    // Test 6: Display sample image info
    printf("TEST 6: Sample image details...\n");
    printf("First training image:\n");
    printf("  Label: %d\n", train_data->labels[0]);
    printf("  First 10 pixel values (R channel): ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", train_data->data[i]);
    }
    printf("\n\n");

    printf("First test image:\n");
    printf("  Label: %d\n", test_data->labels[0]);
    printf("  First 10 pixel values (R channel): ");
    for (int i = 0; i < 10; i++) {
        printf("%.3f ", test_data->data[i]);
    }
    printf("\n\n");

    // Cleanup
    cifar10_free_batch_iterator(batch_iter);
    cifar10_free_dataset(train_data);
    cifar10_free_dataset(test_data);

    printf("=================================================\n");
    printf("All tests completed successfully!\n");
    printf("=================================================\n");

    return 0;
}
