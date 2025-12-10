#include "cifar10.h"
#include "config.h"
#include "device.h"
#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// Test function to benchmark data loading and device transfer
void test_device_performance(const char* data_dir, DeviceType device) {
    printf("\n");
    printf("╔════════════════════════════════════════════════╗\n");
    printf("║   Testing Device: %-29s║\n", device_type_to_string(device));
    printf("╚════════════════════════════════════════════════╝\n");

    Timer* timer = timer_create();
    BenchmarkResult* result = benchmark_create(device);

    // Initialize device
    printf("\nInitializing device...\n");
    if (device_init(device, 0) != 0) {
        fprintf(stderr, "Failed to initialize device\n");
        timer_free(timer);
        benchmark_free(result);
        return;
    }
    device_print_info(device);

    // Benchmark data loading
    printf("Step 1: Loading CIFAR-10 data...\n");
    timer_start(timer);

    CIFAR10Dataset* train_data = cifar10_load_train_data(data_dir);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        device_cleanup(device);
        timer_free(timer);
        benchmark_free(result);
        return;
    }

    CIFAR10Dataset* test_data = cifar10_load_test_data(data_dir);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\n");
        cifar10_free_dataset(train_data);
        device_cleanup(device);
        timer_free(timer);
        benchmark_free(result);
        return;
    }

    timer_stop(timer);
    result->data_loading_time = timer_elapsed(timer);
    printf("✓ Data loaded in %.4f seconds\n", result->data_loading_time);

    // Print dataset info
    cifar10_print_dataset_info(train_data);

    // Benchmark device transfer (if not CPU)
    if (device != DEVICE_CPU) {
        printf("\nStep 2: Transferring data to %s...\n", device_type_to_string(device));
        timer_reset(timer);
        timer_start(timer);

        if (cifar10_transfer_to_device(train_data, device) != 0) {
            fprintf(stderr, "Failed to transfer training data to device\n");
            cifar10_free_dataset(train_data);
            cifar10_free_dataset(test_data);
            device_cleanup(device);
            timer_free(timer);
            benchmark_free(result);
            return;
        }

        if (cifar10_transfer_to_device(test_data, device) != 0) {
            fprintf(stderr, "Failed to transfer test data to device\n");
            cifar10_free_dataset(train_data);
            cifar10_free_dataset(test_data);
            device_cleanup(device);
            timer_free(timer);
            benchmark_free(result);
            return;
        }

        device_synchronize(device);
        timer_stop(timer);
        double transfer_time = timer_elapsed(timer);
        printf("✓ Data transferred in %.4f seconds\n", transfer_time);
        printf("  Transfer bandwidth: %.2f GB/s\n",
               (train_data->num_images * CIFAR10_IMAGE_SIZE * sizeof(float) +
                test_data->num_images * CIFAR10_IMAGE_SIZE * sizeof(float)) /
               (transfer_time * 1024 * 1024 * 1024));
    } else {
        printf("\nStep 2: Skipping device transfer (CPU mode)\n");
    }

    // Simulate batch iteration
    printf("\nStep 3: Testing batch iteration (10 batches)...\n");
    int batch_size = 128;
    CIFAR10Batch* batch_iter = cifar10_create_batch_iterator(train_data, batch_size, 1);

    timer_reset(timer);
    timer_start(timer);

    int batch_count = 0;
    int actual_size;
    while (batch_count < 10 && (actual_size = cifar10_next_batch(batch_iter, train_data)) > 0) {
        batch_count++;
        // Simulate some work
        if (device != DEVICE_CPU) {
            device_synchronize(device);
        }
    }

    timer_stop(timer);
    double batch_time = timer_elapsed(timer);
    printf("✓ Processed %d batches in %.4f seconds\n", batch_count, batch_time);
    printf("  Average batch time: %.4f ms\n", (batch_time / batch_count) * 1000);

    // Calculate memory usage
    size_t train_mem = train_data->num_images * (CIFAR10_IMAGE_SIZE * sizeof(float) + sizeof(uint8_t));
    size_t test_mem = test_data->num_images * (CIFAR10_IMAGE_SIZE * sizeof(float) + sizeof(uint8_t));
    result->peak_memory_bytes = train_mem + test_mem;

    // Set result metrics
    result->training_time = 0.0;  // Not implemented yet
    result->inference_time = 0.0;  // Not implemented yet
    result->total_time = result->data_loading_time;
    result->num_epochs = 0;
    result->num_batches = batch_count;
    result->avg_batch_time = batch_time / batch_count;
    result->throughput = (batch_count * batch_size) / batch_time;
    result->train_accuracy = 0.0f;  // Not implemented yet
    result->test_accuracy = 0.0f;  // Not implemented yet

    // Print results
    benchmark_print(result);

    // Save results to file
    char filename[256];
    snprintf(filename, sizeof(filename), "benchmark_%s.txt", device_type_to_string(device));
    benchmark_save_to_file(result, filename);
    printf("Results saved to %s\n", filename);

    // Cleanup
    cifar10_free_batch_iterator(batch_iter);
    cifar10_free_dataset(train_data);
    cifar10_free_dataset(test_data);
    device_cleanup(device);
    timer_free(timer);
    benchmark_free(result);

    printf("\n✓ Test completed successfully for %s\n", device_type_to_string(device));
}

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_cifar10_data_directory> [--cpu-only]\n", argv[0]);
        fprintf(stderr, "Example: %s ./cifar-10-batches-bin\n", argv[0]);
        return 1;
    }

    const char* data_dir = argv[1];
    int cpu_only = 0;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--cpu-only") == 0) {
            cpu_only = 1;
        }
    }

    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          CIFAR-10 CPU vs GPU Comparison Test              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    // Test CPU
    test_device_performance(data_dir, DEVICE_CPU);

    // Test CUDA if available and not disabled
    if (!cpu_only && device_is_available(DEVICE_CUDA)) {
        printf("\n");
        printf("════════════════════════════════════════════════════════════\n");
        printf("\n");
        test_device_performance(data_dir, DEVICE_CUDA);

        // Compare results
        // In a real scenario, you would save the benchmark results and compare them here
        printf("\n");
        printf("════════════════════════════════════════════════════════════\n");
        printf("\n");
        printf("To compare results, check the generated benchmark files:\n");
        printf("  - benchmark_CPU.txt\n");
        printf("  - benchmark_CUDA.txt\n");
    } else if (cpu_only) {
        printf("\n");
        printf("CUDA testing skipped (--cpu-only flag)\n");
    } else {
        printf("\n");
        printf("CUDA not available on this system\n");
        printf("To test with CUDA, run on a system with CUDA support\n");
    }

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║              All Tests Completed Successfully!            ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    return 0;
}
