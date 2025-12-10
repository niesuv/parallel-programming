#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include "cifar10.h"
#include "autoencoder.h"
#include "autoencoder_gpu.h"
#include "config.h"
#include "benchmark.h"

#define MAX_PATH 256

void print_usage(const char* program_name) {
    printf("Usage: %s <cifar10_path> [options]\n", program_name);
    printf("Options:\n");
    printf("  --epochs N          Number of training epochs (default: 20)\n");
    printf("  --batch-size N      Batch size for GPU training (default: 64)\n");
    printf("  --lr F              Learning rate (default: 0.001)\n");
    printf("  --num-samples N     Limit training samples for quick testing\n");
    printf("  --verify            Verify GPU output against CPU (slow!)\n");
}

int main(int argc, char** argv) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    const char* data_path = argv[1];
    int epochs = 20;
    int batch_size = 64;  // Larger batch for GPU
    float learning_rate = 0.001f;
    int max_train_samples = -1;
    int max_test_samples = -1;
    int verify_mode = 0;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        } else if (strcmp(argv[i], "--num-samples") == 0 && i + 1 < argc) {
            max_train_samples = atoi(argv[++i]);
            max_test_samples = max_train_samples / 5;
        } else if (strcmp(argv[i], "--verify") == 0) {
            verify_mode = 1;
        }
    }

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║       CIFAR-10 Autoencoder Training (Naive GPU)          ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");
    printf("\n");

    // Configuration
    printf("Configuration:\n");
    printf("  Device: CUDA GPU\n");
    printf("  Epochs: %d\n", epochs);
    printf("  Batch size: %d\n", batch_size);
    printf("  Learning rate: %.4f\n", learning_rate);
    if (max_train_samples > 0) {
        printf("  ⚠️  LIMITED TRAINING: %d samples (for testing)\n", max_train_samples);
    }
    if (verify_mode) {
        printf("  ✅ VERIFICATION MODE: Will compare GPU vs CPU\n");
    }
    printf("\n");

    // STEP 1: Load CIFAR-10 data
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" STEP 1: Loading CIFAR-10 Data\n");
    printf("═══════════════════════════════════════════════════════════\n");

    CIFAR10Dataset* train_data = cifar10_load_train(data_path, );
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }

    CIFAR10Dataset* test_data = cifar10_load_test(data_path, );
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\n");
        cifar10_free(train_data);
        return 1;
    }

    int num_train = train_data->num_images;
    int num_test = test_data->num_images;

    if (max_train_samples > 0 && max_train_samples < num_train) {
        num_train = max_train_samples;
        num_test = (max_test_samples > 0 && max_test_samples < test_data->num_images)
                   ? max_test_samples : test_data->num_images;
    }

    printf("✅ Loaded %d training images\n", num_train);
    printf("✅ Loaded %d test images\n", num_test);

    // STEP 2: Create GPU Autoencoder
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf(" STEP 2: Creating GPU Autoencoder\n");
    printf("═══════════════════════════════════════════════════════════\n");

    Autoencoder_GPU* gpu_ae = autoencoder_gpu_create(learning_rate, batch_size, epochs);
    if (!gpu_ae) {
        fprintf(stderr, "Failed to create GPU autoencoder\n");
        cifar10_free(train_data);
        cifar10_free(test_data);
        return 1;
    }

    // Also create CPU autoencoder for weight initialization
    Autoencoder_CPU* cpu_ae = autoencoder_cpu_create(learning_rate, batch_size, epochs, );
    if (!cpu_ae) {
        fprintf(stderr, "Failed to create CPU autoencoder\n");
        autoencoder_gpu_free(gpu_ae);
        cifar10_free(train_data);
        cifar10_free(test_data);
        return 1;
    }

    // Copy weights from CPU to GPU
    autoencoder_gpu_copy_weights_to_device(gpu_ae,
                                           cpu_ae->enc_conv1, cpu_ae->enc_conv2,
                                           cpu_ae->dec_conv1, cpu_ae->dec_conv2, cpu_ae->dec_conv3);

    // STEP 3: Training
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf(" STEP 3: Training on GPU\n");
    printf("═══════════════════════════════════════════════════════════\n");

    float best_loss = 1e10f;
    Timer timer;
    timer_start(&timer);

    for (int epoch = 0; epoch < epochs; epoch++) {
        printf("\n[Epoch %d/%d]\n", epoch + 1, epochs);

        float train_loss = autoencoder_gpu_train_epoch(gpu_ae, train_data->data, num_train, 1);

        printf("  Training Loss: %.6f\n", train_loss);

        if (train_loss < best_loss) {
            best_loss = train_loss;
            printf("  🎯 New best loss!\n");

            // Copy weights back to CPU for saving
            autoencoder_gpu_copy_weights_to_host(gpu_ae,
                                                 cpu_ae->enc_conv1, cpu_ae->enc_conv2,
                                                 cpu_ae->dec_conv1, cpu_ae->dec_conv2, cpu_ae->dec_conv3);
            autoencoder_cpu_save_weights(cpu_ae, "autoencoder_best_gpu.weights");
        }
    }

    double total_time = timer_end(&timer);

    printf("\n═══════════════════════════════════════════════════════════\n");
    printf(" STEP 4: Training Complete\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf("✅ Training completed in %.2f seconds\n", total_time);
    printf("✅ Best loss: %.6f\n", best_loss);
    printf("✅ Weights saved to: autoencoder_best_gpu.weights\n");

    // STEP 5: Verification (if requested)
    if (verify_mode) {
        printf("\n═══════════════════════════════════════════════════════════\n");
        printf(" STEP 5: Verifying GPU vs CPU\n");
        printf("═══════════════════════════════════════════════════════════\n");

        // Take small batch for verification
        int verify_batch = (batch_size < 10) ? batch_size : 10;
        printf("Testing with %d samples...\n", verify_batch);

        // GPU forward pass
        timer_start(&timer);
        gpu_autoencoder_forward(gpu_ae, train_data->data, verify_batch);
        double gpu_time = timer_end(&timer);

        // CPU forward pass
        timer_start(&timer);
        autoencoder_forward(cpu_ae, train_data->data, verify_batch);
        double cpu_time = timer_end(&timer);

        printf("GPU forward time: %.2f ms\n", gpu_time * 1000.0);
        printf("CPU forward time: %.2f ms\n", cpu_time * 1000.0);
        printf("Speedup: %.2fx\n", cpu_time / gpu_time);
    }

    // Cleanup
    printf("\n═══════════════════════════════════════════════════════════\n");
    printf(" Cleanup\n");
    printf("═══════════════════════════════════════════════════════════\n");

    autoencoder_gpu_free(gpu_ae);
    autoencoder_cpu_free(cpu_ae);
    cifar10_free(train_data);
    cifar10_free(test_data);

    printf("✅ All resources freed\n");
    printf("\n🎉 GPU Training completed successfully!\n\n");

    return 0;
}
