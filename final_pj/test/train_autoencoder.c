#include "cifar10.h"
#include "autoencoder.h"
#include "config.h"
#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc, char** argv) {
    if (argc < 2) {
        fprintf(stderr, "Usage: %s <path_to_cifar10_data> [--epochs N] [--batch-size N] [--lr LR]\n", argv[0]);
        fprintf(stderr, "Example: %s ./cifar-10-batches-bin --epochs 20 --batch-size 32 --lr 0.001\n", argv[0]);
        return 1;
    }

    const char* data_dir = argv[1];

    // Default hyperparameters
    int num_epochs = 20;
    int batch_size = 32;
    float learning_rate = 0.001f;

    // Parse arguments
    for (int i = 2; i < argc; i++) {
        if (strcmp(argv[i], "--epochs") == 0 && i + 1 < argc) {
            num_epochs = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--batch-size") == 0 && i + 1 < argc) {
            batch_size = atoi(argv[++i]);
        } else if (strcmp(argv[i], "--lr") == 0 && i + 1 < argc) {
            learning_rate = atof(argv[++i]);
        }
    }

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          CIFAR-10 Autoencoder Training (CPU)              ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n");

    Timer* timer = timer_create();
    BenchmarkResult* result = benchmark_create(DEVICE_CPU);

    // ===== STEP 1: Load CIFAR-10 Data =====
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" STEP 1: Loading CIFAR-10 Data\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    timer_start(timer);

    printf("Loading training data (50,000 images)...\n");
    CIFAR10Dataset* train_data = cifar10_load_train_data(data_dir);
    if (!train_data) {
        fprintf(stderr, "Failed to load training data\n");
        return 1;
    }

    printf("Loading test data (10,000 images)...\n");
    CIFAR10Dataset* test_data = cifar10_load_test_data(data_dir);
    if (!test_data) {
        fprintf(stderr, "Failed to load test data\n");
        cifar10_free_dataset(train_data);
        return 1;
    }

    timer_stop(timer);
    result->data_loading_time = timer_elapsed(timer);

    printf("\n✓ Data loaded successfully in %.2f seconds\n", result->data_loading_time);
    cifar10_print_dataset_info(train_data);

    // ===== STEP 2: Create Autoencoder =====
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" STEP 2: Creating Autoencoder Model\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    printf("Hyperparameters:\n");
    printf("  Batch size:      %d\n", batch_size);
    printf("  Learning rate:   %.6f\n", learning_rate);
    printf("  Epochs:          %d\n", num_epochs);
    printf("  Device:          CPU\n\n");

    Autoencoder* ae = autoencoder_create(learning_rate, batch_size, num_epochs, DEVICE_CPU);
    if (!ae) {
        fprintf(stderr, "Failed to create autoencoder\n");
        cifar10_free_dataset(train_data);
        cifar10_free_dataset(test_data);
        return 1;
    }

    autoencoder_print_summary(ae);

    // ===== STEP 3: Train Autoencoder =====
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" STEP 3: Training Autoencoder (Unsupervised)\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    printf("Training on all 50,000 images (labels ignored)...\n");
    printf("Total batches per epoch: %d\n\n", (train_data->num_images + batch_size - 1) / batch_size);

    timer_reset(timer);
    timer_start(timer);

    float best_loss = 1e10f;
    for (int epoch = 0; epoch < num_epochs; epoch++) {
        printf("Epoch %d/%d:\n", epoch + 1, num_epochs);

        Timer* epoch_timer = timer_create();
        timer_start(epoch_timer);

        float epoch_loss = autoencoder_train_epoch(ae, train_data->data, train_data->num_images, 1);

        timer_stop(epoch_timer);
        double epoch_time = timer_elapsed(epoch_timer);
        timer_free(epoch_timer);

        printf("\n  Loss: %.6f | Time: %.2f s | Throughput: %.1f imgs/s\n",
               epoch_loss, epoch_time, train_data->num_images / epoch_time);

        if (epoch_loss < best_loss) {
            best_loss = epoch_loss;
            printf("  ✓ New best loss! Saving model...\n");
            autoencoder_save_weights(ae, "autoencoder_best.weights");
        }

        printf("\n");

        // Early stopping check
        if (epoch_loss < 0.001f) {
            printf("✓ Converged! Loss below threshold.\n");
            break;
        }
    }

    timer_stop(timer);
    result->training_time = timer_elapsed(timer);

    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" Training Summary\n");
    printf("═══════════════════════════════════════════════════════════\n\n");
    printf("Total training time:  %.2f seconds\n", result->training_time);
    printf("Average time/epoch:   %.2f seconds\n", result->training_time / num_epochs);
    printf("Best loss achieved:   %.6f\n", best_loss);
    printf("\n");

    // ===== STEP 4: Test Reconstruction =====
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" STEP 4: Testing Reconstruction Quality\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    timer_reset(timer);
    timer_start(timer);

    // Test on test set (in batches)
    int test_batches = (test_data->num_images + batch_size - 1) / batch_size;
    float total_test_loss = 0.0f;

    printf("Testing on %d images...\n", test_data->num_images);
    for (int batch_idx = 0; batch_idx < test_batches; batch_idx++) {
        int start_idx = batch_idx * batch_size;
        int end_idx = start_idx + batch_size;
        if (end_idx > test_data->num_images) end_idx = test_data->num_images;
        int actual_batch_size = end_idx - start_idx;

        float* batch_data = &test_data->data[start_idx * 3 * 32 * 32];

        // Forward pass only
        autoencoder_forward(ae, batch_data, actual_batch_size);
        float batch_loss = mse_loss_cpu(ae->output, batch_data, actual_batch_size * 3 * 32 * 32);
        total_test_loss += batch_loss * actual_batch_size;
    }

    total_test_loss /= test_data->num_images;

    timer_stop(timer);
    result->inference_time = timer_elapsed(timer);

    printf("\nTest Results:\n");
    printf("  Reconstruction Loss: %.6f\n", total_test_loss);
    printf("  Inference Time:      %.2f seconds\n", result->inference_time);
    printf("  Throughput:          %.1f imgs/s\n", test_data->num_images / result->inference_time);

    // ===== STEP 5: Extract Latent Features =====
    printf("\n");
    printf("═══════════════════════════════════════════════════════════\n");
    printf(" STEP 5: Extracting Latent Features\n");
    printf("═══════════════════════════════════════════════════════════\n\n");

    printf("Extracting latent representations from encoder...\n");
    printf("Latent dimension: (batch, 128, 8, 8) = 8192 features\n\n");

    // Extract latent features for first 10 images
    float* latent_features = (float*)malloc(10 * 128 * 8 * 8 * sizeof(float));
    autoencoder_encode(ae, train_data->data, latent_features, 10);

    printf("Sample latent feature statistics (first image):\n");
    float min_val = latent_features[0], max_val = latent_features[0], sum = 0.0f;
    for (int i = 0; i < 128 * 8 * 8; i++) {
        float val = latent_features[i];
        if (val < min_val) min_val = val;
        if (val > max_val) max_val = val;
        sum += val;
    }
    printf("  Min: %.4f, Max: %.4f, Mean: %.4f\n", min_val, max_val, sum / (128 * 8 * 8));

    free(latent_features);

    // ===== Final Benchmark Results =====
    result->total_time = result->data_loading_time + result->training_time + result->inference_time;
    result->num_epochs = num_epochs;
    result->num_batches = (train_data->num_images + batch_size - 1) / batch_size;
    result->avg_batch_time = result->training_time / (num_epochs * result->num_batches);
    result->throughput = train_data->num_images / (result->training_time / num_epochs);
    result->peak_memory_bytes = 0;  // Not tracking for now
    result->train_accuracy = 0.0f;  // Not applicable for autoencoder
    result->test_accuracy = 0.0f;

    printf("\n");
    benchmark_print(result);

    // Save results
    benchmark_save_to_file(result, "autoencoder_benchmark_cpu.txt");
    printf("Benchmark results saved to autoencoder_benchmark_cpu.txt\n");

    // Cleanup
    autoencoder_free(ae);
    cifar10_free_dataset(train_data);
    cifar10_free_dataset(test_data);
    timer_free(timer);
    benchmark_free(result);

    printf("\n");
    printf("╔════════════════════════════════════════════════════════════╗\n");
    printf("║          Training Completed Successfully!                 ║\n");
    printf("╚════════════════════════════════════════════════════════════╝\n\n");

    printf("Saved files:\n");
    printf("  - autoencoder_best.weights (model weights)\n");
    printf("  - autoencoder_benchmark_cpu.txt (benchmark results)\n\n");

    return 0;
}
