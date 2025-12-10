#include "autoencoder_gpu.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

// Helper macro for CUDA error checking
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            exit(EXIT_FAILURE); \
        } \
    } while(0)

// ============================================================================
// 2.5 GPU Training Loop
// ============================================================================

// Fisher-Yates shuffle on host
static void shuffle_indices(int* indices, int n) {
    for (int i = n - 1; i > 0; i--) {
        int j = rand() % (i + 1);
        int temp = indices[i];
        indices[i] = indices[j];
        indices[j] = temp;
    }
}

float autoencoder_gpu_train_epoch(Autoencoder_GPU* gpu_ae, float* train_data, int num_samples, int verbose) {
    int batch_size = gpu_ae->batch_size;
    int num_batches = num_samples / batch_size;
    float total_loss = 0.0f;

    // Create indices for shuffling
    int* indices = (int*)malloc(num_samples * sizeof(int));
    for (int i = 0; i < num_samples; i++) {
        indices[i] = i;
    }
    shuffle_indices(indices, num_samples);

    // Allocate batch buffers on host
    int batch_size_bytes = batch_size * 3 * 32 * 32 * sizeof(float);
    float* h_batch_input = (float*)malloc(batch_size_bytes);
    float* h_batch_target = (float*)malloc(batch_size_bytes);

    // GPU timer events
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));

    float total_time_ms = 0.0f;

    if (verbose) {
        printf("\n[GPU Training] Batch size: %d, Batches: %d\n", batch_size, num_batches);
    }

    // Training loop
    for (int batch = 0; batch < num_batches; batch++) {
        // Prepare batch (gather samples by indices)
        for (int i = 0; i < batch_size; i++) {
            int sample_idx = indices[batch * batch_size + i];
            int sample_offset = sample_idx * 3 * 32 * 32;
            int batch_offset = i * 3 * 32 * 32;

            // Copy sample to batch
            for (int j = 0; j < 3 * 32 * 32; j++) {
                h_batch_input[batch_offset + j] = train_data[sample_offset + j];
                h_batch_target[batch_offset + j] = train_data[sample_offset + j];  // Autoencoder target = input
            }
        }

        // Start GPU timer
        CUDA_CHECK(cudaEventRecord(start));

        // Forward pass
        autoencoder_gpu_forward(gpu_ae, h_batch_input, batch_size);

        // Compute loss
        float batch_loss = autoencoder_gpu_compute_loss(gpu_ae, h_batch_target, batch_size);
        total_loss += batch_loss;

        // Backward pass
        autoencoder_gpu_backward(gpu_ae, h_batch_target, batch_size);

        // Update weights
        autoencoder_gpu_update_weights(gpu_ae);

        // Stop GPU timer
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));

        float batch_time_ms;
        CUDA_CHECK(cudaEventElapsedTime(&batch_time_ms, start, stop));
        total_time_ms += batch_time_ms;

        // Print progress
        if (verbose && (batch % 10 == 0 || batch == num_batches - 1)) {
            float imgs_per_sec = (float)batch_size / (batch_time_ms / 1000.0f);
            printf("  Batch [%4d/%4d] Loss: %.6f  Time: %.2f ms  Speed: %.1f imgs/s\r",
                   batch + 1, num_batches, batch_loss, batch_time_ms, imgs_per_sec);
            fflush(stdout);
        }
    }

    if (verbose) {
        printf("\n");
        float avg_time_per_batch = total_time_ms / num_batches;
        float avg_imgs_per_sec = (float)batch_size / (avg_time_per_batch / 1000.0f);
        printf("[GPU Training] Average: %.2f ms/batch, %.1f imgs/s\n",
               avg_time_per_batch, avg_imgs_per_sec);
    }

    // Cleanup
    free(indices);
    free(h_batch_input);
    free(h_batch_target);
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));

    return total_loss / num_batches;
}
