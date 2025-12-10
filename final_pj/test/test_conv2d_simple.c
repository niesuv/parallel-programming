#include "layers.h"
#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Testing Conv2D correctness and speed...\n\n");
    
    // Simple test: 1 image, 1 input channel, 1 output channel
    printf("=== Simple Test: 1x1x4x4 -> 1x1x4x4 ===\n");
    Conv2DLayer* simple = conv2d_create(1, 1, 3, 1, 1, 4, 4, DEVICE_CPU);
    
    // Set simple weights (all 1.0)
    for (int i = 0; i < 9; i++) {
        simple->weights[i] = 1.0f;
    }
    simple->bias[0] = 0.0f;
    
    // Simple input (all 1.0)
    float input[16];
    for (int i = 0; i < 16; i++) input[i] = 1.0f;
    
    float output[16];
    
    Timer* timer = timer_create();
    timer_start(timer);
    conv2d_forward_cpu(simple, input, output, 1);
    timer_stop(timer);
    
    printf("Time: %.6f s\n", timer_elapsed(timer));
    printf("Output[0] should be 9.0: %.2f\n", output[0]);
    printf("Output[5] (center) should be 9.0: %.2f\n\n", output[5]);
    
    // Realistic test
    printf("=== Realistic Test: 32x3x32x32 -> 32x256x32x32 ===\n");
    Conv2DLayer* conv = conv2d_create(3, 256, 3, 1, 1, 32, 32, DEVICE_CPU);
    conv2d_init_weights(conv);
    
    int batch_size = 32;
    float* big_input = (float*)malloc(batch_size * 3 * 32 * 32 * sizeof(float));
    float* big_output = (float*)malloc(batch_size * 256 * 32 * 32 * sizeof(float));
    
    for (int i = 0; i < batch_size * 3 * 32 * 32; i++) {
        big_input[i] = 0.5f;
    }
    
    // Warm up
    conv2d_forward_cpu(conv, big_input, big_output, batch_size);
    
    // Measure
    timer_reset(timer);
    timer_start(timer);
    conv2d_forward_cpu(conv, big_input, big_output, batch_size);
    timer_stop(timer);
    
    double time_per_batch = timer_elapsed(timer);
    double time_per_image = time_per_batch / batch_size;
    double throughput = batch_size / time_per_batch;
    
    printf("Batch size: %d\n", batch_size);
    printf("Time per batch: %.4f s\n", time_per_batch);
    printf("Time per image: %.4f s (%.2f ms)\n", time_per_image, time_per_image * 1000);
    printf("Throughput: %.2f imgs/s\n\n", throughput);
    
    // Operations count
    long ops = (long)32 * 32 * 256 * 3 * 3 * 3;  // per image
    double gflops = (ops * batch_size) / (time_per_batch * 1e9);
    printf("Operations per image: %ld (%.2f M)\n", ops, ops / 1e6);
    printf("GFLOPS: %.3f\n\n", gflops);
    
    if (throughput < 5.0) {
        printf("⚠️  WARNING: Very slow! Expected >10 imgs/s\n");
        printf("Check:\n");
        printf("  - Is -O3 flag being used?\n");
        printf("  - Are you in debug mode?\n");
        printf("  - Memory allocation in loop?\n");
    } else if (throughput < 15.0) {
        printf("⚠️  Slow but acceptable for naive implementation\n");
    } else {
        printf("✓ Good performance for naive CPU implementation\n");
    }
    
    free(big_input);
    free(big_output);
    conv2d_free(simple);
    conv2d_free(conv);
    timer_free(timer);
    
    return 0;
}
