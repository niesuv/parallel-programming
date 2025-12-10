#include "layers.h"
#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>

int main() {
    printf("Profiling individual layers...\n\n");
    
    int batch_size = 32;
    Timer* timer = timer_create();
    
    // Test Conv2D
    printf("=== Conv2D (3->256, 32x32) ===\n");
    Conv2DLayer* conv1 = conv2d_create(3, 256, 3, 1, 1, 32, 32, DEVICE_CPU);
    conv2d_init_weights(conv1);
    
    float* input = (float*)malloc(batch_size * 3 * 32 * 32 * sizeof(float));
    float* output = (float*)malloc(batch_size * 256 * 32 * 32 * sizeof(float));
    
    // Initialize input
    for (int i = 0; i < batch_size * 3 * 32 * 32; i++) {
        input[i] = (float)rand() / RAND_MAX;
    }
    
    timer_start(timer);
    for (int i = 0; i < 10; i++) {
        conv2d_forward_cpu(conv1, input, output, batch_size);
    }
    timer_stop(timer);
    
    double conv_time = timer_elapsed(timer) / 10.0;
    printf("Forward time: %.4f s (%.2f imgs/s)\n", conv_time, batch_size / conv_time);
    printf("%.2f ms per image\n\n", conv_time * 1000 / batch_size);
    
    // Test ReLU
    printf("=== ReLU (256 channels, 32x32) ===\n");
    timer_reset(timer);
    timer_start(timer);
    for (int i = 0; i < 100; i++) {
        relu_forward_cpu(output, batch_size * 256 * 32 * 32);
    }
    timer_stop(timer);
    printf("Forward time: %.4f s per batch\n", timer_elapsed(timer) / 100.0);
    printf("%.4f ms per image\n\n", timer_elapsed(timer) * 1000 / (100 * batch_size));
    
    // Test MaxPool
    printf("=== MaxPool2D (256 channels, 32x32->16x16) ===\n");
    MaxPool2DLayer* pool = maxpool2d_create(2, 2, 256, 32, 32, DEVICE_CPU);
    float* pooled = (float*)malloc(batch_size * 256 * 16 * 16 * sizeof(float));
    
    timer_reset(timer);
    timer_start(timer);
    for (int i = 0; i < 10; i++) {
        maxpool2d_forward_cpu(pool, output, pooled, batch_size);
    }
    timer_stop(timer);
    printf("Forward time: %.4f s per batch\n", timer_elapsed(timer) / 10.0);
    printf("%.2f ms per image\n\n", timer_elapsed(timer) * 1000 / (10 * batch_size));
    
    // Summary
    printf("=== Analysis ===\n");
    printf("Conv2D is the bottleneck (expected)\n");
    printf("Total forward pass estimate: ~%.2f ms per image\n", conv_time * 1000 / batch_size);
    printf("Expected throughput: ~%.1f imgs/s (batch=%d)\n\n", batch_size / conv_time, batch_size);
    
    printf("Your observed: 2.5 imgs/s\n");
    printf("Expected: ~%.1f imgs/s\n", batch_size / conv_time);
    printf("Ratio: %.1fx slower than expected\n", (batch_size / conv_time) / 2.5);
    
    free(input);
    free(output);
    free(pooled);
    conv2d_free(conv1);
    maxpool2d_free(pool);
    timer_free(timer);
    
    return 0;
}
