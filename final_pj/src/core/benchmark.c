#include "benchmark.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/time.h>

// Get current time in seconds
static double get_time(void) {
    struct timeval tv;
    gettimeofday(&tv, NULL);
    return tv.tv_sec + tv.tv_usec / 1000000.0;
}

// Timer functions
Timer* timer_create(void) {
    Timer* timer = (Timer*)malloc(sizeof(Timer));
    if (!timer) {
        fprintf(stderr, "Failed to allocate Timer\n");
        return NULL;
    }
    timer->start_time = 0.0;
    timer->end_time = 0.0;
    timer->elapsed_time = 0.0;
    return timer;
}

void timer_free(Timer* timer) {
    if (timer) free(timer);
}

void timer_start(Timer* timer) {
    if (timer) {
        timer->start_time = get_time();
    }
}

void timer_stop(Timer* timer) {
    if (timer) {
        timer->end_time = get_time();
        timer->elapsed_time = timer->end_time - timer->start_time;
    }
}

double timer_elapsed(const Timer* timer) {
    if (timer) {
        return timer->elapsed_time;
    }
    return 0.0;
}

void timer_reset(Timer* timer) {
    if (timer) {
        timer->start_time = 0.0;
        timer->end_time = 0.0;
        timer->elapsed_time = 0.0;
    }
}

// Benchmark functions
BenchmarkResult* benchmark_create(DeviceType device) {
    BenchmarkResult* result = (BenchmarkResult*)malloc(sizeof(BenchmarkResult));
    if (!result) {
        fprintf(stderr, "Failed to allocate BenchmarkResult\n");
        return NULL;
    }

    memset(result, 0, sizeof(BenchmarkResult));
    result->device = device;
    return result;
}

void benchmark_free(BenchmarkResult* result) {
    if (result) free(result);
}

void benchmark_print(const BenchmarkResult* result) {
    if (!result) return;

    printf("\n╔════════════════════════════════════════════════╗\n");
    printf("║        Benchmark Results - %s%-16s║\n",
           device_type_to_string(result->device), "");
    printf("╠════════════════════════════════════════════════╣\n");

    // Timing results
    printf("║ Timing:                                        ║\n");
    printf("║   Data Loading:      %10.4f s              ║\n", result->data_loading_time);
    printf("║   Training:          %10.4f s              ║\n", result->training_time);
    printf("║   Inference:         %10.4f s              ║\n", result->inference_time);
    printf("║   Total:             %10.4f s              ║\n", result->total_time);
    printf("╟────────────────────────────────────────────────╢\n");

    // Training metrics
    printf("║ Training Metrics:                              ║\n");
    printf("║   Epochs:            %10d                 ║\n", result->num_epochs);
    printf("║   Batches:           %10d                 ║\n", result->num_batches);
    printf("║   Avg Batch Time:    %10.4f ms            ║\n", result->avg_batch_time * 1000);
    printf("║   Throughput:        %10.2f imgs/s        ║\n", result->throughput);
    printf("╟────────────────────────────────────────────────╢\n");

    // Memory usage
    printf("║ Memory:                                        ║\n");
    printf("║   Peak Usage:        %10.2f MB            ║\n",
           result->peak_memory_bytes / (1024.0 * 1024.0));
    printf("╟────────────────────────────────────────────────╢\n");

    // Accuracy
    printf("║ Accuracy:                                      ║\n");
    printf("║   Training:          %10.2f %%             ║\n", result->train_accuracy * 100);
    printf("║   Test:              %10.2f %%             ║\n", result->test_accuracy * 100);
    printf("╚════════════════════════════════════════════════╝\n\n");
}

void benchmark_compare(const BenchmarkResult* cpu_result,
                       const BenchmarkResult* gpu_result) {
    if (!cpu_result || !gpu_result) {
        fprintf(stderr, "Invalid benchmark results for comparison\n");
        return;
    }

    printf("\n╔══════════════════════════════════════════════════════════════╗\n");
    printf("║                    CPU vs GPU Comparison                     ║\n");
    printf("╠══════════════════════════════════════════════════════════════╣\n");
    printf("║ Metric                    │      CPU      │      GPU      │  ║\n");
    printf("╟───────────────────────────┼───────────────┼───────────────┼──╢\n");

    // Timing comparison
    printf("║ Data Loading Time (s)     │  %9.4f    │  %9.4f    │  ║\n",
           cpu_result->data_loading_time, gpu_result->data_loading_time);
    printf("║ Training Time (s)         │  %9.4f    │  %9.4f    │  ║\n",
           cpu_result->training_time, gpu_result->training_time);
    printf("║ Inference Time (s)        │  %9.4f    │  %9.4f    │  ║\n",
           cpu_result->inference_time, gpu_result->inference_time);
    printf("║ Total Time (s)            │  %9.4f    │  %9.4f    │  ║\n",
           cpu_result->total_time, gpu_result->total_time);

    printf("╟───────────────────────────┼───────────────┼───────────────┼──╢\n");

    // Speedup
    double data_speedup = cpu_result->data_loading_time / gpu_result->data_loading_time;
    double train_speedup = cpu_result->training_time / gpu_result->training_time;
    double inference_speedup = cpu_result->inference_time / gpu_result->inference_time;
    double total_speedup = cpu_result->total_time / gpu_result->total_time;

    printf("║ Speedup:                                                     ║\n");
    printf("║   Data Loading:           │               │  %8.2fx     │  ║\n", data_speedup);
    printf("║   Training:               │               │  %8.2fx     │  ║\n", train_speedup);
    printf("║   Inference:              │               │  %8.2fx     │  ║\n", inference_speedup);
    printf("║   Total:                  │               │  %8.2fx     │  ║\n", total_speedup);

    printf("╟───────────────────────────┼───────────────┼───────────────┼──╢\n");

    // Throughput
    printf("║ Throughput (imgs/s)       │  %9.2f    │  %9.2f    │  ║\n",
           cpu_result->throughput, gpu_result->throughput);

    printf("╟───────────────────────────┼───────────────┼───────────────┼──╢\n");

    // Accuracy
    printf("║ Train Accuracy (%%)        │  %9.2f    │  %9.2f    │  ║\n",
           cpu_result->train_accuracy * 100, gpu_result->train_accuracy * 100);
    printf("║ Test Accuracy (%%)         │  %9.2f    │  %9.2f    │  ║\n",
           cpu_result->test_accuracy * 100, gpu_result->test_accuracy * 100);

    printf("╚══════════════════════════════════════════════════════════════╝\n\n");

    // Summary
    printf("Summary:\n");
    printf("  GPU is %.2fx faster overall than CPU\n", total_speedup);
    printf("  Training speedup: %.2fx\n", train_speedup);
    printf("  Inference speedup: %.2fx\n", inference_speedup);
    printf("\n");
}

int benchmark_save_to_file(const BenchmarkResult* result, const char* filename) {
    if (!result || !filename) return -1;

    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return -1;
    }

    fprintf(file, "Device: %s\n", device_type_to_string(result->device));
    fprintf(file, "Data Loading Time: %.6f\n", result->data_loading_time);
    fprintf(file, "Training Time: %.6f\n", result->training_time);
    fprintf(file, "Inference Time: %.6f\n", result->inference_time);
    fprintf(file, "Total Time: %.6f\n", result->total_time);
    fprintf(file, "Num Epochs: %d\n", result->num_epochs);
    fprintf(file, "Num Batches: %d\n", result->num_batches);
    fprintf(file, "Avg Batch Time: %.6f\n", result->avg_batch_time);
    fprintf(file, "Throughput: %.2f\n", result->throughput);
    fprintf(file, "Peak Memory (MB): %.2f\n", result->peak_memory_bytes / (1024.0 * 1024.0));
    fprintf(file, "Train Accuracy: %.6f\n", result->train_accuracy);
    fprintf(file, "Test Accuracy: %.6f\n", result->test_accuracy);

    fclose(file);
    return 0;
}

int benchmark_save_comparison(const BenchmarkResult* cpu_result,
                              const BenchmarkResult* gpu_result,
                              const char* filename) {
    if (!cpu_result || !gpu_result || !filename) return -1;

    FILE* file = fopen(filename, "w");
    if (!file) {
        fprintf(stderr, "Failed to open file %s for writing\n", filename);
        return -1;
    }

    fprintf(file, "=== CPU vs GPU Comparison ===\n\n");

    fprintf(file, "Metric,CPU,GPU,Speedup\n");
    fprintf(file, "Data Loading Time (s),%.6f,%.6f,%.2f\n",
            cpu_result->data_loading_time, gpu_result->data_loading_time,
            cpu_result->data_loading_time / gpu_result->data_loading_time);
    fprintf(file, "Training Time (s),%.6f,%.6f,%.2f\n",
            cpu_result->training_time, gpu_result->training_time,
            cpu_result->training_time / gpu_result->training_time);
    fprintf(file, "Inference Time (s),%.6f,%.6f,%.2f\n",
            cpu_result->inference_time, gpu_result->inference_time,
            cpu_result->inference_time / gpu_result->inference_time);
    fprintf(file, "Total Time (s),%.6f,%.6f,%.2f\n",
            cpu_result->total_time, gpu_result->total_time,
            cpu_result->total_time / gpu_result->total_time);
    fprintf(file, "Throughput (imgs/s),%.2f,%.2f,%.2f\n",
            cpu_result->throughput, gpu_result->throughput,
            gpu_result->throughput / cpu_result->throughput);
    fprintf(file, "Train Accuracy (%%),%.2f,%.2f,-\n",
            cpu_result->train_accuracy * 100, gpu_result->train_accuracy * 100);
    fprintf(file, "Test Accuracy (%%),%.2f,%.2f,-\n",
            cpu_result->test_accuracy * 100, gpu_result->test_accuracy * 100);

    fclose(file);
    return 0;
}
