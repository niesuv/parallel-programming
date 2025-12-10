#ifndef BENCHMARK_H
#define BENCHMARK_H

#include "config.h"
#include <stdint.h>
#include <stddef.h>

// Timer structure
typedef struct {
    double start_time;
    double end_time;
    double elapsed_time;
} Timer;

// Benchmark results
typedef struct {
    DeviceType device;
    double data_loading_time;
    double training_time;
    double inference_time;
    double total_time;

    // Training metrics
    int num_epochs;
    int num_batches;
    double avg_batch_time;
    double throughput;      // images/second

    // Memory usage
    size_t peak_memory_bytes;

    // Accuracy
    float train_accuracy;
    float test_accuracy;
} BenchmarkResult;

// Timer functions
Timer* timer_create(void);
void timer_free(Timer* timer);
void timer_start(Timer* timer);
void timer_stop(Timer* timer);
double timer_elapsed(const Timer* timer);
void timer_reset(Timer* timer);

// Benchmark functions
BenchmarkResult* benchmark_create(DeviceType device);
void benchmark_free(BenchmarkResult* result);
void benchmark_print(const BenchmarkResult* result);
void benchmark_compare(const BenchmarkResult* cpu_result,
                       const BenchmarkResult* gpu_result);

// Save benchmark results to file
int benchmark_save_to_file(const BenchmarkResult* result, const char* filename);
int benchmark_save_comparison(const BenchmarkResult* cpu_result,
                              const BenchmarkResult* gpu_result,
                              const char* filename);

#endif // BENCHMARK_H
