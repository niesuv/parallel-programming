/**
 * Test and benchmark for FP16 WMMA Conv2D kernel V6 (Optimized)
 * Target: T4 GPU (SM75) and newer
 * 
 * Build:
 *   nvcc -arch=sm_75 -O3 -o test_conv2d_v6 test_conv2d_v6.cu gpu_conv2d_fp16_v6.cu
 * 
 * Run:
 *   ./test_conv2d_v6              # Run all tests
 *   ./test_conv2d_v6 --bench      # Benchmark only
 *   ./test_conv2d_v6 --test       # Correctness test only
 *   ./test_conv2d_v6 --compare    # Compare V5 vs V6 performance
 *   ./test_conv2d_v6 --profile    # Detailed profiling info
 */

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <chrono>
#include <vector>
#include <algorithm>
#include <string>
#include <numeric>

// ============================================================================
// External kernel declarations
// ============================================================================
extern "C" {
    // V5 (original optimized)
    void launch_conv2d_fp16_wmma_optimized(
        const half* input, const half* weight, const half* bias, half* output,
        int N, int H, int W, int C, int K, cudaStream_t stream);
    
    void launch_conv2d_relu_fp16_wmma_optimized(
        const half* input, const half* weight, const half* bias,
        half* output, half* conv_output,
        int N, int H, int W, int C, int K, cudaStream_t stream);
    
    // V6 (new optimized with double buffering)
    void launch_conv2d_fp16_wmma_v6(
        const half* input, const half* weight, const half* bias, half* output,
        int N, int H, int W, int C, int K, cudaStream_t stream);
    
    void launch_conv2d_relu_fp16_wmma_v6(
        const half* input, const half* weight, const half* bias,
        half* output, half* conv_output,
        int N, int H, int W, int C, int K, cudaStream_t stream);
    
    // Reference (simple, correct)
    void launch_conv2d_fp16_reference(
        const half* input, const half* weight, const half* bias, half* output,
        int N, int H, int W, int C, int K, cudaStream_t stream);
}

// ============================================================================
// CUDA error checking
// ============================================================================
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// ============================================================================
// Configuration
// ============================================================================
namespace Config {
    // V6 kernel parameters (must match kernel)
    constexpr int WMMA_M = 16;
    constexpr int WMMA_N = 16;
    constexpr int WMMA_K = 16;
    constexpr int WARPS_PER_BLOCK = 8;
    constexpr int TILE_H = 16;
    constexpr int TILE_W = 8;
    constexpr int OC_TILES_PER_BLOCK = 4;
    constexpr int OC_PER_BLOCK = OC_TILES_PER_BLOCK * WMMA_N;  // 64
    constexpr int STAGES = 2;
    
    // Test tolerances
    constexpr float ABS_TOL = 0.05f;   // Absolute tolerance
    constexpr float REL_TOL = 0.02f;   // Relative tolerance (2%)
    
    // Benchmark settings
    constexpr int WARMUP_ITERS = 20;
    constexpr int BENCH_ITERS = 100;
    
    // T4 theoretical peaks
    constexpr double T4_FP16_TFLOPS = 65.0;
    constexpr double T4_MEM_BW_GBPS = 320.0;
}

// ============================================================================
// Utilities
// ============================================================================

void print_separator(const char* title = nullptr) {
    printf("\n");
    for (int i = 0; i < 80; i++) printf("=");
    printf("\n");
    if (title) {
        printf("  %s\n", title);
        for (int i = 0; i < 80; i++) printf("=");
        printf("\n");
    }
}

void print_subsection(const char* title) {
    printf("\n  --- %s ---\n", title);
}

void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    print_separator("GPU Information");
    printf("  Device:              %s\n", prop.name);
    printf("  Compute Capability:  SM%d%d\n", prop.major, prop.minor);
    printf("  SMs:                 %d\n", prop.multiProcessorCount);
    printf("  Memory:              %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Memory Bandwidth:    %.0f GB/s (theoretical)\n", 
           2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1e6);
    printf("  L2 Cache:            %.1f MB\n", prop.l2CacheSize / 1e6);
    printf("  Shared Mem/Block:    %zu KB\n", prop.sharedMemPerBlock / 1024);
    printf("  Max Shared Mem/SM:   %zu KB\n", prop.sharedMemPerMultiprocessor / 1024);
    printf("  Registers/Block:     %d\n", prop.regsPerBlock);
    
    if (prop.major >= 7) {
        printf("  Tensor Cores:        Yes (SM%d%d)\n", prop.major, prop.minor);
    } else {
        printf("  Tensor Cores:        No (requires SM70+)\n");
    }
    
    // Estimate peak TFLOPS based on GPU
    double peak_tflops = Config::T4_FP16_TFLOPS;
    if (strstr(prop.name, "A100")) peak_tflops = 312.0;
    else if (strstr(prop.name, "A10")) peak_tflops = 125.0;
    else if (strstr(prop.name, "V100")) peak_tflops = 125.0;
    else if (strstr(prop.name, "RTX 3090")) peak_tflops = 142.0;
    else if (strstr(prop.name, "RTX 4090")) peak_tflops = 330.0;
    
    printf("  FP16 Tensor Peak:    ~%.0f TFLOPS (estimated)\n", peak_tflops);
}

// Random initialization with optional patterns for debugging
enum class InitPattern {
    RANDOM,
    ONES,
    SEQUENTIAL,
    IDENTITY_LIKE
};

void init_data(half* data, size_t count, InitPattern pattern = InitPattern::RANDOM, 
               float scale = 0.1f, unsigned seed = 42) {
    srand(seed);
    for (size_t i = 0; i < count; ++i) {
        float val;
        switch (pattern) {
            case InitPattern::RANDOM:
                val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
                break;
            case InitPattern::ONES:
                val = 1.0f;
                break;
            case InitPattern::SEQUENTIAL:
                val = (float)(i % 100) * 0.01f;
                break;
            case InitPattern::IDENTITY_LIKE:
                val = (i % 17 == 0) ? 1.0f : 0.0f;
                break;
        }
        data[i] = __float2half(val);
    }
}

// Comparison result structure
struct CompareResult {
    float max_abs_err;
    float mean_abs_err;
    float max_rel_err;
    float rmse;
    int worst_idx;
    int num_errors;
    size_t total_elements;
    bool passed;
    
    void print() const {
        printf("    Max absolute error:  %.6f\n", max_abs_err);
        printf("    Mean absolute error: %.6f\n", mean_abs_err);
        printf("    Max relative error:  %.4f%%\n", max_rel_err * 100);
        printf("    RMSE:                %.6f\n", rmse);
        printf("    Error count:         %d / %zu (%.4f%%)\n", 
               num_errors, total_elements, 100.0 * num_errors / total_elements);
    }
};

CompareResult compare_outputs(const half* ref, const half* test, size_t count,
                              float abs_tol = Config::ABS_TOL, 
                              float rel_tol = Config::REL_TOL) {
    CompareResult result = {0, 0, 0, 0, -1, 0, count, true};
    double sum_err = 0;
    double sum_sq_err = 0;
    
    for (size_t i = 0; i < count; ++i) {
        float r = __half2float(ref[i]);
        float t = __half2float(test[i]);
        float abs_err = fabsf(r - t);
        float rel_err = abs_err / (fabsf(r) + 1e-6f);
        
        sum_err += abs_err;
        sum_sq_err += abs_err * abs_err;
        
        // Check if this element is an error
        bool is_error = (abs_err > abs_tol) && (rel_err > rel_tol);
        if (is_error) result.num_errors++;
        
        if (abs_err > result.max_abs_err) {
            result.max_abs_err = abs_err;
            result.worst_idx = i;
        }
        result.max_rel_err = fmaxf(result.max_rel_err, rel_err);
    }
    
    result.mean_abs_err = sum_err / count;
    result.rmse = sqrtf(sum_sq_err / count);
    result.passed = (result.num_errors == 0) || 
                    (result.max_abs_err < abs_tol * 2) ||
                    (result.max_rel_err < rel_tol * 2);
    
    return result;
}

// FLOPS calculation for 3x3 conv
double calc_flops(int N, int H, int W, int C, int K) {
    return 2.0 * N * H * W * K * C * 9;
}

// Memory bytes calculation
size_t calc_memory_bytes(int N, int H, int W, int C, int K) {
    size_t input_bytes = (size_t)N * H * W * C * sizeof(half);
    size_t weight_bytes = (size_t)K * C * 9 * sizeof(half);
    size_t bias_bytes = (size_t)K * sizeof(half);
    size_t output_bytes = (size_t)N * H * W * K * sizeof(half);
    return input_bytes + weight_bytes + bias_bytes + output_bytes;
}

// Arithmetic intensity (FLOPs per byte)
double calc_arithmetic_intensity(int N, int H, int W, int C, int K) {
    double flops = calc_flops(N, H, W, C, K);
    double bytes = calc_memory_bytes(N, H, W, C, K);
    return flops / bytes;
}

// ============================================================================
// Correctness Tests
// ============================================================================

bool run_single_correctness_test(int N, int H, int W, int C, int K, 
                                  bool verbose = false, InitPattern pattern = InitPattern::RANDOM) {
    if (!verbose) {
        printf("  Testing N=%d, H=%d, W=%d, C=%d, K=%d ... ", N, H, W, C, K);
        fflush(stdout);
    } else {
        printf("\n  === Detailed Test: N=%d, H=%d, W=%d, C=%d, K=%d ===\n", N, H, W, C, K);
    }
    
    // Sizes
    size_t input_size = (size_t)N * H * W * C;
    size_t weight_size = (size_t)K * C * 9;
    size_t bias_size = (size_t)K;
    size_t output_size = (size_t)N * H * W * K;
    
    // Host memory
    std::vector<half> h_input(input_size);
    std::vector<half> h_weight(weight_size);
    std::vector<half> h_bias(bias_size);
    std::vector<half> h_output_ref(output_size);
    std::vector<half> h_output_v6(output_size);
    
    // Initialize
    init_data(h_input.data(), input_size, pattern, 0.1f, 42);
    init_data(h_weight.data(), weight_size, pattern, 0.1f, 123);
    init_data(h_bias.data(), bias_size, pattern, 0.1f, 456);
    
    // Device memory
    half *d_input, *d_weight, *d_bias, *d_output_ref, *d_output_v6;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_ref, output_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_v6, output_size * sizeof(half)));
    
    // Initialize outputs to NaN to detect untouched elements
    std::vector<half> nan_init(output_size);
    for (size_t i = 0; i < output_size; ++i) {
        nan_init[i] = __float2half(nanf(""));
    }
    CUDA_CHECK(cudaMemcpy(d_output_ref, nan_init.data(), output_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_output_v6, nan_init.data(), output_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Copy inputs to device
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run reference kernel
    launch_conv2d_fp16_reference(d_input, d_weight, d_bias, d_output_ref, N, H, W, C, K, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run V6 kernel
    launch_conv2d_fp16_wmma_v6(d_input, d_weight, d_bias, d_output_v6, N, H, W, C, K, 0);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(h_output_ref.data(), d_output_ref, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_v6.data(), d_output_v6, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Check for NaN outputs (indicates kernel didn't write)
    int nan_count = 0;
    for (size_t i = 0; i < output_size; ++i) {
        if (isnan(__half2float(h_output_v6[i]))) nan_count++;
    }
    
    // Compare
    CompareResult cmp = compare_outputs(h_output_ref.data(), h_output_v6.data(), output_size);
    
    if (verbose) {
        cmp.print();
        if (nan_count > 0) {
            printf("    WARNING: %d NaN values in output!\n", nan_count);
        }
        
        // Print sample values
        printf("\n    Sample comparisons:\n");
        for (int i = 0; i < std::min(5, (int)output_size); ++i) {
            printf("      [%d] ref=%.6f, v6=%.6f, diff=%.6f\n",
                   i, __half2float(h_output_ref[i]), __half2float(h_output_v6[i]),
                   fabsf(__half2float(h_output_ref[i]) - __half2float(h_output_v6[i])));
        }
        
        if (cmp.worst_idx >= 0) {
            printf("\n    Worst mismatch at index %d:\n", cmp.worst_idx);
            printf("      ref=%.6f, v6=%.6f\n",
                   __half2float(h_output_ref[cmp.worst_idx]),
                   __half2float(h_output_v6[cmp.worst_idx]));
            
            // Decode position
            int k = cmp.worst_idx % K;
            int w_out = (cmp.worst_idx / K) % W;
            int h_out = (cmp.worst_idx / (K * W)) % H;
            int n = cmp.worst_idx / (K * W * H);
            printf("      Position: n=%d, h=%d, w=%d, k=%d\n", n, h_out, w_out, k);
        }
    }
    
    bool passed = cmp.passed && (nan_count == 0);
    
    if (!verbose) {
        if (passed) {
            printf("PASSED (max_err=%.6f, mean_err=%.6f)\n", cmp.max_abs_err, cmp.mean_abs_err);
        } else {
            printf("FAILED\n");
            cmp.print();
            if (nan_count > 0) {
                printf("    NaN count: %d\n", nan_count);
            }
        }
    }
    
    // Cleanup
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output_ref));
    CUDA_CHECK(cudaFree(d_output_v6));
    
    return passed;
}

void run_all_correctness_tests() {
    print_separator("Correctness Tests (V6 vs Reference)");
    
    struct TestCase { 
        int N, H, W, C, K; 
        const char* desc;
    };
    
    std::vector<TestCase> tests = {
        // Basic small tests
        {1, 8, 8, 16, 16, "Minimal"},
        {1, 8, 8, 32, 32, "Small square"},
        {2, 8, 8, 16, 32, "Multi-batch small"},
        
        // Tile boundary tests (V6: TILE_H=16, TILE_W=8)
        {1, 16, 8, 32, 64, "Exact tile"},
        {1, 17, 9, 32, 64, "Tile + 1"},
        {1, 15, 7, 32, 64, "Tile - 1"},
        
        // Channel alignment tests (WMMA_K=16)
        {1, 8, 8, 3, 64, "RGB input (C=3)"},
        {1, 8, 8, 17, 32, "C not aligned"},
        {1, 8, 8, 32, 17, "K not aligned"},
        
        // Output channel tests (OC_PER_BLOCK=64)
        {1, 8, 8, 32, 64, "OC = OC_PER_BLOCK"},
        {1, 8, 8, 32, 65, "OC = OC_PER_BLOCK + 1"},
        {1, 8, 8, 32, 128, "OC = 2 * OC_PER_BLOCK"},
        
        // Medium tests
        {4, 16, 16, 32, 64, "Medium batch"},
        {8, 16, 16, 64, 128, "Larger channels"},
        
        // Larger tests
        {16, 32, 32, 64, 128, "Large spatial"},
        {32, 32, 32, 128, 256, "Large all"},
        
        // Edge cases
        {64, 8, 8, 64, 128, "Large batch"},
        {1, 1, 1, 16, 16, "Tiny spatial"},
        {1, 3, 3, 32, 32, "Very small spatial"},
        
        // Stress tests
        {1, 64, 64, 128, 256, "Large spatial single batch"},
        {128, 8, 8, 128, 128, "Large batch small spatial"},

        {64, 32, 32, 3, 256, "Stage 1"},
        {64, 16, 16, 256, 128, "Stage 2"},
        {64, 8, 8, 128, 64, "Stage 3"},
        {64, 8, 8, 64, 128, "Stage 4"},
        {64, 16, 16, 128, 256, "Stage 5"},
        {64, 32, 32, 256, 3, "Stage 6"}
    };
    
    int passed = 0, failed = 0;
    std::vector<std::string> failed_tests;
    
    for (const auto& t : tests) {
        printf("  [%s] ", t.desc);
        if (run_single_correctness_test(t.N, t.H, t.W, t.C, t.K)) {
            passed++;
        } else {
            failed++;
            char buf[256];
            snprintf(buf, sizeof(buf), "%s (N=%d,H=%d,W=%d,C=%d,K=%d)", 
                     t.desc, t.N, t.H, t.W, t.C, t.K);
            failed_tests.push_back(buf);
        }
    }
    
    printf("\n  Summary: %d passed, %d failed\n", passed, failed);
    
    if (!failed_tests.empty()) {
        printf("\n  Failed tests:\n");
        for (const auto& name : failed_tests) {
            printf("    - %s\n", name.c_str());
        }
    }
}

// ============================================================================
// Benchmark
// ============================================================================

struct BenchResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double std_ms;
    double tflops;
    double mem_gbps;
    double efficiency_compute;
    double efficiency_memory;
    double arithmetic_intensity;
    size_t mem_used;
};

typedef void (*KernelLauncher)(const half*, const half*, const half*, half*, 
                                int, int, int, int, int, cudaStream_t);

BenchResult run_kernel_benchmark(KernelLauncher launcher, const char* name,
                                  int N, int H, int W, int C, int K,
                                  int warmup = Config::WARMUP_ITERS, 
                                  int iters = Config::BENCH_ITERS) {
    BenchResult result = {0, 1e9, 0, 0, 0, 0, 0, 0, 0, 0};
    
    // Sizes
    size_t input_size = (size_t)N * H * W * C;
    size_t weight_size = (size_t)K * C * 9;
    size_t bias_size = (size_t)K;
    size_t output_size = (size_t)N * H * W * K;
    
    result.mem_used = (input_size + weight_size + bias_size + output_size) * sizeof(half);
    result.arithmetic_intensity = calc_arithmetic_intensity(N, H, W, C, K);
    
    // Host memory
    std::vector<half> h_input(input_size);
    std::vector<half> h_weight(weight_size);
    std::vector<half> h_bias(bias_size);
    
    init_data(h_input.data(), input_size, InitPattern::RANDOM);
    init_data(h_weight.data(), weight_size, InitPattern::RANDOM);
    init_data(h_bias.data(), bias_size, InitPattern::RANDOM);
    
    // Device memory
    half *d_input, *d_weight, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < warmup; ++i) {
        launcher(d_input, d_weight, d_bias, d_output, N, H, W, C, K, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    std::vector<float> times(iters);
    
    for (int i = 0; i < iters; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launcher(d_input, d_weight, d_bias, d_output, N, H, W, C, K, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;
        
        result.min_ms = fmin(result.min_ms, ms);
        result.max_ms = fmax(result.max_ms, ms);
    }
    
    // Calculate statistics
    std::sort(times.begin(), times.end());
    
    // Use trimmed mean (exclude top/bottom 10%)
    int trim = iters / 10;
    double sum = 0;
    for (int i = trim; i < iters - trim; ++i) {
        sum += times[i];
    }
    result.avg_ms = sum / (iters - 2 * trim);
    
    // Standard deviation
    double sq_sum = 0;
    for (int i = trim; i < iters - trim; ++i) {
        double diff = times[i] - result.avg_ms;
        sq_sum += diff * diff;
    }
    result.std_ms = sqrt(sq_sum / (iters - 2 * trim));
    
    // Calculate metrics
    double flops = calc_flops(N, H, W, C, K);
    result.tflops = flops / (result.avg_ms * 1e9);
    
    size_t bytes = calc_memory_bytes(N, H, W, C, K);
    result.mem_gbps = bytes / (result.avg_ms * 1e6);
    
    result.efficiency_compute = (result.tflops / Config::T4_FP16_TFLOPS) * 100.0;
    result.efficiency_memory = (result.mem_gbps / Config::T4_MEM_BW_GBPS) * 100.0;
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output));
    
    return result;
}

void run_all_benchmarks() {
    print_separator("Performance Benchmarks (V6 Kernel)");
    
    struct BenchConfig {
        int N, H, W, C, K;
        const char* name;
    };
    
    std::vector<BenchConfig> configs = {
        // Primary use cases (batch 64+, various configurations)
        // 8x8 spatial
        {64,  8,  8,   3,  64, "B64   8x8   RGB->64 "},
        {64,  8,  8,  64,  64, "B64   8x8   64->64  "},
        {64,  8,  8,  64, 128, "B64   8x8   64->128 "},
        {64,  8,  8, 128, 256, "B64   8x8  128->256 "},
        {64,  8,  8, 256, 256, "B64   8x8  256->256 "},
        
        // 16x16 spatial
        {64, 16, 16,   3,  64, "B64  16x16  RGB->64 "},
        {64, 16, 16,  64,  64, "B64  16x16  64->64  "},
        {64, 16, 16,  64, 128, "B64  16x16  64->128 "},
        {64, 16, 16, 128, 256, "B64  16x16 128->256 "},
        {64, 16, 16, 256, 128, "B64  16x16 256->128 "},
        {64, 16, 16, 256, 256, "B64  16x16 256->256 "},
        
        // 32x32 spatial  
        {64, 32, 32,   3,  64, "B64  32x32  RGB->64 "},
        {64, 32, 32,  64, 128, "B64  32x32  64->128 "},
        {64, 32, 32, 128, 256, "B64  32x32 128->256 "},
        {64, 32, 32, 256, 256, "B64  32x32 256->256 "},
        
        // Larger batches
        {128, 16, 16, 64, 128, "B128 16x16  64->128 "},
        {256,  8,  8, 64, 128, "B256  8x8   64->128 "},
        {128, 32, 32, 64, 128, "B128 32x32  64->128 "},

        {64, 32, 32, 3, 256, "Stage 1"},
        {64, 16, 16, 256, 128, "Stage 2"},
        {64, 8, 8, 128, 64, "Stage 3"},
        {64, 8, 8, 64, 128, "Stage 4"},
        {64, 16, 16, 128, 256, "Stage 5"},
        {64, 32, 32, 256, 3, "Stage 6"}
    };
    
    printf("\n  %-20s %8s %8s %8s %9s %8s %8s %7s\n", 
           "Config", "Avg(ms)", "Std(ms)", "TFLOPS", "Eff%%", "GB/s", "AI", "MB");
    printf("  ");
    for (int i = 0; i < 88; i++) printf("-");
    printf("\n");
    
    for (const auto& cfg : configs) {
        BenchResult res = run_kernel_benchmark(
            launch_conv2d_fp16_wmma_v6, "V6",
            cfg.N, cfg.H, cfg.W, cfg.C, cfg.K);
        
        printf("  %-20s %8.3f %8.4f %8.2f %8.1f%% %8.1f %7.1f %7.1f\n",
               cfg.name,
               res.avg_ms,
               res.std_ms,
               res.tflops,
               res.efficiency_compute,
               res.mem_gbps,
               res.arithmetic_intensity,
               res.mem_used / 1e6);
    }
    
    printf("\n  Legend:\n");
    printf("  - TFLOPS: Tera floating-point operations per second\n");
    printf("  - Eff%%:   Efficiency vs T4 peak (65 TFLOPS)\n");
    printf("  - GB/s:   Memory bandwidth\n");
    printf("  - AI:     Arithmetic intensity (FLOPs/byte)\n");
    printf("  - MB:     Memory used\n");
}

// ============================================================================
// V5 vs V6 Comparison
// ============================================================================

void run_comparison_benchmark() {
    print_separator("Performance Comparison: V5 vs V6");
    
    struct BenchConfig {
        int N, H, W, C, K;
        const char* name;
    };
    
    std::vector<BenchConfig> configs = {
        {64,  8,  8,  64, 128, "B64   8x8   64->128"},
        {64, 16, 16,  64, 128, "B64  16x16  64->128"},
        {64, 16, 16, 128, 256, "B64  16x16 128->256"},
        {64, 32, 32,  64, 128, "B64  32x32  64->128"},
        {64, 32, 32, 128, 256, "B64  32x32 128->256"},
        {128, 16, 16, 64, 128, "B128 16x16  64->128"},
        {64, 32, 32, 3, 256, "Stage 1"},
        {64, 16, 16, 256, 128, "Stage 2"},
        {64, 8, 8, 128, 64, "Stage 3"},
        {64, 8, 8, 64, 128, "Stage 4"},
        {64, 16, 16, 128, 256, "Stage 5"},
        {64, 32, 32, 256, 3, "Stage 6"}
 
    };
    
    printf("\n  %-20s | %10s %8s | %10s %8s | %8s\n",
           "Config", "V5 (ms)", "TFLOPS", "V6 (ms)", "TFLOPS", "Speedup");
    printf("  ");
    for (int i = 0; i < 78; i++) printf("-");
    printf("\n");
    
    double total_speedup = 0;
    int count = 0;
    
    for (const auto& cfg : configs) {
        BenchResult v5 = run_kernel_benchmark(
            launch_conv2d_fp16_wmma_optimized, "V5",
            cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, 10, 50);
        
        BenchResult v6 = run_kernel_benchmark(
            launch_conv2d_fp16_wmma_v6, "V6",
            cfg.N, cfg.H, cfg.W, cfg.C, cfg.K, 10, 50);
        
        double speedup = v5.avg_ms / v6.avg_ms;
        total_speedup += speedup;
        count++;
        
        const char* indicator = speedup > 1.05 ? " ↑" : (speedup < 0.95 ? " ↓" : "  ");
        
        printf("  %-20s | %10.3f %8.2f | %10.3f %8.2f | %7.2fx%s\n",
               cfg.name,
               v5.avg_ms, v5.tflops,
               v6.avg_ms, v6.tflops,
               speedup, indicator);
    }
    
    printf("  ");
    for (int i = 0; i < 78; i++) printf("-");
    printf("\n");
    printf("  %-20s | %10s %8s | %10s %8s | %7.2fx\n",
           "AVERAGE", "", "", "", "", total_speedup / count);
    
    printf("\n  ↑ = V6 faster, ↓ = V5 faster\n");
}

// ============================================================================
// Detailed Profiling
// ============================================================================

void run_profiling_analysis() {
    print_separator("Detailed Profiling Analysis");
    
    // Test configuration
    int N = 64, H = 16, W = 16, C = 128, K = 256;
    
    printf("\n  Configuration: N=%d, H=%d, W=%d, C=%d, K=%d\n", N, H, W, C, K);
    
    // Calculate theoretical metrics
    double flops = calc_flops(N, H, W, C, K);
    size_t bytes = calc_memory_bytes(N, H, W, C, K);
    double ai = flops / bytes;
    
    printf("\n  Theoretical Analysis:\n");
    printf("    Total FLOPs:           %.2f GFLOPS\n", flops / 1e9);
    printf("    Memory transfer:       %.2f MB\n", bytes / 1e6);
    printf("    Arithmetic intensity:  %.2f FLOPs/byte\n", ai);
    printf("    Roofline bound:        %s\n", 
           ai > (Config::T4_FP16_TFLOPS * 1e12 / Config::T4_MEM_BW_GBPS / 1e9) 
           ? "Compute" : "Memory");
    
    // Kernel configuration
    print_subsection("V6 Kernel Configuration");
    printf("    Warps per block:       %d\n", Config::WARPS_PER_BLOCK);
    printf("    Threads per block:     %d\n", Config::WARPS_PER_BLOCK * 32);
    printf("    Tile size (H x W):     %d x %d\n", Config::TILE_H, Config::TILE_W);
    printf("    OC per block:          %d\n", Config::OC_PER_BLOCK);
    printf("    Double buffer stages:  %d\n", Config::STAGES);
    
    // Grid dimensions
    int tiles_x = (W + Config::TILE_W - 1) / Config::TILE_W;
    int tiles_y = (H + Config::TILE_H - 1) / Config::TILE_H;
    int oc_blocks = (K + Config::OC_PER_BLOCK - 1) / Config::OC_PER_BLOCK;
    int total_blocks = tiles_x * tiles_y * N * oc_blocks;
    
    printf("\n    Grid dimensions:       (%d, %d, %d)\n", tiles_x, tiles_y, N * oc_blocks);
    printf("    Total blocks:          %d\n", total_blocks);
    
    // Run benchmark with detailed timing
    print_subsection("Performance Measurements");
    
    BenchResult res = run_kernel_benchmark(
        launch_conv2d_fp16_wmma_v6, "V6",
        N, H, W, C, K, 50, 200);
    
    printf("    Average time:          %.3f ms (±%.3f)\n", res.avg_ms, res.std_ms);
    printf("    Min/Max time:          %.3f / %.3f ms\n", res.min_ms, res.max_ms);
    printf("    Achieved TFLOPS:       %.2f\n", res.tflops);
    printf("    Achieved bandwidth:    %.1f GB/s\n", res.mem_gbps);
    printf("    Compute efficiency:    %.1f%%\n", res.efficiency_compute);
    printf("    Memory efficiency:     %.1f%%\n", res.efficiency_memory);
    
    // Occupancy analysis (approximate)
    print_subsection("Occupancy Analysis (Approximate)");
    
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    int threads_per_block = Config::WARPS_PER_BLOCK * 32;
    int max_blocks_per_sm = prop.maxThreadsPerMultiProcessor / threads_per_block;
    
    // Estimate shared memory usage for V6
    int input_stage_size = (Config::TILE_H + 2) * (Config::TILE_W + 2) * (Config::WMMA_K + 8);
    int weight_stage_size = 9 * Config::OC_TILES_PER_BLOCK * Config::WMMA_N * (Config::WMMA_K + 8);
    int a_size = Config::WARPS_PER_BLOCK * Config::WMMA_M * (Config::WMMA_K + 8);
    size_t smem_per_block = (Config::STAGES * input_stage_size + Config::STAGES * weight_stage_size + a_size) * sizeof(half);
    
    int smem_limited_blocks = prop.sharedMemPerMultiprocessor / smem_per_block;
    int actual_blocks_per_sm = std::min(max_blocks_per_sm, (int)smem_limited_blocks);
    
    float occupancy = (float)(actual_blocks_per_sm * threads_per_block) / prop.maxThreadsPerMultiProcessor;
    
    printf("    Threads per block:     %d\n", threads_per_block);
    printf("    Shared mem per block:  ~%.1f KB\n", smem_per_block / 1024.0);
    printf("    Max blocks/SM (threads): %d\n", max_blocks_per_sm);
    printf("    Max blocks/SM (smem):    %d\n", smem_limited_blocks);
    printf("    Estimated occupancy:   %.0f%%\n", occupancy * 100);
}

// ============================================================================
// Memory Usage Test
// ============================================================================

void run_memory_test() {
    print_separator("Memory Usage Analysis");
    
    size_t free_before, total;
    CUDA_CHECK(cudaMemGetInfo(&free_before, &total));
    
    printf("\n  GPU Memory: %.1f GB total, %.1f GB free\n", 
           total / 1e9, free_before / 1e9);
    
    // Test various configurations
    struct MemConfig { int N, H, W, C, K; const char* name; };
    std::vector<MemConfig> configs = {
        {64, 8, 8, 64, 128, "Small"},
        {64, 16, 16, 128, 256, "Medium"},
        {64, 32, 32, 256, 512, "Large"},
        {128, 32, 32, 256, 512, "XLarge"},
    };
    
    printf("\n  %-12s %10s %10s %10s %10s %10s\n",
           "Config", "Input", "Weight", "Output", "Total", "% GPU");
    printf("  ");
    for (int i = 0; i < 65; i++) printf("-");
    printf("\n");
    
    for (const auto& cfg : configs) {
        size_t input_bytes = (size_t)cfg.N * cfg.H * cfg.W * cfg.C * sizeof(half);
        size_t weight_bytes = (size_t)cfg.K * cfg.C * 9 * sizeof(half);
        size_t bias_bytes = (size_t)cfg.K * sizeof(half);
        size_t output_bytes = (size_t)cfg.N * cfg.H * cfg.W * cfg.K * sizeof(half);
        size_t total_bytes = input_bytes + weight_bytes + bias_bytes + output_bytes;
        
        printf("  %-12s %9.1f MB %9.1f MB %9.1f MB %9.1f MB %9.1f%%\n",
               cfg.name,
               input_bytes / 1e6,
               weight_bytes / 1e6,
               output_bytes / 1e6,
               total_bytes / 1e6,
               100.0 * total_bytes / free_before);
    }
    
    // Find maximum batch size for a typical config
    print_subsection("Maximum Batch Size Analysis");
    
    int H = 16, W = 16, C = 128, K = 256;
    size_t per_sample = (size_t)H * W * C * sizeof(half) + 
                        (size_t)H * W * K * sizeof(half);
    size_t weight_fixed = (size_t)K * C * 9 * sizeof(half) + K * sizeof(half);
    
    // Leave 20% headroom for kernel workspace
    size_t available = (size_t)(free_before * 0.8) - weight_fixed;
    size_t max_batch = available / per_sample;
    
    printf("\n  For H=%d, W=%d, C=%d, K=%d:\n", H, W, C, K);
    printf("    Memory per sample:     %.2f MB\n", per_sample / 1e6);
    printf("    Fixed weight memory:   %.2f MB\n", weight_fixed / 1e6);
    printf("    Maximum batch size:    ~%zu\n", max_batch);
}

// ============================================================================
// ReLU Fusion Test
// ============================================================================

void run_relu_fusion_test() {
    print_separator("ReLU Fusion Test");
    
    int N = 64, H = 16, W = 16, C = 64, K = 128;
    
    printf("\n  Configuration: N=%d, H=%d, W=%d, C=%d, K=%d\n", N, H, W, C, K);
    
    // Sizes
    size_t input_size = (size_t)N * H * W * C;
    size_t weight_size = (size_t)K * C * 9;
    size_t bias_size = (size_t)K;
    size_t output_size = (size_t)N * H * W * K;
    
    // Host memory
    std::vector<half> h_input(input_size);
    std::vector<half> h_weight(weight_size);
    std::vector<half> h_bias(bias_size);
    std::vector<half> h_conv_out(output_size);
    std::vector<half> h_relu_out(output_size);
    
    init_data(h_input.data(), input_size);
    init_data(h_weight.data(), weight_size);
    init_data(h_bias.data(), bias_size);
    
    // Device memory
    half *d_input, *d_weight, *d_bias, *d_conv_out, *d_relu_out;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_conv_out, output_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_relu_out, output_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run fused kernel
    launch_conv2d_relu_fp16_wmma_v6(d_input, d_weight, d_bias, 
                                     d_relu_out, d_conv_out,
                                     N, H, W, C, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Copy back
    CUDA_CHECK(cudaMemcpy(h_conv_out.data(), d_conv_out, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_relu_out.data(), d_relu_out, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify ReLU is correctly applied
    int relu_errors = 0;
    int negative_inputs = 0;
    
    for (size_t i = 0; i < output_size; ++i) {
        float conv_val = __half2float(h_conv_out[i]);
        float relu_val = __half2float(h_relu_out[i]);
        float expected = fmaxf(conv_val, 0.0f);
        
        if (conv_val < 0) negative_inputs++;
        
        if (fabsf(relu_val - expected) > 1e-4f) {
            relu_errors++;
            if (relu_errors <= 5) {
                printf("  ReLU error at %zu: conv=%.4f, relu=%.4f, expected=%.4f\n",
                       i, conv_val, relu_val, expected);
            }
        }
    }
    
    printf("\n  Results:\n");
    printf("    Negative conv outputs: %d / %zu (%.1f%%)\n", 
           negative_inputs, output_size, 100.0 * negative_inputs / output_size);
    printf("    ReLU errors:           %d\n", relu_errors);
    printf("    Status:                %s\n", relu_errors == 0 ? "PASSED" : "FAILED");
    
    // Benchmark comparison
    print_subsection("Fusion Speedup");
    
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    // Warmup
    for (int i = 0; i < 10; ++i) {
        launch_conv2d_fp16_wmma_v6(d_input, d_weight, d_bias, d_conv_out, N, H, W, C, K, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark unfused (conv only, then separate ReLU would be needed)
    float unfused_time = 0;
    for (int i = 0; i < 50; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_conv2d_fp16_wmma_v6(d_input, d_weight, d_bias, d_conv_out, N, H, W, C, K, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        unfused_time += ms;
    }
    unfused_time /= 50;
    
    // Benchmark fused
    float fused_time = 0;
    for (int i = 0; i < 50; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_conv2d_relu_fp16_wmma_v6(d_input, d_weight, d_bias, d_relu_out, nullptr, N, H, W, C, K, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        fused_time += ms;
    }
    fused_time /= 50;
    
    printf("\n    Conv only:      %.3f ms\n", unfused_time);
    printf("    Conv + ReLU:    %.3f ms\n", fused_time);
    printf("    Overhead:       %.3f ms (%.1f%%)\n", 
           fused_time - unfused_time, 
           100.0 * (fused_time - unfused_time) / unfused_time);
    
    // Cleanup
    CUDA_CHECK(cudaEventDestroy(start));
    CUDA_CHECK(cudaEventDestroy(stop));
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_conv_out));
    CUDA_CHECK(cudaFree(d_relu_out));
}

// ============================================================================
// Main
// ============================================================================

void print_usage(const char* prog) {
    printf("Usage: %s [options]\n", prog);
    printf("\nOptions:\n");
    printf("  --test       Run correctness tests only\n");
    printf("  --bench      Run benchmarks only\n");
    printf("  --compare    Compare V5 vs V6 performance\n");
    printf("  --profile    Detailed profiling analysis\n");
    printf("  --mem        Memory usage analysis\n");
    printf("  --relu       ReLU fusion test\n");
    printf("  --all        Run everything (default)\n");
    printf("  --quick      Quick test (fewer iterations)\n");
    printf("  --verbose    Verbose output for tests\n");
    printf("  -h, --help   Show this help\n");
}

int main(int argc, char** argv) {
    bool run_tests = false;
    bool run_bench = false;
    bool run_compare = false;
    bool run_profile = false;
    bool run_mem = false;
    bool run_relu = false;
    bool run_all = true;
    bool verbose = false;
    
    // Parse args
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test") { run_tests = true; run_all = false; }
        else if (arg == "--bench") { run_bench = true; run_all = false; }
        else if (arg == "--compare") { run_compare = true; run_all = false; }
        else if (arg == "--profile") { run_profile = true; run_all = false; }
        else if (arg == "--mem") { run_mem = true; run_all = false; }
        else if (arg == "--relu") { run_relu = true; run_all = false; }
        else if (arg == "--all") { run_all = true; }
        else if (arg == "--verbose") { verbose = true; }
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
        else {
            printf("Unknown option: %s\n", arg.c_str());
            print_usage(argv[0]);
            return 1;
        }
    }
    
    if (run_all) {
        run_tests = run_bench = run_compare = run_profile = run_mem = run_relu = true;
    }
    
    // Initialize CUDA
    CUDA_CHECK(cudaSetDevice(0));
    
    print_separator("FP16 WMMA Conv2D V6 Test Suite");
    printf("  Kernel:     3x3, stride=1, padding=1\n");
    printf("  Features:   Double buffering, vectorized loads\n");
    printf("  Target:     T4 GPU (SM75) and newer\n");
    
    print_gpu_info();
    
    if (run_tests) {
        run_all_correctness_tests();
    }
    
    if (run_bench) {
        run_all_benchmarks();
    }
    
    if (run_compare) {
        run_comparison_benchmark();
    }
    
    if (run_profile) {
        run_profiling_analysis();
    }
    
    if (run_mem) {
        run_memory_test();
    }
    
    if (run_relu) {
        run_relu_fusion_test();
    }
    
    print_separator();
    printf("  Done!\n\n");
    
    return 0;
}