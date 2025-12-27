/**
 * Test and benchmark for FP16 WMMA Conv2D kernel V6
 * Target: T4 GPU (SM75) and newer
 * 
 * Build:
 *   nvcc -arch=sm_75 -O3 -o test_conv2d_v6 test_conv2d_v6.cu gpu_conv2d_fp16_forward_v6.cu
 * 
 * Run:
 *   ./test_conv2d_v6              # Run all tests
 *   ./test_conv2d_v6 --bench      # Benchmark only
 *   ./test_conv2d_v6 --test       # Correctness test only
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
    // V6 kernels
    void launch_conv2d_fp16_wmma_optimized(
        const half* input, const half* weight, const half* bias, half* output,
        int N, int H, int W, int C, int K, cudaStream_t stream);
    
    void launch_conv2d_relu_fp16_wmma_optimized(
        const half* input, const half* weight, const half* bias,
        half* output, half* conv_output,
        int N, int H, int W, int C, int K, cudaStream_t stream);
}

// ============================================================================
// Reference kernel (simple, correct)
// ============================================================================
__global__ void conv2d_reference_kernel(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H, int W, int C, int K
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * H * W * K;
    if (idx >= total) return;
    
    int k = idx % K;
    int w = (idx / K) % W;
    int h = (idx / (K * W)) % H;
    int n = idx / (K * W * H);
    
    float sum = 0.0f;
    for (int c = 0; c < C; c++) {
        for (int kh = 0; kh < 3; kh++) {
            for (int kw = 0; kw < 3; kw++) {
                int ih = h + kh - 1;
                int iw = w + kw - 1;
                if (ih >= 0 && ih < H && iw >= 0 && iw < W) {
                    float in_val = __half2float(input[((n * H + ih) * W + iw) * C + c]);
                    float w_val = __half2float(weight[((k * 3 + kh) * 3 + kw) * C + c]);
                    sum += in_val * w_val;
                }
            }
        }
    }
    sum += __half2float(bias[k]);
    output[idx] = __float2half(sum);
}

void launch_conv2d_reference(
    const half* input, const half* weight, const half* bias, half* output,
    int N, int H, int W, int C, int K, cudaStream_t stream
) {
    int total = N * H * W * K;
    int block = 256;
    int grid = (total + block - 1) / block;
    conv2d_reference_kernel<<<grid, block, 0, stream>>>(input, weight, bias, output, N, H, W, C, K);
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
    // Test tolerances
    constexpr float ABS_TOL = 0.05f;
    constexpr float REL_TOL = 0.02f;
    
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

void print_gpu_info() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    
    print_separator("GPU Information");
    printf("  Device:              %s\n", prop.name);
    printf("  Compute Capability:  SM%d%d\n", prop.major, prop.minor);
    printf("  SMs:                 %d\n", prop.multiProcessorCount);
    printf("  Memory:              %.1f GB\n", prop.totalGlobalMem / 1e9);
    printf("  Shared Mem/Block:    %zu KB\n", prop.sharedMemPerBlock / 1024);
    
    if (prop.major >= 7) {
        printf("  Tensor Cores:        Yes (SM%d%d)\n", prop.major, prop.minor);
    } else {
        printf("  Tensor Cores:        No (requires SM70+)\n");
    }
}

void init_data(half* data, size_t count, float scale = 0.1f, unsigned seed = 42) {
    srand(seed);
    for (size_t i = 0; i < count; ++i) {
        float val = ((float)rand() / RAND_MAX - 0.5f) * 2.0f * scale;
        data[i] = __float2half(val);
    }
}

struct CompareResult {
    float max_abs_err;
    float mean_abs_err;
    int num_errors;
    size_t total_elements;
    bool passed;
    
    void print() const {
        printf("    Max absolute error:  %.6f\n", max_abs_err);
        printf("    Mean absolute error: %.6f\n", mean_abs_err);
        printf("    Error count:         %d / %zu\n", num_errors, total_elements);
    }
};

CompareResult compare_outputs(const half* ref, const half* test, size_t count) {
    CompareResult result = {0, 0, 0, count, true};
    double sum_err = 0;
    
    for (size_t i = 0; i < count; ++i) {
        float r = __half2float(ref[i]);
        float t = __half2float(test[i]);
        float abs_err = fabsf(r - t);
        float rel_err = abs_err / (fabsf(r) + 1e-6f);
        
        sum_err += abs_err;
        
        bool is_error = (abs_err > Config::ABS_TOL) && (rel_err > Config::REL_TOL);
        if (is_error) result.num_errors++;
        
        if (abs_err > result.max_abs_err) {
            result.max_abs_err = abs_err;
        }
    }
    
    result.mean_abs_err = sum_err / count;
    result.passed = (result.num_errors == 0) || (result.max_abs_err < Config::ABS_TOL * 2);
    
    return result;
}

double calc_flops(int N, int H, int W, int C, int K) {
    return 2.0 * N * H * W * K * C * 9;
}

size_t calc_memory_bytes(int N, int H, int W, int C, int K) {
    size_t input_bytes = (size_t)N * H * W * C * sizeof(half);
    size_t weight_bytes = (size_t)K * C * 9 * sizeof(half);
    size_t bias_bytes = (size_t)K * sizeof(half);
    size_t output_bytes = (size_t)N * H * W * K * sizeof(half);
    return input_bytes + weight_bytes + bias_bytes + output_bytes;
}

// ============================================================================
// Correctness Tests
// ============================================================================

bool run_single_test(int N, int H, int W, int C, int K) {
    printf("  Testing N=%d, H=%d, W=%d, C=%d, K=%d ... ", N, H, W, C, K);
    fflush(stdout);
    
    size_t input_size = (size_t)N * H * W * C;
    size_t weight_size = (size_t)K * C * 9;
    size_t bias_size = (size_t)K;
    size_t output_size = (size_t)N * H * W * K;
    
    std::vector<half> h_input(input_size);
    std::vector<half> h_weight(weight_size);
    std::vector<half> h_bias(bias_size);
    std::vector<half> h_output_ref(output_size);
    std::vector<half> h_output_v6(output_size);
    
    init_data(h_input.data(), input_size, 0.1f, 42);
    init_data(h_weight.data(), weight_size, 0.1f, 123);
    init_data(h_bias.data(), bias_size, 0.1f, 456);
    
    half *d_input, *d_weight, *d_bias, *d_output_ref, *d_output_v6;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_ref, output_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output_v6, output_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Run reference
    launch_conv2d_reference(d_input, d_weight, d_bias, d_output_ref, N, H, W, C, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Run V6
    launch_conv2d_fp16_wmma_optimized(d_input, d_weight, d_bias, d_output_v6, N, H, W, C, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_output_ref.data(), d_output_ref, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_output_v6.data(), d_output_v6, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    CompareResult cmp = compare_outputs(h_output_ref.data(), h_output_v6.data(), output_size);
    
    if (cmp.passed) {
        printf("PASSED (max_err=%.6f)\n", cmp.max_abs_err);
    } else {
        printf("FAILED\n");
        cmp.print();
    }
    
    CUDA_CHECK(cudaFree(d_input));
    CUDA_CHECK(cudaFree(d_weight));
    CUDA_CHECK(cudaFree(d_bias));
    CUDA_CHECK(cudaFree(d_output_ref));
    CUDA_CHECK(cudaFree(d_output_v6));
    
    return cmp.passed;
}

void run_all_tests() {
    print_separator("Correctness Tests (V6 vs Reference)");
    
    struct TestCase { int N, H, W, C, K; const char* desc; };
    
    std::vector<TestCase> tests = {
        // Basic tests
        {1, 8, 8, 16, 16, "Minimal"},
        {1, 8, 8, 32, 32, "Small"},
        {2, 8, 8, 16, 32, "Multi-batch"},
        
        // Channel alignment tests
        {1, 8, 8, 3, 64, "RGB input"},
        {1, 8, 8, 17, 32, "C not aligned"},
        {1, 8, 8, 32, 17, "K not aligned"},
        
        // Medium tests
        {4, 16, 16, 32, 64, "Medium"},
        {8, 16, 16, 64, 128, "Larger"},
        
        // Large tests
        {16, 32, 32, 64, 128, "Large spatial"},
        {32, 32, 32, 128, 256, "Large all"},
        
        // Autoencoder stages
        {64, 32, 32, 3, 256, "AE Stage 1"},
        {64, 16, 16, 256, 128, "AE Stage 2"},
        {64, 8, 8, 128, 128, "AE Stage 3"},
        {64, 16, 16, 128, 256, "AE Stage 4"},
        {64, 32, 32, 256, 3, "AE Stage 5"}
    };
    
    int passed = 0, failed = 0;
    
    for (const auto& t : tests) {
        printf("  [%s] ", t.desc);
        if (run_single_test(t.N, t.H, t.W, t.C, t.K)) {
            passed++;
        } else {
            failed++;
        }
    }
    
    printf("\n  Summary: %d passed, %d failed\n", passed, failed);
}

// ============================================================================
// Benchmark
// ============================================================================

struct BenchResult {
    double avg_ms;
    double min_ms;
    double max_ms;
    double tflops;
    double mem_gbps;
    double efficiency;
};

BenchResult run_benchmark(int N, int H, int W, int C, int K) {
    BenchResult result = {0, 1e9, 0, 0, 0, 0};
    
    size_t input_size = (size_t)N * H * W * C;
    size_t weight_size = (size_t)K * C * 9;
    size_t bias_size = (size_t)K;
    size_t output_size = (size_t)N * H * W * K;
    
    std::vector<half> h_input(input_size);
    std::vector<half> h_weight(weight_size);
    std::vector<half> h_bias(bias_size);
    
    init_data(h_input.data(), input_size);
    init_data(h_weight.data(), weight_size);
    init_data(h_bias.data(), bias_size);
    
    half *d_input, *d_weight, *d_bias, *d_output;
    CUDA_CHECK(cudaMalloc(&d_input, input_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_weight, weight_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_bias, bias_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_output, output_size * sizeof(half)));
    
    CUDA_CHECK(cudaMemcpy(d_input, h_input.data(), input_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_weight, h_weight.data(), weight_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_bias, h_bias.data(), bias_size * sizeof(half), cudaMemcpyHostToDevice));
    
    // Warmup
    for (int i = 0; i < Config::WARMUP_ITERS; ++i) {
        launch_conv2d_fp16_wmma_optimized(d_input, d_weight, d_bias, d_output, N, H, W, C, K, 0);
    }
    CUDA_CHECK(cudaDeviceSynchronize());
    
    // Benchmark
    cudaEvent_t start, stop;
    CUDA_CHECK(cudaEventCreate(&start));
    CUDA_CHECK(cudaEventCreate(&stop));
    
    std::vector<float> times(Config::BENCH_ITERS);
    
    for (int i = 0; i < Config::BENCH_ITERS; ++i) {
        CUDA_CHECK(cudaEventRecord(start));
        launch_conv2d_fp16_wmma_optimized(d_input, d_weight, d_bias, d_output, N, H, W, C, K, 0);
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        times[i] = ms;
        
        result.min_ms = fmin(result.min_ms, ms);
        result.max_ms = fmax(result.max_ms, ms);
    }
    
    // Calculate average (trimmed mean)
    std::sort(times.begin(), times.end());
    int trim = Config::BENCH_ITERS / 10;
    double sum = 0;
    for (int i = trim; i < Config::BENCH_ITERS - trim; ++i) {
        sum += times[i];
    }
    result.avg_ms = sum / (Config::BENCH_ITERS - 2 * trim);
    
    // Calculate metrics
    double flops = calc_flops(N, H, W, C, K);
    result.tflops = flops / (result.avg_ms * 1e9);
    
    size_t bytes = calc_memory_bytes(N, H, W, C, K);
    result.mem_gbps = bytes / (result.avg_ms * 1e6);
    
    result.efficiency = (result.tflops / Config::T4_FP16_TFLOPS) * 100.0;
    
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
    
    struct BenchConfig { int N, H, W, C, K; const char* name; };
    
    std::vector<BenchConfig> configs = {
        // Various configurations
        {64,  8,  8,  64, 128, "B64   8x8   64->128"},
        {64, 16, 16,  64, 128, "B64  16x16  64->128"},
        {64, 16, 16, 128, 256, "B64  16x16 128->256"},
        {64, 32, 32,  64, 128, "B64  32x32  64->128"},
        {64, 32, 32, 128, 256, "B64  32x32 128->256"},
        {128, 16, 16, 64, 128, "B128 16x16  64->128"},
        
        // Autoencoder stages
        {64, 32, 32, 3, 256, "AE Stage 1 (enc)"},
        {64, 16, 16, 256, 128, "AE Stage 2 (enc)"},
        {64, 8, 8, 128, 128, "AE Stage 3 (latent)"},
        {64, 16, 16, 128, 256, "AE Stage 4 (dec)"},
        {64, 32, 32, 256, 3, "AE Stage 5 (dec)"}
    };
    
    printf("\n  %-22s %8s %8s %8s %8s\n", 
           "Config", "Avg(ms)", "TFLOPS", "Eff%", "GB/s");
    printf("  ");
    for (int i = 0; i < 60; i++) printf("-");
    printf("\n");
    
    for (const auto& cfg : configs) {
        BenchResult res = run_benchmark(cfg.N, cfg.H, cfg.W, cfg.C, cfg.K);
        
        printf("  %-22s %8.3f %8.2f %7.1f%% %8.1f\n",
               cfg.name,
               res.avg_ms,
               res.tflops,
               res.efficiency,
               res.mem_gbps);
    }
    
    printf("\n  Eff%% = Efficiency vs T4 peak (65 TFLOPS)\n");
}

// ============================================================================
// Profiling Analysis
// ============================================================================

void run_profiling() {
    print_separator("Profiling Analysis");
    
    int N = 64, H = 16, W = 16, C = 128, K = 256;
    
    printf("\n  Configuration: N=%d, H=%d, W=%d, C=%d, K=%d\n", N, H, W, C, K);
    
    double flops = calc_flops(N, H, W, C, K);
    size_t bytes = calc_memory_bytes(N, H, W, C, K);
    double ai = flops / bytes;
    
    printf("\n  Theoretical Analysis:\n");
    printf("    Total FLOPs:           %.2f GFLOPS\n", flops / 1e9);
    printf("    Memory transfer:       %.2f MB\n", bytes / 1e6);
    printf("    Arithmetic intensity:  %.2f FLOPs/byte\n", ai);
    
    BenchResult res = run_benchmark(N, H, W, C, K);
    
    printf("\n  Performance:\n");
    printf("    Average time:          %.3f ms\n", res.avg_ms);
    printf("    Achieved TFLOPS:       %.2f\n", res.tflops);
    printf("    Achieved bandwidth:    %.1f GB/s\n", res.mem_gbps);
    printf("    Compute efficiency:    %.1f%%\n", res.efficiency);
}

// ============================================================================
// ReLU Fusion Test
// ============================================================================

void run_relu_test() {
    print_separator("ReLU Fusion Test");
    
    int N = 64, H = 16, W = 16, C = 64, K = 128;
    
    printf("\n  Configuration: N=%d, H=%d, W=%d, C=%d, K=%d\n", N, H, W, C, K);
    
    size_t input_size = (size_t)N * H * W * C;
    size_t weight_size = (size_t)K * C * 9;
    size_t bias_size = (size_t)K;
    size_t output_size = (size_t)N * H * W * K;
    
    std::vector<half> h_input(input_size);
    std::vector<half> h_weight(weight_size);
    std::vector<half> h_bias(bias_size);
    std::vector<half> h_conv_out(output_size);
    std::vector<half> h_relu_out(output_size);
    
    init_data(h_input.data(), input_size);
    init_data(h_weight.data(), weight_size);
    init_data(h_bias.data(), bias_size);
    
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
    launch_conv2d_relu_fp16_wmma_optimized(d_input, d_weight, d_bias, 
                                            d_relu_out, d_conv_out,
                                            N, H, W, C, K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    
    CUDA_CHECK(cudaMemcpy(h_conv_out.data(), d_conv_out, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_relu_out.data(), d_relu_out, output_size * sizeof(half), cudaMemcpyDeviceToHost));
    
    // Verify ReLU
    int relu_errors = 0;
    for (size_t i = 0; i < output_size; ++i) {
        float conv_val = __half2float(h_conv_out[i]);
        float relu_val = __half2float(h_relu_out[i]);
        float expected = fmaxf(conv_val, 0.0f);
        
        if (fabsf(relu_val - expected) > 1e-4f) {
            relu_errors++;
        }
    }
    
    printf("\n  ReLU errors: %d\n", relu_errors);
    printf("  Status: %s\n", relu_errors == 0 ? "PASSED" : "FAILED");
    
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
    printf("  --profile    Profiling analysis\n");
    printf("  --relu       ReLU fusion test\n");
    printf("  --all        Run everything (default)\n");
    printf("  -h, --help   Show this help\n");
}

int main(int argc, char** argv) {
    bool run_tests = false;
    bool run_bench = false;
    bool run_profile = false;
    bool run_relu = false;
    bool run_all = true;
    
    for (int i = 1; i < argc; ++i) {
        std::string arg = argv[i];
        if (arg == "--test") { run_tests = true; run_all = false; }
        else if (arg == "--bench") { run_bench = true; run_all = false; }
        else if (arg == "--profile") { run_profile = true; run_all = false; }
        else if (arg == "--relu") { run_relu = true; run_all = false; }
        else if (arg == "--all") { run_all = true; }
        else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            return 0;
        }
    }
    
    if (run_all) {
        run_tests = run_bench = run_profile = run_relu = true;
    }
    
    CUDA_CHECK(cudaSetDevice(0));
    
    print_separator("FP16 WMMA Conv2D V6 Test Suite");
    printf("  Kernel: 3x3, stride=1, padding=1\n");
    
    print_gpu_info();
    
    if (run_tests) run_all_tests();
    if (run_bench) run_all_benchmarks();
    if (run_profile) run_profiling();
    if (run_relu) run_relu_test();
    
    print_separator();
    printf("  Done!\n\n");
    
    return 0;
}
