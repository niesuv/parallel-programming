// test_backward_v2.cu
// Test suite for optimized FP16 backward pass

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <iomanip>
#include <algorithm>

#define CUDA_CHECK(x) do { \
    cudaError_t err = (x); \
    if (err != cudaSuccess) { \
        std::cerr << "CUDA error " << cudaGetErrorString(err) << " at " << __FILE__ << ":" << __LINE__ << std::endl; \
        exit(1); \
    } \
} while(0)

// Forward declarations
extern "C" {
void launch_relu_backward_opt(const half*, const half*, half*, int, cudaStream_t);
void launch_conv2d_backward_input_opt(const half*, const half*, half*, int, int, int, int, int, int, int, cudaStream_t);
void launch_conv2d_backward_weight_opt(const half*, const half*, float*, float*, int, int, int, int, int, int, int, cudaStream_t);
void launch_fused_relu_backward_input_opt(const half*, const half*, const half*, half*, int, int, int, int, int, int, int, cudaStream_t);
void launch_fused_relu_backward_input_twopass(const half*, const half*, const half*, half*, half*, int, int, int, int, int, int, int, cudaStream_t);
}

// CPU References
void cpu_relu_backward(const float* g, const float* c, float* o, int size) {
    for (int i = 0; i < size; i++) o[i] = (c[i] > 0) ? g[i] : 0;
}

void cpu_conv_backward_input(const float* grad_out, const float* weight, float* grad_in,
                              int N, int H, int W, int C, int H_out, int W_out, int K) {
    std::fill(grad_in, grad_in + N*H*W*C, 0.0f);
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < C; c++) {
                    float sum = 0;
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++) {
                            int oh = h - kh + 1, ow = w - kw + 1;
                            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out)
                                for (int k = 0; k < K; k++)
                                    sum += grad_out[n*H_out*W_out*K + oh*W_out*K + ow*K + k] *
                                           weight[k*9*C + (2-kh)*3*C + (2-kw)*C + c];
                        }
                    grad_in[n*H*W*C + h*W*C + w*C + c] = sum;
                }
}

void cpu_conv_backward_weight(const float* grad_out, const float* input, float* grad_w, float* grad_b,
                               int N, int H, int W, int C, int H_out, int W_out, int K) {
    std::fill(grad_w, grad_w + K*9*C, 0.0f);
    std::fill(grad_b, grad_b + K, 0.0f);
    for (int k = 0; k < K; k++) {
        for (int kh = 0; kh < 3; kh++)
            for (int kw = 0; kw < 3; kw++)
                for (int c = 0; c < C; c++) {
                    float sum = 0;
                    for (int n = 0; n < N; n++)
                        for (int oh = 0; oh < H_out; oh++)
                            for (int ow = 0; ow < W_out; ow++) {
                                int ih = oh + kh, iw = ow + kw;
                                if (ih < H && iw < W)
                                    sum += grad_out[n*H_out*W_out*K + oh*W_out*K + ow*K + k] *
                                           input[n*H*W*C + ih*W*C + iw*C + c];
                            }
                    grad_w[k*9*C + kh*3*C + kw*C + c] = sum;
                }
        float bsum = 0;
        for (int n = 0; n < N; n++)
            for (int oh = 0; oh < H_out; oh++)
                for (int ow = 0; ow < W_out; ow++)
                    bsum += grad_out[n*H_out*W_out*K + oh*W_out*K + ow*K + k];
        grad_b[k] = bsum;
    }
}

void cpu_fused_relu_backward_input(const float* upstream, const float* conv_out, const float* weight,
                                    float* grad_in, int N, int H, int W, int C, int H_out, int W_out, int K) {
    std::fill(grad_in, grad_in + N*H*W*C, 0.0f);
    for (int n = 0; n < N; n++)
        for (int h = 0; h < H; h++)
            for (int w = 0; w < W; w++)
                for (int c = 0; c < C; c++) {
                    float sum = 0;
                    for (int kh = 0; kh < 3; kh++)
                        for (int kw = 0; kw < 3; kw++) {
                            int oh = h - kh + 1, ow = w - kw + 1;
                            if (oh >= 0 && oh < H_out && ow >= 0 && ow < W_out)
                                for (int k = 0; k < K; k++) {
                                    int idx = n*H_out*W_out*K + oh*W_out*K + ow*K + k;
                                    float g = (conv_out[idx] > 0) ? upstream[idx] : 0;
                                    sum += g * weight[k*9*C + (2-kh)*3*C + (2-kw)*C + c];
                                }
                        }
                    grad_in[n*H*W*C + h*W*C + w*C + c] = sum;
                }
}

// Utilities
void to_fp16(const float* src, half* dst, size_t n) { for (size_t i = 0; i < n; i++) dst[i] = __float2half(src[i]); }
void to_fp32(const half* src, float* dst, size_t n) { for (size_t i = 0; i < n; i++) dst[i] = __half2float(src[i]); }
float rel_error(const float* ref, const float* test, size_t n) {
    float err = 0, mag = 0;
    for (size_t i = 0; i < n; i++) { err += std::abs(ref[i] - test[i]); mag += std::abs(ref[i]); }
    return mag > 1e-8f ? err / mag : err;
}

struct Config { int N, H, W, C, K; const char* name; };

// Tests
bool test_relu(const Config& cfg) {
    int size = cfg.N * cfg.H * cfg.W * cfg.K;
    std::vector<float> h_g(size), h_c(size), h_ref(size), h_gpu(size);
    std::mt19937 gen(42); std::uniform_real_distribution<float> dist(-1, 1);
    for (int i = 0; i < size; i++) { h_g[i] = dist(gen); h_c[i] = dist(gen); }
    
    cpu_relu_backward(h_g.data(), h_c.data(), h_ref.data(), size);
    
    half *d_g, *d_c, *d_o;
    CUDA_CHECK(cudaMalloc(&d_g, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_o, size * sizeof(half)));
    
    std::vector<half> h_g16(size), h_c16(size);
    to_fp16(h_g.data(), h_g16.data(), size);
    to_fp16(h_c.data(), h_c16.data(), size);
    CUDA_CHECK(cudaMemcpy(d_g, h_g16.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c16.data(), size * sizeof(half), cudaMemcpyHostToDevice));
    
    launch_relu_backward_opt(d_g, d_c, d_o, size, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<half> h_o16(size);
    CUDA_CHECK(cudaMemcpy(h_o16.data(), d_o, size * sizeof(half), cudaMemcpyDeviceToHost));
    to_fp32(h_o16.data(), h_gpu.data(), size);
    
    float err = rel_error(h_ref.data(), h_gpu.data(), size);
    bool ok = err < 0.01f;
    std::cout << "[ReLU] " << cfg.name << ": err=" << err << (ok ? " PASS" : " FAIL") << std::endl;
    
    cudaFree(d_g); cudaFree(d_c); cudaFree(d_o);
    return ok;
}

bool test_backward_input(const Config& cfg) {
    int H_out = cfg.H, W_out = cfg.W;
    int g_size = cfg.N * H_out * W_out * cfg.K;
    int w_size = cfg.K * 9 * cfg.C;
    int o_size = cfg.N * cfg.H * cfg.W * cfg.C;
    
    std::vector<float> h_g(g_size), h_w(w_size), h_ref(o_size), h_gpu(o_size);
    std::mt19937 gen(42); std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < g_size; i++) h_g[i] = dist(gen);
    for (int i = 0; i < w_size; i++) h_w[i] = dist(gen);
    
    cpu_conv_backward_input(h_g.data(), h_w.data(), h_ref.data(), cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K);
    
    half *d_g, *d_w, *d_o;
    CUDA_CHECK(cudaMalloc(&d_g, g_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_o, o_size * sizeof(half)));
    
    std::vector<half> h_g16(g_size), h_w16(w_size);
    to_fp16(h_g.data(), h_g16.data(), g_size);
    to_fp16(h_w.data(), h_w16.data(), w_size);
    CUDA_CHECK(cudaMemcpy(d_g, h_g16.data(), g_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w16.data(), w_size * sizeof(half), cudaMemcpyHostToDevice));
    
    launch_conv2d_backward_input_opt(d_g, d_w, d_o, cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<half> h_o16(o_size);
    CUDA_CHECK(cudaMemcpy(h_o16.data(), d_o, o_size * sizeof(half), cudaMemcpyDeviceToHost));
    to_fp32(h_o16.data(), h_gpu.data(), o_size);
    
    float err = rel_error(h_ref.data(), h_gpu.data(), o_size);
    bool ok = err < 0.05f;
    std::cout << "[BackIn] " << cfg.name << ": err=" << err << (ok ? " PASS" : " FAIL") << std::endl;
    
    cudaFree(d_g); cudaFree(d_w); cudaFree(d_o);
    return ok;
}

bool test_backward_weight(const Config& cfg) {
    int H_out = cfg.H, W_out = cfg.W;
    int g_size = cfg.N * H_out * W_out * cfg.K;
    int i_size = cfg.N * cfg.H * cfg.W * cfg.C;
    int w_size = cfg.K * 9 * cfg.C;
    
    std::vector<float> h_g(g_size), h_i(i_size), h_wref(w_size), h_wgpu(w_size), h_bref(cfg.K), h_bgpu(cfg.K);
    std::mt19937 gen(42); std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < g_size; i++) h_g[i] = dist(gen);
    for (int i = 0; i < i_size; i++) h_i[i] = dist(gen);
    
    cpu_conv_backward_weight(h_g.data(), h_i.data(), h_wref.data(), h_bref.data(), cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K);
    
    half *d_g, *d_i; float *d_w, *d_b;
    CUDA_CHECK(cudaMalloc(&d_g, g_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_i, i_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_b, cfg.K * sizeof(float)));
    
    std::vector<half> h_g16(g_size), h_i16(i_size);
    to_fp16(h_g.data(), h_g16.data(), g_size);
    to_fp16(h_i.data(), h_i16.data(), i_size);
    CUDA_CHECK(cudaMemcpy(d_g, h_g16.data(), g_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_i, h_i16.data(), i_size * sizeof(half), cudaMemcpyHostToDevice));
    
    launch_conv2d_backward_weight_opt(d_g, d_i, d_w, d_b, cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    CUDA_CHECK(cudaMemcpy(h_wgpu.data(), d_w, w_size * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_bgpu.data(), d_b, cfg.K * sizeof(float), cudaMemcpyDeviceToHost));
    
    float w_err = rel_error(h_wref.data(), h_wgpu.data(), w_size);
    float b_err = rel_error(h_bref.data(), h_bgpu.data(), cfg.K);
    bool ok = w_err < 0.05f && b_err < 0.05f;
    std::cout << "[BackW] " << cfg.name << ": w_err=" << w_err << " b_err=" << b_err << (ok ? " PASS" : " FAIL") << std::endl;
    
    cudaFree(d_g); cudaFree(d_i); cudaFree(d_w); cudaFree(d_b);
    return ok;
}

bool test_fused(const Config& cfg) {
    int H_out = cfg.H, W_out = cfg.W;
    int u_size = cfg.N * H_out * W_out * cfg.K;
    int w_size = cfg.K * 9 * cfg.C;
    int o_size = cfg.N * cfg.H * cfg.W * cfg.C;
    
    std::vector<float> h_u(u_size), h_c(u_size), h_w(w_size), h_ref(o_size), h_gpu(o_size);
    std::mt19937 gen(42); std::uniform_real_distribution<float> dist(-0.5f, 0.5f);
    for (int i = 0; i < u_size; i++) { h_u[i] = dist(gen); h_c[i] = dist(gen); }
    for (int i = 0; i < w_size; i++) h_w[i] = dist(gen);
    
    cpu_fused_relu_backward_input(h_u.data(), h_c.data(), h_w.data(), h_ref.data(), cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K);
    
    half *d_u, *d_c, *d_w, *d_o;
    CUDA_CHECK(cudaMalloc(&d_u, u_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, u_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_o, o_size * sizeof(half)));
    
    std::vector<half> h_u16(u_size), h_c16(u_size), h_w16(w_size);
    to_fp16(h_u.data(), h_u16.data(), u_size);
    to_fp16(h_c.data(), h_c16.data(), u_size);
    to_fp16(h_w.data(), h_w16.data(), w_size);
    CUDA_CHECK(cudaMemcpy(d_u, h_u16.data(), u_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_c, h_c16.data(), u_size * sizeof(half), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_w, h_w16.data(), w_size * sizeof(half), cudaMemcpyHostToDevice));
    
    launch_fused_relu_backward_input_opt(d_u, d_c, d_w, d_o, cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K, 0);
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaGetLastError());
    
    std::vector<half> h_o16(o_size);
    CUDA_CHECK(cudaMemcpy(h_o16.data(), d_o, o_size * sizeof(half), cudaMemcpyDeviceToHost));
    to_fp32(h_o16.data(), h_gpu.data(), o_size);
    
    float err = rel_error(h_ref.data(), h_gpu.data(), o_size);
    bool ok = err < 0.05f;
    std::cout << "[Fused] " << cfg.name << ": err=" << err << (ok ? " PASS" : " FAIL") << std::endl;
    
    cudaFree(d_u); cudaFree(d_c); cudaFree(d_w); cudaFree(d_o);
    return ok;
}

// Benchmark
void benchmark_config(const Config& cfg) {
    int H_out = cfg.H, W_out = cfg.W;
    size_t g_size = (size_t)cfg.N * H_out * W_out * cfg.K;
    size_t i_size = (size_t)cfg.N * cfg.H * cfg.W * cfg.C;
    size_t w_size = (size_t)cfg.K * 9 * cfg.C;
    size_t relu_size = g_size;
    
    half *d_g, *d_c, *d_i, *d_w, *d_o_relu, *d_o_conv, *d_temp;
    float *d_gw, *d_gb;
    CUDA_CHECK(cudaMalloc(&d_g, g_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_c, g_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_i, i_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_w, w_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_o_relu, relu_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_o_conv, i_size * sizeof(half)));
    CUDA_CHECK(cudaMalloc(&d_temp, g_size * sizeof(half)));  // For two-pass
    CUDA_CHECK(cudaMalloc(&d_gw, w_size * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&d_gb, cfg.K * sizeof(float)));
    CUDA_CHECK(cudaMemset(d_g, 0, g_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_c, 0, g_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_i, 0, i_size * sizeof(half)));
    CUDA_CHECK(cudaMemset(d_w, 0, w_size * sizeof(half)));
    
    auto bench = [](auto fn, int warmup=5, int iters=50) {
        for (int i = 0; i < warmup; i++) fn();
        CUDA_CHECK(cudaDeviceSynchronize());
        CUDA_CHECK(cudaGetLastError());
        
        cudaEvent_t start, stop;
        CUDA_CHECK(cudaEventCreate(&start));
        CUDA_CHECK(cudaEventCreate(&stop));
        CUDA_CHECK(cudaEventRecord(start));
        for (int i = 0; i < iters; i++) fn();
        CUDA_CHECK(cudaEventRecord(stop));
        CUDA_CHECK(cudaEventSynchronize(stop));
        float ms;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
        CUDA_CHECK(cudaEventDestroy(start));
        CUDA_CHECK(cudaEventDestroy(stop));
        return ms / iters;
    };
    
    float relu_ms = bench([&]() { launch_relu_backward_opt(d_g, d_c, d_o_relu, g_size, 0); });
    float back_in_ms = bench([&]() { launch_conv2d_backward_input_opt(d_g, d_w, d_o_conv, cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K, 0); });
    float back_w_ms = bench([&]() { launch_conv2d_backward_weight_opt(d_g, d_i, d_gw, d_gb, cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K, 0); });
    float fused_ms = bench([&]() { launch_fused_relu_backward_input_opt(d_g, d_c, d_w, d_o_conv, cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K, 0); });
    float twopass_ms = bench([&]() { launch_fused_relu_backward_input_twopass(d_g, d_c, d_w, d_o_conv, d_temp, cfg.N, cfg.H, cfg.W, cfg.C, H_out, W_out, cfg.K, 0); });
    
    double flops_in = 2.0 * cfg.N * cfg.H * cfg.W * cfg.C * cfg.K * 9;
    double flops_w = 2.0 * cfg.N * H_out * W_out * cfg.K * 9 * cfg.C;
    
    std::cout << std::left << std::setw(25) << cfg.name
              << std::fixed << std::setprecision(3)
              << std::setw(10) << relu_ms
              << std::setw(10) << back_in_ms
              << std::setw(10) << back_w_ms
              << std::setw(10) << fused_ms
              << std::setw(10) << twopass_ms
              << std::setprecision(2)
              << std::setw(8) << (flops_in/1e12)/(back_in_ms/1e3)
              << std::setw(8) << (flops_w/1e12)/(back_w_ms/1e3)
              << std::endl;
    
    cudaFree(d_g); cudaFree(d_c); cudaFree(d_i); cudaFree(d_w); 
    cudaFree(d_o_relu); cudaFree(d_o_conv); cudaFree(d_temp); cudaFree(d_gw); cudaFree(d_gb);
}

int main() {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, 0));
    std::cout << "Device: " << prop.name << " (SM" << prop.major << prop.minor << ")" << std::endl;
    
    Config configs[] = {
        {2, 16, 16, 32, 32, "Tiny"},
        {4, 32, 32, 64, 64, "Small"},
        {64, 32, 32, 3, 256, "Stage1"},
        {64, 16, 16, 256, 128, "Stage2"},
        {64, 8, 8, 128, 64, "Stage3"},
        {64, 8, 8, 64, 128, "Stage4"},
        {64, 16, 16, 128, 256, "Stage5"},
        {64, 32, 32, 256, 3, "Stage6"},
    };
    
    std::cout << "\n=== CORRECTNESS TESTS ===" << std::endl;
    int pass = 0, total = 0;
    for (auto& c : configs) {
        total += 4;
        if (test_relu(c)) pass++;
        if (test_backward_input(c)) pass++;
        if (test_backward_weight(c)) pass++;
        if (test_fused(c)) pass++;
    }
    std::cout << "\nPassed: " << pass << "/" << total << std::endl;
    
    std::cout << "\n=== PERFORMANCE BENCHMARKS ===" << std::endl;
    std::cout << std::left << std::setw(25) << "Config"
              << std::setw(10) << "ReLU"
              << std::setw(10) << "BackIn"
              << std::setw(10) << "BackW"
              << std::setw(10) << "Fused"
              << std::setw(10) << "2Pass"
              << std::setw(8) << "TF(In)"
              << std::setw(8) << "TF(W)"
              << std::endl;
    std::cout << std::string(91, '-') << std::endl;
    
    Config bench_configs[] = {
        {64, 32, 32, 3, 256, "Stage1 (C=3)"},
        {64, 16, 16, 256, 128, "Stage2"},
        {64, 8, 8, 128, 64, "Stage3"},
        {64, 8, 8, 64, 128, "Stage4"},
        {64, 16, 16, 128, 256, "Stage5"},
        {64, 32, 32, 256, 3, "Stage6 (K=3)"},
    };
    for (auto& c : bench_configs) benchmark_config(c);
    
    return (pass == total) ? 0 : 1;
}
