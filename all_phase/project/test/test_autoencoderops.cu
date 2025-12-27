// test_autoencoder_ops.cu
// Test MaxPool2D, Upsample2D, MSELoss, and SGD kernels

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <vector>

// Declarations from autoencoder_ops.cu
extern "C" {
void launch_maxpool2d_forward(const half* input, half* output, int8_t* max_indices,
                               int N, int H, int W, int C, cudaStream_t stream);
void launch_maxpool2d_backward(const half* grad_output, const int8_t* max_indices,
                                half* grad_input, int N, int H, int W, int C, cudaStream_t stream);
void launch_upsample2d_forward(const half* input, half* output,
                                int N, int H, int W, int C, cudaStream_t stream);
void launch_upsample2d_backward(const half* grad_output, half* grad_input,
                                 int N, int H, int W, int C, cudaStream_t stream);
void launch_mse_loss_grad(const half* pred, const half* target, half* grad,
                          float* loss, float* partial_buffer, int size, cudaStream_t stream);
void launch_sgd_update_conv2d(float* master_weight, half* weight, const float* grad_weight,
                              float* master_bias, half* bias, const float* grad_bias,
                              float lr, int weight_size, int bias_size, cudaStream_t stream);
}

// ============================================================================
// CPU Reference Implementations
// ============================================================================

void maxpool2d_forward_cpu(const float* input, float* output, int* indices,
                           int N, int H, int W, int C) {
    int H_out = H / 2, W_out = W / 2;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                for (int c = 0; c < C; c++) {
                    int ih = oh * 2, iw = ow * 2;
                    float v00 = input[n*H*W*C + ih*W*C + iw*C + c];
                    float v01 = input[n*H*W*C + ih*W*C + (iw+1)*C + c];
                    float v10 = input[n*H*W*C + (ih+1)*W*C + iw*C + c];
                    float v11 = input[n*H*W*C + (ih+1)*W*C + (iw+1)*C + c];
                    
                    float max_val = v00; int max_idx = 0;
                    if (v01 > max_val) { max_val = v01; max_idx = 1; }
                    if (v10 > max_val) { max_val = v10; max_idx = 2; }
                    if (v11 > max_val) { max_val = v11; max_idx = 3; }
                    
                    int out_idx = n*H_out*W_out*C + oh*W_out*C + ow*C + c;
                    output[out_idx] = max_val;
                    indices[out_idx] = max_idx;
                }
            }
        }
    }
}

void maxpool2d_backward_cpu(const float* grad_out, const int* indices, float* grad_in,
                            int N, int H, int W, int C) {
    int H_out = H / 2, W_out = W / 2;
    for (int i = 0; i < N*H*W*C; i++) grad_in[i] = 0.0f;
    
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                for (int c = 0; c < C; c++) {
                    int out_idx = n*H_out*W_out*C + oh*W_out*C + ow*C + c;
                    int max_idx = indices[out_idx];
                    int ih = oh * 2 + max_idx / 2;
                    int iw = ow * 2 + max_idx % 2;
                    grad_in[n*H*W*C + ih*W*C + iw*C + c] = grad_out[out_idx];
                }
            }
        }
    }
}

void upsample2d_forward_cpu(const float* input, float* output, int N, int H, int W, int C) {
    int H_out = H * 2, W_out = W * 2;
    for (int n = 0; n < N; n++) {
        for (int oh = 0; oh < H_out; oh++) {
            for (int ow = 0; ow < W_out; ow++) {
                for (int c = 0; c < C; c++) {
                    output[n*H_out*W_out*C + oh*W_out*C + ow*C + c] = 
                        input[n*H*W*C + (oh/2)*W*C + (ow/2)*C + c];
                }
            }
        }
    }
}

void upsample2d_backward_cpu(const float* grad_out, float* grad_in, int N, int H, int W, int C) {
    int H_out = H * 2, W_out = W * 2;
    for (int n = 0; n < N; n++) {
        for (int ih = 0; ih < H; ih++) {
            for (int iw = 0; iw < W; iw++) {
                for (int c = 0; c < C; c++) {
                    int oh = ih * 2, ow = iw * 2;
                    float sum = grad_out[n*H_out*W_out*C + oh*W_out*C + ow*C + c];
                    sum += grad_out[n*H_out*W_out*C + oh*W_out*C + (ow+1)*C + c];
                    sum += grad_out[n*H_out*W_out*C + (oh+1)*W_out*C + ow*C + c];
                    sum += grad_out[n*H_out*W_out*C + (oh+1)*W_out*C + (ow+1)*C + c];
                    grad_in[n*H*W*C + ih*W*C + iw*C + c] = sum;
                }
            }
        }
    }
}

float mse_loss_cpu(const float* pred, const float* target, int size) {
    float sum = 0.0f;
    for (int i = 0; i < size; i++) {
        float diff = pred[i] - target[i];
        sum += diff * diff;
    }
    return sum / size;
}

void mse_grad_cpu(const float* pred, const float* target, float* grad, int size) {
    float scale = 2.0f / size;
    for (int i = 0; i < size; i++) {
        grad[i] = (pred[i] - target[i]) * scale;
    }
}

void sgd_update_cpu(float* master, const float* grad, float lr, int size) {
    for (int i = 0; i < size; i++) master[i] -= lr * grad[i];
}

// ============================================================================
// Test Functions
// ============================================================================

bool test_maxpool2d(int N, int H, int W, int C) {
    printf("Testing MaxPool2D: N=%d, H=%d, W=%d, C=%d\n", N, H, W, C);
    
    int in_size = N * H * W * C;
    int out_size = N * (H/2) * (W/2) * C;
    
    std::vector<float> h_in(in_size), h_out_ref(out_size), h_out(out_size);
    std::vector<int> h_idx_ref(out_size);
    std::vector<int8_t> h_idx(out_size);
    
    for (int i = 0; i < in_size; i++) h_in[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    maxpool2d_forward_cpu(h_in.data(), h_out_ref.data(), h_idx_ref.data(), N, H, W, C);
    
    half *d_in, *d_out; int8_t *d_idx;
    cudaMalloc(&d_in, in_size * sizeof(half));
    cudaMalloc(&d_out, out_size * sizeof(half));
    cudaMalloc(&d_idx, out_size * sizeof(int8_t));
    
    std::vector<half> h_in_half(in_size);
    for (int i = 0; i < in_size; i++) h_in_half[i] = __float2half(h_in[i]);
    cudaMemcpy(d_in, h_in_half.data(), in_size * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_maxpool2d_forward(d_in, d_out, d_idx, N, H, W, C, 0);
    cudaDeviceSynchronize();
    
    std::vector<half> h_out_half(out_size);
    cudaMemcpy(h_out_half.data(), d_out, out_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_idx.data(), d_idx, out_size * sizeof(int8_t), cudaMemcpyDeviceToHost);
    for (int i = 0; i < out_size; i++) h_out[i] = __half2float(h_out_half[i]);
    
    float max_err = 0.0f; int idx_err = 0;
    for (int i = 0; i < out_size; i++) {
        float err = fabs(h_out[i] - h_out_ref[i]);
        if (err > max_err) max_err = err;
        if (h_idx[i] != h_idx_ref[i]) idx_err++;
    }
    bool pass_fwd = (max_err < 0.01f && idx_err == 0);
    printf("  Forward: max_err=%.6f, idx_errors=%d - %s\n", max_err, idx_err, pass_fwd ? "PASS" : "FAIL");
    
    // Backward
    std::vector<float> h_grad_out(out_size), h_grad_in_ref(in_size), h_grad_in(in_size);
    for (int i = 0; i < out_size; i++) h_grad_out[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    maxpool2d_backward_cpu(h_grad_out.data(), h_idx_ref.data(), h_grad_in_ref.data(), N, H, W, C);
    
    half *d_grad_out, *d_grad_in;
    cudaMalloc(&d_grad_out, out_size * sizeof(half));
    cudaMalloc(&d_grad_in, in_size * sizeof(half));
    
    std::vector<half> h_grad_out_half(out_size);
    for (int i = 0; i < out_size; i++) h_grad_out_half[i] = __float2half(h_grad_out[i]);
    cudaMemcpy(d_grad_out, h_grad_out_half.data(), out_size * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_maxpool2d_backward(d_grad_out, d_idx, d_grad_in, N, H, W, C, 0);
    cudaDeviceSynchronize();
    
    std::vector<half> h_grad_in_half(in_size);
    cudaMemcpy(h_grad_in_half.data(), d_grad_in, in_size * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < in_size; i++) h_grad_in[i] = __half2float(h_grad_in_half[i]);
    
    max_err = 0.0f;
    for (int i = 0; i < in_size; i++) {
        float err = fabs(h_grad_in[i] - h_grad_in_ref[i]);
        if (err > max_err) max_err = err;
    }
    bool pass_bwd = (max_err < 0.01f);
    printf("  Backward: max_err=%.6f - %s\n", max_err, pass_bwd ? "PASS" : "FAIL");
    
    cudaFree(d_in); cudaFree(d_out); cudaFree(d_idx);
    cudaFree(d_grad_out); cudaFree(d_grad_in);
    return pass_fwd && pass_bwd;
}

bool test_upsample2d(int N, int H, int W, int C) {
    printf("Testing Upsample2D: N=%d, H=%d, W=%d, C=%d\n", N, H, W, C);
    
    int in_size = N * H * W * C;
    int out_size = N * (H*2) * (W*2) * C;
    
    std::vector<float> h_in(in_size), h_out_ref(out_size), h_out(out_size);
    for (int i = 0; i < in_size; i++) h_in[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    upsample2d_forward_cpu(h_in.data(), h_out_ref.data(), N, H, W, C);
    
    half *d_in, *d_out;
    cudaMalloc(&d_in, in_size * sizeof(half));
    cudaMalloc(&d_out, out_size * sizeof(half));
    
    std::vector<half> h_in_half(in_size);
    for (int i = 0; i < in_size; i++) h_in_half[i] = __float2half(h_in[i]);
    cudaMemcpy(d_in, h_in_half.data(), in_size * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_upsample2d_forward(d_in, d_out, N, H, W, C, 0);
    cudaDeviceSynchronize();
    
    std::vector<half> h_out_half(out_size);
    cudaMemcpy(h_out_half.data(), d_out, out_size * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < out_size; i++) h_out[i] = __half2float(h_out_half[i]);
    
    float max_err = 0.0f;
    for (int i = 0; i < out_size; i++) {
        float err = fabs(h_out[i] - h_out_ref[i]);
        if (err > max_err) max_err = err;
    }
    bool pass_fwd = (max_err < 0.01f);
    printf("  Forward: max_err=%.6f - %s\n", max_err, pass_fwd ? "PASS" : "FAIL");
    
    // Backward
    std::vector<float> h_grad_out(out_size), h_grad_in_ref(in_size), h_grad_in(in_size);
    for (int i = 0; i < out_size; i++) h_grad_out[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    upsample2d_backward_cpu(h_grad_out.data(), h_grad_in_ref.data(), N, H, W, C);
    
    half *d_grad_out, *d_grad_in;
    cudaMalloc(&d_grad_out, out_size * sizeof(half));
    cudaMalloc(&d_grad_in, in_size * sizeof(half));
    
    std::vector<half> h_grad_out_half(out_size);
    for (int i = 0; i < out_size; i++) h_grad_out_half[i] = __float2half(h_grad_out[i]);
    cudaMemcpy(d_grad_out, h_grad_out_half.data(), out_size * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_upsample2d_backward(d_grad_out, d_grad_in, N, H, W, C, 0);
    cudaDeviceSynchronize();
    
    std::vector<half> h_grad_in_half(in_size);
    cudaMemcpy(h_grad_in_half.data(), d_grad_in, in_size * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < in_size; i++) h_grad_in[i] = __half2float(h_grad_in_half[i]);
    
    max_err = 0.0f;
    for (int i = 0; i < in_size; i++) {
        float err = fabs(h_grad_in[i] - h_grad_in_ref[i]);
        if (err > max_err) max_err = err;
    }
    // Higher tolerance for backward: summing 4 FP16 values accumulates error
    bool pass_bwd = (max_err < 0.02f);
    printf("  Backward: max_err=%.6f - %s\n", max_err, pass_bwd ? "PASS" : "FAIL");
    
    cudaFree(d_in); cudaFree(d_out);
    cudaFree(d_grad_out); cudaFree(d_grad_in);
    return pass_fwd && pass_bwd;
}

bool test_mse_loss(int size) {
    printf("Testing MSELoss: size=%d\n", size);
    
    std::vector<float> h_pred(size), h_target(size), h_grad_ref(size), h_grad(size);
    for (int i = 0; i < size; i++) {
        h_pred[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
        h_target[i] = (float)(rand() % 1000) / 100.0f - 5.0f;
    }
    
    float loss_ref = mse_loss_cpu(h_pred.data(), h_target.data(), size);
    mse_grad_cpu(h_pred.data(), h_target.data(), h_grad_ref.data(), size);
    
    half *d_pred, *d_target, *d_grad;
    float *d_loss, *d_partial;
    int num_blocks = (size + 255) / 256;
    
    cudaMalloc(&d_pred, size * sizeof(half));
    cudaMalloc(&d_target, size * sizeof(half));
    cudaMalloc(&d_grad, size * sizeof(half));
    cudaMalloc(&d_loss, sizeof(float));
    cudaMalloc(&d_partial, num_blocks * sizeof(float));
    
    std::vector<half> h_pred_half(size), h_target_half(size);
    for (int i = 0; i < size; i++) {
        h_pred_half[i] = __float2half(h_pred[i]);
        h_target_half[i] = __float2half(h_target[i]);
    }
    cudaMemcpy(d_pred, h_pred_half.data(), size * sizeof(half), cudaMemcpyHostToDevice);
    cudaMemcpy(d_target, h_target_half.data(), size * sizeof(half), cudaMemcpyHostToDevice);
    
    launch_mse_loss_grad(d_pred, d_target, d_grad, d_loss, d_partial, size, 0);
    cudaDeviceSynchronize();
    
    float loss_gpu;
    cudaMemcpy(&loss_gpu, d_loss, sizeof(float), cudaMemcpyDeviceToHost);
    
    std::vector<half> h_grad_half(size);
    cudaMemcpy(h_grad_half.data(), d_grad, size * sizeof(half), cudaMemcpyDeviceToHost);
    for (int i = 0; i < size; i++) h_grad[i] = __half2float(h_grad_half[i]);
    
    float loss_err = fabs(loss_gpu - loss_ref) / (fabs(loss_ref) + 1e-6f);
    float max_grad_err = 0.0f;
    for (int i = 0; i < size; i++) {
        float err = fabs(h_grad[i] - h_grad_ref[i]);
        if (err > max_grad_err) max_grad_err = err;
    }
    
    bool pass = (loss_err < 0.01f && max_grad_err < 0.01f);
    printf("  Loss: ref=%.6f, gpu=%.6f, rel_err=%.6f\n", loss_ref, loss_gpu, loss_err);
    printf("  Grad: max_err=%.6f - %s\n", max_grad_err, pass ? "PASS" : "FAIL");
    
    cudaFree(d_pred); cudaFree(d_target); cudaFree(d_grad);
    cudaFree(d_loss); cudaFree(d_partial);
    return pass;
}

bool test_sgd_update(int K, int C) {
    printf("Testing SGD Update: K=%d, C=%d (weight_size=%d)\n", K, C, K*9*C);
    
    int weight_size = K * 9 * C;
    int bias_size = K;
    float lr = 0.01f;
    
    std::vector<float> h_master_w(weight_size), h_grad_w(weight_size);
    std::vector<float> h_master_b(bias_size), h_grad_b(bias_size);
    
    for (int i = 0; i < weight_size; i++) {
        h_master_w[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        h_grad_w[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    for (int i = 0; i < bias_size; i++) {
        h_master_b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
        h_grad_b[i] = (float)(rand() % 1000) / 1000.0f - 0.5f;
    }
    
    std::vector<float> h_master_w_ref = h_master_w;
    std::vector<float> h_master_b_ref = h_master_b;
    sgd_update_cpu(h_master_w_ref.data(), h_grad_w.data(), lr, weight_size);
    sgd_update_cpu(h_master_b_ref.data(), h_grad_b.data(), lr, bias_size);
    
    float *d_master_w, *d_grad_w, *d_master_b, *d_grad_b;
    half *d_w, *d_b;
    cudaMalloc(&d_master_w, weight_size * sizeof(float));
    cudaMalloc(&d_w, weight_size * sizeof(half));
    cudaMalloc(&d_grad_w, weight_size * sizeof(float));
    cudaMalloc(&d_master_b, bias_size * sizeof(float));
    cudaMalloc(&d_b, bias_size * sizeof(half));
    cudaMalloc(&d_grad_b, bias_size * sizeof(float));
    
    cudaMemcpy(d_master_w, h_master_w.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_w, h_grad_w.data(), weight_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_master_b, h_master_b.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_grad_b, h_grad_b.data(), bias_size * sizeof(float), cudaMemcpyHostToDevice);
    
    launch_sgd_update_conv2d(d_master_w, d_w, d_grad_w, d_master_b, d_b, d_grad_b,
                             lr, weight_size, bias_size, 0);
    cudaDeviceSynchronize();
    
    std::vector<float> h_master_w_gpu(weight_size), h_master_b_gpu(bias_size);
    std::vector<half> h_w_half(weight_size), h_b_half(bias_size);
    
    cudaMemcpy(h_master_w_gpu.data(), d_master_w, weight_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_w_half.data(), d_w, weight_size * sizeof(half), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_master_b_gpu.data(), d_master_b, bias_size * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(h_b_half.data(), d_b, bias_size * sizeof(half), cudaMemcpyDeviceToHost);
    
    float max_err_w = 0.0f, max_err_b = 0.0f, max_err_fp16 = 0.0f;
    for (int i = 0; i < weight_size; i++) {
        float err = fabs(h_master_w_gpu[i] - h_master_w_ref[i]);
        if (err > max_err_w) max_err_w = err;
        err = fabs(__half2float(h_w_half[i]) - h_master_w_ref[i]);
        if (err > max_err_fp16) max_err_fp16 = err;
    }
    for (int i = 0; i < bias_size; i++) {
        float err = fabs(h_master_b_gpu[i] - h_master_b_ref[i]);
        if (err > max_err_b) max_err_b = err;
    }
    
    bool pass = (max_err_w < 1e-5f && max_err_b < 1e-5f && max_err_fp16 < 0.01f);
    printf("  Master err: %.2e, FP16 err: %.6f, Bias err: %.2e - %s\n",
           max_err_w, max_err_fp16, max_err_b, pass ? "PASS" : "FAIL");
    
    cudaFree(d_master_w); cudaFree(d_w); cudaFree(d_grad_w);
    cudaFree(d_master_b); cudaFree(d_b); cudaFree(d_grad_b);
    return pass;
}

// ============================================================================
// Benchmarks
// ============================================================================

void benchmark_ops() {
    printf("\n=== PERFORMANCE BENCHMARKS ===\n");
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    const int WARMUP = 10, ITERS = 100;
    
    printf("\n%-40s %10s %10s\n", "Operation", "Time(ms)", "BW(GB/s)");
    printf("--------------------------------------------------------------\n");

    // MaxPool Fwd 64x32x32x256
    { int N=64,H=32,W=32,C=256; int in_s=N*H*W*C, out_s=N*(H/2)*(W/2)*C;
      half *d_i,*d_o; int8_t *d_x; cudaMalloc(&d_i,in_s*2); cudaMalloc(&d_o,out_s*2); cudaMalloc(&d_x,out_s);
      for(int i=0;i<WARMUP;i++) launch_maxpool2d_forward(d_i,d_o,d_x,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_maxpool2d_forward(d_i,d_o,d_x,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","MaxPool Fwd (64,32,32,256)",ms,(in_s*2+out_s*3)/(ms*1e6));
      cudaFree(d_i);cudaFree(d_o);cudaFree(d_x); }

    // MaxPool Fwd 64x16x16x128
    { int N=64,H=16,W=16,C=128; int in_s=N*H*W*C, out_s=N*(H/2)*(W/2)*C;
      half *d_i,*d_o; int8_t *d_x; cudaMalloc(&d_i,in_s*2); cudaMalloc(&d_o,out_s*2); cudaMalloc(&d_x,out_s);
      for(int i=0;i<WARMUP;i++) launch_maxpool2d_forward(d_i,d_o,d_x,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_maxpool2d_forward(d_i,d_o,d_x,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","MaxPool Fwd (64,16,16,128)",ms,(in_s*2+out_s*3)/(ms*1e6));
      cudaFree(d_i);cudaFree(d_o);cudaFree(d_x); }

    // MaxPool Bwd 64x32x32x256
    { int N=64,H=32,W=32,C=256; int in_s=N*H*W*C, out_s=N*(H/2)*(W/2)*C;
      half *d_go,*d_gi; int8_t *d_x; cudaMalloc(&d_go,out_s*2); cudaMalloc(&d_gi,in_s*2); cudaMalloc(&d_x,out_s);
      for(int i=0;i<WARMUP;i++) launch_maxpool2d_backward(d_go,d_x,d_gi,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_maxpool2d_backward(d_go,d_x,d_gi,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","MaxPool Bwd (64,32,32,256)",ms,(out_s*3+in_s*4)/(ms*1e6));
      cudaFree(d_go);cudaFree(d_gi);cudaFree(d_x); }

    // MaxPool Bwd 64x16x16x128
    { int N=64,H=16,W=16,C=128; int in_s=N*H*W*C, out_s=N*(H/2)*(W/2)*C;
      half *d_go,*d_gi; int8_t *d_x; cudaMalloc(&d_go,out_s*2); cudaMalloc(&d_gi,in_s*2); cudaMalloc(&d_x,out_s);
      for(int i=0;i<WARMUP;i++) launch_maxpool2d_backward(d_go,d_x,d_gi,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_maxpool2d_backward(d_go,d_x,d_gi,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","MaxPool Bwd (64,16,16,128)",ms,(out_s*3+in_s*4)/(ms*1e6));
      cudaFree(d_go);cudaFree(d_gi);cudaFree(d_x); }

    // Upsample Fwd 64x8x8x128
    { int N=64,H=8,W=8,C=128; int in_s=N*H*W*C, out_s=N*(H*2)*(W*2)*C;
      half *d_i,*d_o; cudaMalloc(&d_i,in_s*2); cudaMalloc(&d_o,out_s*2);
      for(int i=0;i<WARMUP;i++) launch_upsample2d_forward(d_i,d_o,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_upsample2d_forward(d_i,d_o,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","Upsample Fwd (64,8,8,128)",ms,(in_s*2+out_s*2)/(ms*1e6));
      cudaFree(d_i);cudaFree(d_o); }

    // Upsample Fwd 64x16x16x256
    { int N=64,H=16,W=16,C=256; int in_s=N*H*W*C, out_s=N*(H*2)*(W*2)*C;
      half *d_i,*d_o; cudaMalloc(&d_i,in_s*2); cudaMalloc(&d_o,out_s*2);
      for(int i=0;i<WARMUP;i++) launch_upsample2d_forward(d_i,d_o,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_upsample2d_forward(d_i,d_o,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","Upsample Fwd (64,16,16,256)",ms,(in_s*2+out_s*2)/(ms*1e6));
      cudaFree(d_i);cudaFree(d_o); }

    // Upsample Bwd 64x8x8x128
    { int N=64,H=8,W=8,C=128; int in_s=N*H*W*C, out_s=N*(H*2)*(W*2)*C;
      half *d_go,*d_gi; cudaMalloc(&d_go,out_s*2); cudaMalloc(&d_gi,in_s*2);
      for(int i=0;i<WARMUP;i++) launch_upsample2d_backward(d_go,d_gi,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_upsample2d_backward(d_go,d_gi,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","Upsample Bwd (64,8,8,128)",ms,(out_s*2+in_s*2)/(ms*1e6));
      cudaFree(d_go);cudaFree(d_gi); }

    // Upsample Bwd 64x16x16x256
    { int N=64,H=16,W=16,C=256; int in_s=N*H*W*C, out_s=N*(H*2)*(W*2)*C;
      half *d_go,*d_gi; cudaMalloc(&d_go,out_s*2); cudaMalloc(&d_gi,in_s*2);
      for(int i=0;i<WARMUP;i++) launch_upsample2d_backward(d_go,d_gi,N,H,W,C,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_upsample2d_backward(d_go,d_gi,N,H,W,C,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","Upsample Bwd (64,16,16,256)",ms,(out_s*2+in_s*2)/(ms*1e6));
      cudaFree(d_go);cudaFree(d_gi); }

    // MSE Loss+Grad 64x32x32x3
    { int sz=64*32*32*3; int nb=(sz+255)/256;
      half *d_p,*d_t,*d_g; float *d_l,*d_pa;
      cudaMalloc(&d_p,sz*2); cudaMalloc(&d_t,sz*2); cudaMalloc(&d_g,sz*2);
      cudaMalloc(&d_l,4); cudaMalloc(&d_pa,nb*4);
      for(int i=0;i<WARMUP;i++) launch_mse_loss_grad(d_p,d_t,d_g,d_l,d_pa,sz,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_mse_loss_grad(d_p,d_t,d_g,d_l,d_pa,sz,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","MSE Loss+Grad (64,32,32,3)",ms,(sz*6)/(ms*1e6));
      cudaFree(d_p);cudaFree(d_t);cudaFree(d_g);cudaFree(d_l);cudaFree(d_pa); }

    // SGD Update 256x3x3x256
    { int ws=256*9*256, bs=256;
      float *d_mw,*d_gw,*d_mb,*d_gb; half *d_w,*d_b;
      cudaMalloc(&d_mw,ws*4); cudaMalloc(&d_w,ws*2); cudaMalloc(&d_gw,ws*4);
      cudaMalloc(&d_mb,bs*4); cudaMalloc(&d_b,bs*2); cudaMalloc(&d_gb,bs*4);
      for(int i=0;i<WARMUP;i++) launch_sgd_update_conv2d(d_mw,d_w,d_gw,d_mb,d_b,d_gb,0.01f,ws,bs,0); cudaDeviceSynchronize();
      cudaEventRecord(start); for(int i=0;i<ITERS;i++) launch_sgd_update_conv2d(d_mw,d_w,d_gw,d_mb,d_b,d_gb,0.01f,ws,bs,0);
      cudaEventRecord(stop); cudaEventSynchronize(stop); float ms; cudaEventElapsedTime(&ms,start,stop); ms/=ITERS;
      printf("%-40s %10.4f %10.2f\n","SGD Update (256,3,3,256)",ms,(ws*14+bs*14)/(ms*1e6));
      cudaFree(d_mw);cudaFree(d_w);cudaFree(d_gw);cudaFree(d_mb);cudaFree(d_b);cudaFree(d_gb); }

    cudaEventDestroy(start); cudaEventDestroy(stop);
    printf("\nT4 theoretical memory bandwidth: ~320 GB/s\n");
}

int main() {
    printf("=== AUTOENCODER OPS TESTS ===\n\n");
    srand(42);
    int passed = 0, total = 0;
    
    total++; if (test_maxpool2d(2, 32, 32, 256)) passed++;
    total++; if (test_maxpool2d(64, 32, 32, 256)) passed++;
    total++; if (test_maxpool2d(64, 16, 16, 128)) passed++;
    total++; if (test_upsample2d(2, 8, 8, 128)) passed++;
    total++; if (test_upsample2d(64, 8, 8, 128)) passed++;
    total++; if (test_upsample2d(64, 16, 16, 256)) passed++;
    total++; if (test_mse_loss(1024)) passed++;
    total++; if (test_mse_loss(64 * 32 * 32 * 3)) passed++;
    total++; if (test_sgd_update(256, 3)) passed++;
    total++; if (test_sgd_update(128, 256)) passed++;
    total++; if (test_sgd_update(256, 128)) passed++;
    
    printf("\n=== SUMMARY: %d/%d tests passed ===\n", passed, total);
    if (passed == total) benchmark_ops();
    return (passed == total) ? 0 : 1;
}
