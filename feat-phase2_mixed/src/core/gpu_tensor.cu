#include "gpu_layer.h"
#include "cuda_utils.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstring>
#include <stdexcept>


GPUTensor4D::GPUTensor4D(int n_, int c_, int h_, int w_, GPUDtype dtype_)
    : n(n_), c(c_), h(h_), w(w_), dtype(dtype_) {
    allocate(n_, c_, h_, w_, dtype_);
}

GPUTensor4D::~GPUTensor4D() {
    free();
}


GPUTensor4D::GPUTensor4D(GPUTensor4D &&other) noexcept {
    n = other.n;
    c = other.c;
    h = other.h;
    w = other.w;
    dtype = other.dtype;
    d_data = other.d_data;

    other.d_data = nullptr;
    other.n = other.c = other.h = other.w = 0;
}

GPUTensor4D &GPUTensor4D::operator=(GPUTensor4D &&other) noexcept {
    if (this != &other) {
        free();

        n = other.n;
        c = other.c;
        h = other.h;
        w = other.w;
        dtype = other.dtype;
        d_data = other.d_data;

        other.d_data = nullptr;
        other.n = other.c = other.h = other.w = 0;
    }
  return *this;
}


void GPUTensor4D::allocate(int n_, int c_, int h_, int w_, GPUDtype dtype_) {
    free();

    n = n_;
    c = c_;
    h = h_;
    w = w_;
    dtype = dtype_;

    if (n == 0 || c == 0 || h == 0 || w == 0)
        return;

    CUDA_CHECK(cudaMalloc(&d_data, bytes()));
}

void GPUTensor4D::free() {
    if (d_data) {
        CUDA_CHECK(cudaFree(d_data));
        d_data = nullptr;
    }
    n = c = h = w = 0;
}

/* =========================
    HOST <-> DEVICE COPY
    Host data is FP32
========================= */

void GPUTensor4D::copy_from_host_fp32(const float *h_data) {
    if (!d_data)
        throw std::runtime_error("GPUTensor4D not allocated");

    size_t count = size();

    if (dtype == GPUDtype::FP32) {
        CUDA_CHECK(cudaMemcpy(
            d_data, h_data,
            count * sizeof(float),
            cudaMemcpyHostToDevice));
    } else {
        // FP32 → FP16
        std::vector<__half> tmp(count);
        for (size_t i = 0; i < count; ++i)
        tmp[i] = __float2half(h_data[i]);

        CUDA_CHECK(cudaMemcpy(
            d_data, tmp.data(),
            count * sizeof(__half),
            cudaMemcpyHostToDevice));
    }
}

void GPUTensor4D::copy_to_host_fp32(float *h_data) const {
    if (!d_data)
        throw std::runtime_error("GPUTensor4D not allocated");

    size_t count = size();

    if (dtype == GPUDtype::FP32) {
        CUDA_CHECK(cudaMemcpy(
            h_data, d_data,
            count * sizeof(float),
            cudaMemcpyDeviceToHost));
    } else {
        // FP16 → FP32
        std::vector<__half> tmp(count);

        CUDA_CHECK(cudaMemcpy(
            tmp.data(), d_data,
            count * sizeof(__half),
            cudaMemcpyDeviceToHost));

        for (size_t i = 0; i < count; ++i)
        h_data[i] = __half2float(tmp[i]);
    }
}

