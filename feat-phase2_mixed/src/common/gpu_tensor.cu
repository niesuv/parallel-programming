#include "gpu_tensor.h"
#include "cuda_utils.h"
#include <cuda_fp16.h>
#include <cuda_runtime.h>
#include <iostream>

// ============================================================================
// CONSTRUCTORS / DESTRUCTOR
// ============================================================================

template <typename T>
GPUTensor4D<T>::GPUTensor4D() = default;

template <typename T>
GPUTensor4D<T>::GPUTensor4D(size_t dim0, size_t dim1, size_t dim2, size_t dim3,
                             TensorLayout layout)
{
    allocate_internal(dim0, dim1, dim2, dim3, layout, 0);
}

template <typename T>
GPUTensor4D<T>::~GPUTensor4D()
{
    release();
}

// ============================================================================
// COPY SEMANTICS
// ============================================================================

template <typename T>
GPUTensor4D<T>::GPUTensor4D(const GPUTensor4D& other)
    : layout_(other.layout_)
{
    for (int i = 0; i < 4; ++i) dims_[i] = other.dims_[i];
    
    if (other.d_data && size() > 0)
    {
        CUDA_CHECK(cudaMalloc(&d_data, bytes()));
        CUDA_CHECK(cudaMemcpy(
            d_data,
            other.d_data,
            bytes(),
            cudaMemcpyDeviceToDevice));
    }
}

template <typename T>
GPUTensor4D<T>& GPUTensor4D<T>::operator=(const GPUTensor4D& other)
{
    if (this != &other)
    {
        if (bytes() != other.bytes())
        {
            release();
            if (other.size() > 0)
            {
                CUDA_CHECK(cudaMalloc(&d_data, other.bytes()));
            }
        }

        for (int i = 0; i < 4; ++i) dims_[i] = other.dims_[i];
        layout_ = other.layout_;

        if (d_data && other.d_data && size() > 0)
        {
            CUDA_CHECK(cudaMemcpy(
                d_data,
                other.d_data,
                bytes(),
                cudaMemcpyDeviceToDevice));
        }
    }
    return *this;
}

// ============================================================================
// MOVE SEMANTICS
// ============================================================================

template <typename T>
GPUTensor4D<T>::GPUTensor4D(GPUTensor4D&& other) noexcept
    : d_data(std::exchange(other.d_data, nullptr)),
      alloc_stream(std::exchange(other.alloc_stream, nullptr)),
      layout_(other.layout_)
{
    for (int i = 0; i < 4; ++i) {
        dims_[i] = std::exchange(other.dims_[i], 0);
    }
}

template <typename T>
GPUTensor4D<T>& GPUTensor4D<T>::operator=(GPUTensor4D&& other) noexcept
{
    if (this != &other)
    {
        release();
        for (int i = 0; i < 4; ++i) {
            dims_[i] = std::exchange(other.dims_[i], 0);
        }
        d_data = std::exchange(other.d_data, nullptr);
        alloc_stream = std::exchange(other.alloc_stream, nullptr);
        layout_ = other.layout_;
    }
    return *this;
}

// ============================================================================
// MEMORY ALLOCATION
// ============================================================================

template <typename T>
void GPUTensor4D<T>::allocate_internal(
    size_t d0, size_t d1, size_t d2, size_t d3,
    TensorLayout layout, cudaStream_t stream)
{
    size_t new_size = d0 * d1 * d2 * d3;

    // Reuse buffer if same total size
    if (new_size == size() && d_data) {
        dims_[0] = d0;
        dims_[1] = d1;
        dims_[2] = d2;
        dims_[3] = d3;
        layout_ = layout;
        return;
    }

    release();

    dims_[0] = d0;
    dims_[1] = d1;
    dims_[2] = d2;
    dims_[3] = d3;
    layout_ = layout;
    alloc_stream = stream;

    if (new_size > 0)
    {
        CUDA_CHECK(cudaMallocFromPoolAsync(
            reinterpret_cast<void**>(&d_data),
            bytes(),
            CUDAMemoryPool::pool(),
            stream));
    }
}

template <typename T>
void GPUTensor4D<T>::allocate_nhwc(size_t batch, size_t height, size_t width, 
                                    size_t channels, cudaStream_t stream)
{
    // NHWC: dims = [batch, height, width, channels]
    allocate_internal(batch, height, width, channels, TensorLayout::NHWC, stream);
}

template <typename T>
void GPUTensor4D<T>::allocate_nchw(size_t batch, size_t channels, size_t height, 
                                    size_t width, cudaStream_t stream)
{
    // NCHW: dims = [batch, channels, height, width]
    allocate_internal(batch, channels, height, width, TensorLayout::NCHW, stream);
}

template <typename T>
void GPUTensor4D<T>::allocate(size_t n, size_t c, size_t h, size_t w, 
                               cudaStream_t stream)
{
    // Legacy: assume NCHW for backward compatibility
    allocate_nchw(n, c, h, w, stream);
}

template <typename T>
void GPUTensor4D<T>::release()
{
    if (d_data)
    {
        CUDA_CHECK(cudaFreeAsync(d_data, alloc_stream));
        d_data = nullptr;
    }
    for (int i = 0; i < 4; ++i) dims_[i] = 0;
}

// ============================================================================
// DATA TRANSFER
// ============================================================================

template <typename T>
void GPUTensor4D<T>::copy_from_host(const T* h_data)
{
    if (d_data && size() > 0)
    {
        CUDA_CHECK(cudaMemcpy(
            d_data,
            h_data,
            bytes(),
            cudaMemcpyHostToDevice));
    }
}

template <typename T>
void GPUTensor4D<T>::copy_from_host_async(const T* h_data, cudaStream_t stream)
{
    if (d_data && size() > 0)
    {
        CUDA_CHECK(cudaMemcpyAsync(
            d_data,
            h_data,
            bytes(),
            cudaMemcpyHostToDevice,
            stream));
    }
}

template <typename T>
void GPUTensor4D<T>::copy_to_host(T* h_data) const
{
    if (d_data && size() > 0)
    {
        CUDA_CHECK(cudaMemcpy(
            h_data,
            d_data,
            bytes(),
            cudaMemcpyDeviceToHost));
    }
}

template <typename T>
void GPUTensor4D<T>::zero(cudaStream_t stream)
{
    if (d_data && size() > 0)
    {
        CUDA_CHECK(cudaMemsetAsync(d_data, 0, bytes(), stream));
    }
}

// ============================================================================
// SEMANTIC ACCESSORS (layout-aware)
// ============================================================================

template <typename T>
size_t GPUTensor4D<T>::batch() const noexcept
{
    return dims_[0];  // First dimension is always batch
}

template <typename T>
size_t GPUTensor4D<T>::channels() const noexcept
{
    // NCHW: channels at index 1
    // NHWC: channels at index 3
    return (layout_ == TensorLayout::NCHW) ? dims_[1] : dims_[3];
}

template <typename T>
size_t GPUTensor4D<T>::height() const noexcept
{
    // NCHW: height at index 2
    // NHWC: height at index 1
    return (layout_ == TensorLayout::NCHW) ? dims_[2] : dims_[1];
}

template <typename T>
size_t GPUTensor4D<T>::width() const noexcept
{
    // NCHW: width at index 3
    // NHWC: width at index 2
    return (layout_ == TensorLayout::NCHW) ? dims_[3] : dims_[2];
}

// ============================================================================
// SIZE ACCESSORS
// ============================================================================

template <typename T>
size_t GPUTensor4D<T>::size() const noexcept
{
    return dims_[0] * dims_[1] * dims_[2] * dims_[3];
}

template <typename T>
size_t GPUTensor4D<T>::bytes() const noexcept
{
    return size() * sizeof(T);
}

template <typename T>
T* GPUTensor4D<T>::data() noexcept
{
    return d_data;
}

template <typename T>
const T* GPUTensor4D<T>::data() const noexcept
{
    return d_data;
}

template <typename T>
bool GPUTensor4D<T>::empty() const noexcept
{
    return d_data == nullptr;
}

// ============================================================================
// TYPE CONVERSION KERNELS
// ============================================================================

template<typename DST, typename SRC>
__global__ void convert_kernel(
    DST* __restrict__ dst,
    const SRC* __restrict__ src,
    size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = static_cast<DST>(src[idx]);
    }
}

// Specialization: float → half (use __float2half for efficiency)
template<>
__global__ void convert_kernel<half, float>(
    half* __restrict__ dst,
    const float* __restrict__ src,
    size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __float2half(src[idx]);
    }
}

// Specialization: half → float (use __half2float for efficiency)
template<>
__global__ void convert_kernel<float, half>(
    float* __restrict__ dst,
    const half* __restrict__ src,
    size_t size)
{
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        dst[idx] = __half2float(src[idx]);
    }
}

// ============================================================================
// CONVERSION METHOD IMPLEMENTATION
// ============================================================================

template <typename T>
template <typename U>
void GPUTensor4D<T>::convert_from(const GPUTensor4D<U>& other, cudaStream_t stream)
{
    // Allocate if needed (preserving layout from source)
    if (dims_[0] != other.dim0() || dims_[1] != other.dim1() ||
        dims_[2] != other.dim2() || dims_[3] != other.dim3()) {
        allocate_internal(other.dim0(), other.dim1(), other.dim2(), other.dim3(),
                          other.layout(), stream);
    }
    
    if (size() == 0) return;
    
    // Launch conversion kernel
    int threads = 256;
    int blocks = (size() + threads - 1) / threads;
    
    convert_kernel<T, U><<<blocks, threads, 0, stream>>>(
        d_data, 
        other.data(), 
        size()
    );
    
    CUDA_CHECK(cudaGetLastError());
}

// ============================================================================
// EXPLICIT TEMPLATE INSTANTIATIONS
// ============================================================================

template class GPUTensor4D<float>;
template class GPUTensor4D<half>; 
template class GPUTensor4D<uint8_t>;

template void GPUTensor4D<float>::convert_from<half>(const GPUTensor4D<half>&, cudaStream_t);
template void GPUTensor4D<half>::convert_from<float>(const GPUTensor4D<float>&, cudaStream_t);
