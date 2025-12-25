// gpu_tensor.h

#pragma once
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cstddef>
#include <utility>
#include <type_traits>
#include "cuda_memory_pool.h"

// ============================================================================
// Layout enum for clear tensor format specification
// ============================================================================
enum class TensorLayout {
    NCHW,  // Batch, Channels, Height, Width (default for FP32)
    NHWC   // Batch, Height, Width, Channels (preferred for FP16/Tensor Cores)
};

// ============================================================================
// GPUTensor4D: 4D tensor storage with explicit layout support
// ============================================================================
// This class stores tensors with explicit NHWC or NCHW layout.
// 
// NHWC Layout (preferred for FP16 convolutions):
//   Memory order: [batch][height][width][channels]
//   Use: allocate_nhwc(batch, height, width, channels)
//   Access: batch(), height(), width(), channels()
//
// NCHW Layout (traditional):
//   Memory order: [batch][channels][height][width]
//   Use: allocate_nchw(batch, channels, height, width)
//   Access: batch(), channels(), height(), width()
//
// For backward compatibility, allocate() defaults to NCHW interpretation:
//   allocate(n, c, h, w) == allocate_nchw(n, c, h, w)
// ============================================================================

template <typename T>
class GPUTensor4D
{
    static_assert(
        std::is_trivially_copyable<T>::value,
        "GPUTensor4D requires trivially copyable types");

public:
    // Constructors / Destructor
    GPUTensor4D();
    GPUTensor4D(size_t dim0, size_t dim1, size_t dim2, size_t dim3, 
                TensorLayout layout = TensorLayout::NCHW);
    ~GPUTensor4D();

    // Copy semantics
    GPUTensor4D(const GPUTensor4D& other);
    GPUTensor4D& operator=(const GPUTensor4D& other);

    // Move semantics
    GPUTensor4D(GPUTensor4D&& other) noexcept;
    GPUTensor4D& operator=(GPUTensor4D&& other) noexcept;

    // ========================================================================
    // Memory allocation - EXPLICIT layout versions (RECOMMENDED)
    // ========================================================================
    
    // Allocate as NHWC: dimensions are (batch, height, width, channels)
    void allocate_nhwc(size_t batch, size_t height, size_t width, size_t channels,
                       cudaStream_t stream = 0);
    
    // Allocate as NCHW: dimensions are (batch, channels, height, width)
    void allocate_nchw(size_t batch, size_t channels, size_t height, size_t width,
                       cudaStream_t stream = 0);
    
    // Legacy allocation (NCHW interpretation for backward compatibility)
    void allocate(size_t n, size_t c, size_t h, size_t w, cudaStream_t stream = 0);
    
    void release();

    // Data transfer
    void copy_from_host(const T* h_data);
    void copy_from_host_async(const T* h_data, cudaStream_t stream);
    void copy_to_host(T* h_data) const;
    void zero(cudaStream_t stream = 0);

    // Type conversion
    template<typename U>
    void convert_from(const GPUTensor4D<U>& other, cudaStream_t stream = 0);

    // ========================================================================
    // Semantic accessors (layout-aware)
    // ========================================================================
    size_t batch() const noexcept;      // Always returns batch size
    size_t channels() const noexcept;   // Returns channel count
    size_t height() const noexcept;     // Returns spatial height
    size_t width() const noexcept;      // Returns spatial width
    
    // ========================================================================
    // Raw dimension accessors (layout-independent)
    // ========================================================================
    size_t dim0() const noexcept { return dims_[0]; }
    size_t dim1() const noexcept { return dims_[1]; }
    size_t dim2() const noexcept { return dims_[2]; }
    size_t dim3() const noexcept { return dims_[3]; }
    
    // ========================================================================
    // Legacy accessors (DEPRECATED - use semantic accessors instead)
    // For backward compatibility: N/C/H/W always returns dims[0]/[1]/[2]/[3]
    // regardless of layout. Use batch()/channels()/height()/width() instead.
    // ========================================================================
    size_t N() const noexcept { return dims_[0]; }
    size_t C() const noexcept { return dims_[1]; }
    size_t H() const noexcept { return dims_[2]; }
    size_t W() const noexcept { return dims_[3]; }

    // Size and data accessors
    size_t size() const noexcept;
    size_t bytes() const noexcept;
    T* data() noexcept;
    const T* data() const noexcept;
    bool empty() const noexcept;
    
    // Layout query
    TensorLayout layout() const noexcept { return layout_; }
    bool is_nhwc() const noexcept { return layout_ == TensorLayout::NHWC; }
    bool is_nchw() const noexcept { return layout_ == TensorLayout::NCHW; }

private:
    size_t dims_[4] = {0, 0, 0, 0};  // Raw dimensions in memory order
    T* d_data{nullptr};
    cudaStream_t alloc_stream{0};
    TensorLayout layout_{TensorLayout::NCHW};
    
    void allocate_internal(size_t d0, size_t d1, size_t d2, size_t d3,
                           TensorLayout layout, cudaStream_t stream);
};
