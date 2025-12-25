//cuda_utils.h

#ifndef CUDA_UTILS_H
#define CUDA_UTILS_H

#ifdef __CUDACC__
#include <cuda_runtime.h>
#include <iostream>
#include <sstream>
#include <stdexcept>

// ============================================================================
// CUDA ERROR CHECKING MACROS
// ============================================================================

// Basic error check - prints error and continues
#define CUDA_CHECK(expr)                                                            \
    do {                                                                            \
        cudaError_t err__ = (expr);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::cerr << "CUDA error: " << cudaGetErrorString(err__)                \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        }                                                                           \
    } while (0)

// Strict error check - throws exception on error
#define CUDA_CHECK_THROW(expr)                                                      \
    do {                                                                            \
        cudaError_t err__ = (expr);                                                 \
        if (err__ != cudaSuccess) {                                                 \
            std::ostringstream oss;                                                 \
            oss << "CUDA error: " << cudaGetErrorString(err__)                      \
                << " at " << __FILE__ << ":" << __LINE__;                           \
            throw std::runtime_error(oss.str());                                    \
        }                                                                           \
    } while (0)

// Check last CUDA error (for kernel launches)
#define CUDA_CHECK_LAST()                                                           \
    do {                                                                            \
        cudaError_t err__ = cudaGetLastError();                                     \
        if (err__ != cudaSuccess) {                                                 \
            std::cerr << "CUDA kernel error: " << cudaGetErrorString(err__)         \
                      << " at " << __FILE__ << ":" << __LINE__ << std::endl;        \
        }                                                                           \
    } while (0)

// ============================================================================
// CUDA SYNCHRONIZATION UTILITIES
// ============================================================================

// Synchronize device and check for errors
inline void cuda_sync_check() {
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK_LAST();
}

// ============================================================================
// CUDA DEVICE UTILITIES
// ============================================================================

// Get number of CUDA devices
inline int cuda_device_count() {
    int count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&count));
    return count;
}

// Get current device
inline int cuda_current_device() {
    int device = 0;
    CUDA_CHECK(cudaGetDevice(&device));
    return device;
}

// Set current device
inline void cuda_set_device(int device) {
    CUDA_CHECK(cudaSetDevice(device));
}

// ============================================================================
// CUDA STREAM UTILITIES  
// ============================================================================

// RAII wrapper for CUDA stream
class CUDAStream {
public:
    CUDAStream() {
        CUDA_CHECK(cudaStreamCreate(&stream_));
    }
    
    explicit CUDAStream(unsigned int flags) {
        CUDA_CHECK(cudaStreamCreateWithFlags(&stream_, flags));
    }
    
    ~CUDAStream() {
        if (stream_) {
            cudaStreamDestroy(stream_);
        }
    }
    
    // Non-copyable
    CUDAStream(const CUDAStream&) = delete;
    CUDAStream& operator=(const CUDAStream&) = delete;
    
    // Movable
    CUDAStream(CUDAStream&& other) noexcept 
        : stream_(std::exchange(other.stream_, nullptr)) {}
    
    CUDAStream& operator=(CUDAStream&& other) noexcept {
        if (this != &other) {
            if (stream_) cudaStreamDestroy(stream_);
            stream_ = std::exchange(other.stream_, nullptr);
        }
        return *this;
    }
    
    cudaStream_t get() const { return stream_; }
    operator cudaStream_t() const { return stream_; }
    
    void synchronize() const {
        CUDA_CHECK(cudaStreamSynchronize(stream_));
    }

private:
    cudaStream_t stream_ = nullptr;
};

// ============================================================================
// CUDA EVENT UTILITIES
// ============================================================================

// RAII wrapper for CUDA event
class CUDAEvent {
public:
    CUDAEvent() {
        CUDA_CHECK(cudaEventCreate(&event_));
    }
    
    explicit CUDAEvent(unsigned int flags) {
        CUDA_CHECK(cudaEventCreateWithFlags(&event_, flags));
    }
    
    ~CUDAEvent() {
        if (event_) {
            cudaEventDestroy(event_);
        }
    }
    
    // Non-copyable
    CUDAEvent(const CUDAEvent&) = delete;
    CUDAEvent& operator=(const CUDAEvent&) = delete;
    
    // Movable
    CUDAEvent(CUDAEvent&& other) noexcept 
        : event_(std::exchange(other.event_, nullptr)) {}
    
    CUDAEvent& operator=(CUDAEvent&& other) noexcept {
        if (this != &other) {
            if (event_) cudaEventDestroy(event_);
            event_ = std::exchange(other.event_, nullptr);
        }
        return *this;
    }
    
    cudaEvent_t get() const { return event_; }
    operator cudaEvent_t() const { return event_; }
    
    void record(cudaStream_t stream = 0) const {
        CUDA_CHECK(cudaEventRecord(event_, stream));
    }
    
    void synchronize() const {
        CUDA_CHECK(cudaEventSynchronize(event_));
    }
    
    // Returns elapsed time in milliseconds between start and this event
    float elapsed_ms(const CUDAEvent& start) const {
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, start.event_, event_));
        return ms;
    }

private:
    cudaEvent_t event_ = nullptr;
};

#else
// Non-CUDA builds: no-op macros
#define CUDA_CHECK(expr) (expr)
#define CUDA_CHECK_THROW(expr) (expr)
#define CUDA_CHECK_LAST() ((void)0)
#endif

#endif  // CUDA_UTILS_H
