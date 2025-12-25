#include "cuda_memory_pool.h"
#include "cuda_utils.h"
#include <cstdint>
#include <iostream>

// ============================================================================
// STATIC MEMBER INITIALIZATION
// ============================================================================

cudaMemPool_t CUDAMemoryPool::mempool_ = nullptr;
bool CUDAMemoryPool::initialized_ = false;

// ============================================================================
// IMPLEMENTATION
// ============================================================================

void CUDAMemoryPool::init(int device)
{
    if (initialized_) {
        return;  // Already initialized
    }
    
    // Set the device first
    CUDA_CHECK(cudaSetDevice(device));
    
    // Get the default memory pool for this device
    CUDA_CHECK(cudaDeviceGetDefaultMemPool(&mempool_, device));
    
    // Configure pool attributes for optimal performance:
    
    // 1. Keep all memory - don't release back to OS until explicitly trimmed
    size_t threshold = SIZE_MAX;
    CUDA_CHECK(cudaMemPoolSetAttribute(
        mempool_,
        cudaMemPoolAttrReleaseThreshold,
        &threshold));
    
    // 2. Allow the pool to grow as needed (default behavior)
    // No explicit setting needed
    
    initialized_ = true;
}

cudaMemPool_t CUDAMemoryPool::pool()
{
    if (!initialized_) {
        // Auto-initialize on first use (device 0)
        init(0);
    }
    return mempool_;
}

void CUDAMemoryPool::destroy()
{
    // The default memory pool is owned by the CUDA runtime
    // We don't destroy it, but we can trim unused memory if needed
    if (initialized_ && mempool_) {
        // Optional: Release unused memory back to OS
        // cudaMemPoolTrimTo(mempool_, 0);
    }
    
    // Reset state
    mempool_ = nullptr;
    initialized_ = false;
}

bool CUDAMemoryPool::is_initialized()
{
    return initialized_;
}
