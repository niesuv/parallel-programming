//cuda_memory_pool.h

#pragma once
#include <cuda_runtime.h>

// ============================================================================
// CUDAMemoryPool: Singleton wrapper for CUDA memory pooling
// ============================================================================
// Usage:
//   CUDAMemoryPool::init(device_id);  // Call once at startup
//   cudaMemPool_t pool = CUDAMemoryPool::pool();
//   CUDAMemoryPool::destroy();  // Call at shutdown (optional for default pool)
//
// The memory pool enables async allocation/deallocation with cudaMallocFromPoolAsync
// and cudaFreeAsync, which is more efficient for repeated allocations.
// ============================================================================

class CUDAMemoryPool{
public:
    // Initialize the memory pool for the specified device
    // Call once before using GPUTensor4D allocations
    static void init(int device = 0);
    
    // Get the memory pool handle
    static cudaMemPool_t pool();
    
    // Release the memory pool (no-op for default pool)
    static void destroy();
    
    // Query if pool is initialized
    static bool is_initialized();

private:
    static cudaMemPool_t mempool_;
    static bool initialized_;
};