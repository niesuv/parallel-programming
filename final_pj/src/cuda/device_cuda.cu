#include "device.h"
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

// CUDA error checking macro
#define CUDA_CHECK(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
            return -1; \
        } \
    } while(0)

#define CUDA_CHECK_VOID(call) \
    do { \
        cudaError_t err = call; \
        if (err != cudaSuccess) { \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                    cudaGetErrorString(err)); \
        } \
    } while(0)

// Initialize CUDA device
extern "C" int cuda_device_init(int device_id) {
    int device_count;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));

    if (device_count == 0) {
        fprintf(stderr, "No CUDA devices found\n");
        return -1;
    }

    if (device_id >= device_count) {
        fprintf(stderr, "Invalid device ID: %d (max: %d)\n", device_id, device_count - 1);
        return -1;
    }

    CUDA_CHECK(cudaSetDevice(device_id));
    printf("CUDA device %d initialized\n", device_id);

    return 0;
}

// Cleanup CUDA device
extern "C" void cuda_device_cleanup(void) {
    CUDA_CHECK_VOID(cudaDeviceReset());
}

// Check if CUDA is available
extern "C" int cuda_device_is_available(void) {
    int device_count;
    cudaError_t err = cudaGetDeviceCount(&device_count);
    return (err == cudaSuccess && device_count > 0) ? 1 : 0;
}

// Print CUDA device information
extern "C" void cuda_device_print_info(void) {
    int device;
    cudaGetDevice(&device);

    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, device);

    printf("\n=== CUDA Device Info ===\n");
    printf("Device ID:              %d\n", device);
    printf("Device Name:            %s\n", prop.name);
    printf("Compute Capability:     %d.%d\n", prop.major, prop.minor);
    printf("Total Global Memory:    %.2f GB\n", prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Shared Memory per Block: %.2f KB\n", prop.sharedMemPerBlock / 1024.0);
    printf("Max Threads per Block:  %d\n", prop.maxThreadsPerBlock);
    printf("Max Block Dimensions:   (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Max Grid Dimensions:    (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Warp Size:              %d\n", prop.warpSize);
    printf("Number of SMs:          %d\n", prop.multiProcessorCount);
    printf("Clock Rate:             %.2f GHz\n", prop.clockRate / 1000000.0);
    printf("Memory Clock Rate:      %.2f GHz\n", prop.memoryClockRate / 1000000.0);
    printf("Memory Bus Width:       %d-bit\n", prop.memoryBusWidth);
    printf("=======================\n\n");
}

// Allocate device memory
extern "C" void* cuda_malloc(size_t size) {
    void* ptr;
    cudaError_t err = cudaMalloc(&ptr, size);
    if (err != cudaSuccess) {
        fprintf(stderr, "CUDA malloc failed: %s\n", cudaGetErrorString(err));
        return NULL;
    }
    return ptr;
}

// Free device memory
extern "C" void cuda_free(void* ptr) {
    if (ptr) {
        CUDA_CHECK_VOID(cudaFree(ptr));
    }
}

// Copy from host to device
extern "C" int cuda_memcpy_host_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    return 0;
}

// Copy from device to host
extern "C" int cuda_memcpy_device_to_host(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    return 0;
}

// Copy from device to device
extern "C" int cuda_memcpy_device_to_device(void* dst, const void* src, size_t size) {
    CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    return 0;
}

// Synchronize device
extern "C" void cuda_synchronize(void) {
    CUDA_CHECK_VOID(cudaDeviceSynchronize());
}
