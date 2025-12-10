#include "device.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

// CPU implementation
int device_init(DeviceType device, int device_id) {
    if (device == DEVICE_CPU) {
        printf("Initializing CPU device...\n");
        return 0;
    }
    else if (device == DEVICE_CUDA) {
#ifdef __NVCC__
        // CUDA initialization will be implemented in device_cuda.cu
        extern int cuda_device_init(int device_id);
        return cuda_device_init(device_id);
#else
        fprintf(stderr, "Error: CUDA support not compiled in\n");
        return -1;
#endif
    }
    return -1;
}

void device_cleanup(DeviceType device) {
    if (device == DEVICE_CPU) {
        // No cleanup needed for CPU
        return;
    }
    else if (device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern void cuda_device_cleanup(void);
        cuda_device_cleanup();
#endif
    }
}

int device_is_available(DeviceType device) {
    if (device == DEVICE_CPU) {
        return 1;  // CPU always available
    }
    else if (device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern int cuda_device_is_available(void);
        return cuda_device_is_available();
#else
        return 0;  // CUDA not compiled
#endif
    }
    return 0;
}

void device_print_info(DeviceType device) {
    if (device == DEVICE_CPU) {
        printf("\n=== CPU Device Info ===\n");
        printf("Device Type: CPU\n");
        // Could add more CPU info here (cores, cache, etc.)
        printf("======================\n\n");
    }
    else if (device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern void cuda_device_print_info(void);
        cuda_device_print_info();
#else
        printf("CUDA not available\n");
#endif
    }
}

// Memory management
DeviceMemory* device_malloc(size_t size, DeviceType device) {
    DeviceMemory* mem = (DeviceMemory*)malloc(sizeof(DeviceMemory));
    if (!mem) {
        fprintf(stderr, "Failed to allocate DeviceMemory structure\n");
        return NULL;
    }

    mem->size = size;
    mem->device = device;

    if (device == DEVICE_CPU) {
        mem->ptr = malloc(size);
        if (!mem->ptr) {
            fprintf(stderr, "Failed to allocate CPU memory\n");
            free(mem);
            return NULL;
        }
    }
    else if (device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern void* cuda_malloc(size_t size);
        mem->ptr = cuda_malloc(size);
        if (!mem->ptr) {
            free(mem);
            return NULL;
        }
#else
        fprintf(stderr, "CUDA not available\n");
        free(mem);
        return NULL;
#endif
    }

    return mem;
}

void device_free(DeviceMemory* mem) {
    if (!mem) return;

    if (mem->device == DEVICE_CPU) {
        if (mem->ptr) free(mem->ptr);
    }
    else if (mem->device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern void cuda_free(void* ptr);
        if (mem->ptr) cuda_free(mem->ptr);
#endif
    }

    free(mem);
}

// Memory transfers
int device_memcpy_host_to_device(DeviceMemory* dst, const void* src, size_t size) {
    if (!dst || !src) return -1;

    if (dst->device == DEVICE_CPU) {
        memcpy(dst->ptr, src, size);
        return 0;
    }
    else if (dst->device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern int cuda_memcpy_host_to_device(void* dst, const void* src, size_t size);
        return cuda_memcpy_host_to_device(dst->ptr, src, size);
#else
        fprintf(stderr, "CUDA not available\n");
        return -1;
#endif
    }

    return -1;
}

int device_memcpy_device_to_host(void* dst, const DeviceMemory* src, size_t size) {
    if (!dst || !src) return -1;

    if (src->device == DEVICE_CPU) {
        memcpy(dst, src->ptr, size);
        return 0;
    }
    else if (src->device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern int cuda_memcpy_device_to_host(void* dst, const void* src, size_t size);
        return cuda_memcpy_device_to_host(dst, src->ptr, size);
#else
        fprintf(stderr, "CUDA not available\n");
        return -1;
#endif
    }

    return -1;
}

int device_memcpy_device_to_device(DeviceMemory* dst, const DeviceMemory* src, size_t size) {
    if (!dst || !src) return -1;

    if (dst->device == DEVICE_CPU && src->device == DEVICE_CPU) {
        memcpy(dst->ptr, src->ptr, size);
        return 0;
    }
    else if (dst->device == DEVICE_CUDA && src->device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern int cuda_memcpy_device_to_device(void* dst, const void* src, size_t size);
        return cuda_memcpy_device_to_device(dst->ptr, src->ptr, size);
#else
        fprintf(stderr, "CUDA not available\n");
        return -1;
#endif
    }

    fprintf(stderr, "Cross-device copy not supported directly\n");
    return -1;
}

void device_synchronize(DeviceType device) {
    if (device == DEVICE_CUDA) {
#ifdef __NVCC__
        extern void cuda_synchronize(void);
        cuda_synchronize();
#endif
    }
    // CPU doesn't need synchronization
}
