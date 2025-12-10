#ifndef DEVICE_H
#define DEVICE_H

#include "config.h"
#include <stddef.h>

// Device memory management
typedef struct {
    void* ptr;              // Pointer to memory
    size_t size;            // Size in bytes
    DeviceType device;      // Which device owns this memory
} DeviceMemory;

// Initialize device
int device_init(DeviceType device, int device_id);
void device_cleanup(DeviceType device);

// Query device properties
int device_is_available(DeviceType device);
void device_print_info(DeviceType device);

// Memory management
DeviceMemory* device_malloc(size_t size, DeviceType device);
void device_free(DeviceMemory* mem);

// Memory transfers
int device_memcpy_host_to_device(DeviceMemory* dst, const void* src, size_t size);
int device_memcpy_device_to_host(void* dst, const DeviceMemory* src, size_t size);
int device_memcpy_device_to_device(DeviceMemory* dst, const DeviceMemory* src, size_t size);

// Synchronization
void device_synchronize(DeviceType device);

#endif // DEVICE_H
