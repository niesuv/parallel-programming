#include "config.h"
#include <stdio.h>

const char* device_type_to_string(DeviceType device) {
    switch (device) {
        case DEVICE_CPU:
            return "CPU";
        case DEVICE_CUDA:
            return "CUDA";
        default:
            return "Unknown";
    }
}

void config_print(const Config* cfg) {
    printf("\n=== Configuration ===\n");
    printf("Device:          %s\n", device_type_to_string(cfg->device));
    if (cfg->device == DEVICE_CUDA) {
        printf("CUDA Device ID:  %d\n", cfg->cuda_device_id);
    }
    if (cfg->device == DEVICE_CPU) {
        printf("CPU Threads:     %d\n", cfg->num_threads);
    }
    printf("Batch Size:      %d\n", cfg->batch_size);
    printf("Num Epochs:      %d\n", cfg->num_epochs);
    printf("Learning Rate:   %.6f\n", cfg->learning_rate);
    printf("Verbose Level:   %d\n", cfg->verbose);
    printf("====================\n\n");
}
