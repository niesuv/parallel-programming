#ifndef CONFIG_H
#define CONFIG_H

// Device types
typedef enum {
    DEVICE_CPU = 0,
    DEVICE_CUDA = 1
} DeviceType;

// Configuration structure
typedef struct {
    DeviceType device;
    int cuda_device_id;     // Which GPU to use (if multiple)
    int num_threads;        // Number of CPU threads for OpenMP
    int batch_size;
    int num_epochs;
    float learning_rate;
    int verbose;            // Print level: 0=quiet, 1=normal, 2=verbose
} Config;

// Default configuration
static inline Config config_default(void) {
    Config cfg;
    cfg.device = DEVICE_CPU;
    cfg.cuda_device_id = 0;
    cfg.num_threads = 4;
    cfg.batch_size = 128;
    cfg.num_epochs = 10;
    cfg.learning_rate = 0.001f;
    cfg.verbose = 1;
    return cfg;
}

// Print configuration
void config_print(const Config* cfg);

// Device name helpers
const char* device_type_to_string(DeviceType device);

#endif // CONFIG_H
