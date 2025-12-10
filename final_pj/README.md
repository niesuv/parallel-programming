# CIFAR-10 Deep Learning Project in C/CUDA

A high-performance deep learning implementation for CIFAR-10 classification using C and CUDA, with built-in CPU/GPU comparison capabilities.

## Project Structure

```
final_pj/
├── include/              # Header files
│   ├── cifar10.h        # CIFAR-10 data structures and API
│   ├── config.h         # Configuration and device selection
│   ├── device.h         # Device abstraction layer
│   └── benchmark.h      # Benchmarking and comparison utilities
├── src/
│   ├── data/            # Data loading and preprocessing
│   │   └── cifar10.c
│   ├── cpu/             # CPU implementations (future)
│   ├── cuda/            # CUDA implementations
│   │   └── device_cuda.cu
│   └── utils/           # Utility functions
│       ├── config.c
│       ├── device.c
│       └── benchmark.c
├── test/                # Test programs
│   ├── test_cifar10.c           # Basic data loading test
│   └── test_device_compare.c   # CPU vs GPU comparison
├── build/               # Build artifacts (generated)
├── bin/                 # Executables (generated)
├── Makefile             # Build system
└── README.md
```

## Features

### Phase 1: Data Pipeline ✅

- **CIFAR-10 Binary File Parser**
  - Reads binary format: 1 byte label + 3072 bytes image
  - Handles 5 training batches (50,000 images) + 1 test batch (10,000 images)
  - Automatic memory management

- **Data Preprocessing**
  - Converts uint8 [0, 255] to float [0, 1] normalization
  - Efficient memory layout for batch processing

- **Batch Generation**
  - Configurable batch size
  - Optional data shuffling using Fisher-Yates algorithm
  - Efficient batch iteration with reset capability

### Device Management & Configuration ✅

- **Device Abstraction Layer**
  - Unified API for CPU and CUDA operations
  - Automatic device detection
  - Transparent memory management

- **Configuration System**
  - Easy device selection (CPU/CUDA)
  - Configurable training parameters
  - Runtime device switching

- **Benchmark & Comparison Tools**
  - Automatic performance measurement
  - Side-by-side CPU vs GPU comparison
  - Export results to files
  - Pretty-printed comparison tables

## Building

### Prerequisites

**For CPU-only:**
```bash
gcc (or any C11-compatible compiler)
make
```

**For CUDA support:**
```bash
gcc
make
CUDA Toolkit (nvcc)
NVIDIA GPU with compute capability >= 7.5
```

### Compile

```bash
# Build everything (auto-detects CUDA)
make

# Build CPU-only version
make all  # (when CUDA not available)

# Clean build artifacts
make clean

# Show help
make help
```

## Running Tests

### 1. Download CIFAR-10 Dataset

```bash
# Download binary version
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz

# Extract
tar -xzf cifar-10-binary.tar.gz

# This creates: cifar-10-batches-bin/
#   - data_batch_1.bin ... data_batch_5.bin
#   - test_batch.bin
#   - batches.meta.txt
```

### 2. Basic Data Loading Test

```bash
make test DATA_PATH=./cifar-10-batches-bin
```

### 3. CPU vs GPU Comparison

```bash
# Run comparison (auto-detects CUDA)
make compare DATA_PATH=./cifar-10-batches-bin

# Run CPU-only comparison
make compare-cpu DATA_PATH=./cifar-10-batches-bin
```

## Configuration

### Device Selection

You can easily configure which device to use:

```c
#include "config.h"

// Create configuration
Config cfg = config_default();

// Choose device
cfg.device = DEVICE_CPU;   // or DEVICE_CUDA
cfg.batch_size = 128;
cfg.num_epochs = 10;

config_print(&cfg);
```

### Example: CPU vs GPU Workflow

```c
// Load data
CIFAR10Dataset* train_data = cifar10_load_train_data("./cifar-10-batches-bin");

// Transfer to GPU (if needed)
if (cfg.device == DEVICE_CUDA) {
    cifar10_transfer_to_device(train_data, DEVICE_CUDA);
}

// Create batch iterator
CIFAR10Batch* batch_iter = cifar10_create_batch_iterator(train_data, 128, 1);

// Training loop
while (cifar10_next_batch(batch_iter, train_data) > 0) {
    // Your training code here...
}

// Transfer back to CPU
cifar10_transfer_to_host(train_data);
```

## Benchmarking

The project includes comprehensive benchmarking tools:

```bash
# Results are saved to:
#   - benchmark_CPU.txt
#   - benchmark_CUDA.txt
```

### Benchmark Output Example

```
╔════════════════════════════════════════════════╗
║        Benchmark Results - CPU                 ║
╠════════════════════════════════════════════════╣
║ Timing:                                        ║
║   Data Loading:          2.1234 s              ║
║   Training:              0.0000 s              ║
║   Inference:             0.0000 s              ║
║   Total:                 2.1234 s              ║
╟────────────────────────────────────────────────╢
║ Training Metrics:                              ║
║   Throughput:          5123.45 imgs/s          ║
╚════════════════════════════════════════════════╝
```

## Google Colab Integration

### Setup on Colab

```bash
# 1. Upload project files to Colab

# 2. Install build tools (if needed)
!apt-get update
!apt-get install -y build-essential

# 3. Download CIFAR-10 data
!wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz

# 4. Check CUDA availability
!nvcc --version

# 5. Build with CUDA
!make clean
!make

# 6. Run comparison
!make compare DATA_PATH=./cifar-10-batches-bin
```

### Enable GPU on Colab

1. Runtime → Change runtime type
2. Hardware accelerator → GPU
3. Save

## Dataset Details

- **Image format**: 32×32 RGB (3072 values per image)
- **Channel order**: Red (1024) → Green (1024) → Blue (1024)
- **Storage order**: Row-major
- **Classes**: 10 (airplane, automobile, bird, cat, deer, dog, frog, horse, ship, truck)
- **Training set**: 50,000 images (5 batches × 10,000)
- **Test set**: 10,000 images

## API Reference

### Configuration

```c
Config cfg = config_default();        // Create default config
config_print(&cfg);                    // Print configuration
const char* name = device_type_to_string(DEVICE_CPU);
```

### Device Management

```c
device_init(DEVICE_CUDA, 0);          // Initialize device
device_is_available(DEVICE_CUDA);     // Check availability
device_print_info(DEVICE_CUDA);       // Print device info
device_cleanup(DEVICE_CUDA);          // Cleanup

DeviceMemory* mem = device_malloc(size, DEVICE_CUDA);
device_free(mem);
device_synchronize(DEVICE_CUDA);
```

### Data Loading

```c
CIFAR10Dataset* train = cifar10_load_train_data(path);
CIFAR10Dataset* test = cifar10_load_test_data(path);
cifar10_print_dataset_info(train);

// Device transfer
cifar10_transfer_to_device(train, DEVICE_CUDA);
cifar10_transfer_to_host(train);

cifar10_free_dataset(train);
```

### Batch Processing

```c
CIFAR10Batch* batch = cifar10_create_batch_iterator(dataset, 128, 1);

int size;
while ((size = cifar10_next_batch(batch, dataset)) > 0) {
    // batch->data contains batch images
    // batch->labels contains batch labels
}

cifar10_reset_batch_iterator(batch);
cifar10_free_batch_iterator(batch);
```

### Benchmarking

```c
Timer* timer = timer_create();
timer_start(timer);
// ... code to benchmark ...
timer_stop(timer);
double elapsed = timer_elapsed(timer);

BenchmarkResult* result = benchmark_create(DEVICE_CPU);
// ... fill in metrics ...
benchmark_print(result);
benchmark_save_to_file(result, "results.txt");
benchmark_compare(cpu_result, gpu_result);
```

## Next Steps

- **Phase 2**: Neural network implementation (CPU baseline)
- **Phase 3**: CUDA kernels for forward/backward pass
- **Phase 4**: Optimization and advanced features

## Contributing

This is an educational project for parallel programming course at VNU-HCM, University of Science.

## License

Educational project.
