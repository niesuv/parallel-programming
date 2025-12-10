# Makefile for CIFAR-10 Deep Learning Project
# Supports both CPU and CUDA compilation

# Compiler settings
CC = gcc
NVCC = nvcc
CFLAGS = -Wall -O3 -std=c11 -Iinclude
NVCCFLAGS = -O3 -Iinclude -arch=sm_75
LDFLAGS = -lm
CUDA_LDFLAGS = -lcudart

# Directories
SRC_DIR = src
LAYERS_DIR = $(SRC_DIR)/layers
LAYERS_CPU_DIR = $(LAYERS_DIR)/cpu
LAYERS_GPU_DIR = $(LAYERS_DIR)/gpu
MODELS_DIR = $(SRC_DIR)/models
CORE_DIR = $(SRC_DIR)/core
TEST_DIR = test
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include

# Source files
DATA_SRC = $(CORE_DIR)/cifar10.c
CPU_SRC = $(wildcard $(LAYERS_CPU_DIR)/*.c)
CUDA_SRC = $(wildcard $(LAYERS_GPU_DIR)/*.cu)
UTILS_SRC = $(CORE_DIR)/config.c $(CORE_DIR)/device.c $(CORE_DIR)/benchmark.c

# Object files
DATA_OBJ = $(BUILD_DIR)/cifar10.o
CPU_OBJ = $(patsubst $(LAYERS_CPU_DIR)/%.c, $(BUILD_DIR)/%.o, $(CPU_SRC))
CUDA_OBJ = $(patsubst $(LAYERS_GPU_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRC))
UTILS_OBJ = $(BUILD_DIR)/config.o $(BUILD_DIR)/device.o $(BUILD_DIR)/benchmark.o

# Common objects
COMMON_OBJ = $(DATA_OBJ) $(UTILS_OBJ)

# Common objects with CUDA support (for CUDA executables)
ifdef CUDA_AVAILABLE
COMMON_OBJ_CUDA = $(DATA_OBJ) $(BUILD_DIR)/config.o $(BUILD_DIR)/device_cuda_enabled.o $(BUILD_DIR)/benchmark.o
endif

# Autoencoder objects (layers + autoencoder)
AUTOENCODER_OBJ = $(BUILD_DIR)/conv2d_cpu.o $(BUILD_DIR)/relu_cpu.o $(BUILD_DIR)/maxpool_cpu.o \
                  $(BUILD_DIR)/upsample_cpu.o $(BUILD_DIR)/loss_cpu.o $(BUILD_DIR)/autoencoder.o

# GPU Autoencoder objects (CUDA kernels + operations + training)
ifdef CUDA_AVAILABLE
GPU_AUTOENCODER_OBJ = $(BUILD_DIR)/autoencoder_gpu.o $(BUILD_DIR)/autoencoder_gpu_ops.o $(BUILD_DIR)/autoencoder_gpu_train.o
endif

# Test executables
TEST_CIFAR10 = $(BIN_DIR)/test_cifar10
TEST_DEVICE_COMPARE = $(BIN_DIR)/test_device_compare
TEST_DEVICE_COMPARE_CUDA = $(BIN_DIR)/test_device_compare_cuda
TRAIN_AUTOENCODER = $(BIN_DIR)/train_autoencoder
TRAIN_AUTOENCODER_GPU = $(BIN_DIR)/train_autoencoder_gpu
TEST_CONV2D = $(BIN_DIR)/test_conv2d_simple

# Detect CUDA availability
CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)

# Default target
.PHONY: all
ifdef CUDA_AVAILABLE
all: dirs $(TEST_CIFAR10) $(TEST_DEVICE_COMPARE_CUDA) $(TRAIN_AUTOENCODER) $(TRAIN_AUTOENCODER_GPU)
	@echo "Built with CUDA support (including GPU autoencoder)"
else
all: dirs $(TEST_CIFAR10) $(TEST_DEVICE_COMPARE) $(TRAIN_AUTOENCODER)
	@echo "Built CPU-only version (CUDA not available)"
endif

# Create necessary directories
.PHONY: dirs
dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build test_cifar10 (basic CPU-only test) - now needs COMMON_OBJ for device functions
$(TEST_CIFAR10): $(TEST_DIR)/test_cifar10.c $(COMMON_OBJ) | dirs
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@chmod +x $@
	@echo "Built: $@"

# Build test_device_compare (CPU-only version)
$(TEST_DEVICE_COMPARE): $(TEST_DIR)/test_device_compare.c $(COMMON_OBJ) | dirs
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@chmod +x $@
	@echo "Built: $@ (CPU-only)"

# Build test_device_compare (CUDA version)
ifdef CUDA_AVAILABLE
$(TEST_DEVICE_COMPARE_CUDA): $(TEST_DIR)/test_device_compare.c $(COMMON_OBJ_CUDA) $(CUDA_OBJ) | dirs
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
	@chmod +x $@
	@echo "Built: $@ (with CUDA)"
endif

# Build train_autoencoder
$(TRAIN_AUTOENCODER): $(TEST_DIR)/train_autoencoder.c $(COMMON_OBJ) $(AUTOENCODER_OBJ) | dirs
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@chmod +x $@
	@echo "Built: $@"

# Build test_conv2d_simple
$(TEST_CONV2D): $(TEST_DIR)/test_conv2d_simple.c $(COMMON_OBJ) $(AUTOENCODER_OBJ) | dirs
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@chmod +x $@
	@echo "Built: $@"

# Build train_autoencoder_gpu (CUDA version)
ifdef CUDA_AVAILABLE
$(TRAIN_AUTOENCODER_GPU): $(TEST_DIR)/train_autoencoder_gpu.c $(COMMON_OBJ) $(AUTOENCODER_OBJ) $(GPU_AUTOENCODER_OBJ) | dirs
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
	@chmod +x $@
	@echo "Built: $@ (GPU Naive Implementation)"
endif

# Compile data source files
$(BUILD_DIR)/cifar10.o: $(CORE_DIR)/cifar10.c $(INCLUDE_DIR)/cifar10.h $(INCLUDE_DIR)/config.h $(INCLUDE_DIR)/device.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CPU source files (layers and autoencoder)
$(BUILD_DIR)/conv2d_cpu.o: $(LAYERS_CPU_DIR)/conv2d_cpu.c $(INCLUDE_DIR)/layers.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/relu_cpu.o: $(LAYERS_CPU_DIR)/relu_cpu.c $(INCLUDE_DIR)/layers.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/maxpool_cpu.o: $(LAYERS_CPU_DIR)/maxpool_cpu.c $(INCLUDE_DIR)/layers.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/upsample_cpu.o: $(LAYERS_CPU_DIR)/upsample_cpu.c $(INCLUDE_DIR)/layers.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/loss_cpu.o: $(LAYERS_CPU_DIR)/loss_cpu.c $(INCLUDE_DIR)/layers.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/autoencoder.o: $(MODELS_DIR)/autoencoder_cpu.c $(INCLUDE_DIR)/autoencoder.h $(INCLUDE_DIR)/layers.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files (if CUDA is available)
ifdef CUDA_AVAILABLE
$(BUILD_DIR)/device_cuda.o: $(CORE_DIR)/device_cuda.cu $(INCLUDE_DIR)/device.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# GPU Layer CUDA files
$(BUILD_DIR)/conv2d_gpu.o: $(LAYERS_GPU_DIR)/conv2d_gpu.cu $(INCLUDE_DIR)/layers.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/relu_gpu.o: $(LAYERS_GPU_DIR)/relu_gpu.cu $(INCLUDE_DIR)/layers.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/maxpool_gpu.o: $(LAYERS_GPU_DIR)/maxpool_gpu.cu $(INCLUDE_DIR)/layers.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/upsample_gpu.o: $(LAYERS_GPU_DIR)/upsample_gpu.cu $(INCLUDE_DIR)/layers.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/loss_gpu.o: $(LAYERS_GPU_DIR)/loss_gpu.cu $(INCLUDE_DIR)/layers.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/gpu_layer_conv2d.o: $(LAYERS_GPU_DIR)/gpu_layer_conv2d.cu $(INCLUDE_DIR)/gpu_layer.h $(INCLUDE_DIR)/layers.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

# GPU Autoencoder CUDA files
$(BUILD_DIR)/autoencoder_gpu.o: $(MODELS_DIR)/autoencoder_gpu.cu $(INCLUDE_DIR)/autoencoder_gpu.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/autoencoder_gpu_ops.o: $(MODELS_DIR)/autoencoder_gpu_ops.cu $(INCLUDE_DIR)/autoencoder_gpu.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@

$(BUILD_DIR)/autoencoder_gpu_train.o: $(MODELS_DIR)/autoencoder_gpu_train.cu $(INCLUDE_DIR)/autoencoder_gpu.h | dirs
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# Compile utility source files
$(BUILD_DIR)/config.o: $(CORE_DIR)/config.c $(INCLUDE_DIR)/config.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/device.o: $(CORE_DIR)/device.c $(INCLUDE_DIR)/device.h $(INCLUDE_DIR)/config.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

# Compile device.o with CUDA support (for CUDA executables)
ifdef CUDA_AVAILABLE
$(BUILD_DIR)/device_cuda_enabled.o: $(CORE_DIR)/device.c $(INCLUDE_DIR)/device.h $(INCLUDE_DIR)/config.h | dirs
	$(CC) $(CFLAGS) -D__NVCC__ -c $< -o $@
endif

$(BUILD_DIR)/benchmark.o: $(CORE_DIR)/benchmark.c $(INCLUDE_DIR)/benchmark.h $(INCLUDE_DIR)/config.h | dirs
	$(CC) $(CFLAGS) -c $< -o $@

# Clean build artifacts
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR) $(BIN_DIR)
	rm -f benchmark_*.txt
	rm -f autoencoder_*.weights autoencoder_*.txt
	@echo "Cleaned build artifacts"

# Run the basic CIFAR-10 test
.PHONY: test
test: $(TEST_CIFAR10)
	@echo "Running CIFAR-10 data loading test..."
	@echo "Usage: make test DATA_PATH=<path_to_cifar10_data>"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: Please specify DATA_PATH"; \
		echo "Example: make test DATA_PATH=./cifar-10-batches-bin"; \
		exit 1; \
	fi
	$(TEST_CIFAR10) $(DATA_PATH)

# Run CPU vs GPU comparison test
.PHONY: compare
ifdef CUDA_AVAILABLE
compare: $(TEST_DEVICE_COMPARE_CUDA)
	@echo "Running CPU vs GPU comparison test..."
	@echo "Usage: make compare DATA_PATH=<path_to_cifar10_data>"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: Please specify DATA_PATH"; \
		echo "Example: make compare DATA_PATH=./cifar-10-batches-bin"; \
		exit 1; \
	fi
	$(TEST_DEVICE_COMPARE_CUDA) $(DATA_PATH)
else
compare: $(TEST_DEVICE_COMPARE)
	@echo "Running device comparison test (CPU-only)..."
	@echo "Usage: make compare DATA_PATH=<path_to_cifar10_data>"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: Please specify DATA_PATH"; \
		echo "Example: make compare DATA_PATH=./cifar-10-batches-bin"; \
		exit 1; \
	fi
	$(TEST_DEVICE_COMPARE) $(DATA_PATH) --cpu-only
endif

# Run CPU-only comparison
.PHONY: compare-cpu
compare-cpu: $(TEST_DEVICE_COMPARE)
	@echo "Running CPU-only comparison test..."
	@echo "Usage: make compare-cpu DATA_PATH=<path_to_cifar10_data>"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: Please specify DATA_PATH"; \
		echo "Example: make compare-cpu DATA_PATH=./cifar-10-batches-bin"; \
		exit 1; \
	fi
	$(TEST_DEVICE_COMPARE) $(DATA_PATH) --cpu-only

# Train autoencoder (CPU)
.PHONY: train
train: $(TRAIN_AUTOENCODER)
	@echo "Training autoencoder on CPU..."
	@echo "Usage: make train DATA_PATH=<path> [EPOCHS=N] [BATCH_SIZE=N] [LR=f] [NUM_SAMPLES=N]"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: Please specify DATA_PATH"; \
		echo "Example: make train DATA_PATH=./cifar-10-batches-bin EPOCHS=20"; \
		echo "Quick test: make train DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=1000 EPOCHS=2"; \
		exit 1; \
	fi
	$(TRAIN_AUTOENCODER) $(DATA_PATH) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(LR),--lr $(LR),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)

# Train autoencoder on GPU (Naive Implementation)
ifdef CUDA_AVAILABLE
.PHONY: train-gpu
train-gpu: $(TRAIN_AUTOENCODER_GPU)
	@echo "Training autoencoder on GPU (Naive Implementation)..."
	@echo "Usage: make train-gpu DATA_PATH=<path> [EPOCHS=N] [BATCH_SIZE=N] [LR=f] [NUM_SAMPLES=N]"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: Please specify DATA_PATH"; \
		echo "Example: make train-gpu DATA_PATH=./cifar-10-batches-bin EPOCHS=20"; \
		echo "Quick test: make train-gpu DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=2000 EPOCHS=2"; \
		exit 1; \
	fi
	$(TRAIN_AUTOENCODER_GPU) $(DATA_PATH) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(LR),--lr $(LR),) \
		$(if $(NUM_SAMPLES),--num-samples $(NUM_SAMPLES),)
endif

# Help target
.PHONY: help
help:
	@echo "CIFAR-10 Deep Learning Project Makefile"
	@echo ""
	@echo "Targets:"
	@echo "  all              - Build all targets (default)"
	@echo "  test             - Run CIFAR-10 data loading test"
	@echo "  compare          - Run CPU vs GPU comparison test"
	@echo "  compare-cpu      - Run CPU-only comparison test"
	@echo "  train            - Train autoencoder (CPU)"
ifdef CUDA_AVAILABLE
	@echo "  train-gpu        - Train autoencoder on GPU (Naive)"
endif
	@echo "  clean            - Remove build artifacts"
	@echo "  help             - Show this help message"
	@echo ""
	@echo "Examples:"
	@echo "  make                                          # Build everything"
	@echo "  make test DATA_PATH=./cifar-10-batches-bin   # Run basic test"
	@echo "  make compare DATA_PATH=./cifar-10-batches-bin # Compare CPU vs GPU"
	@echo "  make train DATA_PATH=./cifar-10-batches-bin EPOCHS=20  # Train autoencoder"
	@echo "  make clean                                    # Clean build"
	@echo ""
	@echo "Environment:"
	@echo "  CC      = $(CC)"
ifdef CUDA_AVAILABLE
	@echo "  NVCC    = $(NVCC) (CUDA available)"
	@echo "  NVCCFLAGS = $(NVCCFLAGS)"
else
	@echo "  NVCC    = not found (CUDA not available)"
endif
	@echo "  CFLAGS  = $(CFLAGS)"
	@echo "  LDFLAGS = $(LDFLAGS)"
