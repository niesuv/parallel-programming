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
DATA_DIR = $(SRC_DIR)/data
CPU_DIR = $(SRC_DIR)/cpu
CUDA_DIR = $(SRC_DIR)/cuda
UTILS_DIR = $(SRC_DIR)/utils
TEST_DIR = test
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include

# Source files
DATA_SRC = $(DATA_DIR)/cifar10.c
CPU_SRC = $(wildcard $(CPU_DIR)/*.c)
CUDA_SRC = $(wildcard $(CUDA_DIR)/*.cu)
UTILS_SRC = $(UTILS_DIR)/config.c $(UTILS_DIR)/device.c $(UTILS_DIR)/benchmark.c

# Object files
DATA_OBJ = $(BUILD_DIR)/cifar10.o
CPU_OBJ = $(patsubst $(CPU_DIR)/%.c, $(BUILD_DIR)/%.o, $(CPU_SRC))
CUDA_OBJ = $(patsubst $(CUDA_DIR)/%.cu, $(BUILD_DIR)/%.o, $(CUDA_SRC))
UTILS_OBJ = $(BUILD_DIR)/config.o $(BUILD_DIR)/device.o $(BUILD_DIR)/benchmark.o

# Common objects
COMMON_OBJ = $(DATA_OBJ) $(UTILS_OBJ)

# Autoencoder objects (layers + autoencoder)
AUTOENCODER_OBJ = $(BUILD_DIR)/conv2d_cpu.o $(BUILD_DIR)/relu_cpu.o $(BUILD_DIR)/maxpool_cpu.o \
                  $(BUILD_DIR)/upsample_cpu.o $(BUILD_DIR)/loss_cpu.o $(BUILD_DIR)/autoencoder.o

# Test executables
TEST_CIFAR10 = $(BIN_DIR)/test_cifar10
TEST_DEVICE_COMPARE = $(BIN_DIR)/test_device_compare
TEST_DEVICE_COMPARE_CUDA = $(BIN_DIR)/test_device_compare_cuda
TRAIN_AUTOENCODER = $(BIN_DIR)/train_autoencoder

# Detect CUDA availability
CUDA_AVAILABLE := $(shell which nvcc 2>/dev/null)

# Default target
.PHONY: all
ifdef CUDA_AVAILABLE
all: dirs $(TEST_CIFAR10) $(TEST_DEVICE_COMPARE_CUDA) $(TRAIN_AUTOENCODER)
	@echo "Built with CUDA support"
else
all: dirs $(TEST_CIFAR10) $(TEST_DEVICE_COMPARE) $(TRAIN_AUTOENCODER)
	@echo "Built CPU-only version (CUDA not available)"
endif

# Create necessary directories
.PHONY: dirs
dirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

# Build test_cifar10 (basic CPU-only test)
$(TEST_CIFAR10): $(TEST_DIR)/test_cifar10.c $(DATA_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

# Build test_device_compare (CPU-only version)
$(TEST_DEVICE_COMPARE): $(TEST_DIR)/test_device_compare.c $(COMMON_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@ (CPU-only)"

# Build test_device_compare (CUDA version)
ifdef CUDA_AVAILABLE
$(TEST_DEVICE_COMPARE_CUDA): $(TEST_DIR)/test_device_compare.c $(COMMON_OBJ) $(CUDA_OBJ)
	$(NVCC) $(NVCCFLAGS) -o $@ $^ $(LDFLAGS) $(CUDA_LDFLAGS)
	@echo "Built: $@ (with CUDA)"
endif

# Build train_autoencoder
$(TRAIN_AUTOENCODER): $(TEST_DIR)/train_autoencoder.c $(COMMON_OBJ) $(AUTOENCODER_OBJ)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS)
	@echo "Built: $@"

# Compile data source files
$(BUILD_DIR)/cifar10.o: $(DATA_DIR)/cifar10.c $(INCLUDE_DIR)/cifar10.h $(INCLUDE_DIR)/config.h $(INCLUDE_DIR)/device.h
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CPU source files (layers and autoencoder)
$(BUILD_DIR)/conv2d_cpu.o: $(CPU_DIR)/conv2d_cpu.c $(INCLUDE_DIR)/layers.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/relu_cpu.o: $(CPU_DIR)/relu_cpu.c $(INCLUDE_DIR)/layers.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/maxpool_cpu.o: $(CPU_DIR)/maxpool_cpu.c $(INCLUDE_DIR)/layers.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/upsample_cpu.o: $(CPU_DIR)/upsample_cpu.c $(INCLUDE_DIR)/layers.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/loss_cpu.o: $(CPU_DIR)/loss_cpu.c $(INCLUDE_DIR)/layers.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/autoencoder.o: $(CPU_DIR)/autoencoder.c $(INCLUDE_DIR)/autoencoder.h $(INCLUDE_DIR)/layers.h
	$(CC) $(CFLAGS) -c $< -o $@

# Compile CUDA source files (if CUDA is available)
ifdef CUDA_AVAILABLE
$(BUILD_DIR)/device_cuda.o: $(CUDA_DIR)/device_cuda.cu $(INCLUDE_DIR)/device.h
	$(NVCC) $(NVCCFLAGS) -c $< -o $@
endif

# Compile utility source files
$(BUILD_DIR)/config.o: $(UTILS_DIR)/config.c $(INCLUDE_DIR)/config.h
	$(CC) $(CFLAGS) -c $< -o $@

$(BUILD_DIR)/device.o: $(UTILS_DIR)/device.c $(INCLUDE_DIR)/device.h $(INCLUDE_DIR)/config.h
ifdef CUDA_AVAILABLE
	$(CC) $(CFLAGS) -D__NVCC__ -c $< -o $@
else
	$(CC) $(CFLAGS) -c $< -o $@
endif

$(BUILD_DIR)/benchmark.o: $(UTILS_DIR)/benchmark.c $(INCLUDE_DIR)/benchmark.h $(INCLUDE_DIR)/config.h
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

# Train autoencoder
.PHONY: train
train: $(TRAIN_AUTOENCODER)
	@echo "Training autoencoder..."
	@echo "Usage: make train DATA_PATH=<path_to_cifar10_data> [EPOCHS=N] [BATCH_SIZE=N] [LR=f]"
	@if [ -z "$(DATA_PATH)" ]; then \
		echo "Error: Please specify DATA_PATH"; \
		echo "Example: make train DATA_PATH=./cifar-10-batches-bin EPOCHS=20 BATCH_SIZE=32 LR=0.001"; \
		exit 1; \
	fi
	$(TRAIN_AUTOENCODER) $(DATA_PATH) \
		$(if $(EPOCHS),--epochs $(EPOCHS),) \
		$(if $(BATCH_SIZE),--batch-size $(BATCH_SIZE),) \
		$(if $(LR),--lr $(LR),)

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
