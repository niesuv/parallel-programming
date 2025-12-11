# CIFAR-10 Autoencoder with GPU Acceleration

A high-performance implementation of a convolutional autoencoder for CIFAR-10 image reconstruction and feature extraction, with both CPU and GPU training support.

## Project Structure

```
final_pj/
├── include/
│   ├── data_loader.h          # CIFAR-10 dataset loading
│   ├── cpu_layers.h           # CPU layer implementations
│   ├── autoencoder.h          # Autoencoder architecture
│   └── gpu_autoencoder.h      # GPU autoencoder kernels
├── src/
│   ├── data_loader.cpp        # Dataset loading implementation
│   ├── cpu_layers.cpp         # CPU layer implementations
│   ├── autoencoder.cpp        # Autoencoder class
│   ├── gpu_autoencoder.cu     # GPU kernels and implementation
│   ├── train_cpu.cpp          # CPU training loop
│   ├── train_gpu.cu           # GPU training loop
│   ├── feature_extraction.cu  # Feature extraction (GPU)
│   └── svm_classifier.cpp     # SVM classifier
├── data/
│   └── cifar-10-batches-bin/  # CIFAR-10 dataset
├── build/                      # Build directory
├── CMakeLists.txt             # CMake configuration
└── README.md                  # This file
```

## Requirements

### Hardware

- NVIDIA GPU with CUDA Compute Capability 7.5+ (recommended: RTX series)
- 8GB+ GPU memory (16GB recommended for batch size 128)
- 8GB+ CPU RAM for dataset loading

### Software

- CMake 3.20+
- CUDA Toolkit 11.0+
- C++17 compatible compiler
- LIBSVM (optional, for SVM classification)

### macOS/Linux

```bash
brew install cmake  # macOS
apt-get install cmake  # Linux

# CUDA installation
# Visit https://developer.nvidia.com/cuda-downloads
```

## Installation & Building

### 1. Download CIFAR-10 Dataset

```bash
cd final_pj/data
# Download from https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar xzf cifar-10-binary.tar.gz
cd cifar-10-batches-bin
```

Verify you have:

- `data_batch_1.bin` through `data_batch_5.bin` (50,000 training images)
- `test_batch.bin` (10,000 test images)
- `batches.meta.txt` (class names)

### 2. Install LIBSVM (optional, for SVM classifier)

```bash
# macOS
brew install libsvm

# Linux
apt-get install libsvm-dev

# Or build from source:
git clone https://github.com/cjlin1/libsvm.git
cd libsvm
make
```

### 3. Build the Project

```bash
cd final_pj
mkdir -p build
cd build
cmake ..
make
```

This will create:

- `train_cpu` - CPU training executable
- `train_gpu` - GPU training executable
- `extract_features` - Feature extraction executable
- `svm_classifier` - SVM classification executable

## Usage

### CPU Training (for testing/debugging)

```bash
cd build

# Basic training (20 epochs, batch size 32)
./train_cpu --data-dir ../data/cifar-10-batches-bin

# Custom parameters
./train_cpu --data-dir ../data/cifar-10-batches-bin \
            --epochs 5 \
            --batch-size 32 \
            --lr 0.001 \
            --num-samples 5000  # Use only 5000 samples for quick testing

# Help
./train_cpu --help
```

**Parameters:**

- `--epochs N`: Number of training epochs (default: 20)
- `--batch-size N`: Batch size (default: 32)
- `--lr LR`: Learning rate (default: 0.001)
- `--num-samples N`: Number of training samples (default: 50000)
- `--data-dir PATH`: Path to CIFAR-10 data directory

### GPU Training (full dataset)

```bash
cd build

# Basic training (20 epochs, batch size 64)
./train_gpu --data-dir ../data/cifar-10-batches-bin

# Custom parameters
./train_gpu --data-dir ../data/cifar-10-batches-bin \
            --epochs 20 \
            --batch-size 128 \
            --lr 0.001

# Monitor GPU usage in another terminal
nvidia-smi -l 1  # Update every 1 second
```

**Expected Performance:**

- **GPU Training Time**: < 10 minutes for 20 epochs (batch size 128)
- **Final Loss**: < 0.01
- **GPU Speedup**: 20-50x vs CPU

### Feature Extraction

```bash
cd build

# Extract features using pre-trained weights
./extract_features --data-dir ../data/cifar-10-batches-bin \
                   --output ./cifar10_features.bin \
                   --weights ./autoencoder_gpu.weights

# Expected time: < 20 seconds for all 60K images
```

### SVM Classification

Requires pre-extracted features:

```bash
cd build

# First extract features
./extract_features --data-dir ../data/cifar-10-batches-bin \
                   --output ./cifar10_features.bin

# Then run SVM classifier
./svm_classifier --data-dir ../data/cifar-10-batches-bin \
                 --features ./cifar10_features.bin
```

**Expected Accuracy:** 60-65% on test set

## Architecture Details

### Encoder

```
Input (32×32×3)
    ↓
Conv2D(256, 3×3, pad=1) + ReLU → (32×32×256) [7,168 params]
    ↓
MaxPool2D(2×2) → (16×16×256)
    ↓
Conv2D(128, 3×3, pad=1) + ReLU → (16×16×128) [295,040 params]
    ↓
MaxPool2D(2×2) → (8×8×128)
    ↓
Latent Features (8×8×128 = 8,192 dimensions)
```

### Decoder

```
Latent (8×8×128)
    ↓
Conv2D(128, 3×3, pad=1) + ReLU → (8×8×128) [147,584 params]
    ↓
UpSample2D(2×2) → (16×16×128)
    ↓
Conv2D(256, 3×3, pad=1) + ReLU → (16×16×256) [295,168 params]
    ↓
UpSample2D(2×2) → (32×32×256)
    ↓
Conv2D(3, 3×3, pad=1) [no activation] → (32×32×3) [6,915 params]
    ↓
Output (32×32×3)
```

**Total Parameters:** 751,875 (trainable)

## Training Configuration

- **Loss Function:** MSE (Mean Squared Error)
- **Optimizer:** SGD (Stochastic Gradient Descent)
- **Learning Rate:** 0.001
- **Epochs:** 20 (configurable)
- **Data Normalization:** uint8 [0,255] → float [0,1]

## GPU Implementation Details

### Kernels Implemented

1. **naiveConv2D**: 2D convolution with padding and stride support
2. **reluKernel**: ReLU activation function
3. **reluBackwardKernel**: ReLU gradient computation
4. **maxPool2DKernel**: Max pooling with index storage
5. **maxPoolBackwardKernel**: Max pool gradient routing
6. **upSampleKernel**: Nearest-neighbor upsampling
7. **upSampleBackwardKernel**: Upsampling gradient computation
8. **mseLossKernel**: MSE loss with reduction
9. **sgdUpdateKernel**: SGD weight updates

### Memory Usage

- **GPU Memory per batch (batch_size=128):**
  - Input: ~12 MB
  - Activations: ~300 MB
  - Weights: ~12 MB
  - Gradients: ~300 MB
  - **Total**: ~625 MB (leaves headroom on 8GB GPU)

## Performance Optimization Tips

1. **Batch Size:** Use 128 for GPU training (default 64)
2. **Learning Rate:** Reduce to 0.0001 if loss diverges
3. **Number of Epochs:** Monitor convergence, 20 epochs usually sufficient
4. **GPU Compute:** Run with `nvidia-smi` to verify GPU utilization (>80% is good)

## Output Files

After training and feature extraction:

```
build/
├── autoencoder_cpu.weights      # CPU model weights
├── autoencoder_gpu.weights      # GPU model weights
└── cifar10_features.bin         # Extracted features (60K × 8192 floats)
```

## Troubleshooting

### CUDA Out of Memory

- Reduce batch size: `--batch-size 32`
- Use fewer samples: `--num-samples 10000`
- Run GPU memory check: `nvidia-smi`

### Slow Training

- Ensure you're using GPU version: `./train_gpu`
- Check GPU usage: `nvidia-smi` (should be >80%)
- Try larger batch size: `--batch-size 128`

### Data Loading Errors

- Verify CIFAR-10 data path: `ls data/cifar-10-batches-bin/`
- Check file permissions: `chmod 644 data/cifar-10-batches-bin/*`
- Ensure binary files are not corrupted

### Build Errors

- Update CUDA: `nvidia-smi` shows CUDA version
- Verify CMake: `cmake --version`
- Check GPU compute capability: `nvidia-smi --query-gpu=compute_cap`

## Performance Targets (MANDATORY)

✓ Autoencoder training time: < 10 minutes  
✓ Feature extraction time: < 20 seconds for all 60K images  
✓ Test classification accuracy: 60-65%  
✓ GPU speedup over CPU: > 20x

## LIBSVM Integration

The SVM classifier uses LIBSVM with RBF kernel:

- **C (regularization):** 10.0
- **gamma:** 1.0 / 8192.0
- **Kernel:** RBF (Radial Basis Function)

Training time: ~5-10 minutes for 50K samples

## References

- CIFAR-10 Dataset: https://www.cs.toronto.edu/~kriz/cifar.html
- LIBSVM: https://www.csie.ntu.edu.tw/~cjlin/libsvm/

## License

This project is provided for educational and research purposes.
