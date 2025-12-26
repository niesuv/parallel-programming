# CIFAR-10 Autoencoder with SVM Classification

A high-performance CUDA implementation of a convolutional autoencoder for CIFAR-10 feature extraction, followed by SVM classification. Uses FP16 (half precision) for fast training on NVIDIA GPUs.

## Overview

This project implements a complete pipeline for unsupervised feature learning:

1. **Autoencoder Training** - Learn compressed representations of CIFAR-10 images
2. **Feature Extraction** - Extract latent features using the trained encoder
3. **SVM Classification** - Train a classifier on the learned features

### Architecture

```
Input (32×32×3) 
    ↓
[Encoder]
    Conv1 (3→256) + ReLU + MaxPool → 16×16×256
    Conv2 (256→128) + ReLU + MaxPool → 8×8×128
    Conv3 (128→128) + ReLU → 8×8×128 (Latent: 8192-dim)
    ↓
[Decoder]
    Upsample → 16×16×128
    Conv4 (128→256) + ReLU
    Upsample → 32×32×256
    Conv5 (256→3) → 32×32×3
    ↓
Output (32×32×3)
```

### Dataset Organization

| Stage | Data | Labels | Task |
|-------|------|--------|------|
| Autoencoder Training | 50,000 train images | ❌ Ignored | Unsupervised reconstruction |
| Feature Extraction | 50,000 train + 10,000 test | ✅ Saved | Encode to 8192-dim latent |
| SVM Training | 50,000 train features | ✅ Used | 10-class classification |
| SVM Evaluation | 10,000 test features | ✅ Used | Accuracy measurement |

## Project Structure

```
project/
├── CMakeLists.txt
├── README.md
├── include/
│   └── autoencoder.h          # Autoencoder class declaration
├── src/
│   ├── kernel/                # CUDA kernels
│   │   ├── gpu_conv2d_fp16_forward_v6.cu
│   │   ├── gpu_conv2d_fp16_backward_v8.cu
│   │   └── autoencoder_ops.cu
│   ├── layer/
│   │   └── autoencoder.cu     # Autoencoder implementation
│   ├── train/
│   │   ├── train_AE.cu        # Autoencoder training
│   │   └── train_svm.py       # SVM training (Python)
│   └── inference/
│       └── extract_features.cu # Feature extraction
├── data/                      # CIFAR-10 dataset (after download)
├── weights/                   # Saved model weights
└── features/                  # Extracted features
```

## Requirements

### Hardware
- NVIDIA GPU with Compute Capability 7.5+ (e.g., T4, RTX 2080, V100)
- 4GB+ GPU memory

### Software
- CUDA Toolkit 11.0+
- CMake 3.18+
- Python 3.7+
- NumPy
- scikit-learn (CPU) or cuML (GPU) for SVM

## Quick Start

### 1. Build

```bash
# Clone/extract project
cd project

# Build
mkdir build && cd build
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75  # Use 75 for T4, 86 for A100
make -j4
```

### 2. Download CIFAR-10

```bash
make download_data
```

Or manually:
```bash
mkdir -p data && cd data
curl -L -O https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* .
cd ..
```

### 3. Run Full Pipeline

```bash
./bin/run_all_stages.sh
```

Or with custom settings:
```bash
LR=0.005 EPOCHS_AE=50 EPOCHS_SVM=30 ./bin/run_all_stages.sh
```

## Usage

### Run Individual Stages

#### Stage 1: Train Autoencoder
```bash
./bin/train_autoencoder \
    --data data \
    --weights_dir weights \
    --lr 0.003 \
    --epochs 30 \
    --batch 64
```

Options:
- `--data` - Path to CIFAR-10 data directory
- `--weights_dir` - Directory to save weights
- `--lr` - Learning rate (default: 0.003)
- `--epochs` - Number of epochs (default: 30)
- `--batch` - Batch size (default: 64)
- `--load` - Load pre-trained weights to continue training

#### Stage 2: Extract Features
```bash
./bin/extract_features \
    --data data \
    --weights weights/autoencoder_best.bin \
    --output features/cifar10
```

Options:
- `--data` - Path to CIFAR-10 data directory
- `--weights` - Path to trained autoencoder weights
- `--output` - Output prefix for feature files
- `--batch` - Batch size (default: 64)

#### Stage 3: Train SVM
```bash
python3 bin/train_svm.py \
    --features features/cifar10 \
    --epochs 20 \
    --kernel linear \
    --save_model weights/svm_model.pkl
```

Options:
- `--features` - Prefix for feature files
- `--epochs` - Number of C value iterations (default: 20)
- `--kernel` - SVM kernel: `linear` or `rbf` (default: linear)
- `--C` - Base regularization parameter (default: 1.0)
- `--save_model` - Path to save best model

## Configuration

### Environment Variables for `run_all_stages.sh`

| Variable | Default | Description |
|----------|---------|-------------|
| `LR` | 0.003 | Learning rate for autoencoder |
| `EPOCHS_AE` | 30 | Autoencoder training epochs |
| `EPOCHS_SVM` | 20 | SVM C-value iterations |
| `DATA_DIR` | build/data | CIFAR-10 data location |
| `WEIGHTS_DIR` | build/weights | Model weights directory |
| `FEATURES_DIR` | build/features | Extracted features directory |

### CUDA Architecture

Set according to your GPU:
```bash
cmake .. -DCMAKE_CUDA_ARCHITECTURES=75   # T4
cmake .. -DCMAKE_CUDA_ARCHITECTURES=80   # A100
cmake .. -DCMAKE_CUDA_ARCHITECTURES=86   # RTX 3090
cmake .. -DCMAKE_CUDA_ARCHITECTURES=89   # RTX 4090
```

## Output Files

| File | Location | Description |
|------|----------|-------------|
| `autoencoder_best.bin` | weights/ | Best autoencoder weights (lowest loss) |
| `autoencoder_final.bin` | weights/ | Final autoencoder weights |
| `svm_model.pkl` | weights/ | Trained SVM model (pickle) |
| `cifar10_train_features.bin` | features/ | Training features (50K × 8192) |
| `cifar10_train_labels.bin` | features/ | Training labels (50K) |
| `cifar10_test_features.bin` | features/ | Test features (10K × 8192) |
| `cifar10_test_labels.bin` | features/ | Test labels (10K) |

## Training Details

### Autoencoder
- **Loss**: Mean Squared Error (MSE)
- **Optimizer**: SGD with learning rate decay
- **LR Schedule**: Decay by 0.5× after 3 epochs without improvement
- **Precision**: FP16 activations, FP32 master weights

### SVM
- **Kernel**: Linear (faster) or RBF
- **C-values**: Logarithmic sweep from 0.001 to 10
- **Normalization**: Zero mean, unit variance (per feature)
- **Backend**: cuML (GPU) or scikit-learn (CPU)

## Example Output

```
=== CIFAR-10 Autoencoder Training ===

Hyperparameters:
  Batch size: 64
  Learning rate: 0.003000
  Epochs: 30
  Data dir: data
  Weights dir: weights

Loaded 50000 samples from training set
Creating model...
Model created.

Training: 50000 samples, 781 batches per epoch

  Epoch 1 [781/781] loss: 0.143452
Epoch 1 complete: avg_loss=0.202030, lr=0.003000, time=52s [NEW BEST - saved]

  Epoch 2 [781/781] loss: 0.079552
Epoch 2 complete: avg_loss=0.106764, lr=0.003000, time=53s [NEW BEST - saved]

...

Epoch 30 complete: avg_loss=0.047123, lr=0.000750, time=53s [NEW BEST - saved]

=== Training Complete ===
```

```
=== SVM Training on Autoencoder Features ===

Loading features...
  Train: 50000 samples, 8192 features
  Test:  10000 samples, 8192 features

Normalizing features...
  Before - Train mean: 0.0921, std: 254.8619
  After  - Train mean: 0.000000, std: 1.000000

Using sklearn LinearSVC (CPU)
============================================================
Epoch  1/20: C= 0.00100 | Train Acc: 35.20% | Test Acc: 34.50% | Time: 1.2s
Epoch  2/20: C= 0.00215 | Train Acc: 42.30% | Test Acc: 41.80% | Time: 1.3s [BEST]
...
Epoch 20/20: C=10.00000 | Train Acc: 52.40% | Test Acc: 48.90% | Time: 2.1s
============================================================
Final Best Test Accuracy: 48.90%
```

## Troubleshooting
All kind of craps that we has faced so far
### CUDA Out of Memory
- Reduce batch size: `--batch 32`
- Use smaller GPU architecture

### Training Explodes (NaN loss)
- Reduce learning rate: `LR=0.001`
- The FP16 precision has limited range (~65504 max)

### cuML Not Available
- Install: `! pip install cuml-cu11` (or `cuml-cu12`)
- Or use CPU fallback (slower but works)

### Download Failed
```bash
# Manual download
cd data
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
mv cifar-10-batches-bin/* .
```

## Performance

Tested on NVIDIA T4 GPU:

| Stage                   | Time    | Throughput   |
|-------------------------|---------|--------------|
| Autoencoder (20 epochs) | ~17 min | ~950 img/s   |
| Feature Extraction      | ~7 sec  | ~ 8100 img/s |
| SVM Training (20 iters) | ~25 min | -            |


## Acknowledgments

- CIFAR-10 dataset: [Alex Krizhevsky](https://www.cs.toronto.edu/~kriz/cifar.html)
- CUDA programming: NVIDIA Developer Documentation
