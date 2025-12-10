# CIFAR-10 Autoencoder Training Guide

## Quick Start

### 1. Build the Project
```bash
make clean
make
```

### 2. Download CIFAR-10 Data
```bash
wget https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
tar -xzf cifar-10-binary.tar.gz
# This creates: cifar-10-batches-bin/
```

### 3. Train Autoencoder (CPU Baseline)

**Default training (20 epochs, batch=32, lr=0.001):**
```bash
make train DATA_PATH=./cifar-10-batches-bin
```

**Custom hyperparameters:**
```bash
# Quick test: 5 epochs
make train DATA_PATH=./cifar-10-batches-bin EPOCHS=5

# Large batch size
make train DATA_PATH=./cifar-10-batches-bin BATCH_SIZE=64

# Lower learning rate
make train DATA_PATH=./cifar-10-batches-bin LR=0.0005

# Combined
make train DATA_PATH=./cifar-10-batches-bin EPOCHS=10 BATCH_SIZE=64 LR=0.0005
```

## Architecture

```
INPUT: (32, 32, 3)
  ↓
ENCODER:
  Conv2D(3→256, 3×3) + ReLU → (32, 32, 256)
  MaxPool2D(2×2) → (16, 16, 256)
  Conv2D(256→128, 3×3) + ReLU → (16, 16, 128)
  MaxPool2D(2×2) → (8, 8, 128)  [LATENT: 8,192 features]
  ↓
DECODER:
  Conv2D(128→128, 3×3) + ReLU → (8, 8, 128)
  UpSample2D(2×) → (16, 16, 128)
  Conv2D(128→256, 3×3) + ReLU → (16, 16, 256)
  UpSample2D(2×) → (32, 32, 256)
  Conv2D(256→3, 3×3) → (32, 32, 3)
  ↓
OUTPUT: (32, 32, 3)

Total Parameters: 751,875
```

## Training Output

The training process logs:

1. **STEP 1**: Data Loading (50k train + 10k test)
2. **STEP 2**: Model Creation & Initialization
3. **STEP 3**: Training Loop
   - Epoch progress with loss
   - Time per epoch
   - Throughput (imgs/s)
   - Auto-saves best model
4. **STEP 4**: Test Reconstruction
5. **STEP 5**: Latent Feature Extraction

## Output Files

After training:
- `autoencoder_best.weights` - Trained model weights
- `autoencoder_benchmark_cpu.txt` - Performance metrics

## Expected Performance (CPU)

On a typical modern CPU:
- Training time: ~5-10 minutes/epoch (batch_size=32)
- Throughput: ~100-200 imgs/s
- Reconstruction loss: Should decrease below 0.1 after 20 epochs

## Next Steps

After CPU baseline:
1. Implement CUDA kernels for layers
2. Compare CPU vs GPU performance
3. Use trained encoder for classification task

## Troubleshooting

**Out of memory:**
```bash
make train DATA_PATH=./cifar-10-batches-bin BATCH_SIZE=16
```

**Too slow:**
- Reduce epochs for testing: `EPOCHS=5`
- GPU implementation coming soon!

**Check help:**
```bash
make help
./bin/train_autoencoder
```
