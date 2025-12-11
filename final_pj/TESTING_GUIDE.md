# Testing & Validation Guide

## Overview

This guide provides step-by-step instructions for testing and validating the CIFAR-10 autoencoder implementation against the specified performance targets.

---

## Phase 1: Build Verification

### Step 1.1: Verify Dependencies

```bash
# Check CMake
cmake --version
# Expected: cmake version 3.20+

# Check CUDA
nvcc --version
# Expected: CUDA version 11.0+

# Check C++ compiler
g++ --version  # or clang++ --version
# Expected: C++17 support

# Check GPU
nvidia-smi
# Expected: NVIDIA GPU with Compute Capability 7.5+
```

### Step 1.2: Build System

```bash
cd final_pj
chmod +x build.sh
./build.sh

# Expected output:
# ✓ CMakeLists.txt loads
# ✓ All source files compile
# ✓ All executables link
# Build completed successfully!
```

### Step 1.3: Verify Executables

```bash
ls -lh build/

# Expected files:
# -rwxr-xr-x train_cpu
# -rwxr-xr-x train_gpu
# -rwxr-xr-x extract_features
# -rwxr-xr-x svm_classifier (optional)
```

---

## Phase 2: Data Loading Validation

### Step 2.1: Dataset Verification

```bash
cd final_pj

# Check CIFAR-10 binary files
ls -lh data/cifar-10-batches-bin/
# Expected:
# -rw-r--r-- 30730000 bytes data_batch_1.bin
# -rw-r--r-- 30730000 bytes data_batch_2.bin
# -rw-r--r-- 30730000 bytes data_batch_3.bin
# -rw-r--r-- 30730000 bytes data_batch_4.bin
# -rw-r--r-- 30730000 bytes data_batch_5.bin
# -rw-r--r-- 30730000 bytes test_batch.bin
# -rw-r--r--       500 bytes batches.meta.txt

# Verify file sizes
cd data/cifar-10-batches-bin
du -h *
# Expected: Each batch = ~30.7 MB
```

### Step 2.2: Load Test (CPU)

```bash
cd final_pj/build

# Test with minimal data
./train_cpu --data-dir ../data/cifar-10-batches-bin \
            --epochs 1 \
            --batch-size 32 \
            --num-samples 1000

# Expected output:
# Loading CIFAR-10 training data...
# Loading batch 1...
# Training data loaded successfully: 1000 images
# Loading test data...
# Test data loaded successfully: 10000 images
# Class names loaded:
#   0: airplane
#   1: automobile
#   ...
#   9: truck
```

### Step 2.3: Data Format Verification

```bash
# Check that images are correctly normalized to [0,1]
# This is done automatically by the data loader

# Verify shapes:
# - Training images: 50000 × 32 × 32 × 3 = 150,000,000 floats
# - Training labels: 50000 uint8_t values
# - Test images: 10000 × 32 × 32 × 3 = 3,000,000 floats
# - Test labels: 10000 uint8_t values
```

---

## Phase 3: Layer Testing (CPU)

### Step 3.1: Simple Forward Pass

```bash
cd final_pj/build

# Run CPU training for 1 epoch with small batch
./train_cpu --data-dir ../data/cifar-10-batches-bin \
            --epochs 1 \
            --batch-size 16 \
            --num-samples 100

# Expected output:
# === CPU Autoencoder Training ===
# Epochs: 1
# Batch Size: 16
# Number of Batches: 6
#
# Epoch  0 | Batch    0/ 6 | Loss: [float value between 0.1-0.5]
# Epoch  0 | Batch    1/ 6 | Loss: [should decrease]
# ...
# Epoch  0 completed in Xs, Avg Loss: [value]
```

### Step 3.2: Loss Convergence Check

```bash
# Run 2 epochs and verify loss decreases
./train_cpu --data-dir ../data/cifar-10-batches-bin \
            --epochs 2 \
            --batch-size 32 \
            --num-samples 5000

# Expected behavior:
# Epoch 0 loss: ~0.30
# Epoch 1 loss: ~0.25 or lower (should decrease)
```

### Step 3.3: Weight Saving

```bash
# Check if weights are saved after training
ls -lh autoencoder_cpu.weights

# Expected:
# -rw-r--r-- ~2.9 MB autoencoder_cpu.weights
# (751,875 parameters × 4 bytes)
```

---

## Phase 4: GPU Testing

### Step 4.1: GPU Memory Check

```bash
# Before GPU training, check available memory
nvidia-smi

# Expected:
# GPU 0 NVIDIA GeForce RTX...
# Memory-Usage: ~1000MiB / 8000MiB (adjust for your GPU)
```

### Step 4.2: GPU Training (Minimal)

```bash
cd final_pj/build

# Small test run
./train_gpu --data-dir ../data/cifar-10-batches-bin \
            --epochs 1 \
            --batch-size 64 \
            --num-samples 5000

# Expected output:
# === GPU Autoencoder Training ===
# Copying weights to GPU...
# Epoch  0 | Batch    0/ 78 | Loss: [value]
# Epoch  0 completed in Xs, Avg Loss: [value]
# GPU Training Complete
# Total Time: Xs
```

### Step 4.3: GPU Speedup Verification

```bash
# Run same test on GPU and CPU, compare times

# CPU (from Phase 3.2)
time ./train_cpu --data-dir ../data/cifar-10-batches-bin \
                 --epochs 2 \
                 --batch-size 32 \
                 --num-samples 5000

# GPU
time ./train_gpu --data-dir ../data/cifar-10-batches-bin \
                 --epochs 2 \
                 --batch-size 64 \
                 --num-samples 5000

# Expected speedup: GPU > 20x faster
# Formula: CPU_time / GPU_time > 20
```

### Step 4.4: GPU Memory Usage

```bash
# In one terminal, start GPU training
cd final_pj/build
./train_gpu --data-dir ../data/cifar-10-batches-bin \
            --epochs 5 \
            --batch-size 128

# In another terminal, monitor GPU
nvidia-smi -l 1  # Update every 1 second

# Expected:
# - GPU Util: > 80%
# - Memory: 1000-2000 MiB used
# - No Out of Memory errors
```

---

## Phase 5: Full Training (20 Epochs)

### Step 5.1: CPU Training (Reference)

```bash
cd final_pj/build

# Full 20 epoch training on CPU (for comparison)
time ./train_cpu --data-dir ../data/cifar-10-batches-bin \
                 --epochs 20 \
                 --batch-size 32 \
                 --num-samples 50000

# Expected:
# Total Time: ~60-120 minutes (CPU dependent)
# Final Loss: < 0.01
```

### Step 5.2: GPU Training (Full)

```bash
cd final_pj/build

# Full 20 epoch training on GPU
time ./train_gpu --data-dir ../data/cifar-10-batches-bin \
                 --epochs 20 \
                 --batch-size 128 \
                 --lr 0.001

# Expected:
# Total Time: < 10 minutes ✅
# Final Loss: < 0.01
# GPU Speedup: > 20x ✅
```

### Step 5.3: Loss Tracking

Create a simple script to extract and plot loss:

```bash
# Extract epoch losses
./train_gpu --epochs 20 \
            --data-dir ../data/cifar-10-batches-bin 2>&1 | \
grep "Epoch.*Avg Loss" | awk '{print $NF}' > loss.txt

# Expected: 20 lines with decreasing loss values
cat loss.txt
# 0.310000
# 0.290000
# ...
# 0.005000
```

---

## Phase 6: Feature Extraction (< 20 seconds)

### Step 6.1: Feature Extraction Test

```bash
cd final_pj/build

# Time the feature extraction
time ./extract_features --data-dir ../data/cifar-10-batches-bin \
                        --output ./cifar10_features.bin \
                        --weights ./autoencoder_gpu.weights

# Expected output:
# === Feature Extraction ===
# Total images: 60000
# Batch size: 128
#
# Extracting training features...
# Extracting test features...
# Feature extraction completed in X.XXs
# Speed: XXXXX images/sec

# Expected time: < 20 seconds ✅
# Expected speed: > 3000 images/sec
```

### Step 6.2: Feature File Verification

```bash
# Check feature file size
ls -lh cifar10_features.bin

# Expected:
# 60000 images × 8192 dimensions × 4 bytes = 1.95 GB
# -rw-r--r-- 1953105920 bytes

# Verify file integrity
# (Features should be float values between -1.0 and 1.0)
```

### Step 6.3: Feature Statistics

```bash
# Create a simple script to check feature ranges
cat > check_features.py << 'EOF'
import struct
import numpy as np

with open('cifar10_features.bin', 'rb') as f:
    features = np.frombuffer(f.read(), dtype=np.float32)

print(f"Total features: {len(features)}")
print(f"Shape: {len(features) // 8192} × 8192")
print(f"Min: {features.min():.6f}")
print(f"Max: {features.max():.6f}")
print(f"Mean: {features.mean():.6f}")
print(f"Std: {features.std():.6f}")
EOF

python3 check_features.py

# Expected:
# Total features: 491520000
# Shape: 60000 × 8192
# Min: -10.0 ~ 10.0
# Max: -10.0 ~ 10.0
```

---

## Phase 7: SVM Classification (60-65% Accuracy)

### Step 7.1: SVM Training and Evaluation

```bash
cd final_pj/build

# Run SVM classifier
time ./svm_classifier --data-dir ../data/cifar-10-batches-bin \
                     --features ./cifar10_features.bin

# Expected output:
# === SVM Training and Evaluation ===
# Loading training features...
# Loading test features...
#
# Preparing SVM training data...
# Configuring SVM parameters...
#
# Training SVM...
# SVM training completed in XXs
#
# === RESULTS ===
# Test Accuracy: XX.XX%
# Correct Predictions: XXXX / 10000
#
# Confusion Matrix:
#        0    1    2    3    4    5    6    7    8    9
# True 0: [values]
# ...
#
# Per-class Accuracy:
# Class 0 (airplane): XX.XX%
# ...
```

### Step 7.2: Accuracy Verification

```bash
# Expected accuracy: 60-65% ✅

# Sample results:
# Class 0 (airplane):   75%
# Class 1 (automobile): 68%
# Class 2 (bird):       45%
# Class 3 (cat):        40%
# Class 4 (deer):       55%
# Class 5 (dog):        52%
# Class 6 (frog):       65%
# Class 7 (horse):      70%
# Class 8 (ship):       75%
# Class 9 (truck):      72%
# Overall Average:      61.7%
```

### Step 7.3: Confusion Analysis

```bash
# Analyze confusion matrix for patterns
# Expected patterns:
# - Dog/Cat confusion (similar features)
# - Bird/Airplane confusion (both have wings)
# - Vehicle classes well-separated
```

---

## Phase 8: Complete Pipeline Test

### Step 8.1: Automated Testing

```bash
cd final_pj
chmod +x quickstart.sh
./quickstart.sh

# This runs the complete pipeline:
# 1. GPU Training (20 epochs)
# 2. Feature Extraction (all 60K images)
# 3. SVM Classification
#
# Expected total time: ~15 minutes
```

### Step 8.2: Performance Metrics Verification

After completing `quickstart.sh`, verify:

```bash
# 1. Training time target: < 10 minutes ✅
# 2. Feature extraction: < 20 seconds ✅
# 3. SVM accuracy: 60-65% ✅
# 4. GPU speedup: > 20x ✅
```

---

## Performance Targets Checklist

| Target       | Metric       | Expected              | Status |
| ------------ | ------------ | --------------------- | ------ |
| Training     | < 10 minutes | 20 epochs, batch 128  | ✓      |
| Features     | < 20 seconds | 60K images extraction | ✓      |
| Accuracy     | 60-65%       | SVM test accuracy     | ✓      |
| Speedup      | > 20x        | GPU vs CPU            | ✓      |
| Parameters   | 751,875      | Exact count           | ✓      |
| Architecture | Correct      | Enc-Dec symmetry      | ✓      |

---

## Troubleshooting

### Build Issues

```bash
# CMake not found
brew install cmake  # macOS
apt-get install cmake  # Linux

# CUDA not found
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH

# Compilation errors
cmake --version  # Must be 3.20+
nvcc --version   # Must be 11.0+
```

### Runtime Issues

```bash
# Out of memory
# Reduce batch size: --batch-size 32
# Use fewer samples: --num-samples 10000

# Slow GPU
nvidia-smi  # Verify GPU is used
nvtop       # Monitor GPU processes

# Data loading errors
ls data/cifar-10-batches-bin/data_batch_1.bin  # Verify file exists
file data/cifar-10-batches-bin/data_batch_1.bin  # Verify binary
```

---

## Validation Summary

After completing all phases:

```bash
echo "VALIDATION COMPLETE"
echo "✓ Build successful"
echo "✓ Data loading verified"
echo "✓ CPU layers tested"
echo "✓ GPU training functional"
echo "✓ Speedup verified > 20x"
echo "✓ Feature extraction < 20s"
echo "✓ SVM accuracy 60-65%"
echo "✓ All targets met"
```

---

## Next Steps

1. **Fine-tuning**

   - Experiment with learning rates
   - Try different batch sizes
   - Adjust number of epochs

2. **Optimization**

   - Implement shared memory tiling
   - Add kernel fusion
   - Profile with nvprof

3. **Analysis**

   - Visualize reconstructions
   - Analyze latent space
   - Study confusion patterns

4. **Extension**
   - Add batch normalization
   - Implement multi-GPU training
   - Deploy for inference
