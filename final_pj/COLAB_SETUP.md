# Google Colab Setup Guide

## Setup Instructions

### 1. Upload Code to Colab

```python
# In Colab notebook
from google.colab import drive
drive.mount('/content/drive')

# Or upload directly
!git clone <your-repo-url>
# Or upload zip file
```

### 2. Install Build Tools

```bash
!apt-get update -qq
!apt-get install -y build-essential wget
```

### 3. Download CIFAR-10 Data

```bash
!wget -q https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
!ls cifar-10-batches-bin/
```

### 4. Build Project

```bash
# Navigate to project directory
%cd /content/final_pj

# Build (permissions are automatically set by Makefile)
!make clean
!make
```

**Note:** The Makefile now automatically sets execute permissions on all binaries, so `chmod` is no longer needed!

### 5. Quick Test

```bash
# Test with 500 samples
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --epochs 2 \
    --num-samples 500 \
    --batch-size 32
```

### 6. Full Training (if you have time)

```bash
# This will take ~2-3 hours on Colab CPU
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --epochs 20 \
    --batch-size 32
```

## Complete Colab Notebook Example

```python
# Cell 1: Setup
!apt-get update -qq
!apt-get install -y build-essential wget

# Cell 2: Download CIFAR-10
!wget -q https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
!ls cifar-10-batches-bin/

# Cell 3: Upload your code
# Option A: From GitHub
!git clone https://github.com/your-username/your-repo.git
%cd your-repo

# Option B: Upload zip
from google.colab import files
uploaded = files.upload()  # Upload your final_pj.zip
!unzip final_pj.zip
%cd final_pj

# Cell 4: Build
!make clean
!make

# Cell 5: Verify build
!ls -lh bin/
!./bin/train_autoencoder

# Cell 6: Quick test (2 minutes)
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --epochs 2 \
    --num-samples 500

# Cell 7: Check results
!ls -lh *.weights
!cat autoencoder_benchmark_cpu.txt

# Cell 8: Download results
from google.colab import files
files.download('autoencoder_best.weights')
files.download('autoencoder_benchmark_cpu.txt')
```

## Troubleshooting

### Permission Denied Error
**This should no longer happen!** The Makefile automatically sets execute permissions.

If you still see this error:
```bash
# Rebuild
!make clean && make
```

### "Command not found"
```bash
# Make sure you're in the right directory
!pwd
%cd /content/final_pj

# Check if binary exists
!ls -la bin/
```

### Build errors
```bash
# Install dependencies
!apt-get install -y build-essential

# Check compiler
!gcc --version
```

### Out of memory
```bash
# Reduce batch size
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --num-samples 500 \
    --batch-size 16
```

## Performance on Colab CPU

- **Quick test (500 samples, 2 epochs):** ~2 minutes
- **Medium test (1000 samples, 3 epochs):** ~8 minutes  
- **Full training (50k samples, 20 epochs):** ~2-3 hours

## Tips

1. **Always start with quick test:**
   ```bash
   !./bin/train_autoencoder ./cifar-10-batches-bin --num-samples 500 --epochs 2
   ```

2. **Save results to Drive:**
   ```python
   !cp autoencoder_best.weights /content/drive/MyDrive/
   ```

3. **Monitor progress:**
   - Watch for "New best loss!" messages
   - Loss should decrease each epoch

4. **Download results:**
   ```python
   from google.colab import files
   files.download('autoencoder_best.weights')
   ```

## Using GPU (When CUDA Implemented)

```python
# Check GPU
!nvidia-smi

# Build with CUDA (when implemented)
!make clean
!make  # Will auto-detect CUDA

# Run with GPU
!./bin/train_autoencoder_cuda ./cifar-10-batches-bin \
    --epochs 20 \
    --batch-size 128
```

Expected GPU speedup: **50-100x faster!**
- Full 50k training: **10-20 minutes** (vs 2-3 hours)

---

**TL;DR Quick Commands:**
```bash
!make clean && make
!./bin/train_autoencoder ./cifar-10-batches-bin --num-samples 500 --epochs 2
```
