# Quick Test Guide - CIFAR-10 Autoencoder

## For Fast Testing (Recommended for CPU)

### Ultra Quick Test (1-2 minutes)
Test với chỉ 500 samples, 2 epochs:
```bash
make train DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=500 EPOCHS=2
```

### Quick Test (5-10 minutes)
Test với 1000 samples, 3 epochs:
```bash
make train DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=1000 EPOCHS=3
```

### Medium Test (30-60 minutes)
Test với 5000 samples, 5 epochs:
```bash
make train DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=5000 EPOCHS=5
```

### Full Training (2-3 hours)
Train với tất cả 50,000 samples:
```bash
make train DATA_PATH=./cifar-10-batches-bin EPOCHS=20
```

## Direct Command Examples

### Quick smoke test
```bash
./bin/train_autoencoder ./cifar-10-batches-bin --epochs 2 --num-samples 500
```

### Test with custom batch size
```bash
./bin/train_autoencoder ./cifar-10-batches-bin --epochs 2 --num-samples 1000 --batch-size 64
```

### Full training with custom params
```bash
./bin/train_autoencoder ./cifar-10-batches-bin --epochs 20 --batch-size 32 --lr 0.001
```

## What the `--num-samples` Parameter Does

- Limits training to first N images (instead of all 50,000)
- Automatically limits test set to N/5 images
- Perfect for quick iteration and debugging
- Much faster on CPU!

### Example Output:
```
⚠️  Using only 1000 training samples (out of 50000) for quick testing
⚠️  Using only 200 test samples (out of 10000) for quick testing
```

## Expected Times (Apple M1/M2)

| Samples | Epochs | Time (CPU) |
|---------|--------|------------|
| 500     | 2      | ~2 min     |
| 1000    | 3      | ~8 min     |
| 5000    | 5      | ~45 min    |
| 50000   | 20     | ~3 hours   |

## Recommended Workflow

1. **First run** - Ultra quick test to verify everything works:
   ```bash
   make train DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=500 EPOCHS=2
   ```

2. **Debug/tune** - Quick test with more samples:
   ```bash
   make train DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=1000 EPOCHS=3
   ```

3. **Verify quality** - Medium test to check convergence:
   ```bash
   make train DATA_PATH=./cifar-10-batches-bin NUM_SAMPLES=5000 EPOCHS=5
   ```

4. **Full training** - When ready for final results:
   ```bash
   make train DATA_PATH=./cifar-10-batches-bin EPOCHS=20
   ```

## Tips

- Use smaller batch sizes for memory: `BATCH_SIZE=16`
- Fewer epochs for quick tests: `EPOCHS=2`
- Combine flags: `NUM_SAMPLES=1000 EPOCHS=2 BATCH_SIZE=64`
- Watch for "New best loss!" messages - model is learning!
- Check output files: `autoencoder_best.weights` and `autoencoder_benchmark_cpu.txt`

## When to Use Full 50k Samples

- Final model training
- Benchmarking CPU performance
- Before implementing CUDA (to establish baseline)
- When you have time to wait 😊

## CUDA Implementation (Coming Soon)

Once CUDA kernels are implemented:
- Full 50k training should take ~5-10 minutes on GPU
- 10-30x speedup expected
- Same `NUM_SAMPLES` parameter will work with CUDA too!

---

**TL;DR**: Always start with `NUM_SAMPLES=500 EPOCHS=2` for quick testing! ⚡
