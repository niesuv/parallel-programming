CIFAR-10 Autoencoder Project (scaffold)

Overview

- Project scaffolds a CPU/GPU autoencoder pipeline for CIFAR-10 feature learning.

What's included so far

- C++ project scaffold with CMake
- `CIFAR10Dataset` data loader that parses CIFAR-10 binary files into CHW float arrays
- `test_data_loader` tool to verify loading

Build & Run

1. From project root:

```bash
mkdir -p build && cd build
cmake ..
make -j
```

2. Run test (ensure CIFAR-10 binary files are placed in `data/cifar-10-batches-bin`):

```bash
./test_data_loader ../data/cifar-10-batches-bin
```

Next steps (proposed)

- Implement CPU layer primitives (Conv2D, ReLU, MaxPool, UpSample, MSELoss)
- Implement `Autoencoder` and `train_cpu.cpp`
- Add CUDA project files for GPU implementation
- Add feature extraction + LIBSVM integration

If you'd like, I'll implement the Conv2D / ReLU / MaxPool CPU layers next and wire up the training loop.
