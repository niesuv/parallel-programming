# CIFAR-10 Autoencoder + SVM Classification Pipeline

> **Project:** Phase 2, 3, 4 - Parallel Programming Final Project  
> **Goal:** Train autoencoder on CIFAR-10, extract features, classify with SVM

---

## 📂 Project Structure

```
feat-phase2-3-4/
├── include/                    # Header files
│   ├── layer.h                 # CPU tensor & layer definitions
│   ├── autoencoder.h           # CPU Autoencoder class
│   ├── gpu_layer.h             # GPU tensor & layer definitions
│   ├── gpu_autoencoder.h       # GPU Autoencoder class
│   ├── cuda_utils.h            # CUDA error checking macros
│   ├── dataset.h               # CIFAR-10 dataset loader
│   └── svm_wrapper.h           # SVM classifier (ThunderSVM/LIBSVM)
│
├── src/                        # Source implementations
│   ├── main.cpp                # Entry: CPU training
│   ├── main_gpu.cu             # Entry: GPU training + SVM pipeline
│   ├── autoencoder.cpp         # CPU Autoencoder forward/backward
│   ├── gpu_autoencoder.cu      # GPU Autoencoder forward/backward/encode
│   ├── layers_cpu.cpp          # CPU layer implementations
│   ├── layers_gpu.cu           # GPU layers (naive CUDA)
│   ├── layers_gpu_opt.cu       # GPU layers (optimized CUDA)
│   ├── dataset.cpp             # CIFAR-10 binary file loader
│   ├── svm_wrapper.cpp         # SVM wrapper implementations
│   └── verify_gpu.cu           # GPU correctness verification
│
└── external/thundersvm/        # GPU-accelerated SVM (submodule)
```

---

## 🏗️ Autoencoder Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                      ENCODER                                     │
├─────────────────────────────────────────────────────────────────┤
│  Input:  [N, 3, 32, 32]   ← RGB CIFAR-10 images                 │
│     ↓                                                            │
│  Conv2D(3→256, 3×3, pad=1) + ReLU                               │
│     ↓ [N, 256, 32, 32]                                          │
│  MaxPool2D(2×2)                                                  │
│     ↓ [N, 256, 16, 16]                                          │
│  Conv2D(256→128, 3×3, pad=1) + ReLU                             │
│     ↓ [N, 128, 16, 16]                                          │
│  MaxPool2D(2×2)                                                  │
│     ↓ [N, 128, 8, 8]   ← LATENT SPACE (8192 features)           │
├─────────────────────────────────────────────────────────────────┤
│                      DECODER                                     │
├─────────────────────────────────────────────────────────────────┤
│  Conv2D(128→128, 3×3, pad=1) + ReLU                             │
│     ↓ [N, 128, 8, 8]                                            │
│  UpSample2D(scale=2)                                            │
│     ↓ [N, 128, 16, 16]                                          │
│  Conv2D(128→256, 3×3, pad=1) + ReLU                             │
│     ↓ [N, 256, 16, 16]                                          │
│  UpSample2D(scale=2)                                            │
│     ↓ [N, 256, 32, 32]                                          │
│  Conv2D(256→3, 3×3, pad=1)                                      │
│     ↓                                                            │
│  Output: [N, 3, 32, 32]  ← Reconstructed images                 │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🔄 Pipeline Flows

### Pipeline 1: CPU Training (`cpu_train`)

```
┌──────────────────┐
│ Load CIFAR-10    │
│ (50K train, 10K  │
│  test images)    │
└────────┬─────────┘
         ↓
┌──────────────────┐
│ Init CPU         │
│ Autoencoder      │
│ (random weights) │
└────────┬─────────┘
         ↓
┌──────────────────┐     ┌───────────────────┐
│ For each epoch   │────→│ For each batch    │
└────────┬─────────┘     │  • Forward pass   │
         │               │  • MSE loss       │
         │               │  • Backward pass  │
         │               │  • Update weights │
         │               └───────────────────┘
         ↓
┌──────────────────┐
│ Save weights to  │
│ binary file      │
└──────────────────┘
```

**Build:** `make cpu_train` or CMake target `cpu_train`  
**Run:** `./cpu_train data/ 5 32 0.001`

---

### Pipeline 2: GPU Training (`gpu_train` / `gpu_train_opt`)

```
┌────────────────────────────────────────────────────────────────┐
│                        INITIALIZATION                           │
└────────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────┐     ┌──────────────────┐
│ Check CUDA GPU   │────→│ Load CIFAR-10    │
│ Tesla T4, etc.   │     │ dataset          │
└──────────────────┘     └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Init GPU         │
                         │ Autoencoder      │
                         │ + Load weights?  │
                         └────────┬─────────┘
                                  ↓
┌────────────────────────────────────────────────────────────────┐
│                        TRAINING LOOP                            │
└────────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────┐     ┌───────────────────────────────────────┐
│ For each epoch   │────→│ For each batch:                       │
└────────┬─────────┘     │  1. Allocate pinned host memory       │
         │               │  2. cudaMemcpyAsync → GPU              │
         │               │  3. Forward (Conv+ReLU+Pool+Up)        │
         │               │  4. MSE Loss via parallel reduction    │
         │               │  5. Backward (gradient descent)        │
         │               │  6. cudaStreamSynchronize              │
         │               └───────────────────────────────────────┘
         ↓
┌──────────────────┐
│ Save weights     │
│ (binary format)  │
└──────────────────┘

[If WITH_SVM defined, continue to Pipeline 3]
```

**Build:** `./build.sh` (CMake)  
**Run:** `./build/bin/full_pipeline --epochs 10 --data ./data/cifar-10-batches-bin`

**Optimizations (Phase 3 vs Phase 2):**
- Memory coalescing in Conv2D
- Warp shuffle for reduction
- Tiled shared memory
- Loop unrolling for 3×3 kernels
- Pinned host memory for async transfers

---

### Pipeline 3: Feature Extraction + SVM Classification

```
┌────────────────────────────────────────────────────────────────┐
│                   FEATURE EXTRACTION                            │
└────────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────┐
│ Load trained     │
│ autoencoder      │
│ weights          │
└────────┬─────────┘
         ↓
┌──────────────────┐     ┌───────────────────────────────────────┐
│ For each image   │────→│ encode() function:                    │
│ (train + test)   │     │  1. Copy image → GPU                  │
│                  │     │  2. Conv1 + ReLU → Pool1              │
│                  │     │  3. Conv2 + ReLU → Pool2              │
│                  │     │  4. Output: 128×8×8 = 8192 features   │
│                  │     │  5. cudaStreamSynchronize             │
│                  │     │  6. Copy features → CPU               │
└────────┬─────────┘     └───────────────────────────────────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│                   FEATURE NORMALIZATION                         │
└────────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────┐
│ Standardize:     │
│ mean = 0         │
│ std = 1          │
└────────┬─────────┘
         ↓
┌────────────────────────────────────────────────────────────────┐
│                   SVM CLASSIFICATION                            │
└────────────────────────────────────────────────────────────────┘
         ↓
┌──────────────────┐     ┌──────────────────┐
│ Train SVM on     │────→│ Predict test     │
│ 50K train        │     │ features         │
│ features+labels  │     │                  │
└──────────────────┘     └────────┬─────────┘
                                  ↓
                         ┌──────────────────┐
                         │ Compute          │
                         │ Accuracy         │
                         │ (expected >40%)  │
                         └──────────────────┘
```

**Run (SVM-only mode):**
```bash
./build/bin/full_pipeline --load-weights autoencoder_gpu.weights --svm-only --data ./data/cifar-10-batches-bin
```

---

## ⚙️ Build Commands

### CMake (Recommended)
```bash
./build.sh               # Build all targets
./build.sh --clean       # Clean build
```

### Executables
| Binary | Description |
|--------|-------------|
| `gpu_train` | Train autoencoder only (no SVM) |
| `full_pipeline` | Train autoencoder + SVM classification |
| `thundersvm-train` | Standalone ThunderSVM training tool |

---

## 🔧 Key Configuration Flags

| Flag | Description |
|------|-------------|
| `USE_OPTIMIZED_KERNELS` | Enable Phase 3 CUDA optimizations |
| `WITH_SVM` | Enable SVM classification pipeline |
| `WITH_THUNDERSVM` | Use GPU-accelerated ThunderSVM |
| `WITH_LIBSVM` | Fallback to CPU LIBSVM |

---

## 📊 Layer Implementations Summary

| Layer | CPU File | GPU Naive | GPU Optimized |
|-------|----------|-----------|---------------|
| Conv2D | `layers_cpu.cpp` | `layers_gpu.cu` | `layers_gpu_opt.cu` (tiled + coalesced) |
| ReLU | `layers_cpu.cpp` | `layers_gpu.cu` | `layers_gpu_opt.cu` (fused with Conv) |
| MaxPool2D | `layers_cpu.cpp` | `layers_gpu.cu` | `layers_gpu_opt.cu` (2D blocks) |
| UpSample2D | `layers_cpu.cpp` | `layers_gpu.cu` | `layers_gpu_opt.cu` |
| MSE Loss | `layers_cpu.cpp` | `layers_gpu.cu` | `layers_gpu_opt.cu` (warp shuffle) |

---

## 📝 Weight File Format

Binary format with magic number verification:
```
[4 bytes] Magic: 0x48414557 ("WEAH")
[4 bytes] Version: 1
[4 bytes] Num layers: 5

For each conv layer:
  [4 bytes] in_channels
  [4 bytes] out_channels
  [4 bytes] kernel_size
  [4 bytes] weights_count
  [N floats] weights data
  [4 bytes] bias_count
  [M floats] bias data
```
