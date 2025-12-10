# Hướng Dẫn Chạy Trên Google Colab

## Chuẩn Bị

### 1. Tải CIFAR-10
```bash
!wget -q https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
```

### 2. Upload Code
```python
# Cách 1: Upload file zip
from google.colab import files
uploaded = files.upload()  # Chọn file final_pj.zip
!unzip -q final_pj.zip
%cd final_pj

# Cách 2: Clone từ GitHub
!git clone https://github.com/your-username/your-repo.git
%cd your-repo
```

### 3. Build
```bash
!make clean && make
```

### 4. Test Nhanh (2 phút)
```bash
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --num-samples 500 \
    --epochs 2
```

## Các Lệnh Thường Dùng

### Test với nhiều mẫu hơn
```bash
# 1000 mẫu, 3 epochs (~8 phút)
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --num-samples 1000 \
    --epochs 3

# 5000 mẫu, 5 epochs (~40 phút)
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --num-samples 5000 \
    --epochs 5
```

### Training đầy đủ
```bash
# 50,000 mẫu, 20 epochs (~2-3 giờ)
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --epochs 20
```

### Xem kết quả
```bash
# Xem các file weights đã lưu
!ls -lh *.weights

# Xem benchmark
!cat autoencoder_benchmark_cpu.txt
```

### Tải kết quả về
```python
from google.colab import files

# Tải weights tốt nhất
files.download('autoencoder_best.weights')

# Tải benchmark
files.download('autoencoder_benchmark_cpu.txt')
```

## Notebook Mẫu Hoàn Chỉnh

```python
# ===== CELL 1: Cài đặt công cụ =====
!apt-get update -qq
!apt-get install -y build-essential wget

# ===== CELL 2: Tải CIFAR-10 =====
!wget -q https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz
!tar -xzf cifar-10-binary.tar.gz
!ls cifar-10-batches-bin/

# ===== CELL 3: Upload code =====
from google.colab import files
uploaded = files.upload()
!unzip -q final_pj.zip
%cd final_pj

# ===== CELL 4: Build =====
!make clean && make

# ===== CELL 5: Kiểm tra build =====
!ls -lh bin/

# ===== CELL 6: Test nhanh =====
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --num-samples 500 \
    --epochs 2

# ===== CELL 7: Xem kết quả =====
!ls -lh *.weights
!cat autoencoder_benchmark_cpu.txt

# ===== CELL 8: Tải về =====
from google.colab import files
files.download('autoencoder_best.weights')
files.download('autoencoder_benchmark_cpu.txt')
```

## Giải Quyết Lỗi

### Lỗi "Permission denied"
Makefile đã tự động set quyền, không cần làm gì thêm. Nếu vẫn lỗi:
```bash
!make clean && make
```

### Lỗi "Command not found"
```bash
# Kiểm tra đường dẫn
!pwd
!ls -la bin/

# Chuyển đến thư mục đúng
%cd /content/final_pj
```

### Lỗi build
```bash
# Cài đặt lại compiler
!apt-get install -y build-essential

# Kiểm tra phiên bản
!gcc --version
```

### Hết RAM
```bash
# Giảm batch size
!./bin/train_autoencoder ./cifar-10-batches-bin \
    --num-samples 500 \
    --batch-size 16
```

## Thời Gian Chạy Dự Kiến

| Số mẫu | Epochs | Thời gian |
|--------|--------|-----------|
| 500    | 2      | ~2 phút   |
| 1000   | 3      | ~8 phút   |
| 5000   | 5      | ~40 phút  |
| 50000  | 20     | ~2-3 giờ  |

## Lưu Ý

1. **Luôn bắt đầu với test nhanh** để đảm bảo code chạy đúng
2. **Lưu kết quả vào Drive** nếu training lâu:
   ```python
   from google.colab import drive
   drive.mount('/content/drive')
   !cp autoencoder_best.weights /content/drive/MyDrive/
   ```
3. **Theo dõi loss** - phải giảm dần mỗi epoch
4. **Tải kết quả về** trước khi đóng notebook

## Lệnh Nhanh

```bash
# Build và test trong 1 dòng
!make clean && make && ./bin/train_autoencoder ./cifar-10-batches-bin --num-samples 500 --epochs 2
```

---

**Chúc may mắn!** 🚀
