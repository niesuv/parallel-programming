# Bài giảng HW3: CUDA Memories & Streams

## 1. Mục tiêu học tập
- Hiểu vì sao bài toán làm mờ ảnh 2D cần tận dụng **Shared Memory** và **Constant Memory** để giảm truy cập Global Memory.
- Phân tích các mô hình pipeline với **CUDA Stream** để chồng chéo truyền dữ liệu và tính toán.
- Thực hành nâng cấp lần lượt từ kernel cơ bản (naive) tới kernel tối ưu, đúng với tinh thần ProgressiveComplexity.

## 2. Bài giảng chuyên đề

### 2.1 Shared Memory Tiling cho Blur 2D
#### AnalogyFirst
Hãy nghĩ mỗi **Block** như một đội thợ sơn đang xử lý một ô tường. Global Memory giống kho sơn dưới tầng trệt, trong khi **Shared Memory** là tủ sơn mini ngay trên sàn thi công. Nếu đội thợ lấy từng thìa sơn từ kho chính cho mỗi nét cọ, họ sẽ tốn thời gian xếp hàng. Thay vào đó, cả đội hợp tác chuyển đủ sơn cho cả ô vào tủ chung, rồi mỗi thợ dùng dần — nhanh hơn rất nhiều.

#### Giải thích kỹ thuật
- Với blur 2D kích thước `FILTER_WIDTH`, mỗi Thread (worker) tính một pixel trong ô `blockDim.x × blockDim.y`.
- Ta cần cả **vùng halo**: thêm `filterRadius` điểm biên quanh ô để đủ dữ liệu cho convolution. Điều này khiến vùng nạp vào Shared Memory có kích thước `(blockDim.x + 2*radius) × (blockDim.y + 2*radius)`.
- Cấu trúc lệnh:
  1. `extern __shared__ uchar3 s_inPixels[];` cấp phát động theo kích thước ở launch.
  2. Mỗi Thread copy một hoặc nhiều phần tử từ Global Memory vào ô tương ứng trong Shared Memory bằng công thức:
     ```cpp
     int sharedPitch = blockDim.x + 2 * radius;
     int sharedRow = threadIdx.y + radius;
     int sharedCol = threadIdx.x + radius;
     s_inPixels[sharedRow * sharedPitch + sharedCol] = inPixels[globalIdx];
     ```
  3. Những Thread nằm ở rìa Block cần copy thêm halo (dùng điều kiện `threadIdx.x < radius`, ...).
  4. `__syncthreads()` để đảm bảo mọi dữ liệu đã sẵn sàng trước khi tích chập.
  5. Khi tính kết quả, mỗi Thread chỉ đọc từ `s_inPixels`, nhờ vậy giảm lượng truy cập Global Memory xuống ~1 lần tải/pixel thay vì `FILTER_WIDTH^2` lần.

#### Liên hệ Grid/Block/Thread
- **Grid** phủ toàn ảnh, **Block** trùng với một tile của ảnh, **Thread** trùng 1 pixel output trong tile.
- Thiết kế block vuông (16×16 hoặc 32×16) giúp coalesced load khi copy vào Shared Memory.

### 2.2 Constant Memory cho Filter
#### AnalogyFirst
Filter giống một bảng hướng dẫn chung treo ở sảnh tòa nhà. Mọi đội thợ cần xem cùng nội dung, vì vậy tốt nhất hãy đóng khung nó ở vị trí ai cũng nhìn thấy thay vì phát bản photo cho từng người.

#### Giải thích kỹ thuật
- `__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];` lưu filter trên cache đọc-chung nhanh.
- Khi tất cả Thread trong một Warp đọc cùng địa chỉ (ví dụ `dc_filter[k]`), phần cứng broadcast giá trị chỉ với 1 giao dịch.
- Thao tác copy: `CHECK(cudaMemcpyToSymbol(dc_filter, filter, filterSize));`
- Trong `blurImgKernel3`, bỏ tham số `float *filter`, thay bằng `float coeff = dc_filter[filterIdx];`.

### 2.3 Đồng bộ và xử lý biên (Halo Cooperation)
#### AnalogyFirst
Tường nhà bịt bằng băng keo xung quanh để không lem. Mỗi thợ chịu trách nhiệm dán một phần băng. Họ phải báo nhau xong xuôi (điểm danh) trước khi bất kỳ ai bắt đầu sơn.

#### Giải thích kỹ thuật
- Các Thread phải clamp tọa độ khi truy cập Global Memory: `int rClamped = min(max(globalRow, 0), height-1);`
- Sau khi copy dữ liệu và halo vào Shared Memory, bắt buộc `__syncthreads();` để tránh đọc giá trị chưa được ghi.
- Khi chỉ còn một Warp hoạt động (stride ≤ 32 trong reduction) ta có thể bỏ đồng bộ, nhưng trong blur 2D mọi Thread vẫn cần dữ liệu hàng xóm nên phải giữ `__syncthreads()` trước pha tích chập.

### 2.4 CUDA Streams & Pipeline
#### AnalogyFirst
Hệ thống kho vận có nhiều băng chuyền độc lập. Mỗi băng chuyền (Stream) gồm các trạm: nhận hàng (H2D), gia công 1, gia công 2, đóng gói (D2H). Làm tuần tự đồng nghĩa bạn dùng duy nhất một băng chuyền và các kiện phải chờ; dùng nhiều băng chuyền cho phép vận hành chồng chéo.

#### Giải thích kỹ thuật
- **Stream** là hàng đợi lệnh riêng biệt. Thao tác trong cùng một Stream giữ thứ tự; thao tác ở Stream khác có thể song song nếu GPU hỗ trợ copy engine kép và concurrent kernels.
- `cudaMemcpyAsync` cho phép truyền dữ liệu gắn với Stream.
- **Breadth-first**: phase 1 copy tất cả vectors, phase 2 chạy kernel 1… Giống xử lý theo từng công đoạn cho toàn bộ kiện -> khó che giấu latency giữa phase.
- **Depth-first**: mỗi vector dùng Stream riêng và thực hiện đủ H2D → scale → add → square → D2H trước khi chuyển sang vector tiếp theo. Điều này tận dụng tính chồng chéo tốt hơn vì copy của vector A có thể trùng thời gian với compute của vector B.

### 2.5 Pinned Memory & Đồng bộ
#### AnalogyFirst
Pinned memory là “khoá chỗ đỗ xe” ngay trước cửa kho. Xe tải (DMA engine) biết chính xác vị trí để ghé nên không cần tìm chỗ, giúp giao nhận nhanh và song song.

#### Giải thích kỹ thuật
- Dùng `cudaMallocHost` để tạo buffer host; nếu không, `cudaMemcpyAsync` sẽ fallback sang đường đồng bộ.
- Sau khi enqueue lệnh trong Stream, đồng bộ bằng `cudaStreamSynchronize(streams[v])` hoặc đồng bộ toàn bộ bằng `cudaDeviceSynchronize()` tùy nhu cầu.
- Khi đo thời gian pipeline, đặt `cudaEventRecord` trước và sau toàn bộ pha để tránh tính nhầm.

## 3. Hướng dẫn giải chi tiết HW3_P1 (Blur với 3 kernel)

### Bước 0 – Hoàn thiện blur trên Host
1. Trong `blurImg(...)`, nhánh `useDevice == false`, duyệt `r` và `c` trên ảnh.
2. Với mỗi pixel, cộng dồn `filterWidth^2` phần tử, áp dụng clamp biên.
3. Gán kết quả vào `outPixels` để làm ground truth cho hàm `printError`.

```cpp
if (useDevice == false)
{
   int radius = filterWidth / 2;
   for (int r = 0; r < height; r++)
   {
      for (int c = 0; c < width; c++)
      {
         float3 sum = {0.0f, 0.0f, 0.0f};
         for (int fr = 0; fr < filterWidth; fr++)
         {
            for (int fc = 0; fc < filterWidth; fc++)
            {
               int inR = min(max(r + fr - radius, 0), height - 1);
               int inC = min(max(c + fc - radius, 0), width - 1);
               uchar3 inPix = inPixels[inR * width + inC];
               float coeff = filter[fr * filterWidth + fc];
               sum.x += inPix.x * coeff;
               sum.y += inPix.y * coeff;
               sum.z += inPix.z * coeff;
            }
         }
         outPixels[r * width + c] = make_uchar3(sum.x, sum.y, sum.z);
      }
   }
   return;
}
```

### Bước 1 – Kernel 1 (GMEM-only, baseline)
1. Trong `blurImgKernel1`, tính `c` và `r` bằng công thức chuẩn của Grid/Block.
2. Nếu ngoài phạm vi, return.
3. Lặp qua `filterR`, `filterC`, tính `imgR = min(max(r + filterR - radius, 0), height-1)`.
4. Đọc `uchar3` từ `inPixels[imgR * width + imgC]`, cộng dồn với hệ số `filter[filterIdx]`.
5. Sau vòng lặp, ghi `make_uchar3((unsigned char)sum.x, ...)` vào Global Memory.

```cpp
__global__ void blurImgKernel1(uchar3 *inPixels, int width, int height,
      float *filter, int filterWidth,
      uchar3 *outPixels)
{
   int c = blockIdx.x * blockDim.x + threadIdx.x;
   int r = blockIdx.y * blockDim.y + threadIdx.y;
   if (c >= width || r >= height) return;

   int radius = filterWidth / 2;
   float3 sum = make_float3(0.f, 0.f, 0.f);
   for (int fr = 0; fr < filterWidth; fr++)
   {
      for (int fc = 0; fc < filterWidth; fc++)
      {
         int inR = min(max(r + fr - radius, 0), height - 1);
         int inC = min(max(c + fc - radius, 0), width - 1);
         uchar3 inPix = inPixels[inR * width + inC];
         float coeff = filter[fr * filterWidth + fc];
         sum.x += inPix.x * coeff;
         sum.y += inPix.y * coeff;
         sum.z += inPix.z * coeff;
      }
   }
   outPixels[r * width + c] = make_uchar3((unsigned char)sum.x,
                                 (unsigned char)sum.y,
                                 (unsigned char)sum.z);
}
```

### Bước 2 – Kernel 2 (Shared Memory, GMEM filter)
1. Khai báo `extern __shared__ uchar3 s_inPixels[];`.
2. Tính kích thước Shared Memory khi launch: `size_t smemBytes = (blockSize.x + 2*radius) * (blockSize.y + 2*radius) * sizeof(uchar3);` rồi truyền làm tham số thứ 3 trong `<<<grid, block, smemBytes>>>`.
3. Mỗi Thread copy pixel “chính” của mình vào `s_inPixels` ở vị trí offset `+radius`.
4. Thread ở viền additionally copy halo theo 4 hướng và 4 góc (có thể chia thành nhiều nhiệm vụ cho đủ tất cả phần tử). Lưu ý clamp tọa độ khi đọc GMEM.
5. `__syncthreads()`.
6. Thực hiện convolution bằng cách đọc từ `s_inPixels` thay vì `inPixels`. Chỉ cần truy cập vùng `blockDim` vì halo đã có.

```cpp
__global__ void blurImgKernel2(uchar3 *inPixels, int width, int height,
      float *filter, int filterWidth,
      uchar3 *outPixels)
{
   extern __shared__ uchar3 tile[];
   int radius = filterWidth / 2;
   int tileW = blockDim.x + 2 * radius;
   int tileH = blockDim.y + 2 * radius;
   int sharedSize = tileW * tileH;

   int tileOriginR = blockIdx.y * blockDim.y - radius;
   int tileOriginC = blockIdx.x * blockDim.x - radius;

   int linearThread = threadIdx.y * blockDim.x + threadIdx.x;
   int threadsPerBlock = blockDim.x * blockDim.y;
   for (int idx = linearThread; idx < sharedSize; idx += threadsPerBlock)
   {
      int localR = idx / tileW;
      int localC = idx % tileW;
      int globalR = min(max(tileOriginR + localR, 0), height - 1);
      int globalC = min(max(tileOriginC + localC, 0), width - 1);
      tile[idx] = inPixels[globalR * width + globalC];
   }

   __syncthreads();

   int globalR = blockIdx.y * blockDim.y + threadIdx.y;
   int globalC = blockIdx.x * blockDim.x + threadIdx.x;
   if (globalR >= height || globalC >= width) return;

   float3 sum = make_float3(0.f, 0.f, 0.f);
   int sharedRow = threadIdx.y + radius;
   int sharedCol = threadIdx.x + radius;
   for (int fr = 0; fr < filterWidth; fr++)
   {
      for (int fc = 0; fc < filterWidth; fc++)
      {
         int tileRow = sharedRow + fr - radius;
         int tileCol = sharedCol + fc - radius;
         uchar3 inPix = tile[tileRow * tileW + tileCol];
         float coeff = filter[fr * filterWidth + fc];
         sum.x += inPix.x * coeff;
         sum.y += inPix.y * coeff;
         sum.z += inPix.z * coeff;
      }
   }
   outPixels[globalR * width + globalC] = make_uchar3((unsigned char)sum.x,
                                          (unsigned char)sum.y,
                                          (unsigned char)sum.z);
}
```

### Bước 3 – Kernel 3 (Shared Memory + Constant Memory)
1. Sao chép filter lên constant: `CHECK(cudaMemcpyToSymbol(dc_filter, filter, filterSize));` ở nhánh `else` trong `blurImg(...)`.
2. Kernel giống kernel 2 nhưng bỏ tham số `filter` và thay `filterVal = dc_filter[filterIdx];`.
3. Khi launch kernel 3, vẫn truyền `smemBytes` vì Shared Memory dùng cho ảnh.

```cpp
__global__ void blurImgKernel3(uchar3 *inPixels, int width, int height,
      int filterWidth,
      uchar3 *outPixels)
{
   extern __shared__ uchar3 tile[];
   int radius = filterWidth / 2;
   int tileW = blockDim.x + 2 * radius;
   int tileH = blockDim.y + 2 * radius;
   int sharedSize = tileW * tileH;

   int tileOriginR = blockIdx.y * blockDim.y - radius;
   int tileOriginC = blockIdx.x * blockDim.x - radius;

   int linearThread = threadIdx.y * blockDim.x + threadIdx.x;
   int threadsPerBlock = blockDim.x * blockDim.y;
   for (int idx = linearThread; idx < sharedSize; idx += threadsPerBlock)
   {
      int localR = idx / tileW;
      int localC = idx % tileW;
      int globalR = min(max(tileOriginR + localR, 0), height - 1);
      int globalC = min(max(tileOriginC + localC, 0), width - 1);
      tile[idx] = inPixels[globalR * width + globalC];
   }

   __syncthreads();

   int globalR = blockIdx.y * blockDim.y + threadIdx.y;
   int globalC = blockIdx.x * blockDim.x + threadIdx.x;
   if (globalR >= height || globalC >= width) return;

   float3 sum = make_float3(0.f, 0.f, 0.f);
   int sharedRow = threadIdx.y + radius;
   int sharedCol = threadIdx.x + radius;
   for (int fr = 0; fr < filterWidth; fr++)
   {
      for (int fc = 0; fc < filterWidth; fc++)
      {
         int tileRow = sharedRow + fr - radius;
         int tileCol = sharedCol + fc - radius;
         uchar3 inPix = tile[tileRow * tileW + tileCol];
         float coeff = dc_filter[fr * filterWidth + fc];
         sum.x += inPix.x * coeff;
         sum.y += inPix.y * coeff;
         sum.z += inPix.z * coeff;
      }
   }
   outPixels[globalR * width + globalC] = make_uchar3((unsigned char)sum.x,
                                          (unsigned char)sum.y,
                                          (unsigned char)sum.z);
}
```

### Bước 4 – Gọi kernel và thu kết quả
1. Tính `gridSize` bằng `(width + blockDim.x - 1)/blockDim.x` cho mỗi chiều.
2. `blurImg(..., true, blockSize, 1)` gọi kernel 1 (không cần `smemBytes`).
3. Với kernel 2 và 3, bổ sung đối số thứ ba: `blurImgKernel2<<<gridSize, blockSize, smemBytes>>>(...)`.
4. Sau mỗi lần chạy, `cudaDeviceSynchronize()` và `cudaGetLastError()` đã có sẵn; chỉ cần đảm bảo `GpuTimer` bao quanh.
5. Kết thúc bằng cách ghi các file `_device1/2/3.pnm` và quan sát sai số nhỏ.

```cpp
int radius = filterWidth / 2;
dim3 gridSize((width + blockSize.x - 1) / blockSize.x,
        (height + blockSize.y - 1) / blockSize.y);

if (kernelType == 1)
{
   blurImgKernel1<<<gridSize, blockSize>>>(d_inPixels, width, height,
      d_filter, filterWidth, d_outPixels);
}
else if (kernelType == 2)
{
   size_t smemBytes = (blockSize.x + 2 * radius) *
         (blockSize.y + 2 * radius) * sizeof(uchar3);
   blurImgKernel2<<<gridSize, blockSize, smemBytes>>>(d_inPixels, width, height,
      d_filter, filterWidth, d_outPixels);
}
else
{
   CHECK(cudaMemcpyToSymbol(dc_filter, filter, filterWidth * filterWidth * sizeof(float)));
   size_t smemBytes = (blockSize.x + 2 * radius) *
         (blockSize.y + 2 * radius) * sizeof(uchar3);
   blurImgKernel3<<<gridSize, blockSize, smemBytes>>>(d_inPixels, width, height,
      filterWidth, d_outPixels);
}
CHECK(cudaDeviceSynchronize());
CHECK(cudaGetLastError());
```

## 4. Hướng dẫn giải chi tiết HW3_P2 (Streams)

### Bước 0 – Chuẩn bị bộ nhớ host
1. Dùng `cudaMallocHost` để cấp phát `h_input`, `h_output_seq`, `h_output_bf`, `h_output_df`.
2. Mỗi `h_input[v]` có `vectorSize` phần tử float (~16 MB mặc định). Cũng cấp phát chỗ tương ứng cho các output.

```cpp
float **h_input, **h_output_seq, **h_output_bf, **h_output_df;
CHECK(cudaMallocHost(&h_input, vectorCount * sizeof(float *)));
CHECK(cudaMallocHost(&h_output_seq, vectorCount * sizeof(float *)));
CHECK(cudaMallocHost(&h_output_bf, vectorCount * sizeof(float *)));
CHECK(cudaMallocHost(&h_output_df, vectorCount * sizeof(float *)));

size_t bytes = vectorSize * sizeof(float);
for (int v = 0; v < vectorCount; v++)
{
   CHECK(cudaMallocHost(&h_input[v], bytes));
   CHECK(cudaMallocHost(&h_output_seq[v], bytes));
   CHECK(cudaMallocHost(&h_output_bf[v], bytes));
   CHECK(cudaMallocHost(&h_output_df[v], bytes));
   for (int i = 0; i < vectorSize; i++)
   {
      h_input[v][i] = (float)(i % 100) / 10.0f;
   }
}
```

### Bước 1 – `processSequential`
1. `cudaMalloc(&d_data, vectorSize * sizeof(float));`
2. Với từng vector `v`:
   - `CHECK(cudaMemcpy(d_data, h_input[v], bytes, cudaMemcpyHostToDevice));`
   - Cài đặt `grid = (vectorSize + blockDim - 1) / blockDim`.
   - Gọi `scaleKernel<<<grid, block>>>(d_data, vectorSize, 2.0f);` rồi `addKernel`, `squareKernel`.
   - `CHECK(cudaDeviceSynchronize());` để đảm bảo đã hoàn tất trước khi copy về.
   - `CHECK(cudaMemcpy(h_output_seq[v], d_data, bytes, cudaMemcpyDeviceToHost));`
3. Sau vòng lặp, giải phóng `d_data` và events.

```cpp
float processSequential(float **h_input, float **h_output,
      int vectorCount, int vectorSize)
{
   cudaEvent_t start, stop;
   CHECK(cudaEventCreate(&start));
   CHECK(cudaEventCreate(&stop));

   float *d_data;
   size_t bytes = vectorSize * sizeof(float);
   CHECK(cudaMalloc(&d_data, bytes));

   CHECK(cudaEventRecord(start));
   dim3 block(256);
   dim3 grid((vectorSize + block.x - 1) / block.x);

   for (int v = 0; v < vectorCount; v++)
   {
      CHECK(cudaMemcpy(d_data, h_input[v], bytes, cudaMemcpyHostToDevice));
      scaleKernel<<<grid, block>>>(d_data, vectorSize, 2.0f);
      addKernel<<<grid, block>>>(d_data, vectorSize, 10.0f);
      squareKernel<<<grid, block>>>(d_data, vectorSize);
      CHECK(cudaDeviceSynchronize());
      CHECK(cudaMemcpy(h_output[v], d_data, bytes, cudaMemcpyDeviceToHost));
   }

   CHECK(cudaEventRecord(stop));
   CHECK(cudaEventSynchronize(stop));
   float ms = 0.f;
   CHECK(cudaEventElapsedTime(&ms, start, stop));

   CHECK(cudaFree(d_data));
   CHECK(cudaEventDestroy(start));
   CHECK(cudaEventDestroy(stop));
   return ms;
}
```

### Bước 2 – `processBreadthFirst`
1. Tạo mảng stream: `cudaStream_t *streams = new cudaStream_t[vectorCount];` và `cudaStreamCreate(&streams[v]);`.
2. Tạo mảng thiết bị: `float **d_data = (float **)malloc(vectorCount * sizeof(float*));` và `cudaMalloc(&d_data[v], bytes);`
3. **Phase 1:** `cudaMemcpyAsync(d_data[v], h_input[v], bytes, cudaMemcpyHostToDevice, streams[v]);`
4. **Phase 2-4:** Với từng pha, launch kernel ứng với tất cả `v`, kèm stream tương ứng: `scaleKernel<<<grid, block, 0, streams[v]>>>(...)`.
5. **Phase 5:** `cudaMemcpyAsync(h_output_bf[v], d_data[v], bytes, cudaMemcpyDeviceToHost, streams[v]);`
6. Đồng bộ: lặp `cudaStreamSynchronize(streams[v]);`
7. Giải phóng `d_data[v]`, hủy stream.

```cpp
float processBreadthFirst(float **h_input, float **h_output,
      int vectorCount, int vectorSize)
{
   size_t bytes = vectorSize * sizeof(float);
   cudaStream_t *streams = (cudaStream_t *)malloc(vectorCount * sizeof(cudaStream_t));
   float **d_data = (float **)malloc(vectorCount * sizeof(float *));
   dim3 block(256);
   dim3 grid((vectorSize + block.x - 1) / block.x);

   for (int v = 0; v < vectorCount; v++)
   {
      CHECK(cudaStreamCreate(&streams[v]));
      CHECK(cudaMalloc(&d_data[v], bytes));
   }

   for (int v = 0; v < vectorCount; v++)
      CHECK(cudaMemcpyAsync(d_data[v], h_input[v], bytes,
            cudaMemcpyHostToDevice, streams[v]));

   for (int v = 0; v < vectorCount; v++)
      scaleKernel<<<grid, block, 0, streams[v]>>>(d_data[v], vectorSize, 2.0f);

   for (int v = 0; v < vectorCount; v++)
      addKernel<<<grid, block, 0, streams[v]>>>(d_data[v], vectorSize, 10.0f);

   for (int v = 0; v < vectorCount; v++)
      squareKernel<<<grid, block, 0, streams[v]>>>(d_data[v], vectorSize);

   for (int v = 0; v < vectorCount; v++)
      CHECK(cudaMemcpyAsync(h_output[v], d_data[v], bytes,
            cudaMemcpyDeviceToHost, streams[v]));

   for (int v = 0; v < vectorCount; v++)
   {
      CHECK(cudaStreamSynchronize(streams[v]));
      CHECK(cudaFree(d_data[v]));
      CHECK(cudaStreamDestroy(streams[v]));
   }

   free(streams);
   free(d_data);
   // trả về thời gian đo bằng cudaEvent tương tự processSequential
}
```

### Bước 3 – `processDepthFirst`
1. Tạo stream và thiết bị tương tự bước 2.
2. Trong vòng lặp `v`:
   - `cudaMemcpyAsync(d_data[v], h_input[v], bytes, cudaMemcpyHostToDevice, streams[v]);`
   - Gọi cả 3 kernel liên tiếp trên cùng `streams[v]`.
   - `cudaMemcpyAsync(h_output_df[v], d_data[v], bytes, cudaMemcpyDeviceToHost, streams[v]);`
3. Sau vòng lặp, đồng bộ tất cả stream một lần.
4. Giải phóng bộ nhớ & stream.

```cpp
float processDepthFirst(float **h_input, float **h_output,
      int vectorCount, int vectorSize)
{
   size_t bytes = vectorSize * sizeof(float);
   cudaStream_t *streams = (cudaStream_t *)malloc(vectorCount * sizeof(cudaStream_t));
   float **d_data = (float **)malloc(vectorCount * sizeof(float *));
   dim3 block(256);
   dim3 grid((vectorSize + block.x - 1) / block.x);

   for (int v = 0; v < vectorCount; v++)
   {
      CHECK(cudaStreamCreate(&streams[v]));
      CHECK(cudaMalloc(&d_data[v], bytes));
      CHECK(cudaMemcpyAsync(d_data[v], h_input[v], bytes,
            cudaMemcpyHostToDevice, streams[v]));
      scaleKernel<<<grid, block, 0, streams[v]>>>(d_data[v], vectorSize, 2.0f);
      addKernel<<<grid, block, 0, streams[v]>>>(d_data[v], vectorSize, 10.0f);
      squareKernel<<<grid, block, 0, streams[v]>>>(d_data[v], vectorSize);
      CHECK(cudaMemcpyAsync(h_output[v], d_data[v], bytes,
            cudaMemcpyDeviceToHost, streams[v]));
   }

   for (int v = 0; v < vectorCount; v++)
   {
      CHECK(cudaStreamSynchronize(streams[v]));
      CHECK(cudaFree(d_data[v]));
      CHECK(cudaStreamDestroy(streams[v]));
   }
   free(streams);
   free(d_data);
   // trả về thời gian đo tương tự
}
```

### Bước 4 – Hàm `main`
1. Sau cấp phát và khởi tạo input, gọi lần lượt 3 hàm xử lý để thu thời gian.
2. Gọi `verifyResults` để chắc chắn kết quả trùng khớp.
3. In bảng tổng kết tốc độ và ghi nhận speedup `timeSeq / timeBF`, `timeSeq / timeDF`.
4. Thu dọn bộ nhớ host bằng `cudaFreeHost` hoặc `free` tùy cách cấp phát.

```cpp
int main(int argc, char **argv)
{
   int vectorCount = 8;
   int vectorSize = 1 << 22;
   // ... đọc tham số dòng lệnh nếu có

   float **h_input, **h_seq, **h_bf, **h_df;
   allocatePinnedBuffers(vectorCount, vectorSize,
         &h_input, &h_seq, &h_bf, &h_df);

   float timeSeq = processSequential(h_input, h_seq, vectorCount, vectorSize);
   float timeBF = processBreadthFirst(h_input, h_bf, vectorCount, vectorSize);
   float timeDF = processDepthFirst(h_input, h_df, vectorCount, vectorSize);

   verifyResults(h_seq, h_bf, vectorCount, vectorSize);
   verifyResults(h_seq, h_df, vectorCount, vectorSize);

   printf("Sequential: %.2f ms\n", timeSeq);
   printf("Breadth-first: %.2f ms (%.2fx)\n", timeBF, timeSeq / timeBF);
   printf("Depth-first: %.2f ms (%.2fx)\n", timeDF, timeSeq / timeDF);

   freePinnedBuffers(vectorCount, &h_input, &h_seq, &h_bf, &h_df);
   return 0;
}
```

### Bước 5 – Báo cáo trong Notebook
- Chụp log thời gian của 3 phương pháp trong các khối code cell.
- Chạy thêm thí nghiệm với `vectorCount = 16, 32` và lưu biểu đồ.
- Ghi chú quan sát: Depth-first thường thắng do mỗi stream giữ GPU bận liên tục và tận dụng copy engine song song.
- Nếu đo bằng Nsight Systems, so sánh timeline 1 stream vs n stream để chứng minh overlap.

---
Tài liệu này tập trung vào kiến thức và bước giải mới xuất hiện ở HW3. Bạn có thể quay lại `lesson-HW0/HW1/HW2` để ôn lại nền tảng Grid/Block/Thread hoặc luồng 5 bước tiêu chuẩn khi cần.