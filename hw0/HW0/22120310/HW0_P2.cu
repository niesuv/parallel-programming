#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) { \
    const cudaError_t e = call; \
    if (e != cudaSuccess) { \
        fprintf(stderr, "Error %s:%d: %s\n", __FILE__, __LINE__, cudaGetErrorString(e)); \
        exit(EXIT_FAILURE); \
    } \
}

struct GpuTimer {
    cudaEvent_t start, stop;
    GpuTimer() { cudaEventCreate(&start); cudaEventCreate(&stop); }
    ~GpuTimer() { cudaEventDestroy(start); cudaEventDestroy(stop); }
    void Start() { cudaEventRecord(start); }
    void Stop() { cudaEventRecord(stop); }
    float Elapsed() {
        float t;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);
        return t;
    }
};

void addHost(float *a, float *b, float *c, int n) {
    for (int i = 0; i < n; i++)
        c[i] = a[i] + b[i];
}

__global__ void add_v1(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    int base = blockIdx.x * (blockDim.x * 2);
    int i1 = base + tid;
    int i2 = base + tid + blockDim.x;
    if (i1 < n) c[i1] = a[i1] + b[i1];
    if (i2 < n) c[i2] = a[i2] + b[i2];
}

__global__ void add_v2(float *a, float *b, float *c, int n) {
    int tid = threadIdx.x;
    int base = blockIdx.x * (blockDim.x * 2);
    int i1 = base + 2 * tid;
    int i2 = i1 + 1;
    if (i1 < n) c[i1] = a[i1] + b[i1];
    if (i2 < n) c[i2] = a[i2] + b[i2];
}

int main(int argc, char **argv) {
    if (argc != 2) {
        printf("Usage: %s <vector_size>\n", argv[0]);
        return 1;
    }

    int N = atoi(argv[1]);
    size_t bytes = N * sizeof(float);

    float *a = (float*)malloc(bytes);
    float *b = (float*)malloc(bytes);
    float *h_out = (float*)malloc(bytes);
    float *d_out1 = (float*)malloc(bytes);
    float *d_out2 = (float*)malloc(bytes);

    for (int i = 0; i < N; i++) {
        a[i] = (float)rand() / RAND_MAX;
        b[i] = (float)rand() / RAND_MAX;
    }

    GpuTimer t;
    t.Start();
    addHost(a, b, h_out, N);
    t.Stop();
    float host_t = t.Elapsed();

    float *da, *db, *dc;
    CHECK(cudaMalloc(&da, bytes));
    CHECK(cudaMalloc(&db, bytes));
    CHECK(cudaMalloc(&dc, bytes));

    CHECK(cudaMemcpy(da, a, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, b, bytes, cudaMemcpyHostToDevice));

    dim3 block(256);
    dim3 grid((N + block.x * 2 - 1) / (block.x * 2));

    t.Start();
    add_v1<<<grid, block>>>(da, db, dc, N);
    CHECK(cudaDeviceSynchronize());
    t.Stop();
    float dev1 = t.Elapsed();
    CHECK(cudaMemcpy(d_out1, dc, bytes, cudaMemcpyDeviceToHost));

    t.Start();
    add_v2<<<grid, block>>>(da, db, dc, N);
    CHECK(cudaDeviceSynchronize());
    t.Stop();
    float dev2 = t.Elapsed();
    CHECK(cudaMemcpy(d_out2, dc, bytes, cudaMemcpyDeviceToHost));

    for (int i = 0; i < N; i++) {
        if (fabs(d_out1[i] - h_out[i]) > 1e-5 || fabs(d_out2[i] - h_out[i]) > 1e-5) {
            printf("INCORRECT\n");
            return 1;
        }
    }

    printf(
        "Vector size: %-10d | Host: %8.3f ms | Device v1: %8.3f ms | Device v2: %8.3f ms\n",
        N, host_t, dev1, dev2
    );

    free(a);
    free(b);
    free(h_out);
    free(d_out1);
    free(d_out2);
    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    return 0;
}
