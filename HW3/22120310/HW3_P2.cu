#include <stdio.h>
#include <stdlib.h>
#include <cuda_runtime.h>

#define CHECK(call) \
{ \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__, \
                cudaGetErrorString(err)); \
        exit(EXIT_FAILURE); \
    } \
}

// ========================================
// Kernel 1: Scale
// ========================================
__global__ void scaleKernel(float *data, int n, float factor) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * factor;
    }
}

// ========================================
// Kernel 2: Add
// ========================================
__global__ void addKernel(float *data, int n, float value) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] + value;
    }
}

// ========================================
// Kernel 3: Square
// ========================================
__global__ void squareKernel(float *data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * data[idx];
    }
}

// ========================================
// CPU Reference Implementation
// ========================================
void processVectorCPU(float *data, int n) {
    // Scale
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * 2.0f;
    }
    
    // Add
    for (int i = 0; i < n; i++) {
        data[i] = data[i] + 10.0f;
    }
    
    // Square
    for (int i = 0; i < n; i++) {
        data[i] = data[i] * data[i];
    }
}

// ========================================
// PART A: Sequential GPU Version (No Streams)
// ========================================
float processSequential(float **h_input, float **h_output, int vectorCount, int vectorSize) 
{
    printf("\n=== SEQUENTIAL (NO STREAMS) ===\n");
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // TODO: Allocate device memory for ONE vector
    float *d_data;
    // YOUR CODE HERE
    CHECK(cudaMalloc(&d_data, vectorSize * sizeof(float)));
    
    
    CHECK(cudaEventRecord(start));
    
    // Process each vector sequentially
    dim3 blockSize(256);
    dim3 gridSize((vectorSize + blockSize.x - 1) / blockSize.x);
    for (int v = 0; v < vectorCount; v++) {
        // 1. Copy input vector to device
        CHECK(cudaMemcpy(d_data, h_input[v],
                         sizeof(float) * vectorSize,
                         cudaMemcpyHostToDevice));

        // 2/3/4. Launch kernels in sequence
        scaleKernel<<<gridSize, blockSize>>>(d_data, vectorSize, 2.0f);
        addKernel<<<gridSize, blockSize>>>(d_data, vectorSize, 10.0f);
        squareKernel<<<gridSize, blockSize>>>(d_data, vectorSize);

        // 5. Copy result back to host
        CHECK(cudaMemcpy(h_output[v], d_data,
                         sizeof(float) * vectorSize,
                         cudaMemcpyDeviceToHost));

        // 6. Make sure the device finished the work for this vector
        CHECK(cudaDeviceSynchronize());
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Time: %.2f ms\n", milliseconds);
    
    // Cleanup
    CHECK(cudaFree(d_data));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    return milliseconds;
}

// ========================================
// PART B: Breadth-First with Streams
// ========================================
float processBreadthFirst(float **h_input, float **h_output, int vectorCount, int vectorSize) {
    printf("\n=== BREADTH-FIRST PIPELINE ===\n");
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // TODO: Create streams
    // YOUR CODE HERE
    // Create per-vector streams for the breadth-first pipeline
    cudaStream_t *bfStreams = (cudaStream_t*)malloc(vectorCount * sizeof(cudaStream_t));
    for (int s = 0; s < vectorCount; ++s) {
        CHECK(cudaStreamCreate(&bfStreams[s]));
    }

    // TODO: Allocate device memory for each stream
    // float **d_data; // Array of device pointers
    // YOUR CODE HERE
    // Allocate device memory buffers (one per vector)
    float **d_data_bf = (float **)malloc(vectorCount * sizeof(float *));
    for (int i = 0; i < vectorCount; ++i) {
        CHECK(cudaMalloc(&d_data_bf[i], vectorSize * sizeof(float)));
    }

    CHECK(cudaEventRecord(start));
    
    // Phase 1: All H2D transfers (async)
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaMemcpyAsync(d_data_bf[v], h_input[v], vectorSize * sizeof(float),
                              cudaMemcpyHostToDevice, bfStreams[v]));
    }
    
    // Phase 2: All scale operations
    dim3 bfBlock(256);
    dim3 bfGrid((vectorSize + bfBlock.x - 1) / bfBlock.x);
    for (int v = 0; v < vectorCount; v++) {
        scaleKernel<<<bfGrid, bfBlock, 0, bfStreams[v]>>>(d_data_bf[v], vectorSize, 2.0f);
    }
    
    // Phase 3: All add operations
    for (int v = 0; v < vectorCount; v++) {
        addKernel<<<bfGrid, bfBlock, 0, bfStreams[v]>>>(d_data_bf[v], vectorSize, 10.0f);
    }
    
    // Phase 4: All square operations
    for (int v = 0; v < vectorCount; v++) {
        squareKernel<<<bfGrid, bfBlock, 0, bfStreams[v]>>>(d_data_bf[v], vectorSize);
    }
    
    // Phase 5: All D2H transfers (async)
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaMemcpyAsync(h_output[v], d_data_bf[v], vectorSize * sizeof(float),
                              cudaMemcpyDeviceToHost, bfStreams[v]));
    }
    
    // Synchronize all streams
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaStreamSynchronize(bfStreams[v]));
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Time: %.2f ms\n", milliseconds);
    
    // Cleanup breadth-first resources
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaFree(d_data_bf[v]));
        CHECK(cudaStreamDestroy(bfStreams[v]));
    }
    free(d_data_bf);
    free(bfStreams);
    
    return milliseconds;
}

// ========================================
// PART C: Depth-First with Streams
// ========================================
float processDepthFirst(float **h_input, float **h_output, int vectorCount, int vectorSize) {
    printf("\n=== DEPTH-FIRST PIPELINE ===\n");
    
    cudaEvent_t start, stop;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    
    // TODO: Create streams
    // YOUR CODE HERE
    // Create streams for depth-first pipeline
    cudaStream_t *dfStreams = (cudaStream_t*)malloc(vectorCount * sizeof(cudaStream_t));
    for (int s = 0; s < vectorCount; ++s) {
        CHECK(cudaStreamCreate(&dfStreams[s]));
    }

    // TODO: Allocate device memory
    // YOUR CODE HERE
    // Allocate device memory buffers (one per vector)
    float **d_data_df = (float **)malloc(vectorCount * sizeof(float *));
    for (int i = 0; i < vectorCount; ++i) {
        CHECK(cudaMalloc(&d_data_df[i], vectorSize * sizeof(float)));
    }
   
    CHECK(cudaEventRecord(start));
    
    // Process each vector completely in its own stream
    dim3 dfBlock(256);
    dim3 dfGrid((vectorSize + dfBlock.x - 1) / dfBlock.x);
    for (int v = 0; v < vectorCount; v++) {
        // 1. H2D transfer
        CHECK(cudaMemcpyAsync(d_data_df[v], h_input[v], vectorSize * sizeof(float),
                              cudaMemcpyHostToDevice, dfStreams[v]));

        // 2. Scale
        scaleKernel<<<dfGrid, dfBlock, 0, dfStreams[v]>>>(d_data_df[v], vectorSize, 2.0f);

        // 3. Add
        addKernel<<<dfGrid, dfBlock, 0, dfStreams[v]>>>(d_data_df[v], vectorSize, 10.0f);

        // 4. Square
        squareKernel<<<dfGrid, dfBlock, 0, dfStreams[v]>>>(d_data_df[v], vectorSize);

        // 5. D2H transfer
        CHECK(cudaMemcpyAsync(h_output[v], d_data_df[v], vectorSize * sizeof(float),
                              cudaMemcpyDeviceToHost, dfStreams[v]));
    }

    // Wait for all streams to complete
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaStreamSynchronize(dfStreams[v]));
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&milliseconds, start, stop));
    
    printf("Time: %.2f ms\n", milliseconds);
    
    // Cleanup depth-first resources
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaFree(d_data_df[v]));
        CHECK(cudaStreamDestroy(dfStreams[v]));
    }
    free(d_data_df);
    free(dfStreams);
    
    return milliseconds;
}

// ========================================
// Verification Function
// ========================================
bool verifyResults(float **result1, float **result2, int vectorCount, int vectorSize) {
    for (int v = 0; v < vectorCount; v++) {
        for (int i = 0; i < vectorSize; i++) {
            if (fabs(result1[v][i] - result2[v][i]) > 1e-3) {
                printf("Mismatch at vector %d, index %d: %.2f vs %.2f\n",
                       v, i, result1[v][i], result2[v][i]);
                return false;
            }
        }
    }
    return true;
}

// ========================================
// Main Function
// ========================================
int main(int argc, char **argv) {
    // Configuration
    int vectorCount = 8;
    int vectorSize = 1 << 22;  // 4M elements = 16 MB per vector
    
    if (argc > 1) vectorCount = atoi(argv[1]);
    if (argc > 2) vectorSize = atoi(argv[2]);
    
    printf("╔════════════════════════════════════════╗\n");
    printf("║  VECTOR OPERATIONS PIPELINE EXERCISE   ║\n");
    printf("╚════════════════════════════════════════╝\n\n");
    
    printf("Configuration:\n");
    printf("  Number of vectors: %d\n", vectorCount);
    printf("  Elements per vector: %d (%.2f MB)\n", 
           vectorSize, vectorSize * sizeof(float) / 1024.0 / 1024.0);
    printf("  Total data: %.2f MB\n\n",
           vectorCount * vectorSize * sizeof(float) / 1024.0 / 1024.0);
    
    // Print GPU info
    cudaDeviceProp prop;
    CHECK(cudaGetDeviceProperties(&prop, 0));
    printf("GPU: %s\n", prop.name);
    printf("Concurrent kernels supported: %s\n", 
           prop.concurrentKernels ? "Yes" : "No");
    printf("Number of copy engines: %d\n\n", prop.asyncEngineCount);
    
    // Allocate host memory (pinned for async transfers)
    float **h_input = (float **)malloc(vectorCount * sizeof(float *));
    float **h_output_seq = (float **)malloc(vectorCount * sizeof(float *));
    float **h_output_bf = (float **)malloc(vectorCount * sizeof(float *));
    float **h_output_df = (float **)malloc(vectorCount * sizeof(float *));
    
    for (int v = 0; v < vectorCount; v++) {
        //TODO: Allocate memory for h_input[v], h_output_seq[v]
        // h_output_bf[v], h_output_df[v]
        // YOUR CODE HERE
        // Allocate pinned host buffers for input and outputs
        CHECK(cudaMallocHost(&h_input[v], vectorSize * sizeof(float)));
        CHECK(cudaMallocHost(&h_output_seq[v], vectorSize * sizeof(float)));
        CHECK(cudaMallocHost(&h_output_bf[v], vectorSize * sizeof(float)));
        CHECK(cudaMallocHost(&h_output_df[v], vectorSize * sizeof(float)));

        // Initialize input
        for (int i = 0; i < vectorSize; i++) {
            h_input[v][i] = (float)(i % 100) / 10.0f;
        }
    }
    
    // Run all implementations
    float timeSeq = processSequential(h_input, h_output_seq, vectorCount, vectorSize);
    float timeBF = processBreadthFirst(h_input, h_output_bf, vectorCount, vectorSize);
    float timeDF = processDepthFirst(h_input, h_output_df, vectorCount, vectorSize);
    
    // Verify results
    printf("\n=== VERIFICATION ===\n");
    bool seqVsBF = verifyResults(h_output_seq, h_output_bf, vectorCount, vectorSize);
    bool seqVsDF = verifyResults(h_output_seq, h_output_df, vectorCount, vectorSize);
    
    if (seqVsBF && seqVsDF) {
        printf("✓ All results match!\n");
    } else {
        printf("✗ Results don't match!\n");
    }
    
    // Performance summary
    printf("\n╔════════════════════════════════════════╗\n");
    printf("║         PERFORMANCE SUMMARY            ║\n");
    printf("╚════════════════════════════════════════╝\n");
    printf("Sequential:     %.2f ms\n", timeSeq);
    printf("Breadth-First:  %.2f ms  (%.2fx speedup)\n", 
           timeBF, timeSeq / timeBF);
    printf("Depth-First:    %.2f ms  (%.2fx speedup)\n", 
           timeDF, timeSeq / timeDF);
    printf("\nDepth-First vs Breadth-First: %.2fx\n", timeBF / timeDF);
    
    // Cleanup all pinned host memory
    for (int v = 0; v < vectorCount; v++) {
        cudaFreeHost(h_input[v]);
        cudaFreeHost(h_output_seq[v]);
        cudaFreeHost(h_output_bf[v]);
        cudaFreeHost(h_output_df[v]);
    }
    free(h_input);
    free(h_output_seq);
    free(h_output_bf);
    free(h_output_df);
    
    return 0;
}