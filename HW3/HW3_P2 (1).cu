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
    
    cudaEvent_t start, stop, startCompute, stopCompute, startH2D, stopH2D, startD2H, stopD2H;
    float totalTime = 0, transferTime = 0, kernelTime = 0, milliseconds = 0;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));
    CHECK(cudaEventCreate(&startCompute));
    CHECK(cudaEventCreate(&stopCompute));
    CHECK(cudaEventCreate(&startH2D));
    CHECK(cudaEventCreate(&stopH2D));
    CHECK(cudaEventCreate(&startD2H));
    CHECK(cudaEventCreate(&stopD2H));

    
    // TODO: Allocate device memory for ONE vector
    float *d_data;
    // YOUR CODE HERE
    CHECK(cudaMalloc(&d_data, vectorSize * sizeof(float)));
    
    
    CHECK(cudaEventRecord(start));
    
    // Process each vector sequentially
    dim3 blockSize(256);
    dim3 gridSize((vectorSize - 1)/blockSize.x + 1);
    for (int v = 0; v < vectorCount; v++) {
        // TODO: 
        // 1. Copy input vector to device
        // 2. Launch scale kernel
        // 3. Launch add kernel
        // 4. Launch square kernel
        // 5. Copy result back to host
        // 6. Synchronize
        
        // YOUR CODE HERE
        // YOUR CODE HERE
        CHECK(cudaEventRecord(startH2D));
        CHECK(cudaMemcpy(d_data, h_input[v],
                        sizeof(float) * vectorSize,
                        cudaMemcpyHostToDevice));
        CHECK(cudaEventRecord(stopH2D));

        CHECK(cudaEventRecord(startCompute));
        scaleKernel<<<gridSize, blockSize>>>(d_data, vectorSize, 2.0f);
        addKernel<<<gridSize, blockSize>>>(d_data, vectorSize, 10.0f);
        squareKernel<<<gridSize, blockSize>>>(d_data, vectorSize);
        CHECK(cudaEventRecord(stopCompute));

        CHECK(cudaEventRecord(startD2H));
        CHECK(cudaMemcpy(h_output[v], d_data,
                        sizeof(float) * vectorSize,
                        cudaMemcpyDeviceToHost));
        CHECK(cudaEventRecord(stopD2H));

        CHECK(cudaDeviceSynchronize());

    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    // float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&totalTime, start, stop));
    
    printf("Time: %.2f ms\n", totalTime);
    // printf("  Transfer time: %.2f ms\n", transferTime);
    // printf("  Kernel time:   %.2f ms\n", kernelTime);
    
    // Cleanup
    CHECK(cudaFree(d_data));
    CHECK(cudaEventDestroy(start));
    CHECK(cudaEventDestroy(stop));
    
    return totalTime;
}

// ========================================
// PART B: Breadth-First with Streams
// ========================================
float processBreadthFirst(float **h_input, float **h_output, int vectorCount, int vectorSize) {
    printf("\n=== BREADTH-FIRST PIPELINE ===\n");
    
    cudaEvent_t start, stop;
    cudaEvent_t * startH2D = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopH2D = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startD2H = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopD2H = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startKernel1 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopKernel1 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startKernel2 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopKernel2 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startKernel3 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopKernel3 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));

    float totalTime = 0, milliseconds = 0,
    maxH2D = 0.0, maxD2H = 0.0, maxKernel1 = 0.0, maxKernel2 = 0.0, maxKernel3 = 0.0;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    
    // TODO: Create streams
    // YOUR CODE HERE
    cudaStream_t *streams = (cudaStream_t*)malloc(vectorCount * sizeof(cudaStream_t));
    for (int s = 0; s < vectorCount; ++s) {
        CHECK(cudaStreamCreate(&streams[s]));

        CHECK(cudaEventCreate(&startH2D[s]));
        CHECK(cudaEventCreate(&stopH2D[s]));
        CHECK(cudaEventCreate(&startD2H[s]));
        CHECK(cudaEventCreate(&stopD2H[s]));
        CHECK(cudaEventCreate(&startKernel1[s]));
        CHECK(cudaEventCreate(&stopKernel1[s]));
        CHECK(cudaEventCreate(&startKernel2[s]));
        CHECK(cudaEventCreate(&stopKernel2[s]));
        CHECK(cudaEventCreate(&startKernel3[s]));
        CHECK(cudaEventCreate(&stopKernel3[s]));
    }
    
    // TODO: Allocate device memory for each stream
    float **d_data; // Array of device pointers
    // YOUR CODE HERE
    d_data = (float **)malloc(vectorCount * sizeof(float *));
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaMalloc(&d_data[i], vectorSize * sizeof(float)));
    }
    
    CHECK(cudaEventRecord(start));
    
    for (int v = 0; v < vectorCount; v++) {
         // Phase 1: All H2D transfers
    // TODO: Copy all vectors to device
    // YOUR CODE HERE
    }

   
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaEventRecord(startH2D[v], streams[v]));
        CHECK(cudaMemcpyAsync(d_data[v], h_input[v], vectorSize * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[v]));
        CHECK(cudaEventRecord(stopH2D[v], streams[v]));
    }
    
    // Phase 2: All scale operations
    // TODO: Launch scale kernel for all vectors
    // YOUR CODE HERE
    dim3 blockSize(256);
    dim3 gridSize((vectorSize - 1)/blockSize.x + 1);

    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaEventRecord(startKernel1[v], streams[v]));
        scaleKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 2.0f);
        CHECK(cudaEventRecord(stopKernel1[v], streams[v]));
    }
    
    // Phase 3: All add operations
    // TODO: Launch add kernel for all vectors
    // YOUR CODE HERE
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaEventRecord(startKernel2[v], streams[v]));
        addKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 10.0f);
        CHECK(cudaEventRecord(stopKernel2[v], streams[v]));
    }
    
    // Phase 4: All square operations
    // TODO: Launch square kernel for all vectors
    // YOUR CODE HERE
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaEventRecord(startKernel3[v], streams[v]));
        squareKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize);
        CHECK(cudaEventRecord(stopKernel3[v], streams[v]));
    }

    
    // Phase 5: All D2H transfers
    // TODO: Copy all results back to host
    // YOUR CODE HERE
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaEventRecord(startD2H[v], streams[v]));
        CHECK(cudaMemcpyAsync(h_output[v], d_data[v], vectorSize * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[v]));
        CHECK(cudaEventRecord(stopD2H[v], streams[v]));
    }
    
    // Synchronize all streams
    // YOUR CODE HERE
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaStreamSynchronize(streams[v]));
    }
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    CHECK(cudaEventElapsedTime(&totalTime, start, stop));

    // Calculate transfer and kernel times
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaEventSynchronize(stopH2D[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startH2D[v], stopH2D[v]));
        if (milliseconds > maxH2D) maxH2D = milliseconds;

        CHECK(cudaEventSynchronize(stopD2H[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startD2H[v], stopD2H[v]));
        if (milliseconds > maxD2H) maxD2H = milliseconds;

        CHECK(cudaEventSynchronize(stopKernel1[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startKernel1[v], stopKernel1[v]));
        if (milliseconds > maxKernel1) maxKernel1 = milliseconds;

        CHECK(cudaEventSynchronize(stopKernel2[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startKernel2[v], stopKernel2[v]));
        if (milliseconds > maxKernel2) maxKernel2 = milliseconds;

        CHECK(cudaEventSynchronize(stopKernel3[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startKernel3[v], stopKernel3[v]));
        if (milliseconds > maxKernel3) maxKernel3 = milliseconds;
    }

    float transferTime = maxH2D + maxD2H;
    float kernelTime   = maxKernel1 + maxKernel2 + maxKernel3;

    
    printf("Time: %.2f ms\n", totalTime);
    printf("  Transfer time: %.2f ms\n", transferTime);
    printf("  Kernel time:   %.2f ms\n", kernelTime);
    
    // TODO: Cleanup
    // YOUR CODE HERE
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaFree(d_data[v]));
        CHECK(cudaStreamDestroy(streams[v]));
    }
    free(d_data);
    free(streams);
    
    return totalTime;
}

// ========================================
// PART C: Depth-First with Streams
// ========================================
float processDepthFirst(float **h_input, float **h_output, int vectorCount, int vectorSize) {
    printf("\n=== DEPTH-FIRST PIPELINE ===\n");
    
    cudaEvent_t start, stop;
    cudaEvent_t * startH2D = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopH2D = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startD2H = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopD2H = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startKernel1 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopKernel1 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startKernel2 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopKernel2 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * startKernel3 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));
    cudaEvent_t * stopKernel3 = (cudaEvent_t *)malloc(vectorCount * sizeof(cudaEvent_t));

    float totalTime = 0, milliseconds = 0,
    maxH2D = 0.0, maxD2H = 0.0, maxKernel1 = 0.0, maxKernel2 = 0.0, maxKernel3 = 0.0;
    CHECK(cudaEventCreate(&start));
    CHECK(cudaEventCreate(&stop));

    // TODO: Create streams
    // YOUR CODE HERE
    cudaStream_t *streams = (cudaStream_t *)malloc(vectorCount * sizeof(cudaStream_t));
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaStreamCreate(&streams[i]));
        CHECK(cudaEventCreate(&startH2D[i]));
        CHECK(cudaEventCreate(&stopH2D[i]));
        CHECK(cudaEventCreate(&startD2H[i]));
        CHECK(cudaEventCreate(&stopD2H[i]));
        CHECK(cudaEventCreate(&startKernel1[i]));
        CHECK(cudaEventCreate(&stopKernel1[i]));
        CHECK(cudaEventCreate(&startKernel2[i]));
        CHECK(cudaEventCreate(&stopKernel2[i]));
        CHECK(cudaEventCreate(&startKernel3[i]));
        CHECK(cudaEventCreate(&stopKernel3[i]));
    }
    
    // TODO: Allocate device memory
    // YOUR CODE HERE
    float **d_data = (float **)malloc(vectorCount * sizeof(float *));
    for (int i = 0; i < vectorCount; i++) {
        CHECK(cudaMalloc(&d_data[i], vectorSize * sizeof(float)));
    }

   
    CHECK(cudaEventRecord(start));
    
    // Process each vector completely in its own stream
    dim3 blockSize(256);
    dim3 gridSize((vectorSize - 1)/blockSize.x + 1);
    for (int v = 0; v < vectorCount; v++) {
        // TODO: For each vector, issue ALL operations in sequence:
        // 1. H2D transfer
        // 2. Scale kernel
        // 3. Add kernel
        // 4. Square kernel
        // 5. D2H transfer
        // All operations should use streams[v]
        
        // YOUR CODE HERE
        CHECK(cudaEventRecord(startH2D[v], streams[v]));
        CHECK(cudaMemcpyAsync(d_data[v], h_input[v], vectorSize * sizeof(float), 
                              cudaMemcpyHostToDevice, streams[v]));
        CHECK(cudaEventRecord(stopH2D[v], streams[v]));

        CHECK(cudaEventRecord(startKernel1[v], streams[v]));
        scaleKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 2.0f);
        CHECK(cudaEventRecord(stopKernel1[v], streams[v]));

        CHECK(cudaEventRecord(startKernel2[v], streams[v]));
        addKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize, 10.0f);
        CHECK(cudaEventRecord(stopKernel2[v], streams[v]));

        CHECK(cudaEventRecord(startKernel3[v], streams[v]));
        squareKernel<<<gridSize, blockSize, 0, streams[v]>>>(d_data[v], vectorSize);
        CHECK(cudaEventRecord(stopKernel3[v], streams[v]));

        CHECK(cudaEventRecord(startD2H[v], streams[v]));
        CHECK(cudaMemcpyAsync(h_output[v], d_data[v], vectorSize * sizeof(float), 
                              cudaMemcpyDeviceToHost, streams[v]));
        CHECK(cudaEventRecord(startD2H[v], streams[v]));
    }
    
    // Wait for all streams to complete
    // YOUR CODE HERE
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaStreamSynchronize(streams[v]));
    }

    // Calculate transfer and kernel times
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaEventSynchronize(stopH2D[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startH2D[v], stopH2D[v]));
        if (milliseconds > maxH2D) maxH2D = milliseconds;

        CHECK(cudaEventSynchronize(stopD2H[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startD2H[v], stopD2H[v]));
        if (milliseconds > maxD2H) maxD2H = milliseconds;

        CHECK(cudaEventSynchronize(stopKernel1[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startKernel1[v], stopKernel1[v]));
        if (milliseconds > maxKernel1) maxKernel1 = milliseconds;

        CHECK(cudaEventSynchronize(stopKernel2[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startKernel2[v], stopKernel2[v]));
        if (milliseconds > maxKernel2) maxKernel2 = milliseconds;

        CHECK(cudaEventSynchronize(stopKernel3[v]));
        CHECK(cudaEventElapsedTime(&milliseconds, startKernel3[v], stopKernel3[v]));
        if (milliseconds > maxKernel3) maxKernel3 = milliseconds;
    }

    float transferTime = maxH2D + maxD2H;
    float kernelTime   = maxKernel1 + maxKernel2 + maxKernel3;
    
    CHECK(cudaEventRecord(stop));
    CHECK(cudaEventSynchronize(stop));
    
    // float milliseconds = 0;
    CHECK(cudaEventElapsedTime(&totalTime, start, stop));

    transferTime = maxH2D + maxD2H;
    kernelTime   = maxCompute;
    
    printf("Time: %.2f ms\n", totalTime);
    printf("  Transfer time: %.2f ms\n", transferTime);
    printf("  Kernel time:   %.2f ms\n", kernelTime);
    
    // TODO: Cleanup
    // YOUR CODE HERE
    for (int v = 0; v < vectorCount; v++) {
        CHECK(cudaFree(d_data[v]));
        CHECK(cudaStreamDestroy(streams[v]));
    }
    free(d_data);
    free(streams);
    
    return totalTime;
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
    float **h_input = (float **)malloc(vectorCount * sizeof(float *)); //TODO
    float **h_output_seq = (float **)malloc(vectorCount * sizeof(float *)); //TODO
    float **h_output_bf = (float **)malloc(vectorCount * sizeof(float *)); //TODO
    float **h_output_df = (float **)malloc(vectorCount * sizeof(float *)); //TODO
    
    for (int v = 0; v < vectorCount; v++) {
		//TODO: Allocate memory for h_input[v], h_output_seq[v]
		// h_output_bf[v], h_output_df[v]
		// YOUR CODE HERE
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
    
    // TODO: Cleanup all memory that has been allocated
	// YOUR CODE HERE
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