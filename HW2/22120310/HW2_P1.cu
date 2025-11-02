#include <stdio.h>

#define CHECK(call)\
{\
    const cudaError_t error = call;\
    if (error != cudaSuccess)\
    {\
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);\
        fprintf(stderr, "code: %d, reason: %s\n", error,\
                cudaGetErrorString(error));\
        exit(EXIT_FAILURE);\
    }\
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
        cudaEventSynchronize(start);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};


__global__ void reduceBlksKernel1(int * in, int * out, int n)
{
	// TODO
	int start = blockIdx.x * 2 * blockDim.x;
    int end   = min(start + 2 * blockDim.x, n);
    int i     = start + 2 * threadIdx.x;

	if (i >= n) return;
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2) {
        if (threadIdx.x % stride == 0) {
            int j = i + stride;
            if (j < end)
                in[i] += in[j];
        }
        __syncthreads();
    }

    if (threadIdx.x == 0)
        atomicAdd(out, in[start]);
}


__global__ void reduceBlksKernel2(int * in, int * out, int n)
{
	// TODO
    int start = blockIdx.x * 2 * blockDim.x;
    int end   = min(start + 2 * blockDim.x, n);
    int tid   = threadIdx.x;
	int len = end - start;

    for (int stride = 1; stride < len; stride *= 2) {
        int i = start + 2 * tid * stride;
        if (i + stride < end) {
            in[i] += in[i + stride];
        }
        __syncthreads();
    }

    if (tid == 0)
        atomicAdd(out, in[start]);
}


__global__ void reduceBlksKernel3(int *in, int *out, int n)
{
	// TODO
	int start = blockIdx.x * 2 * blockDim.x;
	int end   = min(start + 2 * blockDim.x, n);
    int tid   = threadIdx.x;
	int len = end - start;
	 if (len <= 0) return;
    if (start >= n) return;

	for (int curlen = len; curlen > 1; curlen = (curlen + 1) / 2) {
        int active = (curlen + 1) / 2; 
        int stride = active;
		
		
        if (tid < active) {
            int i = start + tid;
            int j = i + stride;
            if (j < start + curlen) {
                in[i] += in[j];
            }
        }
        __syncthreads();
    }
	if (tid == 0)
        atomicAdd(out, in[start]);
}



__global__ void reduceBlksKernel4_UnrollWarp(int * in, int * out,int n)
{
	// TODO: Modify from previous best version
	int start = blockIdx.x * 2 * blockDim.x;
    if (start >= n) return;
    int end = min(start + 2 * blockDim.x, n);
    int tid = threadIdx.x;
    int len = end - start;
    if (len <= 0) return;

    int curlen = len;
    while (curlen > 64) {
        int active = (curlen + 1) / 2;
        if (tid < active) {
            int i = start + tid;
            int j = i + active;
            
            if (j < start + curlen)
                in[i] += in[j];
        }
        __syncthreads();
        curlen = (curlen + 1) / 2;
    }

    // final warp (curlen <= 32): sync warp then unroll without __syncthreads()
    if (tid < 32) {
        int i = start + tid;
        int curend = start + curlen;
        if (i + 32 < curend) in[i] += in[i + 32];
        if (i + 16 < curend) in[i] += in[i + 16];
        if (i + 8  < curend) in[i] += in[i + 8];
        if (i + 4  < curend) in[i] += in[i + 4];
        if (i + 2  < curend) in[i] += in[i + 2];
        if (i + 1  < curend) in[i] += in[i + 1];
    }

    if (tid == 0)
        atomicAdd(out, in[start]);
}


__global__ void reduceBlksKernel5_CompleteUnroll(int * in, int * out, int n)
{
	// TODO: Modify from previous best version
    int start = blockIdx.x * 2 * blockDim.x;
    if (start >= n) return;

    int end = min(start + 2 * blockDim.x, n);
	
    int tid = threadIdx.x;
    int i = start + tid;
	if (i >= end) return;

	if (tid < 1024 && i + 1024 < end) in[i] += in[i + 1024];
    __syncthreads();

    if (tid < 512 && i + 512 < end) in[i] += in[i + 512];
    __syncthreads();

    if (tid < 256 && i + 256 < end) in[i] += in[i + 256];
    __syncthreads();

   if (tid < 128 && i + 128 < end) in[i] += in[i + 128];
    __syncthreads();

    if (tid < 64 && i + 64 < end) in[i] += in[i + 64];
    __syncthreads();

    if (tid < 32) {
        if (i + 32 < end) in[i] += in[i + 32];
        if (i + 16 < end) in[i] += in[i + 16];
        if (i + 8  < end) in[i] += in[i + 8];
        if (i + 4  < end) in[i] += in[i + 4];
        if (i + 2  < end) in[i] += in[i + 2];
        if (i + 1  < end) in[i] += in[i + 1];
    }

    if (tid == 0)
        atomicAdd(out, in[start]);
}


__global__ void reduceBlksKernel6_MultiplePerThread(int * in, int * out,int n, int elemsPerThread)
{
	int B = blockDim.x;
    int start = blockIdx.x * 2 * B * elemsPerThread;
    if (start >= n) return;

    int end = min(start + 2 * B * elemsPerThread, n);
    int tid = threadIdx.x;
    if (tid >= B) return;

	int localSum = 0;
    for (int e = 0; e < elemsPerThread; ++e) {
        int idx1 = start + tid + e * B;
        if (idx1 < end) localSum += in[idx1];

        int idx2 = start + B * elemsPerThread + tid + e * B;
        if (idx2 < end) localSum += in[idx2];
    }

    
    in[start + tid] = localSum;
    __syncthreads();

    if (B > 512) {
        if (tid < 512 && (tid + 512) < B) in[start + tid] += in[start + tid + 512];
        __syncthreads();
    }
    if (B > 256) {
        if (tid < 256 && (tid + 256) < B) in[start + tid] += in[start + tid + 256];
        __syncthreads();
    }
    if (B > 128) {
        if (tid < 128 && (tid + 128) < B) in[start + tid] += in[start + tid + 128];
        __syncthreads();
    }
    if (B > 64) {
        if (tid < 64 && (tid + 64) < B) in[start + tid] += in[start + tid + 64];
        __syncthreads();
    }

    if (tid < 32) {
        int i = start + tid;
        if (tid + 32 < B) in[i] += in[i + 32];
        if (tid + 16 < B) in[i] += in[i + 16];
        if (tid + 8  < B) in[i] += in[i + 8];
        if (tid + 4  < B) in[i] += in[i + 4];
        if (tid + 2  < B) in[i] += in[i + 2];
        if (tid + 1  < B) in[i] += in[i + 1];
    }


	if (tid == 0)
		atomicAdd(out, in[start]);
}



int reduce(int const * in, int n,
        bool useDevice=false, dim3 blockSize=dim3(1), int kernelType=1)
{

	GpuTimer timer;
	int result = 0; // Init
	if (useDevice == false)
	{
		timer.Start();
		result = in[0];
		for (int i = 1; i < n; i++)
		{
			result += in[i];
		}
		timer.Stop();
		float hostTime = timer.Elapsed();
		printf("Host time: %f ms\n",hostTime);
	}
	else // Use device
	{
		// Allocate device memories
		int * d_in, * d_out;
		dim3 gridSize(1); // TODO: Compute gridSize from n and blockSize
		gridSize = (n + blockSize.x * 2 - 1) / (blockSize.x * 2) ;

		// TODO: Allocate device memories
		CHECK(cudaMalloc(&d_in, n * sizeof(int)));
		CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));

		// TODO: Copy data to device memories
		CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

		// Call kernel
		timer.Start();
		
		switch(kernelType)
		{
			case 1:
				reduceBlksKernel1<<<gridSize, blockSize>>>(d_in, d_out, n);
				break;
				
			case 2:
				reduceBlksKernel2<<<gridSize, blockSize>>>(d_in, d_out, n);
				break;

			case 3:
				reduceBlksKernel3<<<gridSize, blockSize>>>(d_in, d_out, n);
				break;
				
			case 4:
				reduceBlksKernel4_UnrollWarp<<<gridSize, blockSize>>>(d_in, d_out, n);				
				break;
				
			case 5:
				reduceBlksKernel5_CompleteUnroll<<<gridSize, blockSize>>>(d_in, d_out, n);
				break;
				
			case 6:
			{
				int elemsPerThread = 4;
				//TODO: Re calculate gridSize based on elemsPerThread
				gridSize.x = (n + blockSize.x * elemsPerThread * 2 - 1) / (blockSize.x * 2 * elemsPerThread);
				reduceBlksKernel6_MultiplePerThread<<<gridSize, blockSize>>>(d_in, d_out, n, elemsPerThread);
				break;
			}
		}

		cudaDeviceSynchronize();
		timer.Stop();
		float kernelTime = timer.Elapsed();

		CHECK(cudaGetLastError());
		
		// TODO: Copy result from device memories
		int * partialSums = (int *)malloc(gridSize.x * sizeof(int));
		CHECK(cudaMemcpy(partialSums, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

		result = 0;
		for (int i = 0; i < gridSize.x; i++)
			result += partialSums[i];
		free(partialSums);

		// TODO: Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));

		// Print info
		printf("\nKernel %d\n", kernelType);
		printf("Grid size: %d, block size: %d\n", gridSize.x, blockSize.x);
		printf("Kernel time = %f ms\n", kernelTime);
	}

	return result;
}

void checkCorrectness(int r1, int r2)
{
	printf("r1 = %i, r2 = %i \n", r1, r2);
	if (r1 == r2)
		printf("CORRECT :)\n");
	else
		printf("INCORRECT :(\n");
}

void printDeviceInfo()
{
	cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n\n");
}

int main(int argc, char ** argv)
{
	printDeviceInfo();

	// Set up input size
    int n = (1 << 24)+1;
	if (argc == 3)
    	n = atoi(argv[2]); 
    printf("Input size: %d\n", n);
	
    // Set up input data
    int * in = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        // Generate a random integer in [0, 255]
        in[i] = (int)(rand() & 0xFF);
    }

    // Reduce NOT using device
    int correctResult = reduce(in, n);

    // Reduce using device, kernel1
    dim3 blockSize(1024); // Default
    if (argc == 2 || argc == 3)
    	blockSize.x = atoi(argv[1]); 
 	
	printf("========== Testing Kernel Versions ==========\n\n");
	
	for (int v = 1; v <= 6; v++)
	{
		int result = reduce(in, n, true, blockSize, v);
		checkCorrectness(result, correctResult);
	}
	
	printf("====================================================\n");
    // Free memories
    free(in);
}
