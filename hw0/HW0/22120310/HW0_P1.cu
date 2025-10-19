#include <stdio.h>
#include <cuda_runtime.h>

void printDeviceInfo()
{
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);

    printf("===== GPU Information =====\n");
    printf("GPU card's name: %s\n", prop.name);
    printf("GPU computation capabilities: %d.%d\n", prop.major, prop.minor);
    printf("Maximum number of block dimensions: (%d, %d, %d)\n",
           prop.maxThreadsDim[0], prop.maxThreadsDim[1], prop.maxThreadsDim[2]);
    printf("Maximum number of grid dimensions: (%d, %d, %d)\n",
           prop.maxGridSize[0], prop.maxGridSize[1], prop.maxGridSize[2]);
    printf("Maximum size of GPU memory: %zu bytes (%.2f GB)\n",
           prop.totalGlobalMem, prop.totalGlobalMem / (1024.0 * 1024.0 * 1024.0));
    printf("Amount of constant memory: %zu bytes\n", prop.totalConstMem);
    printf("Amount of shared memory per block: %zu bytes\n", prop.sharedMemPerBlock);
    printf("Warp size: %d\n", prop.warpSize);
    printf("============================\n");
}

int main()
{
    printDeviceInfo();
    return 0;
}
