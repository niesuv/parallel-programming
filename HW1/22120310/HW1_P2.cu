#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <stdlib.h>
#include <cuda_runtime.h>

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

void readPnm(char * fileName, 
		int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0) // In this exercise, we don't touch other types
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	int max_val;
	fscanf(f, "%i", &max_val);
	if (max_val > 255) // In this exercise, we assume 1 byte per value
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); 
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(uchar3 * pixels, int width, int height, 
		char * fileName)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

__global__ void blurImgKernel(uchar3 * inPixels, int width, int height, 
		float * filter, int filterWidth, 
		uchar3 * outPixels)
{
	// TODO
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height) return;

	int half = filterWidth / 2;
	float sumR = 0.0f;
	float sumG = 0.0f;
	float sumB = 0.0f;

	for (int fr = 0; fr < filterWidth; fr++)
	{
		for (int fc = 0; fc < filterWidth; fc++)
		{
			int imgR = y + fr - half;
			int imgC = x + fc - half;

			if (imgR < 0) imgR = 0;
			if (imgR >= height) imgR = height - 1;
			if (imgC < 0) imgC = 0;
			if (imgC >= width) imgC = width - 1;

			int idx = imgR * width + imgC;
			float w = filter[fr * filterWidth + fc];

			sumR += w * (float)inPixels[idx].x;
			sumG += w * (float)inPixels[idx].y;
			sumB += w * (float)inPixels[idx].z;
		}
	}

	int outIdx = y * width + x;
	outPixels[outIdx].x = (unsigned char)fminf(fmaxf(sumR, 0.0f), 255.0f);
	outPixels[outIdx].y = (unsigned char)fminf(fmaxf(sumG, 0.0f), 255.0f);
	outPixels[outIdx].z = (unsigned char)fminf(fmaxf(sumB, 0.0f), 255.0f);
}

void blurImg(uchar3 * inPixels, int width, int height, float * filter, int filterWidth, 
		uchar3 * outPixels,
		bool useDevice=false, dim3 blockSize=dim3(1, 1))
{
	GpuTimer timer;
	timer.Start();
	if (useDevice == false)
	{
		// TODO
		int half = filterWidth / 2;
		for (int y = 0; y < height; y++)
		{
			for (int x = 0; x < width; x++)
			{
				float sumR = 0.0f, sumG = 0.0f, sumB = 0.0f;
				for (int fr = 0; fr < filterWidth; fr++)
				{
					for (int fc = 0; fc < filterWidth; fc++)
					{
						int imgR = y + fr - half;
						int imgC = x + fc - half;
						if (imgR < 0) imgR = 0;
						if (imgR >= height) imgR = height - 1;
						if (imgC < 0) imgC = 0;
						if (imgC >= width) imgC = width - 1;

						int idx = imgR * width + imgC;
						float w = filter[fr * filterWidth + fc];
						sumR += w * (float)inPixels[idx].x;
						sumG += w * (float)inPixels[idx].y;
						sumB += w * (float)inPixels[idx].z;
					}
				}
				int outIdx = y * width + x;
				outPixels[outIdx].x = (unsigned char)fminf(fmaxf(sumR, 0.0f), 255.0f);
				outPixels[outIdx].y = (unsigned char)fminf(fmaxf(sumG, 0.0f), 255.0f);
				outPixels[outIdx].z = (unsigned char)fminf(fmaxf(sumB, 0.0f), 255.0f);
			}
		}
	}
	else // Use device
	{
		cudaDeviceProp devProp;
		
		CHECK(cudaGetDeviceProperties(&devProp, 0));
		printf("GPU name: %s\n", devProp.name);
		printf("GPU compute capability: %d.%d\n", devProp.major, devProp.minor);
		int maxThreadsPerBlock = devProp.maxThreadsPerBlock;
		int maxBlockDimX = devProp.maxThreadsDim[0];
		int maxBlockDimY = devProp.maxThreadsDim[1];


		if (blockSize.x <= 0 || blockSize.y <= 0)
		{
			fprintf(stderr, "Error: block size must be positive (got %d, %d)\n",
					blockSize.x, blockSize.y);
			exit(EXIT_FAILURE);
		}

		if (blockSize.x > maxBlockDimX || blockSize.y > maxBlockDimY)
		{
			fprintf(stderr, "Error: block size (%d, %d) exceeds device limit (%d, %d)\n",
					blockSize.x, blockSize.y, maxBlockDimX, maxBlockDimY);
			exit(EXIT_FAILURE);
		}

		if (blockSize.x * blockSize.y > maxThreadsPerBlock)
		{
			fprintf(stderr, "Error: total threads per block (%d) exceeds maxThreadsPerBlock (%d)\n",
					blockSize.x * blockSize.y, maxThreadsPerBlock);
			exit(EXIT_FAILURE);
		}
		printf("Block: %d x %d, filterWidth=%d\n", blockSize.x, blockSize.y, filterWidth);

		// TODO
		uchar3 *d_in, *d_out;
		float *d_filter;
		size_t imgSize = (size_t)width * height * sizeof(uchar3);
		size_t filterSize = (size_t)filterWidth * filterWidth * sizeof(float);

		CHECK(cudaMalloc(&d_in, imgSize));
		CHECK(cudaMalloc(&d_out, imgSize));
		CHECK(cudaMalloc(&d_filter, filterSize));

		CHECK(cudaMemcpy(d_in, inPixels, imgSize, cudaMemcpyHostToDevice));
		CHECK(cudaMemcpy(d_filter, filter, filterSize, cudaMemcpyHostToDevice));

		dim3 grid((width + blockSize.x - 1) / blockSize.x,
				  (height + blockSize.y - 1) / blockSize.y);
		blurImgKernel<<<grid, blockSize>>>(d_in, width, height, d_filter, filterWidth, d_out);
		CHECK(cudaDeviceSynchronize());
		CHECK(cudaMemcpy(outPixels, d_out, imgSize, cudaMemcpyDeviceToHost));

		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));
		CHECK(cudaFree(d_filter));
	}
	timer.Stop();
	float time = timer.Elapsed();
	printf("Processing time (%s): %f ms\n", 
    		useDevice == true? "use device" : "use host", time);
}

float computeError(uchar3 * a1, uchar3 * a2, int n)
{
	float err = 0;
	for (int i = 0; i < n; i++)
	{
		err += abs((int)a1[i].x - (int)a2[i].x);
		err += abs((int)a1[i].y - (int)a2[i].y);
		err += abs((int)a1[i].z - (int)a2[i].z);
	}
	err /= (n * 3);
	return err;
}

char * concatStr(const char * s1, const char * s2)
{
    char * result = (char *)malloc(strlen(s1) + strlen(s2) + 1);
    strcpy(result, s1);
    strcat(result, s2);
    return result;
}

int main(int argc, char ** argv)
{
	if (argc != 4 && argc != 5 && argc != 6 && argc != 7)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// Read input image file
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// Read correct output image file
	int correctWidth, correctHeight;
	uchar3 * correctOutPixels;
	readPnm(argv[3], correctWidth, correctHeight, correctOutPixels);
	if (correctWidth != width || correctHeight != height)
	{
		printf("The shape of the correct output image is invalid\n");
		return EXIT_FAILURE;
	}

	// Filter setup
	int filterWidth;
	float *filter = NULL;

	if (argc >= 5 && strstr(argv[4], ".txt") != NULL) // Có truyền file filter
	{
		FILE *f = fopen(argv[4], "r");
		if (f == NULL)
		{
			printf("Cannot open filter file %s\n", argv[4]);
			return EXIT_FAILURE;
		}

		fscanf(f, "%d", &filterWidth);
		filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
		for (int i = 0; i < filterWidth * filterWidth; i++)
			fscanf(f, "%f", &filter[i]);
		fclose(f);
	}
	else
	{
		//  default blur filter
		filterWidth = 9;
		filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
		for (int r = 0; r < filterWidth; r++)
		{
			for (int c = 0; c < filterWidth; c++)
			{
				filter[r * filterWidth + c] = 1.0f / (filterWidth * filterWidth);
			}
		}
	}

	// Blur input image using host
	uchar3 * hostOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3)); 
	blurImg(inPixels, width, height, filter, filterWidth, hostOutPixels);
	
	// Compute mean absolute error between host result and correct result
	float hostErr = computeError(hostOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", hostErr);

	// Blur input image using device
	uchar3 * deviceOutPixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	dim3 blockSize(32, 32); // Default

	if (argc == 6 && strstr(argv[4], ".txt") == NULL)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}
	else if (argc == 7)
	{
		blockSize.x = atoi(argv[5]);
		blockSize.y = atoi(argv[6]);
	}


	blurImg(inPixels, width, height, filter, filterWidth, deviceOutPixels, true, blockSize);

	// Compute mean absolute error between device result and correct result
	float deviceErr = computeError(deviceOutPixels, correctOutPixels, width * height);
	printf("Error: %f\n\n", deviceErr);

	// Compute mean absolute error between device result and host result
	float hostDeviceErr = computeError(hostOutPixels, deviceOutPixels, width * height);
	printf("Error between host and device: %f\n\n", hostDeviceErr);

	// Write results to files
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	writePnm(hostOutPixels, width, height, concatStr(outFileNameBase, "_host.pnm"));
	writePnm(deviceOutPixels, width, height, concatStr(outFileNameBase, "_device.pnm"));

	// Free memories
	free(inPixels);
	free(correctOutPixels);
	free(hostOutPixels);
	free(deviceOutPixels);
	free(filter);
}
