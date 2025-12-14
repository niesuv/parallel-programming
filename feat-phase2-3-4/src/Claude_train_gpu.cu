#ifdef USE_ADVANCED_OPTIMIZATIONS

#include "cuda_utils.h"
#include "gpu_layer.h"

#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#include <cublasLt.h>
#include <cudnn.h>
#include <cufft.h>

// ============================================================================
// 1. MIXED PRECISION (FP16 + FP32)
// ============================================================================

// Convert FP32 to FP16
__global__ void float_to_half_kernel(const float* input, __half* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __float2half(input[idx]);
    }
}

// Convert FP16 to FP32
__global__ void half_to_float_kernel(const __half* input, float* output, size_t n) {
    size_t idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        output[idx] = __half2float(input[idx]);
    }
}

// FP16 Conv2D using Tensor Cores (requires compute capability 7.0+)
__global__ void conv2d_fp16_tensor_core_kernel(
    const __half* __restrict__ input,
    const __half* __restrict__ weights,
    const __half* __restrict__ bias,
    __half* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding) {
    
    // Use wmma for tensor core operations
#if __CUDA_ARCH__ >= 700
    #include <mma.h>
    using namespace nvcuda;
    
    // Tensor Core matrix multiply accumulate
    // This is simplified - real implementation needs proper tiling
    const int WMMA_M = 16;
    const int WMMA_N = 16;
    const int WMMA_K = 16;
    
    wmma::fragment<wmma::matrix_a, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> a_frag;
    wmma::fragment<wmma::matrix_b, WMMA_M, WMMA_N, WMMA_K, __half, wmma::row_major> b_frag;
    wmma::fragment<wmma::accumulator, WMMA_M, WMMA_N, WMMA_K, __half> acc_frag;
    
    wmma::fill_fragment(acc_frag, __float2half(0.0f));
    
    // Load and multiply-accumulate
    // wmma::load_matrix_sync(a_frag, input, WMMA_K);
    // wmma::load_matrix_sync(b_frag, weights, WMMA_N);
    // wmma::mma_sync(acc_frag, a_frag, b_frag, acc_frag);
    
    // Store result
    // wmma::store_matrix_sync(output, acc_frag, WMMA_N, wmma::mem_row_major);
#endif
}

// ============================================================================
// 2. IM2COL + GEMM APPROACH
// ============================================================================

// Im2Col: Convert image patches to columns for matrix multiplication
__global__ void im2col_kernel(
    const float* __restrict__ data_im,
    float* __restrict__ data_col,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = height_col * width_col * channels * kernel_h * kernel_w;
    
    if (index >= total) return;
    
    // Calculate indices
    int w_out = index % width_col;
    int h_index = index / width_col;
    int h_out = h_index % height_col;
    int channel_in = h_index / height_col;
    int channel_out = channel_in * kernel_h * kernel_w;
    int h_in = h_out * stride_h - pad_h;
    int w_in = w_out * stride_w - pad_w;
    
    float* data_col_ptr = data_col + (channel_out * height_col + h_out) * width_col + w_out;
    const float* data_im_ptr = data_im + (channel_in * height + h_in) * width + w_in;
    
    for (int i = 0; i < kernel_h; ++i) {
        for (int j = 0; j < kernel_w; ++j) {
            int h = h_in + i * dilation_h;
            int w = w_in + j * dilation_w;
            
            *data_col_ptr = (h >= 0 && w >= 0 && h < height && w < width) ?
                data_im_ptr[i * dilation_h * width + j * dilation_w] : 0.0f;
            
            data_col_ptr += height_col * width_col;
        }
    }
}

// Col2Im: Inverse of Im2Col for backward pass
__global__ void col2im_kernel(
    const float* __restrict__ data_col,
    float* __restrict__ data_im,
    int channels, int height, int width,
    int kernel_h, int kernel_w,
    int pad_h, int pad_w,
    int stride_h, int stride_w,
    int dilation_h, int dilation_w,
    int height_col, int width_col) {
    
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    int total = channels * height * width;
    
    if (index >= total) return;
    
    int w = index % width + pad_w;
    int h = (index / width) % height + pad_h;
    int c = index / (width * height);
    
    int kernel_extent_w = (kernel_w - 1) * dilation_w + 1;
    int kernel_extent_h = (kernel_h - 1) * dilation_h + 1;
    
    int w_col_start = (w < kernel_extent_w) ? 0 : (w - kernel_extent_w) / stride_w + 1;
    int w_col_end = min(w / stride_w + 1, width_col);
    int h_col_start = (h < kernel_extent_h) ? 0 : (h - kernel_extent_h) / stride_h + 1;
    int h_col_end = min(h / stride_h + 1, height_col);
    
    float val = 0.0f;
    
    for (int h_col = h_col_start; h_col < h_col_end; h_col += 1) {
        for (int w_col = w_col_start; w_col < w_col_end; w_col += 1) {
            int h_k = (h - h_col * stride_h);
            int w_k = (w - w_col * stride_w);
            
            if (h_k % dilation_h == 0 && w_k % dilation_w == 0) {
                h_k /= dilation_h;
                w_k /= dilation_w;
                
                int data_col_index = (((c * kernel_h + h_k) * kernel_w + w_k) *
                                     height_col + h_col) * width_col + w_col;
                val += data_col[data_col_index];
            }
        }
    }
    
    data_im[index] = val;
}

// ============================================================================
// 3. WINOGRAD CONVOLUTION (for 3x3 kernels)
// ============================================================================

// Winograd F(2x2, 3x3) - compute 2x2 output from 4x4 input using 3x3 kernel
// Requires only 16 multiplications instead of 36

__device__ void winograd_transform_input(const float* input, float* transformed, 
                                         int stride_h, int stride_w) {
    // BT * d * B where B is the transform matrix
    // Simplified version - full implementation is more complex
    
    // Transform matrix B for F(2x2, 3x3)
    // B = [1,  0, -1,  0]
    //     [0,  1,  1,  0]
    //     [0, -1,  1,  0]
    //     [0,  1,  0, -1]
    
    float d[16]; // 4x4 input tile
    
    // Load input tile
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        #pragma unroll
        for (int j = 0; j < 4; ++j) {
            d[i * 4 + j] = input[i * stride_h + j * stride_w];
        }
    }
    
    // Apply BT * d * B transformation
    float temp[16];
    
    // First: BT * d
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        temp[0 * 4 + i] = d[0 * 4 + i] - d[2 * 4 + i];
        temp[1 * 4 + i] = d[1 * 4 + i] + d[2 * 4 + i];
        temp[2 * 4 + i] = -d[1 * 4 + i] + d[2 * 4 + i];
        temp[3 * 4 + i] = d[1 * 4 + i] - d[3 * 4 + i];
    }
    
    // Second: (BT * d) * B
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        transformed[i * 4 + 0] = temp[i * 4 + 0] - temp[i * 4 + 2];
        transformed[i * 4 + 1] = temp[i * 4 + 1] + temp[i * 4 + 2];
        transformed[i * 4 + 2] = -temp[i * 4 + 1] + temp[i * 4 + 2];
        transformed[i * 4 + 3] = temp[i * 4 + 1] - temp[i * 4 + 3];
    }
}

__device__ void winograd_transform_filter(const float* filter, float* transformed) {
    // G * g * GT where G is the filter transform matrix
    // Similar transformation for 3x3 kernel
    
    float g[9]; // 3x3 kernel
    #pragma unroll
    for (int i = 0; i < 9; ++i) {
        g[i] = filter[i];
    }
    
    // Transform matrix G for F(2x2, 3x3)
    // G = [1,    0,    0]
    //     [1/2,  1/2,  1/2]
    //     [1/2, -1/2,  1/2]
    //     [0,    0,    1]
    
    float temp[12]; // 4x3
    
    #pragma unroll
    for (int i = 0; i < 3; ++i) {
        temp[0 * 3 + i] = g[0 * 3 + i];
        temp[1 * 3 + i] = 0.5f * (g[0 * 3 + i] + g[1 * 3 + i] + g[2 * 3 + i]);
        temp[2 * 3 + i] = 0.5f * (g[0 * 3 + i] - g[1 * 3 + i] + g[2 * 3 + i]);
        temp[3 * 3 + i] = g[2 * 3 + i];
    }
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        transformed[i * 4 + 0] = temp[i * 3 + 0];
        transformed[i * 4 + 1] = 0.5f * (temp[i * 3 + 0] + temp[i * 3 + 1] + temp[i * 3 + 2]);
        transformed[i * 4 + 2] = 0.5f * (temp[i * 3 + 0] - temp[i * 3 + 1] + temp[i * 3 + 2]);
        transformed[i * 4 + 3] = temp[i * 3 + 2];
    }
}

__device__ void winograd_transform_output(const float* transformed, float* output,
                                          int stride_h, int stride_w) {
    // AT * m * A where A is the output transform matrix
    // A = [1,  1,  1,  0]
    //     [0,  1, -1, -1]
    
    float m[16];
    #pragma unroll
    for (int i = 0; i < 16; ++i) {
        m[i] = transformed[i];
    }
    
    float temp[8]; // 2x4
    
    #pragma unroll
    for (int i = 0; i < 4; ++i) {
        temp[0 * 4 + i] = m[0 * 4 + i] + m[1 * 4 + i] + m[2 * 4 + i];
        temp[1 * 4 + i] = m[1 * 4 + i] - m[2 * 4 + i] - m[3 * 4 + i];
    }
    
    float result[4]; // 2x2
    #pragma unroll
    for (int i = 0; i < 2; ++i) {
        result[i * 2 + 0] = temp[i * 4 + 0] + temp[i * 4 + 1] + temp[i * 4 + 2];
        result[i * 2 + 1] = temp[i * 4 + 1] - temp[i * 4 + 2] - temp[i * 4 + 3];
    }
    
    output[0] = result[0];
    output[stride_w] = result[1];
    output[stride_h] = result[2];
    output[stride_h + stride_w] = result[3];
}

__global__ void winograd_conv2d_3x3_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w) {
    
    // Each thread processes one 2x2 output tile
    int tile_x = blockIdx.x * blockDim.x + threadIdx.x;
    int tile_y = blockIdx.y * blockDim.y + threadIdx.y;
    int oc = blockIdx.z % out_c;
    int n = blockIdx.z / out_c;
    
    if (tile_x >= (out_w + 1) / 2 || tile_y >= (out_h + 1) / 2 || n >= batch_size)
        return;
    
    float result[4] = {0.0f}; // 2x2 output tile
    
    // For each input channel
    for (int ic = 0; ic < in_c; ++ic) {
        // Transform input
        float d_transformed[16];
        const float* input_ptr = input + ((n * in_c + ic) * in_h + tile_y * 2) * in_w + tile_x * 2;
        winograd_transform_input(input_ptr, d_transformed, in_w, 1);
        
        // Transform filter
        float g_transformed[16];
        const float* weight_ptr = weights + (oc * in_c + ic) * 9;
        winograd_transform_filter(weight_ptr, g_transformed);
        
        // Element-wise multiplication in transformed domain
        float m[16];
        #pragma unroll
        for (int i = 0; i < 16; ++i) {
            m[i] = d_transformed[i] * g_transformed[i];
        }
        
        // Transform output
        float temp_out[4];
        winograd_transform_output(m, temp_out, 1, 1);
        
        #pragma unroll
        for (int i = 0; i < 4; ++i) {
            result[i] += temp_out[i];
        }
    }
    
    // Add bias and write output
    float b = bias[oc];
    int out_y = tile_y * 2;
    int out_x = tile_x * 2;
    
    if (out_y < out_h && out_x < out_w)
        output[((n * out_c + oc) * out_h + out_y) * out_w + out_x] = result[0] + b;
    if (out_y < out_h && out_x + 1 < out_w)
        output[((n * out_c + oc) * out_h + out_y) * out_w + out_x + 1] = result[1] + b;
    if (out_y + 1 < out_h && out_x < out_w)
        output[((n * out_c + oc) * out_h + out_y + 1) * out_w + out_x] = result[2] + b;
    if (out_y + 1 < out_h && out_x + 1 < out_w)
        output[((n * out_c + oc) * out_h + out_y + 1) * out_w + out_x + 1] = result[3] + b;
}

// ============================================================================
// 4. FFT CONVOLUTION (for large kernels)
// ============================================================================

void gpu_conv2d_fft(const GPUTensor4D& input,
                    const float* d_weights,
                    const float* d_bias,
                    GPUTensor4D& output,
                    int in_c, int out_c, int k,
                    int stride, int padding) {
    
    // FFT-based convolution is efficient for large kernels
    // Using cuFFT library
    
    cufftHandle plan_forward, plan_inverse;
    cufftComplex *d_input_freq, *d_kernel_freq, *d_output_freq;
    
    int fft_h = input.h + k - 1;
    int fft_w = input.w + k - 1;
    
    // Allocate frequency domain buffers
    size_t freq_size = fft_h * (fft_w / 2 + 1) * sizeof(cufftComplex);
    CUDA_CHECK(cudaMalloc(&d_input_freq, freq_size));
    CUDA_CHECK(cudaMalloc(&d_kernel_freq, freq_size));
    CUDA_CHECK(cudaMalloc(&d_output_freq, freq_size));
    
    // Create FFT plans
    cufftPlan2d(&plan_forward, fft_h, fft_w, CUFFT_R2C);
    cufftPlan2d(&plan_inverse, fft_h, fft_w, CUFFT_C2R);
    
    // For each batch and channel pair
    for (int n = 0; n < input.n; ++n) {
        for (int oc = 0; oc < out_c; ++oc) {
            for (int ic = 0; ic < in_c; ++ic) {
                // Transform input to frequency domain
                const float* input_ptr = input.d_data + ((n * in_c + ic) * input.h * input.w);
                cufftExecR2C(plan_forward, (cufftReal*)input_ptr, d_input_freq);
                
                // Transform kernel to frequency domain
                const float* kernel_ptr = d_weights + (oc * in_c + ic) * k * k;
                cufftExecR2C(plan_forward, (cufftReal*)kernel_ptr, d_kernel_freq);
                
                // Complex multiplication in frequency domain
                // d_output_freq = d_input_freq * d_kernel_freq
                
                // Transform back to spatial domain
                cufftExecC2R(plan_inverse, d_output_freq, (cufftReal*)output.d_data);
            }
        }
    }
    
    // Cleanup
    cufftDestroy(plan_forward);
    cufftDestroy(plan_inverse);
    CUDA_CHECK(cudaFree(d_input_freq));
    CUDA_CHECK(cudaFree(d_kernel_freq));
    CUDA_CHECK(cudaFree(d_output_freq));
}

// ============================================================================
// 5. PERSISTENT KERNELS (Grid-Persistent Approach)
// ============================================================================

__global__ void persistent_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int total_work_items,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding) {
    
    // Grid-stride loop: kernel stays resident on GPU
    for (int idx = blockIdx.x * blockDim.x + threadIdx.x;
         idx < total_work_items;
         idx += blockDim.x * gridDim.x) {
        
        // Decode work item
        int ow = idx % out_w;
        int temp = idx / out_w;
        int oh = temp % out_h;
        temp = temp / out_h;
        int oc = temp % out_c;
        int n = temp / out_c;
        
        float sum = bias[oc];
        
        // Compute convolution
        #pragma unroll 4
        for (int ic = 0; ic < in_c; ++ic) {
            #pragma unroll
            for (int kh = 0; kh < k; ++kh) {
                #pragma unroll
                for (int kw = 0; kw < k; ++kw) {
                    int ih = oh * stride + kh - padding;
                    int iw = ow * stride + kw - padding;
                    
                    if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                        size_t in_idx = ((static_cast<size_t>(n) * in_c + ic) * in_h + ih) * in_w + iw;
                        size_t w_idx = ((static_cast<size_t>(oc) * in_c + ic) * k + kh) * k + kw;
                        sum += input[in_idx] * weights[w_idx];
                    }
                }
            }
        }
        
        output[idx] = sum;
    }
}

// ============================================================================
// 6. CUDNN INTEGRATION (Best Performance)
// ============================================================================

class CudnnConvolution {
private:
    cudnnHandle_t cudnn;
    cudnnTensorDescriptor_t input_desc, output_desc;
    cudnnFilterDescriptor_t filter_desc;
    cudnnConvolutionDescriptor_t conv_desc;
    cudnnConvolutionFwdAlgo_t algo;
    void* workspace;
    size_t workspace_size;
    
public:
    CudnnConvolution() {
        cudnnCreate(&cudnn);
        cudnnCreateTensorDescriptor(&input_desc);
        cudnnCreateTensorDescriptor(&output_desc);
        cudnnCreateFilterDescriptor(&filter_desc);
        cudnnCreateConvolutionDescriptor(&conv_desc);
        workspace = nullptr;
        workspace_size = 0;
    }
    
    ~CudnnConvolution() {
        if (workspace) cudaFree(workspace);
        cudnnDestroyConvolutionDescriptor(conv_desc);
        cudnnDestroyFilterDescriptor(filter_desc);
        cudnnDestroyTensorDescriptor(output_desc);
        cudnnDestroyTensorDescriptor(input_desc);
        cudnnDestroy(cudnn);
    }
    
    void setup(int n, int c, int h, int w,
               int k, int out_c, int stride, int padding) {
        
        // Set input descriptor
        cudnnSetTensor4dDescriptor(input_desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   n, c, h, w);
        
        // Set filter descriptor
        cudnnSetFilter4dDescriptor(filter_desc,
                                   CUDNN_DATA_FLOAT,
                                   CUDNN_TENSOR_NCHW,
                                   out_c, c, k, k);
        
        // Set convolution descriptor
        cudnnSetConvolution2dDescriptor(conv_desc,
                                       padding, padding,
                                       stride, stride,
                                       1, 1,
                                       CUDNN_CROSS_CORRELATION,
                                       CUDNN_DATA_FLOAT);
        
        // Enable Tensor Cores if available
        cudnnSetConvolutionMathType(conv_desc, CUDNN_TENSOR_OP_MATH);
        
        // Get output dimensions
        int out_n, out_c_dim, out_h, out_w;
        cudnnGetConvolution2dForwardOutputDim(conv_desc, input_desc, filter_desc,
                                             &out_n, &out_c_dim, &out_h, &out_w);
        
        // Set output descriptor
        cudnnSetTensor4dDescriptor(output_desc,
                                   CUDNN_TENSOR_NCHW,
                                   CUDNN_DATA_FLOAT,
                                   out_n, out_c_dim, out_h, out_w);
        
        // Find best algorithm
        cudnnGetConvolutionForwardAlgorithm_v7(
            cudnn, input_desc, filter_desc, conv_desc, output_desc,
            1, 0, nullptr, &algo);
        
        // Get workspace size
        cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, input_desc, filter_desc, conv_desc, output_desc,
            algo, &workspace_size);
        
        if (workspace_size > 0) {
            cudaMalloc(&workspace, workspace_size);
        }
    }
    
    void forward(const float* input, const float* weights, const float* bias,
                float* output, float alpha = 1.0f, float beta = 0.0f) {
        
        // Convolution
        cudnnConvolutionForward(cudnn,
                               &alpha,
                               input_desc, input,
                               filter_desc, weights,
                               conv_desc,
                               algo,
                               workspace, workspace_size,
                               &beta,
                               output_desc, output);
        
        // Add bias
        if (bias) {
            cudnnTensorDescriptor_t bias_desc;
            cudnnCreateTensorDescriptor(&bias_desc);
            
            int out_n, out_c, out_h, out_w;
            cudnnDataType_t dtype;
            cudnnTensorFormat_t format;
            cudnnGetTensor4dDescriptor(output_desc, &dtype, &out_n, &out_c, &out_h, &out_w,
                                      nullptr, nullptr, nullptr, nullptr);
            
            cudnnSetTensor4dDescriptor(bias_desc, CUDNN_TENSOR_NCHW, CUDNN_DATA_FLOAT,
                                      1, out_c, 1, 1);
            
            alpha = 1.0f;
            beta = 1.0f;
            cudnnAddTensor(cudnn, &alpha, bias_desc, bias, &beta, output_desc, output);
            
            cudnnDestroyTensorDescriptor(bias_desc);
        }
    }
};

// ============================================================================
// 7. WRAPPER FUNCTIONS
// ============================================================================

void gpu_conv2d_im2col_gemm(const GPUTensor4D& input,
                           const float* d_weights,
                           const float* d_bias,
                           GPUTensor4D& output,
                           int in_c, int out_c, int k,
                           int stride, int padding) {
    
    int out_h = (input.h + 2 * padding - k) / stride + 1;
    int out_w = (input.w + 2 * padding - k) / stride + 1;
    
    // Allocate col buffer
    float* d_col;
    size_t col_size = in_c * k * k * out_h * out_w * sizeof(float);
    CUDA_CHECK(cudaMalloc(&d_col, col_size));
    
    cublasHandle_t cublas;
    cublasCreate(&cublas);
    
    for (int n = 0; n < input.n; ++n) {
        // Im2Col
        int threads = 256;
        int blocks = (out_h * out_w * in_c * k * k + threads - 1) / threads;
        
        im2col_kernel<<<blocks, threads>>>(
            input.d_data + n * in_c * input.h * input.w,
            d_col,
            in_c, input.h, input.w,
            k, k, padding, padding,
            stride, stride, 1, 1,
            out_h, out_w);
        
        // GEMM: output = weights * col + bias
        // weights: [out_c, in_c*k*k]
        // col: [in_c*k*k, out_h*out_w]
        // output: [out_c, out_h*out_w]
        
        float alpha = 1.0f, beta = 0.0f;
        int M = out_c;
        int N = out_h * out_w;
        int K = in_c * k * k;
        
        cublasSgemm(cublas,
                   CUBLAS_OP_N, CUBLAS_OP_N,
                   N, M, K,
                   &alpha,
                   d_col, N,
                   d_weights, K,
                   &beta,
                   output.d_data + n * out_c * out_h * out_w, N);
        
        // Add bias (broadcast)
        if (d_bias) {
            // Use custom kernel or cudnn
        }
    }
    
    cublasDestroy(cublas);
    CUDA_CHECK(cudaFree(d_col));
}

// ============================================================================
// 8. MIXED PRECISION TRAINING with AUTOMATIC LOSS SCALING
// ============================================================================

class MixedPrecisionTrainer {
private:
    float loss_scale;
    float loss_scale_factor;
    int loss_scale_window;
    int unskipped_steps;
    
public:
    MixedPrecisionTrainer(float init_scale = 65536.0f) 
        : loss_scale(init_scale)
        , loss_scale_factor(2.0f)
        , loss_scale_window(2000)
        , unskipped_steps(0) {}
    
    // Scale gradients before backward pass
    void scale_loss(float* d_loss, size_t n) {
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scale_kernel<<<blocks, threads>>>(d_loss, loss_scale, n);
    }
    
    // Unscale gradients after backward pass and check for overflow
    bool unscale_and_check_gradients(float* d_grads, size_t n) {
        float inv_scale = 1.0f / loss_scale;
        
        // Unscale
        int threads = 256;
        int blocks = (n + threads - 1) / threads;
        scale_kernel<<<blocks, threads>>>(d_grads, inv_scale, n);
        
        // Check for inf/nan
        bool has_inf_nan = check_inf_nan(d_grads, n);
        
        if (has_inf_nan) {
            // Reduce loss scale
            loss_scale = fmaxf(loss_scale / loss_scale_factor, 1.0f);
            unskipped_steps = 0;
            return false;
        } else {
            unskipped_steps++;
            // Increase loss scale periodically
            if (unskipped_steps >= loss_scale_window) {
                loss_scale = fminf(loss_scale * loss_scale_factor, 65536.0f);
                unskipped_steps = 0;
            }
            return true;
        }
    }
    
private:
    __global__ static void scale_kernel(float* data, float scale, size_t n) {
        int idx = blockIdx.x * blockDim.x + threadIdx.x;
        if (idx < n) {
            data[idx] *= scale;
        }
    }
    
    bool check_inf_nan(float* d_data, size_t n) {
        // Implement parallel reduction to check for inf/nan
        // Return true if any element is inf or nan
        return false; // Placeholder
    }
};

// ============================================================================
// 9. DEPTHWISE SEPARABLE CONVOLUTION (MobileNet style)
// ============================================================================

__global__ void depthwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int channels, int in_h, int in_w,
    int out_h, int out_w, int k, int stride, int padding) {
    
    int ow = blockIdx.x * blockDim.x + threadIdx.x;
    int oh = blockIdx.y * blockDim.y + threadIdx.y;
    int c_n = blockIdx.z;
    int c = c_n % channels;
    int n = c_n / channels;
    
    if (ow >= out_w || oh >= out_h || n >= batch_size) return;
    
    float sum = bias[c];
    
    // Only process one channel (depthwise)
    const size_t in_offset = ((n * channels + c) * in_h) * in_w;
    const size_t w_offset = c * k * k;
    
    #pragma unroll
    for (int kh = 0; kh < k; ++kh) {
        #pragma unroll
        for (int kw = 0; kw < k; ++kw) {
            int ih = oh * stride + kh - padding;
            int iw = ow * stride + kw - padding;
            
            if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                sum += input[in_offset + ih * in_w + iw] * 
                       weights[w_offset + kh * k + kw];
            }
        }
    }
    
    size_t out_idx = ((n * channels + c) * out_h + oh) * out_w + ow;
    output[out_idx] = sum;
}

__global__ void pointwise_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_c, int spatial_size, int out_c) {
    
    int spatial_idx = blockIdx.x * blockDim.x + threadIdx.x;
    int oc = blockIdx.y;
    int n = blockIdx.z;
    
    if (spatial_idx >= spatial_size || n >= batch_size) return;
    
    float sum = bias[oc];
    
    #pragma unroll 8
    for (int ic = 0; ic < in_c; ++ic) {
        size_t in_idx = (n * in_c + ic) * spatial_size + spatial_idx;
        size_t w_idx = oc * in_c + ic;
        sum += input[in_idx] * weights[w_idx];
    }
    
    size_t out_idx = (n * out_c + oc) * spatial_size + spatial_idx;
    output[out_idx] = sum;
}

// ============================================================================
// 10. GROUPED CONVOLUTION (ResNeXt style)
// ============================================================================

__global__ void grouped_conv2d_kernel(
    const float* __restrict__ input,
    const float* __restrict__ weights,
    const float* __restrict__ bias,
    float* __restrict__ output,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding, int groups) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_c * out_h * out_w;
    
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int temp = idx / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int oc = temp % out_c;
    int n = temp / out_c;
    
    int group = oc / (out_c / groups);
    int in_c_per_group = in_c / groups;
    int out_c_per_group = out_c / groups;
    
    int ic_start = group * in_c_per_group;
    int ic_end = ic_start + in_c_per_group;
    
    float sum = bias[oc];
    
    for (int ic = ic_start; ic < ic_end; ++ic) {
        #pragma unroll
        for (int kh = 0; kh < k; ++kh) {
            #pragma unroll
            for (int kw = 0; kw < k; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    size_t in_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                    size_t w_idx = ((oc * in_c_per_group + (ic - ic_start)) * k + kh) * k + kw;
                    sum += input[in_idx] * weights[w_idx];
                }
            }
        }
    }
    
    output[idx] = sum;
}

// ============================================================================
// 11. INT8 QUANTIZED INFERENCE
// ============================================================================

__global__ void quantize_fp32_to_int8_kernel(
    const float* __restrict__ input,
    int8_t* __restrict__ output,
    float scale, int zero_point, size_t n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    float val = input[idx] / scale + zero_point;
    val = fminf(fmaxf(val, -128.0f), 127.0f);
    output[idx] = static_cast<int8_t>(rintf(val));
}

__global__ void dequantize_int8_to_fp32_kernel(
    const int8_t* __restrict__ input,
    float* __restrict__ output,
    float scale, int zero_point, size_t n) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= n) return;
    
    output[idx] = (static_cast<float>(input[idx]) - zero_point) * scale;
}

__global__ void conv2d_int8_kernel(
    const int8_t* __restrict__ input,
    const int8_t* __restrict__ weights,
    const int32_t* __restrict__ bias,
    int8_t* __restrict__ output,
    float input_scale, float weight_scale, float output_scale,
    int input_zero, int weight_zero, int output_zero,
    int batch_size, int in_c, int in_h, int in_w,
    int out_c, int out_h, int out_w,
    int k, int stride, int padding) {
    
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = batch_size * out_c * out_h * out_w;
    
    if (idx >= total) return;
    
    int ow = idx % out_w;
    int temp = idx / out_w;
    int oh = temp % out_h;
    temp = temp / out_h;
    int oc = temp % out_c;
    int n = temp / out_c;
    
    int32_t sum = bias[oc];
    
    for (int ic = 0; ic < in_c; ++ic) {
        for (int kh = 0; kh < k; ++kh) {
            for (int kw = 0; kw < k; ++kw) {
                int ih = oh * stride + kh - padding;
                int iw = ow * stride + kw - padding;
                
                if (ih >= 0 && ih < in_h && iw >= 0 && iw < in_w) {
                    size_t in_idx = ((n * in_c + ic) * in_h + ih) * in_w + iw;
                    size_t w_idx = ((oc * in_c + ic) * k + kh) * k + kw;
                    
                    int32_t in_val = static_cast<int32_t>(input[in_idx]) - input_zero;
                    int32_t w_val = static_cast<int32_t>(weights[w_idx]) - weight_zero;
                    sum += in_val * w_val;
                }
            }
        }
    }
    
    // Requantize: sum * (input_scale * weight_scale / output_scale) + output_zero
    float requant_scale = (input_scale * weight_scale) / output_scale;
    float result = sum * requant_scale + output_zero;
    result = fminf(fmaxf(result, -128.0f), 127.0f);
    
    output[idx] = static_cast<int8_t>(rintf(result));
}

// ============================================================================
// PERFORMANCE COMPARISON TABLE
// ============================================================================

/*
OPTIMIZATION TECHNIQUE COMPARISON:

1. BASELINE (Naive Implementation)
   - Speed: 1x (reference)
   - Memory: High
   - Precision: FP32
   - Use case: Small models, debugging

2. TILED SHARED MEMORY
   - Speed: 2-3x
   - Memory: Reduced global access
   - Precision: FP32
   - Use case: Medium models

3. IM2COL + GEMM (cuBLAS)
   - Speed: 5-10x
   - Memory: High (col buffer)
   - Precision: FP32/FP16
   - Use case: Standard training

4. WINOGRAD (3x3 only)
   - Speed: 2-4x (vs optimized)
   - Memory: Low
   - Precision: FP32 (numerical issues possible)
   - Use case: 3x3 conv inference

5. FFT CONVOLUTION (large kernels)
   - Speed: Fast for k > 9
   - Memory: High
   - Precision: FP32
   - Use case: k >= 11x11

6. CUDNN (Best Overall)
   - Speed: 10-20x
   - Memory: Optimized
   - Precision: FP32/FP16/INT8/TF32
   - Use case: Production (NVIDIA GPUs)
   - Features: Tensor Cores, autotuning

7. MIXED PRECISION (FP16)
   - Speed: 2-3x on Tensor Cores
   - Memory: 50% reduction
   - Precision: FP16 (with FP32 master weights)
   - Use case: Modern GPUs (V100+)

8. INT8 QUANTIZATION
   - Speed: 4-5x
   - Memory: 75% reduction
   - Precision: INT8 (1-2% accuracy loss)
   - Use case: Inference on edge devices

9. PERSISTENT KERNELS
   - Speed: 1.2-1.5x
   - Memory: Same
   - Precision: Any
   - Use case: Reduce launch overhead

10. DEPTHWISE SEPARABLE
    - Speed: 8-9x vs standard conv
    - Memory: Much lower
    - Precision: FP32/FP16
    - Use case: Mobile models (MobileNet)

RECOMMENDATIONS:
- Training: cuDNN + Mixed Precision + Tensor Cores
- Inference (Server): cuDNN + INT8 + TensorRT
- Inference (Mobile): Depthwise Separable + INT8
- Research: IM2COL + cuBLAS for flexibility
- 3x3 only: Consider Winograd
- Large kernels (11x11+): FFT
*/

#endif // USE_ADVANCED_OPTIMIZATIONS