#include "matrix.h"
#include "cuda_operator.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <iostream>

#define RINDEX(r, c, width) (r * width + c)
#define MBLK 4
#define LBLK 8
#define THREADWORK 8
// cuda kernel variables
// store in fast memory

static inline int updiv(int n, int d) {
    return (n + d - 1) / d;
}

__device__ static inline int sigmoid(float x) {
    return 1 / (1 + exp(x));
}

__device__ static inline float devTanh(float x) {
    return tanh(x);
}

// kernel functions
__global__ void cudaMatAddKernel(float *src1, float *src2, float *dst, int bound) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= bound) return;
    for (int i = 0; i < THREADWORK && (offset + i) < bound; i ++) {
        dst[offset + i] = src1[offset + i] * src2[offset + i];
    }
}

__global__ void cudaMatDotKernel(float *src1, float *src2, float *dst, int bound) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= bound) return;
    for (int i = 0; i < THREADWORK && (offset + i) < bound; i ++) {
        dst[offset + i] = src1[offset + i] * src2[offset + i];
    }
}

__global__ void cudaMatTransposeKernel(float *src, float *dst, int M, int N) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M) {
        return;
    }
    for (int k = 0; k < N; k++) {
        dst[RINDEX(k, i, N)] = src[RINDEX(i, k, N)];
    }
}


// vecotr sigmoid
__global__ void cudaSigmoidKernel(float *src, float *dst, int length) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= length) return;
    for (int i = 0; i < THREADWORK && (offset + i) < length; i++) {
        dst[offset + i] = sigmoid(src[offset + i]);
    }
}

__global__ void cudaTanhKernel(float *src, float *dst, int length) {
    int offset = blockIdx.x * blockDim.x + threadIdx.x;
    if (offset >= length) return;
    for (int i = 0; i < THREADWORK && (offset + i) < length; i ++) {
        dst[offset + i] = devTanh(src[offset + i]);
    }
}

__global__ void cudaMatMulKernel(int M, int N, int R, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= R) {
        return;
    }
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += dmatA[RINDEX(i, k, N)] * dmatB[RINDEX(k, j, R)];
    }
    dmatC[RINDEX(i, j, R)] = sum;
}

__global__ void cudaReverseSign(float* src, float* dst, int length) {
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= length) return;
    for (int i = 0; i < THREADWORK && (offset + i) < length; i++) {
        dst[offset + i] = -src[offset + i];
    }
}

__global__ void cudaSubtract(float* src, float* dst, int length, float num) {
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= length) return;
    for (int i = 0; i < THREADWORK && (offset + i) < length; i++) {
        dst[offset + i] = src[offset + i] - num;
    }
}

__global__ void cudaAddition(float* src, float* dst, int length, float num) {
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= length) return;
    for (int i = 0; i < THREADWORK && (offset + i) < length; i++) {
        dst[offset + i] = src[offset + i] + num;
    }
}

void cuNumMinus(float* src, float* dst, int length, float num) {
    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    if (num == 0.0) {
        cudaReverseSign<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);    
    } else {
        cudaSubtract<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length, num);  
    }
}

void cuNumAdd(float* src, float* dst, int length, float num) {
    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    cudaAddition<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length, num);  
}

void cuAdd(float *src1, float *src2, float *dst, int M, int N) {
    int elements = M * N;
 
    // Invoke kernel
    int threadsPerBlock = MBLK;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);
    cudaMatAddKernel<<<blocksPerGrid, threadsPerBlock>>>(src1, src2, dst, elements);
}

void cuMul(float *A, float *B, float *C, int M, int N, int R) {
    // Invoke Kernel
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(M, LBLK), updiv(N, LBLK));
    cudaMatMulKernel<<<blocks, threadsPerBlock>>>(M, N, R, A, B, C);
    cudaDeviceSynchronize();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}

void cuDot(float *A, float *B, float *C, int M, int N) {
    // std::cout << "cuDot()\n";
    int elements = M * N;
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);

    // Invoke
    cudaMatDotKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, elements);
    cudaDeviceSynchronize();
}

void cuT(float *A, float *AT, int M, int N) {
    int elements = M * N;

    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);
    cudaMatTransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(A, AT, M, N);
}

void cuSigmoid(float *src, float *dst, int length) {
    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);
}

void cuTanh(float *src, float *dst, int length) {
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    cudaTanhKernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);
}