#include "matrix.h"
#include "cuda_operator.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <iostream>

#define INDEX(r, c, width) ((r) * (width) + (c))
#define MBLK 16
#define LBLK 32
#define MAXSIZE 1024
// cuda kernel variables
// store in fast memory

__constant__ float cuData[MAXSIZE];

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
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bound) {
        dst[i] = src1[i] + src2[i];
    }
}

__global__ void cudaMatDotKernel(float *src1, float *src2, float *dst, int bound) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < bound) {
        dst[i] = src1[i] * src2[i];
    }
}

__global__ void cudaMatMulKernel(int M, int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) {
        return;
    }
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += dmatA[INDEX(i, k, N)] * dmatB[INDEX(k, j, N)];
    }
    dmatC[INDEX(i, j, N)] = sum;
}

__global__ void cudaSigmoidKernel(float *src, float *dst, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > length) return;
    dst[i] = sigmoid(src[i]);
}

__global__ void cudaTanhKernel(float *src, float *dst, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > length) return;
    dst[i] = devTanh(src[i]);
}

void cuAdd(float *src1, float *src2, float *dst, int M, int N) {
    // std::cout << "cuAdd()\n";
    int elements = M * N;
 
    // Invoke kernel
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = (elements + threadsPerBlock - 1) / threadsPerBlock;
    cudaMatAddKernel<<<blocksPerGrid, threadsPerBlock>>>(src1, src2, dst, elements);
    cudaDeviceSynchronize();
}

void cuMul(float *A, float *B, float *C, int M, int N) {
    // Invoke Kernel
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(M, LBLK), updiv(N, LBLK));
    cudaMatMulKernel<<<blocks, threadsPerBlock>>>(M, N, A, B, C);
    cudaDeviceSynchronize();

    cudaError_t errCode = cudaPeekAtLastError();
    if (errCode != cudaSuccess) {
        fprintf(stderr, "WARNING: A CUDA error occured: code=%d, %s\n", errCode, cudaGetErrorString(errCode));
    }
}

void cuDot(float *A, float *B, float *C, int M, int N) {
    // std::cout << "cuDot()\n";
    int elements = M * N;
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(elements, threadsPerBlock);

    // Invoke
    cudaMatDotKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, elements);
    cudaDeviceSynchronize();
}

void cuSigmoid(float *src, float *dst, int length) {
    // Invoke
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(length, threadsPerBlock);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);
    cudaDeviceSynchronize();
}

void cuTanh(float *src, float *dst, int length) {
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(length, threadsPerBlock);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);
    cudaDeviceSynchronize();
}