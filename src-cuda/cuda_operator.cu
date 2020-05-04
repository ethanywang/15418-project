#include "matrix.h"
#include "cuda_operator.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <iostream>
#define RINDEX(r, c, width) (r * width + c)
#define MBLK 4
#define LBLK 8
#define MAXSIZE 1024
// 0, simple mat mul -> 1 read, 1 write per thread
// 1, modified simple mat mul -> 1 read, 1 write per warp
// 2, blocked mat mul
#define KERNELFUNC 0

// read-only variables should be stored in fast memory
__constant__ float cuData[MAXSIZE];

// cuda kernel functions
static inline int updiv(int n, int d) {
    return (n + d - 1) / d;
}

__device__ static inline int sigmoid(float x) {
    return 1 / (1 + exp(x));
}

__device__ static inline float devTanh(float x) {
    return tanh(x);
}

// basic matrix operation
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

// matrix multiplication
__global__ void cudaSimpleMatMulKernel(int M, int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int j = blockIdx.y * blockDim.y + threadIdx.y;
    if (i >= M || j >= N) {
        return;
    }
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += dmatA[RINDEX(i, k, N)] * dmatB[RINDEX(k, j, N)];
    }
    dmatC[RINDEX(i, j, N)] = sum;
}

__global__ void cudaOptMatMulKernel(int M, int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M || j >= N) {
        return;
    }
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += dmatA[RINDEX(i, k, N)] * dmatB[RINDEX(k, j, N)];
    }
    dmatC[RINDEX(i, j, N)] = sum;
}

__global__ void cudaSimpleBlockedMatMulKernel(int M, int N, float *dmatA, float *dmatB, float *dmatC) {
    // convert (i,j) to the top left of the block
    int i = blockIdx.y * blockDim.y + threadIdx.y; 
    int j = blockIdx.x * blockDim.x + threadIdx.x; 
    i *= LBLK; 
    j *= LBLK;
  
    // keep local copy
    float subA[LBLK * LBLK];
    float subB[LBLK * LBLK];
    float subC[LBLK * LBLK];
  
    for (int bi = 0; bi < LBLK; bi++){
        for (int bj = 0; bj < LBLK; bj++){
            subC[RINDEX(bi,bj,LBLK)] = 0;
        }
    }

    for (int k = 0; k <= N-LBLK; k+=LBLK) { /* Compute product for each submatrix */
        for (int bi = 0; bi < LBLK; bi++) {
            for (int bj = 0; bj < LBLK; bj++) {
                subA[RINDEX(bi,bj,LBLK)] = dmatA[RINDEX(i+bi,k+bj,N)];
                subB[RINDEX(bi,bj,LBLK)] = dmatB[RINDEX(k+bi,j+bj,N)];
            }
        }
  
        for (int bi = 0; bi < LBLK; bi++) {
            for (int bj = 0; bj < LBLK; bj++) {
                float sum = 0.0;
                for (int bk = 0; bk < LBLK; bk++) {
                    sum += subA[RINDEX(bi,bk,LBLK)] * subB[RINDEX(bk,bj,LBLK)];
                }
                subC[RINDEX(bi,bj,LBLK)] += sum;
        }
      }
    }
  
    for (int bi = 0; bi < LBLK; bi++){
        for (int bj = 0; bj < LBLK; bj++){
            dmatC[RINDEX(i+bi,j+bj,N)] = subC[RINDEX(bi,bj,LBLK)];
        }
    }           
}

__global__ void cudaOptBlockedMatMulKernel(int M, int N, float *dmatA, float *dmatB, float *dmatC) {
    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;
  
    int bi = threadIdx.y;
    int bj = threadIdx.x;
  
    __shared__ float subA[LBLK * LBLK];
    __shared__ float subB[LBLK * LBLK];
    float sum = 0;
  
    for (int k = 0; k < N; k += LBLK) {
      subA[RINDEX(bi,bj,LBLK)] = dmatA[RINDEX(i,k+bj,N)];
      subB[RINDEX(bi,bj,LBLK)] = dmatB[RINDEX(k+bi,j,N)];
  
      __syncthreads();
  
      for (int bk = 0; bk < LBLK; bk++) {
        sum += subA[RINDEX(bi,bk,LBLK)] * subB[RINDEX(bk,bj,LBLK)];
      }
  
      __syncthreads();
    }
    dmatC[RINDEX(i,j,N)] = sum;
}

// vecotr sigmoid
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
    int size = elements * sizeof(float);
    // Allocate vectors in device memory
    float *d_A;
    cudaMalloc(&d_A, size);
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, src1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, src2, size, cudaMemcpyHostToDevice);

    // Invoke kernel
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = (elements + threadsPerBlock - 1) / threadsPerBlock;
    cudaMatAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, elements);
    cudaDeviceSynchronize();
    // copy result
    cudaMemcpy(dst, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cuMul(float *A, float *B, float *C, int M, int N) {
    // std::cout << "cuMul()\n";
    int elements = M * N;
    int size = elements * sizeof(float);
    // Allocate vectors in device memory
    float *d_A;
    cudaMalloc(&d_A, size);
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Invoke Kernel
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(M, LBLK), updiv(N, LBLK));
    cudaOptMatMulKernel<<<blocks, threadsPerBlock>>>(M, N, d_A, d_B, d_C);
    cudaDeviceSynchronize();
    // copy result
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
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
    int size = elements * sizeof(float);
    // Allocate vectors in device memory
    float *d_A;
    cudaMalloc(&d_A, size);
    float *d_B;
    cudaMalloc(&d_B, size);
    float *d_C;
    cudaMalloc(&d_C, size);

    // Copy matrix from host memory to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Invoke
    cudaMatDotKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, elements);
    cudaDeviceSynchronize();
    // copy result
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cuSigmoid(float *src, float *dst, int length) {
    // std::cout << "cuSig()\n";
    int size = length * sizeof(float);
    // Allocate vectors in device memory
    float *d_src;
    cudaMalloc(&d_src, size);
    float *d_dst;
    cudaMalloc(&d_dst, size);

    // Copy matrix from host memory to device memory
    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

    // Invoke
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(length, threadsPerBlock);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, length);
    cudaDeviceSynchronize();
    // copy result
    cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
}

void cuTanh(float *src, float *dst, int length) {
    // std::cout << "cuTanh()\n";
    int size = length * sizeof(float);
    // Allocate vectors in device memory
    float *d_src;
    cudaMalloc(&d_src, size);
    float *d_dst;
    cudaMalloc(&d_dst, size);

    // Copy matrix from host memory to device memory
    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(length, threadsPerBlock);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, length);
    cudaDeviceSynchronize();
    
    // copy result
    cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
}