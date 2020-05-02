#include <iostream>
#include "cudaOperator.h"
#define INDEX(r, c, width) ((r) * (width) + (c))
#define LBLK 32
#define MAXSIZE 1024
// cuda kernel variables
// store in fast memory
__constant__ float cuData[MAXSIZE];

static inline int updiv(int n, int d) {
    return (n+d-1)/d;
}

// kernel functions
__global__ void cudaVecAddKernel(double* src1, double* src2, double* dst, int bound) {
    int i = threadIdx.x;
    if (i < bound){
        dst[i] = src1[i] + src2[i];
    }
}

__global__ void cudaMatMulKernel(int M, int N, double* dmatA,double* dmatB, double * dmatC) {
    int i= blockIdx.x* blockDim.x+ threadIdx.x;int j = blockIdx.y* blockDim.y+ threadIdx.y;
    if (i>= N || j >= N) return;float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += dmatA[INDEX(i,k,N)] * dmatB[INDEX(k,j,N)];
    }
    dmatC[INDEX(i,j,N)] = sum;
}

// host functions
void CudaOperator::setup(int size, double* data) {
    std::cout<<"cuda setup...\n";
    cudaMemcpy(cuData, data, sizeof(double) * size, cudaMemcpyHostToDevice);
}

void CudaOperator::cuAdd(double* src1, double *src2, double* dst, int M, int N) {
    std::cout<<"cuAdd()\n";
    int size = M * N * sizeof(double);
    // Allocate vectors in device memory
    double* d_A;
    cudaMalloc(&d_A, size);
    double* d_B;
    cudaMalloc(&d_B, size);
    double* d_C;
    cudaMalloc(&d_C, size);
    
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, src1, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, src2, size, cudaMemcpyHostToDevice);
    
    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid = (size + threadsPerBlock - 1) / threadsPerBlock;
    cudaVecAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, size);
    
    // copy result
    cudaMemcpy(dst, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CudaOperator::cuMul(double* A, double* B, double* C, int M, int N) {
    std::cout<<"cuMul\n";
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(M, LBLK), updiv(N, LBLK));
    cudaMatMulKernel<<<blocks, threadsPerBlock>>>(M, N, A, B, C);
}

void CudaOperator::cuDot(double* A, double* B, double* C, int M, int N) {
    // dim3 threadsPerBlock(LBLK, LBLK);
    // dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    // cudaSimpleKernel<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
    std::cout<<"cuDot\n";
}