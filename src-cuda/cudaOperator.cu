#include <iostream>
#include "matrix.h"
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

__device__ static inline int sigmoid(double x) {
    return 1 / (1 + exp(x));
}

__device__ static inline double devTanh(double x) {
    return tanh(x);
}

// kernel functions
__global__ void cudaMatAddKernel(double* src1, double* src2, double* dst, int bound) {
    int i = blockIdx.x * blockDim.x+ threadIdx.x;
    if (i < bound){
        dst[i] = src1[i] + src2[i];
    }
}

__global__ void cudaMatDotKernel(double* src1, double* src2, double* dst, int bound) {
    int i = blockIdx.x * blockDim.x+ threadIdx.x;
    if (i < bound){
        dst[i] = src1[i] * src2[i];
    }
}

__global__ void cudaMatMulKernel(int M, int N, double* dmatA,double* dmatB, double * dmatC) {
    int i = blockIdx.x * blockDim.x+ threadIdx.x;
    int j = blockIdx.y * blockDim.y+ threadIdx.y;
    if (i>= M || j >= N) {
        return;
    }
    float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += dmatA[INDEX(i,k,N)] * dmatB[INDEX(k,j,N)];
    }
    dmatC[INDEX(i,j,N)] = sum;
}

__global__ void cudaSigmoidKernel(double* src, double* dst, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > length) return;
    dst[i] = sigmoid(src[i]);
}

__global__ void cudaTanhKernel(double* src, double* dst, int length) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > length) return;
    dst[i] = devTanh(src[i]);
}

// host functions
void CudaOperator::setup(int size, double* data) {
    std::cout<<"cuda setup...\n";
    cudaMemcpy(cuData, data, sizeof(double) * size, cudaMemcpyHostToDevice);
}

void CudaOperator::cuAdd(double* src1, double *src2, double* dst, int M, int N) {
    std::cout<<"cuAdd()\n";
    int elements = M * N;
    int size = elements * sizeof(double);
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
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = (elements + threadsPerBlock - 1) / threadsPerBlock;
    cudaMatAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, elements);
    
    // copy result
    cudaMemcpy(dst, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CudaOperator::cuMul(double* A, double* B, double* C, int M, int N) {
    int elements = M * N;
    int size = elements * sizeof(double);
    // Allocate vectors in device memory
    double* d_A;
    cudaMalloc(&d_A, size);
    double* d_B;
    cudaMalloc(&d_B, size);
    double* d_C;
    cudaMalloc(&d_C, size);

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Invoke Kernel
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(M, LBLK), updiv(N, LBLK));
    cudaMatMulKernel<<<blocks, threadsPerBlock>>>(M, N, d_A, d_B, d_C);

    // copy result
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CudaOperator::cuDot(double* A, double* B, double* C, int M, int N) {
    int elements = M * N;
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(elements, threadsPerBlock);
    int size = elements * sizeof(double);
    // Allocate vectors in device memory
    double* d_A;
    cudaMalloc(&d_A, size);
    double* d_B;
    cudaMalloc(&d_B, size);
    double* d_C;
    cudaMalloc(&d_C, size);

    // Copy matrix from host memory to device memory
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, size, cudaMemcpyHostToDevice);

    // Invoke
    cudaMatDotKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, elements);

    // copy result
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void CudaOperator::cuSigmoid(double* src, double* dst, int length) {
    int size = length * sizeof(double);
    // Allocate vectors in device memory
    double* d_src;
    cudaMalloc(&d_src, size);
    double* d_dst;
    cudaMalloc(&d_dst, size);

    // Copy matrix from host memory to device memory
    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

    // Invoke
    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(length, threadsPerBlock);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, length); 
    
    // copy result
    cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
}

void CudaOperator::cuTanh(double* src, double* dst, int length) {
    int size = length * sizeof(double);
    // Allocate vectors in device memory
    double* d_src;
    cudaMalloc(&d_src, size);
    double* d_dst;
    cudaMalloc(&d_dst, size);

    // Copy matrix from host memory to device memory
    cudaMemcpy(d_src, src, size, cudaMemcpyHostToDevice);

    int threadsPerBlock = MBLK * MBLK;
    int blocksPerGrid = updiv(length, threadsPerBlock);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, length);  
    
    // copy result
    cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost); 

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
}