#include <matrix.h>
#define LBLK 32
#define MAXSIZE 1024
// cuda kernel variables
__constant__ float cuData[MAXSIZE];

CudaOperator::setup(int size, double* data) {
    cout<<"cuda setup...\n";
    cudaMemcpy(cuData, data, sizeof(double) * size, cudaMemcpyHostToDevice);
}

CudaOperator::add(double* src1, double src*2, double* dst, size_t size) {
    size = size * sizeof(double);
    // Allocate vectors in device memory
    float* d_A;
    cudaMalloc(&d_A, size);
    float* d_B;
    cudaMalloc(&d_B, size);
    float* d_C;
    cudaMalloc(&d_C, size);
    
    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);
    
    // Invoke kernel
    int threadsPerBlock = 256;
    int blocksPerGrid =
    (N + threadsPerBlock - 1) / threadsPerBlock;
    VecAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // copy result
    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

CudaOperator::mul(double* A, double* B, double* C) {
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(N, LBLK), updiv(N, LBLK));
    cudaSimpleKernel<<<blocks, threadsPerBlock>>>(N, dmatA, dmatB, dmatC);
}

__global__ void VecAdd(double* src1, double* src2, double* dst) {
    int i = threadIdx.x;
    dst[i] = src1[i] + src2[i];
}

__global__ voidcudaSimpleOldKernel(int N, double* dmatA,double* dmatB, double * dmatC) {
    int i= blockIdx.x* blockDim.x+ threadIdx.x;int j = blockIdx.y* blockDim.y+ threadIdx.y;
    if (i>= N || j >= N) return;float sum = 0.0;
    for (int k = 0; k < N; k++) {
        sum += dmatA[RM(i,k,N)] * dmatB[RM(k,j,N)];
    }
    dmatC[RM(i,j,N)] = sum;
}