#include "matrix.h"
#include "cuda_operator.h"
#include "gru_operator.h"
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

#include <iostream>
#define RINDEX(r, c, width) (r * width + c)
#define MBLK 4
#define LBLK 8
#define THREADWORK 8

// 0, simple mat mul -> 1 read, 1 write per thread
// 1, modified simple mat mul -> 1 read, 1 write per warp
// 2, blocked mat mul
#define KERNELFUNC 0

// read-only variables should be stored in fast memory

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

__global__ void cudaOptMatMulKernel(int M, int N, int R, float *dmatA, float *dmatB, float *dmatC) {
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
    int threadsPerBlock = MBLK;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);
    cudaMatAddKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, elements);
    // cudaDeviceSynchronize();
    // copy result
    cudaMemcpy(dst, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cuMul(float *A, float *B, float *C, int M, int N, int R) {
    // std::cout << "cuMul()\n";
    int A_elements = M * N;
    int B_elements = N * R;
    int C_elements = M * R;
    // Allocate vectors in device memory
    float *d_A;
    cudaMalloc(&d_A, A_elements * sizeof(float));
    float *d_B;
    cudaMalloc(&d_B, B_elements * sizeof(float));
    float *d_C;
    cudaMalloc(&d_C, C_elements * sizeof(float));

    // Copy vectors from host memory to device memory
    cudaMemcpy(d_A, A, A_elements * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, B, B_elements * sizeof(float), cudaMemcpyHostToDevice);

    // Invoke Kernel
    dim3 threadsPerBlock(LBLK, LBLK);
    dim3 blocks(updiv(M, LBLK), updiv(N, LBLK));
    cudaOptMatMulKernel<<<blocks, threadsPerBlock>>>(M, N, R, d_A, d_B, d_C);
    // cudaDeviceSynchronize();
    // copy result
    cudaMemcpy(C, d_C, C_elements * sizeof(float), cudaMemcpyDeviceToHost);

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
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);
    cudaMatDotKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, elements);
    // cudaDeviceSynchronize();
    // copy result
    cudaMemcpy(C, d_C, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

void cuT(float *A, float *AT, int M, int N) {
    int elements = M * N;
    int size = elements * sizeof(float);
    // Allocate in device memory
    float *d_A;
    float *d_AT;
    cudaMalloc(&d_A, size);
    cudaMalloc(&d_AT, size);
    
    // Copy source
    cudaMemcpy(d_A, A, size, cudaMemcpyHostToDevice);

    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);
    cudaMatTransposeKernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_AT, M, N);

    // copy result back
    cudaMemcpy(AT, d_AT, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_A);
    cudaFree(d_AT);
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
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, length);
    // cudaDeviceSynchronize();
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

    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(d_src, d_dst, length);
    // cudaDeviceSynchronize();
    
    // copy result
    cudaMemcpy(dst, d_dst, size, cudaMemcpyDeviceToHost);

    // Free device memory
    cudaFree(d_src);
    cudaFree(d_dst);
}

void cu_mat_mul(float* A, float* B, float* C, int M, int N, int R) {
    // Invoke Kernel
    dim3 threadsPerBlock(MBLK, MBLK);
    dim3 blocks(updiv(M, MBLK), updiv(R, MBLK));
    cudaOptMatMulKernel<<<blocks, threadsPerBlock>>>(M, N, R, A, B, C);
}

void cu_mat_dot(float *A, float *B, float *C, int M, int N) {
    int elements = M * N;
    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);
    cudaMatDotKernel<<<blocksPerGrid, threadsPerBlock>>>(A, B, C, elements);
}

void cu_mat_add(float *src1, float *src2, float *dst, int M, int N)  {
    int elements = M * N;
    // Invoke kernel
    int threadsPerBlock = MBLK;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(elements, threadsPerBlock * elementsPerThread);
    cudaMatAddKernel<<<blocksPerGrid, threadsPerBlock>>>(src1, src2, dst, elements);
    // cudaDeviceSynchronize();
}

void cu_mat_sigmoid(float *src, float *dst, int length) {
    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);
}

void cu_mat_tanh(float *src, float *dst, int length) {
    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    cudaSigmoidKernel<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);
}

__global__ void cuReverseSign(float* src, float* dst, int length) {
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= length) return;
    for (int i = 0; i < THREADWORK && (offset + i) < length; i++) {
        dst[offset + i] = -src[offset + i];
    }
}

__global__ void cuSubtract(float* src, float* dst, int length, float num) {
    int offset = blockDim.x * blockIdx.x + threadIdx.x;
    if (offset >= length) return;
    for (int i = 0; i < THREADWORK && (offset + i) < length; i++) {
        dst[offset + i] = src[offset + i] - num;
    }
}

void cu_minus(float* src, float* dst, int length, float num = 0.0) {
    // Invoke
    int threadsPerBlock = 1;
    int elementsPerThread = THREADWORK;
    int blocksPerGrid = updiv(length, threadsPerBlock * elementsPerThread);
    if (num == 0.0) {
        cuReverseSign<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length);    
    } else {
        cuSubtract<<<blocksPerGrid, threadsPerBlock>>>(src, dst, length, num);  
    }
}

void cu_gru_forward(float *input, float* hiddent, float *output, int i_m, int i_n, int h_m, int h_n) {
    gru_forward_setup(i_m, i_n, h_m, h_n);
    // copy input data into GPU
    // this->h_t_1 = h;
    cudaMemcpy(h_t_1, hiddent, h_m * h_n * sizeof(float), cudaMemcpyHostToDevice);
    // this->x = x;
    cudaMemcpy(x, input, i_m * i_n * sizeof(float), cudaMemcpyHostToDevice);

    // allocate tmp values
    int d = i_m;
    int h = h_m;
    // __device__ float tmp1[h * i_n];
    // __device__ float tmp2[h * i_n];
    // __device__ float tmp3[h * i_n];
    // __device__ float tmp4[h * i_n];
    // __device__ float tmp5[h * i_n];
    // __device__ float mid_res1[h * i_n];
    // __device__ float mid_res2[h * i_n];
    // __device__ float mid_res3[h * i_n];
    // __device__ float mid_res4[h * i_n];
    float* tmp1;
    float* tmp2;
    float* tmp3;
    float* tmp4;
    float* tmp5;
    float* mid_res1;
    float* mid_res2;
    float* mid_res3;
    float* mid_res4;

    gru_mat_setup(&tmp1, h, i_n, false);
    gru_mat_setup(&tmp2, h, i_n, false);
    gru_mat_setup(&tmp3, h, i_n, false);
    gru_mat_setup(&tmp4, h, i_n, false);
    gru_mat_setup(&tmp5, h, i_n, false);
    gru_mat_setup(&mid_res1, h, i_n, false);
    gru_mat_setup(&mid_res2, h, i_n, false);
    gru_mat_setup(&mid_res3, h, i_n, false);
    gru_mat_setup(&mid_res4, h, i_n, false);

    // forward process
    // auto tmp1 = this->Wzx.mul(x);
    // auto tmp2 = this->Wrx.mul(x);
    cu_mat_mul(Wzx, x, tmp1, h, d, i_n);
    cu_mat_mul(Wrx, x, tmp2, h, d, i_n);
    cudaDeviceSynchronize();

    // this->z_t = this->z_act.forward(this->Wzh.mul(h_t_1).add(tmp1));
    // this->r_t = this->r_act.forward(this->Wrh.mul(h_t_1).add(tmp2));
    cu_mat_mul(Wzh, h_t_1, mid_res1, h, h, i_n);
    cu_mat_mul(Wrh, h_t_1, mid_res2, h, h, i_n);
    cu_mat_add(mid_res1, tmp1, mid_res3, h, i_n);
    cu_mat_add(mid_res2, tmp2, mid_res4, h, i_n);
    cu_mat_sigmoid(mid_res3, z_t, h * i_n);
    cu_mat_sigmoid(mid_res4, r_t, h * i_n);

    // auto tmp3 = this->r_t.dot(h_t_1);
    cu_mat_dot(r_t, h_t_1, tmp3, h, i_n);
    // auto tmp4 = this->Wx.mul(x);
    cu_mat_mul(Wx, x, tmp4, h, d,i_n);
    // this->h_bar_t = this->h_act.forward(this->Wh.mul(tmp3).add(tmp4));
    cu_mat_mul(Wh, tmp3, mid_res1, h, h, i_n);
    cu_mat_add(mid_res1, tmp4, mid_res3, h, i_n);
    cu_mat_sigmoid(mid_res3, h_bar_t, h * i_n);
    // auto tmp5 = this->z_t.dot(this->h_bar_t);
    cu_mat_dot(z_t, h_bar_t, tmp5, h, i_n);
    // this->h_t = (-this->z_t + 1.0).dot(this->h_t_1).add(tmp5);
    cu_minus(z_t, mid_res2, h * i_n);
    cu_minus(mid_res2, mid_res4, h * i_n, -1.0);
    cu_mat_dot(z_t, h_t_1, mid_res1, h, i_n);
    cu_mat_add(mid_res1, tmp5, h_t, h, i_n);

    // copy result out
    cudaMemcpy(output, h_t, h * i_n * sizeof(float), cudaMemcpyDeviceToHost);
    gru_forward_clear();
    cudaFree(tmp1);
    cudaFree(tmp2);
    cudaFree(tmp3);
    cudaFree(tmp4);
    cudaFree(tmp5);
    cudaFree(mid_res1);
    cudaFree(mid_res2);
    cudaFree(mid_res3);
    cudaFree(mid_res4);
}

void gru_forward_setup(int i_m, int i_n, int h_m, int h_n) {
    int d = i_m;
    int h = h_m;

    // cudaMalloc(&h_t_1, mat_size);
    gru_mat_setup(&h_t_1, h_m, h_n, false);
    
    // cudaMalloc(&x, mat_size);
    gru_mat_setup(&x, i_m, i_n, false);

    // cudaMalloc(&Wzx, mat_size);
    gru_mat_setup(&Wzx, h, d, true);
    
    // cudaMalloc(&Wrx, mat_size);
    gru_mat_setup(&Wrx, h, d, true);
    
    // cudaMalloc(&Wzh, mat_size);
    gru_mat_setup(&Wzh, h, h, true);
    
    // cudaMalloc(&Wrh, mat_size);
    gru_mat_setup(&Wrh, h, h, true);

    // cudaMalloc(&Wx, mat_size);
    gru_mat_setup(&Wx, h, d, true);

    // cudaMalloc(&z_t, sizeof(Mat));
    gru_mat_setup(&z_t, h, i_n, false);

    // cudaMalloc(&r_t, sizeof(Mat));
    gru_mat_setup(&r_t, h, i_n, false);

    // cudaMalloc(&h_bar_t, sizeof(Mat));
    gru_mat_setup(&h_bar_t, h, i_n, false);
}

void gru_mat_setup(float** ptr, int m, int n, bool init) {
    cudaMalloc(ptr, m * n * sizeof(float));
    // cudaError_t code =  printf("%s\n",cudaGetErrorString(code));
    if (!init) return;
    /* initialize */
    cudaMemset(ptr, 0, m * n * sizeof(float));
}

void gru_forward_clear() {
    cudaFree(h_t_1);
    cudaFree(x);
    cudaFree(Wzx);
    cudaFree(Wzh);
    cudaFree(Wrh);
    cudaFree(Wx);
    cudaFree(z_t);
    cudaFree(h_t_1);
    cudaFree(h_bar_t);
}