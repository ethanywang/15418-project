
#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

__global__ void _warm_up_gpu(){
    unsigned int tid = blockIdx.x * blockDim.x + threadIdx.x;
    float ia, ib;
    ia = ib = 0.0f;
    ib += ia + tid; 
}

void warm_up_gpu() {
    _warm_up_gpu<<<65536, 32>>>();
    cudaDeviceSynchronize();
}