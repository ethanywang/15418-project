# set cuda mode
find_package(CUDA REQUIRED)

# set CUDA config
set(CMAKE_CUDA_COMPILER /usr/local/depot/cuda/bin/nvcc)
set(CUDA_NVCC_FLAGS "-O3 --gpu-architecture compute_61 --std c++11")

# set include directories
include_directories(include)

# add the executable
cuda_add_executable(rnn-cuda main.cpp gru.cpp sigmoid.cpp tanh.cpp matrix.cpp lstm.cpp cudaOperator.cu)