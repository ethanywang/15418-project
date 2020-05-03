# set cuda mode
find_package(CUDA REQUIRED)

# set CUDA config
set(CMAKE_CUDA_COMPILER /usr/local/depot/cuda/bin/nvcc)
set(CUDA_NVCC_FLAGS "-O3 -m64 --gpu-architecture compute_61")
set(EXTRA_LIBS ${EXTRA_LIBS} gpu)
set(CUDA_SEPARABLE_COMPILATION ON)

# set include directories
include_directories(include)
include_directories(cuda)

# compile cuda files
file(GLOB CUDA_FILES "src-cuda/" *.cu)
CUDA_COMPILE(CU_O ${CUDA_FILES})

# add the executable
cuda_add_executable(rnn-cuda main.cpp gru.cpp sigmoid.cpp tanh.cpp matrix.cpp lstm.cpp cudaOperator.cu)