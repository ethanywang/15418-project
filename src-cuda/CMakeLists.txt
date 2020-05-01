# set cuda mode
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)
find_package(CUDA REQUIRED)
set(EXTRA_LIBS ${EXTRA_LIBS} gpu)

# set include directories
include_directories(include)
include_directories(cuda)

# add the executable
add_executable(rnn-cuda main.cpp gru.cpp sigmoid.cpp tanh.cpp matrix.cpp lstm.cpp)