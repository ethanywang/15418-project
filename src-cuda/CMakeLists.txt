# set CUDA config
set(CMAKE_CUDA_COMPILER /usr/local/depot/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "-m64 --gpu-architecture compute_61")
set(CMAKE_CUDA_FLAGS_DEBUG "-O3")

# compile CUDA library
add_library(libcuda STATIC
    gru.cu
    sigmoid.cu 
    tanh.cu 
    matrix.cu 
    lstm.cpp 
    cuda_operator.cu
)
target_compile_features(libcuda PUBLIC cxx_std_11)
set_target_properties(libcuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add the executable
add_executable(rnn-cuda main.cpp)
set_target_properties(rnn-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(rnn-cuda PRIVATE libcuda)
