# set CUDA config
set(CMAKE_CUDA_COMPILER /usr/local/depot/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "-O3 -m64 --gpu-architecture compute_61")

# compile CUDA library
add_library(cu STATIC
    gru.cpp 
    sigmoid.cpp 
    tanh.cpp 
    matrix.cpp 
    lstm.cpp 
    cuda_operator.cu
)
target_compile_features(cu PUBLIC cxx_std_11)
set_target_properties(cu PROPERTIES CUDA_SEPARABLE_COMPILATION ON)

# add the executable
add_executable(rnn-cuda main.cpp)
set_target_properties(rnn-cuda PROPERTIES CUDA_SEPARABLE_COMPILATION ON)
target_link_libraries(rnn-cuda PRIVATE cu)
