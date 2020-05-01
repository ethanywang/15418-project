# set the minimum cmake version
cmake_minimum_required(VERSION 2.8)

# set the project name
project(RNN)
find_package(CUDA REQUIRED)

# set debug mode
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")
SET(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)

# specify the C++ standard
if (CMAKE_VERSION VERSION_LESS "3.1")
        if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
                set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
        endif ()
else ()
        set (CMAKE_CXX_STANDARD 11)
endif ()

# set include directories
include_directories(cudasrc/include)
include_directories(cudasrc/cuda)

set(EXTRA_LIBS ${EXTRA_LIBS} gpu)

# set source file
set(SRC_FILES
        cudasrc/main.cpp
        cudasrc/gru.cpp
        cudasrc/sigmoid.cpp
        cudasrc/tanh.cpp
        cudasrc/matrix.cpp
        cudasrc/lstm.cpp
        )

# add the executable
add_executable(rnn ${SRC_FILES})