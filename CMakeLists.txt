# set the minimum cmake version
cmake_minimum_required(VERSION 2.8)

# specify the C++ standard
if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 11)
endif ()

# set the project name
project(RNN)
find_package(CUDA REQUIRED)

# set debug mode
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")

# set cuda flags
set(CUDA_NVCC_FLAGS ${CUDA_NVCC_FLAGS};-gencode arch=compute_61,code=sm_61;)
set(EXTRA_LIBS ${EXTRA_LIBS} gpu)

# include subdirectories
add_subdirectory(cuda-src)
add_subdirectory(src)
