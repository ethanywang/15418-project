# set the minimum cmake version
cmake_minimum_required(VERSION 3.8)
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()

# set the project name
project(RNN LANGUAGES CXX C CUDA VERSION 1.0.0)

# CTest
include(CTest)

# specify the C++ standard
set (CMAKE_CXX_STANDARD 11)

# set common flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -m64 -Wall")

# set debug mode
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")

# add include directory
include_directories(include)
include_directories(src-cuda/include)

# add subdirectories
# do not build tests of benchmarking lib
set(BENCHMARK_ENABLE_TESTING OFF CACHE BOOL "Suppressing benchmark's tests" FORCE)
add_subdirectory(lib/benchmark)
# build google test
add_subdirectory(lib/googletest)
# source
add_subdirectory(src-seq)
add_subdirectory(src-cuda)
add_subdirectory(src-ispc)
# QA
add_subdirectory(test)
add_subdirectory(bench)