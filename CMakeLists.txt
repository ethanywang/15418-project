# set the minimum cmake version
cmake_minimum_required(VERSION 2.8)
if (POLICY CMP0048)
    cmake_policy(SET CMP0048 NEW)
endif()

# set the project name
project(RNN)

# CTest
include(CTest)

# specify the C++ standard
if (CMAKE_VERSION VERSION_LESS "3.1")
    if (CMAKE_CXX_COMPILER_ID STREQUAL "GNU")
        set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=gnu++11")
    endif ()
else ()
    set (CMAKE_CXX_STANDARD 11)
endif ()

# set common flags
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3 -m64 -Wall")

# set debug mode
set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")

# add include directory
include_directories(include)
include_directories(src-cuda/include)

# add subdirectories
add_subdirectory(src-seq)
add_subdirectory(src-cuda)
add_subdirectory(src-ispc)
add_subdirectory(lib/benchmark)
add_subdirectory(test)