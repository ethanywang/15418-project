# set the minimum cmake version
cmake_minimum_required(VERSION 2.8)

# Alternatively, set gcc/g++
#set(CMAKE_C_COMPILER "gcc")
#set(CMAKE_CXX_COMPILER "g++")

# set the project name
project(RNN)

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
#set(CMAKE_BUILD_TYPE Debug)
set(CMAKE_C_FLAGS_DEBUG "${CMAKE_C_FLAGS_DEBUG} -D_DEBUG")

# include subdirectories
#add_subdirectory(src-seq)
add_subdirectory(src-cuda)
#add_subdirectory(src-ispc)
