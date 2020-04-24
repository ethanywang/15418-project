# set the minimum cmake version
cmake_minimum_required(VERSION 2.8)

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

# set include directories
include_directories(src/include)

# set source file
set(SRC_FILES
        src/main.cpp
        src/gru.cpp
        src/sigmoid.cpp
        src/tanh.cpp
        src/matrix.cpp)

# add the executable
add_executable(rnn ${SRC_FILES})

