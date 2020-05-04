# find thread library
find_package(Threads)

# compile to library
add_library(libthread STATIC
    gru.cpp
    lstm.cpp
    matrix.cpp
    sigmoid.cpp
    tanh.cpp
)
target_compile_features(libthread PUBLIC cxx_std_11)

# add the executable
add_executable(rnn-thread main.cpp)
target_link_libraries(rnn-thread
    libthread
    ${CMAKE_THREAD_LIBS_INIT}
)