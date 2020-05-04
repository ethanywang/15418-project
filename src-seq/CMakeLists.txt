# compile to library
add_library(libseq STATIC
    gru.cpp
    lstm.cpp
    matrix.cpp
    sigmoid.cpp
    tanh.cpp
)
target_compile_features(libseq PUBLIC cxx_std_11)

# add the executable
add_executable(rnn-seq main.cpp)
target_link_libraries(rnn-seq PRIVATE libseq)