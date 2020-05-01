# set include directories
include_directories(include)

# add the executable
add_executable(rnn-seq main.cpp gru.cpp sigmoid.cpp tanh.cpp matrix.cpp lstm.cpp)