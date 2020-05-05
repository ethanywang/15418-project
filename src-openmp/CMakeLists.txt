# support OPENMP
find_package(OpenMP)
set(OpenMP_CXX_FLAGS "-fopenmp -DOMP")
set(ENV{OMP_NUM_THREADS} "8")

# compile to library
add_library(libopenmp STATIC
    gru.cpp
    lstm.cpp
    matrix.cpp
    sigmoid.cpp
    tanh.cpp
)
target_compile_features(libopenmp PUBLIC cxx_std_11)
target_link_libraries(libopenmp PUBLIC OpenMP::OpenMP_CXX)

# add the executable
add_executable(rnn-openmp main.cpp)
target_link_libraries(rnn-openmp PRIVATE libopenmp)