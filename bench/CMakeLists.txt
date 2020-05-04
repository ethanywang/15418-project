# bench sources
file(GLOB BM_SRC "bench/*.cpp")

# seq bench
add_executable(rnn-seq-bench seq_bench.cpp)
target_link_libraries(rnn-seq-bench
        benchmark::benchmark
        libseq
)

# cuda bench
add_executable(rnn-cuda-bench cuda_bench.cpp)
target_link_libraries(rnn-cuda-bench
        benchmark::benchmark
        libcuda
)

# ispc bench
add_executable(rnn-ispc-bench ispc_bench.cpp)
target_link_libraries(rnn-ispc-bench
        benchmark::benchmark
        libispc
)

# add "make bench"
add_custom_target(bench
        COMMAND rnn-seq-bench
        COMMAND rnn-cuda-bench
        COMMAND rnn-ispc-bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS rnn-seq-bench rnn-cuda-bench rnn-ispc-bench
        SOURCES ${BM_SRC}
)
