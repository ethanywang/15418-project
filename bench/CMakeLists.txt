# bench sources
file(GLOB BM_SRC "bench/*.cpp")

# seq bench
add_executable(rnn-seq-bench seq_bench.cpp)
target_link_libraries(rnn-seq-bench
        benchmark::benchmark
        libseq
)

# thread bench
add_executable(rnn-thread-bench thread_bench.cpp)
target_link_libraries(rnn-thread-bench
        benchmark::benchmark
        libthread
)

# openmp bench
set(ENV{OMP_NUM_THREADS} "8")
add_executable(rnn-openmp-bench openmp_bench.cpp)
target_link_libraries(rnn-openmp-bench
        benchmark::benchmark
        libopenmp
)

# cuda bench
add_executable(rnn-cuda-bench cuda_bench.cpp warmup.cu)
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
        COMMAND rnn-thread-bench
        COMMAND rnn-openmp-bench
        COMMAND rnn-cuda-bench
        COMMAND rnn-ispc-bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS rnn-seq-bench rnn-thread-bench rnn-openmp-bench rnn-cuda-bench rnn-ispc-bench
        SOURCES ${BM_SRC}
)
