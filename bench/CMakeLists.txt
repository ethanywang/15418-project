
file(GLOB BENCH_SRC "bench/" *_bench.cpp)

add_executable(rnn_bench seq_bench.cpp ${BENCH_SRC})
target_link_libraries(rnn_bench
        benchmark::benchmark
)

add_custom_target(bench
        COMMAND rnn_bench
        WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
        DEPENDS rnn_bench
        SOURCES ${BENCH_SRC}
)