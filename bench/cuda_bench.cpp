//
// Created by yuwang on 5/3/20.
//

#include "benchmark/benchmark.h"
#include "matrix.h"
#include "sigmoid.h"
#include "tanh.h"
#include "gru.h"
#include "lstm.h"

#include <chrono>

static void BM_CUDA_MatDot_100(benchmark::State& state) {
    Matrix m1(10, 10);
    Matrix m2(10, 10);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.dot(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }

}

// Register the function as a benchmark
BENCHMARK(BM_CUDA_MatDot_100)->UseManualTime();

BENCHMARK_MAIN();