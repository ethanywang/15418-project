//
// Created by yuwang on 5/3/20.
//

#include "benchmark/benchmark.h"
#include "matrix.h"
#include "sigmoid.h"
#include "tanh.h"
#include "gru.h"
#include "lstm.h"

static void BM_SEQ_MatDot_100(benchmark::State& state) {
    Matrix m1(10, 10);
    Matrix m2(10, 10);
    for (auto _ : state) {
        m1.dot(m2);
    }
}

// Register the function as a benchmark
BENCHMARK(BM_SEQ_MatDot_100);

BENCHMARK_MAIN();