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

/* dot */
static void BM_ISPC_MatDot_10x10(benchmark::State& state) {
    Matrix m1(10, 10);
    Matrix m2(10, 10);
    for (auto _ : state) {
        m1.dot(m2);
    }
}

static void BM_ISPC_MatDot_100x100(benchmark::State& state) {
    Matrix m1(100, 100);
    Matrix m2(100, 100);
    for (auto _ : state) {
        m1.dot(m2);
    }
}

static void BM_ISPC_MatDot_1000x1000(benchmark::State& state) {
    Matrix m1(1000, 1000);
    Matrix m2(1000, 1000);
    for (auto _ : state) {
        m1.dot(m2);
    }
}

/* add */
static void BM_ISPC_MatAdd_10x10(benchmark::State& state) {
    Matrix m1(10, 10);
    Matrix m2(10, 10);
    for (auto _ : state) {
        m1.add(m2);
    }
}

static void BM_ISPC_MatAdd_100x100(benchmark::State& state) {
    Matrix m1(100, 100);
    Matrix m2(100, 100);
    for (auto _ : state) {
        m1.add(m2);
    }
}

static void BM_ISPC_MatAdd_1000x1000(benchmark::State& state) {
    Matrix m1(1000, 1000);
    Matrix m2(1000, 1000);
    for (auto _ : state) {
        m1.add(m2);
    }
}

/* mul */
static void BM_ISPC_MatMul_10x10(benchmark::State& state) {
    Matrix m1(10, 10);
    Matrix m2(10, 10);
    for (auto _ : state) {
        m1.mul(m2);
    }
}

static void BM_ISPC_MatMul_100x100(benchmark::State& state) {
    Matrix m1(100, 100);
    Matrix m2(100, 100);
    for (auto _ : state) {
        m1.mul(m2);
    }
}

static void BM_ISPC_MatMul_1000x1000(benchmark::State& state) {
    Matrix m1(1000, 1000);
    Matrix m2(1000, 1000);
    for (auto _ : state) {
        m1.mul(m2);
    }
}

/* transpose */
static void BM_ISPC_MatT_10x10(benchmark::State& state) {
    Matrix m(10, 10);
    for (auto _ : state) {
        m.T();
    }
}

static void BM_ISPC_MatT_100x100(benchmark::State& state) {
    Matrix m(100, 100);
    for (auto _ : state) {
        m.T();
    }
}

static void BM_ISPC_MatT_1000x1000(benchmark::State& state) {
    Matrix m(1000, 1000);
    for (auto _ : state) {
        m.T();
    }
}

// Register the function as a benchmark
BENCHMARK(BM_ISPC_MatDot_10x10);
BENCHMARK(BM_ISPC_MatDot_100x100);
BENCHMARK(BM_ISPC_MatDot_1000x1000);
BENCHMARK(BM_ISPC_MatAdd_10x10);
BENCHMARK(BM_ISPC_MatAdd_100x100);
BENCHMARK(BM_ISPC_MatAdd_1000x1000);
BENCHMARK(BM_ISPC_MatMul_10x10);
BENCHMARK(BM_ISPC_MatMul_100x100);
BENCHMARK(BM_ISPC_MatMul_1000x1000);
BENCHMARK(BM_ISPC_MatT_10x10);
BENCHMARK(BM_ISPC_MatT_100x100);
BENCHMARK(BM_ISPC_MatT_1000x1000);

BENCHMARK_MAIN();