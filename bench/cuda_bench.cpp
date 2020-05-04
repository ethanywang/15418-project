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
static void BM_CUDA_MatDot_10x10(benchmark::State& state) {
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

static void BM_CUDA_MatDot_100x100(benchmark::State& state) {
    Matrix m1(100, 100);
    Matrix m2(100, 100);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.dot(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatDot_1000x1000(benchmark::State& state) {
    Matrix m1(1000, 1000);
    Matrix m2(1000, 1000);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.dot(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

/* add */
static void BM_CUDA_MatAdd_10x10(benchmark::State& state) {
    Matrix m1(10, 10);
    Matrix m2(10, 10);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.add(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatAdd_100x100(benchmark::State& state) {
    Matrix m1(100, 100);
    Matrix m2(100, 100);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.add(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatAdd_1000x1000(benchmark::State& state) {
    Matrix m1(1000, 1000);
    Matrix m2(1000, 1000);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.add(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

/* mul */
static void BM_CUDA_MatMul_10x10(benchmark::State& state) {
    Matrix m1(10, 10);
    Matrix m2(10, 10);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.mul(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatMul_100x100(benchmark::State& state) {
    Matrix m1(100, 100);
    Matrix m2(100, 100);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.mul(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatMul_1000x1000(benchmark::State& state) {
    Matrix m1(1000, 1000);
    Matrix m2(1000, 1000);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m1.mul(m2);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

/* transpose */
static void BM_CUDA_MatT_10x10(benchmark::State& state) {
    Matrix m(10, 10);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m.T();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_100x100(benchmark::State& state) {
    Matrix m(100, 100);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m.T();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_1000x1000(benchmark::State& state) {
    Matrix m(1000, 1000);
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        m.T();
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

/* GRU - 10x10 */
static void BM_CUDA_MatT_10x10_b1(benchmark::State& state) {
    GRU gru(10, 10);
    Matrix input_x(10, 1);
    Matrix h(10, 1);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_10x10_b8(benchmark::State& state) {
    GRU gru(10, 10);
    Matrix input_x(10, 8);
    Matrix h(10, 8);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_10x10_b16(benchmark::State& state) {
    GRU gru(10, 10);
    Matrix input_x(10, 16);
    Matrix h(10, 16);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_10x10_b32(benchmark::State& state) {
    GRU gru(10, 10);
    Matrix input_x(10, 32);
    Matrix h(10, 32);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_10x10_b64(benchmark::State& state) {
    GRU gru(10, 10);
    Matrix input_x(10, 64);
    Matrix h(10, 64);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_10x10_b128(benchmark::State& state) {
    GRU gru(10, 10);
    Matrix input_x(10, 128);
    Matrix h(10, 128);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

/* GRU - 100x100 */
static void BM_CUDA_MatT_100x100_b1(benchmark::State& state) {
    GRU gru(100, 100);
    Matrix input_x(100, 1);
    Matrix h(100, 1);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_100x100_b8(benchmark::State& state) {
    GRU gru(100, 100);
    Matrix input_x(100, 8);
    Matrix h(100, 8);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_100x100_b16(benchmark::State& state) {
    GRU gru(100, 100);
    Matrix input_x(100, 16);
    Matrix h(100, 16);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_100x100_b32(benchmark::State& state) {
    GRU gru(100, 100);
    Matrix input_x(100, 32);
    Matrix h(100, 32);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_100x100_b64(benchmark::State& state) {
    GRU gru(100, 100);
    Matrix input_x(100, 64);
    Matrix h(100, 64);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

static void BM_CUDA_MatT_100x100_b128(benchmark::State& state) {
    GRU gru(100, 100);
    Matrix input_x(100, 128);
    Matrix h(100, 128);
    
    for (auto _ : state) {
        auto start = std::chrono::high_resolution_clock::now();
        gru.forward(input_x, h);
        auto end = std::chrono::high_resolution_clock::now();
        auto elapsed_seconds = 
            std::chrono::duration_cast<std::chrono::duration<double>>(end - start);

        state.SetIterationTime(elapsed_seconds.count());
    }
}

// Register the function as a benchmark
BENCHMARK(BM_CUDA_MatDot_10x10)->UseManualTime();
BENCHMARK(BM_CUDA_MatDot_100x100)->UseManualTime();
BENCHMARK(BM_CUDA_MatDot_1000x1000)->UseManualTime();
BENCHMARK(BM_CUDA_MatAdd_10x10)->UseManualTime();
BENCHMARK(BM_CUDA_MatAdd_100x100)->UseManualTime();
BENCHMARK(BM_CUDA_MatAdd_1000x1000)->UseManualTime();
BENCHMARK(BM_CUDA_MatMul_10x10)->UseManualTime();
BENCHMARK(BM_CUDA_MatMul_100x100)->UseManualTime();
BENCHMARK(BM_CUDA_MatMul_1000x1000)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_10x10)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_100x100)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_1000x1000)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_10x10_b1)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_10x10_b8)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_10x10_b16)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_10x10_b32)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_10x10_b64)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_10x10_b128)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_100x100_b1)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_100x100_b8)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_100x100_b16)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_100x100_b32)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_100x100_b64)->UseManualTime();
BENCHMARK(BM_CUDA_MatT_100x100_b128)->UseManualTime();

BENCHMARK_MAIN();