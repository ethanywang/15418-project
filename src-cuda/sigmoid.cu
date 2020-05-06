//
// Created by yuwang4 on 2020-04-21.
//

#include "sigmoid.h"
#include "cuda_operator.h"

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

Matrix Sigmoid::forward(Matrix m) {
    // auto *__data = new float[];
    float *__data;
    cudaMalloc((void**)&__data, m.size() * sizeof(float));

    float *_data = m.data();
    // if (m._dev == SEQ) {
    //     for (int i = 0; i < m.size(); i++) {
    //         __data[i] = 1 / (1 + exp(_data[i]));
    //     }
    // }
    cuSigmoid(_data, __data, m.size());
    return std::move(Matrix(__data, m.size(0), m.size(1)));
}

