//
// Created by yuwang on 2020-04-21.
//

#include "tanh.h"
#include "cudaOperator.h"

Matrix Tanh::forward(Matrix m) {
    auto *__data = new float[m.size()];
    auto *_data = m.data();
    // if (m._dev == SEQ) {
    //     for (int i = 0; i < m.size(); i++) {
    //         __data[i] = tanh(_data[i]);
    //     }
    // }
    cuTanh(_data, __data, m.size());
    _res = Matrix(__data, m.size(0), m.size(1));
    return _res;
}

