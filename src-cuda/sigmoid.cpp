//
// Created by yuwang4 on 2020-04-21.
//

#include <sigmoid.h>

Matrix Sigmoid::forward(Matrix m) {
    auto *__data = new double[m.size()];
    auto *_data = m.data();
    for (int i = 0; i < m.size(); i++) {
        __data[i] = 1 / (1 + exp(_data[i]));
    }
    _res = Matrix(__data, m.size(0), m.size(1));
    return _res;
}

