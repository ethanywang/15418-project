//
// Created by yuwang on 2020-04-21.
//

#include "tanh.h"

Matrix Tanh::forward(Matrix m) {
    auto *__data = new float[m.size()];
    auto *_data = m.data();
    for (int i = 0; i < m.size(); i++) {
        __data[i] = tanh(_data[i]);
    }
    return std::move(Matrix(__data, m.size(0), m.size(1)));
}

