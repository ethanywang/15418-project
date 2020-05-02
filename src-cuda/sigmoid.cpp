//
// Created by yuwang4 on 2020-04-21.
//

#include <sigmoid.h>

Matrix Sigmoid::forward(Matrix m) {
    auto *__data = new double[m.size()];
    auto *_data = m.data();
    if (m._dev == SEQ) {
        for (int i = 0; i < m.size(); i++) {
            __data[i] = 1 / (1 + exp(_data[i]));
        }
    }

    if (m._dev == GPU) {
        m._cu->cuSigmoid(_data, __data, m.size());
    }
    
    _res = Matrix(__data, m.size(0), m.size(1));
    return _res;
}

