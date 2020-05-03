//
// Created by yuwang on 2020-04-21.
//

#ifndef RNN_TANH_H
#define RNN_TANH_H

#include "matrix.h"

class Tanh {
public:
    Matrix forward(Matrix);

    Matrix operator()(const Matrix &mat) {
        return this->forward(mat);
    }

private:
    Matrix _res;
};


#endif //RNN_TANH_H
