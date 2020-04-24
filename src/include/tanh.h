//
// Created by yuwang on 2020-04-21.
//

#ifndef RNN_TANH_H
#define RNN_TANH_H

#include <matrix.h>

class Tanh {
public:
    Tanh() : res(1, 1) {};

    Matrix forward(Matrix);

private:
    Matrix res;
};


#endif //RNN_TANH_H
