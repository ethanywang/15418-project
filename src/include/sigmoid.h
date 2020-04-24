//
// Created by yuwang4 on 2020-04-21.
//

#ifndef RNN_SIGMOID_H
#define RNN_SIGMOID_H

#include <matrix.h>

class Sigmoid {
public:
    Sigmoid() : res(1, 1) {};

    Matrix forward(Matrix);

private:
    Matrix res;
};


#endif //RNN_SIGMOID_H
