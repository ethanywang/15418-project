//
// Created by yuwang4 on 2020-04-21.
//

#ifndef RNN_SIGMOID_H
#define RNN_SIGMOID_H

#include "matrix.h"

class Sigmoid {
public:
    Matrix forward(Matrix);

    Matrix operator()(const Matrix &mat) {
        return this->forward(mat);
    }
private:
    Matrix _res;
};


#endif //RNN_SIGMOID_H
