//
// Created by yuwang4 on 2020-04-20.
//

#include <gru.h>

int main(int argc, char **argv) {
    GRU gru(10, 20);
    Matrix input_x = Matrix(10, 1);
    Matrix h = Matrix(20, 1);
    Matrix output = gru.forward(input_x, h);
    return 0;
}
