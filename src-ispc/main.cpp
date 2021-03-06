//
// Created by yuwang4 on 2020-04-20.
//
#include "gru.h"
#include "lstm.h"

int main(int argc, char **argv) {
    GRU gru(10, 20);
    Matrix input_x(10, 1);
    Matrix h(20, 1);
    Matrix output1 = gru.forward(input_x, h);

    LSTM lstm(10, 20);
    Matrix c0(20, 1);
    Matrix c1(20, 1);
    Matrix output2 = lstm.forward(input_x, h, c0, c1);
    Matrix m3 = input_x.T();
    return 0;
}
