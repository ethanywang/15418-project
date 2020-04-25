//
// Created by yuwang4 on 2020-04-20.
//

#include <gru.h>
#include <lstm.h>

int main(int argc, char **argv) {
    GRU gru(10, 20);
    Matrix input_x(10, 1);
    Matrix h(20, 1);
    Matrix output1 = gru.forward(input_x, h);

//    LSTM lstm(10, 20);
//    Matrix C(20, 1);
//    Matrix output2 = lstm.forward(input_x, h, C);
    return 0;
}
