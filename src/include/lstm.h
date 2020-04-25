//
// Created by yuwang on 4/24/20.
//

#ifndef RNN_LSTM_H
#define RNN_LSTM_H

#include "matrix.h"
#include "sigmoid.h"
#include "tanh.h"

class LSTM {
public:
    LSTM(int input_dim, int hidden_dim) :
            d(input_dim),
            h(hidden_dim),
            Wix(hidden_dim, input_dim, false),
            Wih(hidden_dim, hidden_dim, false),
            Wfx(hidden_dim, input_dim, false),
            Wfh(hidden_dim, hidden_dim, false),
            Wcx(hidden_dim, input_dim, false),
            Wch(hidden_dim, hidden_dim, false),
            Wox(hidden_dim, input_dim, false),
            Woh(hidden_dim, hidden_dim, false),
            f_act(),
            x_act(),
            o_act(),
            c_bar_act(),
            c_act(),
            c_t_1(hidden_dim, hidden_dim)
    {};
    ~LSTM() {};

    Matrix forward(Matrix &, Matrix &, Matrix &);

private:
    int d;
    int h;

    Matrix Wix;
    Matrix Wih;

    Matrix Wfx;
    Matrix Wfh;

    Matrix Wcx;
    Matrix Wch;

    Matrix Wox;
    Matrix Woh;

    Sigmoid f_act;
    Sigmoid x_act;
    Sigmoid o_act;
    Tanh c_bar_act;
    Tanh c_act;

    Matrix f_t;
    Matrix i_t;
    Matrix x;
    Matrix o_t;
    Matrix c_bar_t;
    Matrix h_t;
    Matrix h_t_1;
    Matrix c_t;
    Matrix c_t_1;
};

#endif //RNN_LSTM_H
