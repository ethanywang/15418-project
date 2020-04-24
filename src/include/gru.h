//
// Created by yuwang4 on 2020-04-21.
//

#ifndef RNN_GRU_H
#define RNN_GRU_H

#include <matrix.h>
#include <sigmoid.h>
#include <tanh.h>
#include <random>
#include <cassert>

class GRU {
public:
    GRU(int input_dim, int hidden_dim)
            : d(input_dim),
              h(hidden_dim),
              Wzh(h, h, false),
              Wrh(h, h, false),
              Wh(h, h, false),
              Wzx(h, d, false),
              Wrx(h, d, false),
              Wx(h, d, false),
              dWzh(h, h),
              dWrh(h, h),
              dWh(h, h),
              dWzx(h, d),
              dWrx(h, d),
              dWx(h, d),
              z_act(),
              r_act(),
              h_act() {};

    ~GRU() {};

    Matrix forward(Matrix &, Matrix &);

private:
    int d;
    int h;

    Matrix Wzh;
    Matrix Wrh;
    Matrix Wh;

    Matrix Wzx;
    Matrix Wrx;
    Matrix Wx;

    Matrix dWzh;
    Matrix dWrh;
    Matrix dWh;

    Matrix dWzx;
    Matrix dWrx;
    Matrix dWx;

    Sigmoid z_act;
    Sigmoid r_act;
    Tanh h_act;

    /* Tmp Saving */
    Matrix h_t_1;
    Matrix x;
    Matrix z_t;
    Matrix r_t;
    Matrix h_bar_t;
    Matrix h_t;
};

#endif  // RNN_GRU_H
