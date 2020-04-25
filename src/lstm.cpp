//
// Created by yuwang on 4/24/20.
//

#include <lstm.h>

Matrix LSTM::forward(Matrix &x, Matrix &h, Matrix &c_t) {
    this->h_t_1 = h;
    this->x = x;

    auto tmp1 = this->Wfx.mul(x);
    this->f_t = this->f_act(this->Wfh.mul(h_t_1).add(tmp1));

    auto tmp2 = this->Wix.mul(x);
    this->i_t = this->x_act(this->Wih.mul(h_t_1).add(tmp2));

    auto tmp3 = this->Wcx.mul(x);
    this->c_bar_t = this->c_bar_act(this->Wch.mul(h_t_1).add(tmp3));

    auto tmp4 = (-this->f_t + 1.0).dot(this->c_bar_t);
    this->c_t = this->f_t.dot(this->c_t_1).add(tmp4);
    c_t = this->c_t;

    auto tmp5 = this->Wox.mul(x);
    auto tmp6 = this->c_act(this->c_t);
    this->o_t = this->o_act(this->Woh.mul(h_t_1).add(tmp5));
    this->h_t = this->o_t.dot(tmp6);

    return this->h_t;
}

