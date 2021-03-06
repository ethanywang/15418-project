//
// Created by yuwang4 on 2020-04-21.
//

#include "gru.h"

Matrix GRU::forward(Matrix &x, Matrix &h) {
    this->h_t_1 = h;
    this->x = x;

    auto tmp1 = this->Wzx.mul(x);
    auto tmp2 = this->Wrx.mul(x);

    this->z_t = this->z_act.forward(this->Wzh.mul(h_t_1).add(tmp1));
    this->r_t = this->r_act.forward(this->Wrh.mul(h_t_1).add(tmp2));

    auto tmp3 = this->r_t.dot(h_t_1);
    auto tmp4 = this->Wx.mul(x);
    this->h_bar_t = this->h_act.forward(this->Wh.mul(tmp3).add(tmp4));

    auto tmp5 = this->z_t.dot(this->h_bar_t);
    this->h_t = (-this->z_t + 1.0).dot(this->h_t_1).add(tmp5);

    return this->h_t;

}