//
// Created by yuwang4 on 2020-04-21.
//

#include <gru.h>


Matrix GRU::forward(Matrix &x, Matrix &h) {
    this->h_t_1 = h;
    this->x = x;
    this->z_t = this->z_act.forward(this->Wzh.mul(h_t_1).add(this->Wzx.mul(x)));
    this->r_t = this->r_act.forward(this->Wrh.mul(h_t_1).add(this->Wrx.mul(x)));
    this->h_bar_t = this->h_act.forward(this->Wh.mul(this->r_t.dot(h_t_1)).add(this->Wx.mul(x)));
    this->h_t = this->z_t.dot(this->h_t_1).add(this->z_t.dot(this->h_bar_t));

    return this->h_t;

}