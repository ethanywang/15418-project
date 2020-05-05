//
// Created by yuwang on 4/24/20.
//

#include "lstm.h"

Matrix LSTM::forward(Matrix &x, Matrix &h, Matrix &c_t_1, Matrix &c_t) {
    auto tmp1 = std::move(this->Wfx.mul(x));
    auto tmp7 = std::move(this->f_act(this->Wfh.mul(h).add(tmp1)));

    auto tmp2 = std::move(this->Wix.mul(x));
    auto tmp8 = std::move(this->x_act(this->Wih.mul(h).add(tmp2)));

    auto tmp3 = std::move(this->Wcx.mul(x));
    auto tmp9 = std::move(this->c_bar_act(this->Wch.mul(h).add(tmp3)));

    auto tmp4 = std::move(tmp8.dot(tmp9));
    auto tmp10 = std::move(tmp7.dot(c_t_1).add(tmp4));
    c_t = std::move(tmp10);

    auto tmp5 = std::move(this->Wox.mul(x));
    auto tmp6 = std::move(this->c_act(tmp10));
    auto tmp11 = std::move(this->o_act(this->Woh.mul(h).add(tmp5)));

    return std::move(tmp11.dot(tmp6));
}

