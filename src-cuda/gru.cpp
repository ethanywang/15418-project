//
// Created by yuwang4 on 2020-04-21.
//

#include "gru.h"

Matrix GRU::forward(Matrix &x, Matrix &h) {
    auto tmp1 = std::move(this->Wzx.mul(x));
    auto tmp2 = std::move(this->Wrx.mul(x));

    auto tmp3 = std::move(this->z_act.forward(this->Wzh.mul(h).add(tmp1)));
    auto tmp4 = std::move(this->r_act.forward(this->Wrh.mul(h).add(tmp2)));

    auto tmp5 = std::move(tmp4.dot(h));
    auto tmp6 = std::move(this->Wx.mul(x));
    auto tmp7 = std::move(this->h_act.forward(this->Wh.mul(tmp5).add(tmp6)));

    auto tmp8 = std::move(tmp3.dot(tmp7));

    return std::move((-tmp3 + 1.0).dot(h).add(tmp8));

}