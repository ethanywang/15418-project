//
// Created by yuwang4 on 2020-04-21.
//

#include "gru.h"
#include "cuda_operator.h"

inline static void gru_mat_setup(float** ptr, int m, int n, bool init) {
    cudaMalloc(ptr, m * n * sizeof(float));
    // cudaError_t code =  printf("%s\n",cudaGetErrorString(code));
    if (!init) return;
    /* initialize */
    cudaMemset(ptr, 0, m * n * sizeof(float));
}

Matrix GRU::forward(Matrix &x, Matrix &hid) {
    float *tmp1, *tmp2, *tmp3, *tmp4, *tmp5, *tmp6, *tmp7, *tmp8, *tmp9;
    float *mid1, *mid2, *mid3, *mid4;
    int h = hid.size(0);
    int i_n = x.size(1);
    int d = x.size(0);

    gru_mat_setup(&tmp1, h, i_n, false); 
    gru_mat_setup(&tmp2, h, i_n, false);
    gru_mat_setup(&tmp3, h, i_n, false);
    gru_mat_setup(&tmp4, h, i_n, false);
    gru_mat_setup(&tmp5, h, i_n, false);
    gru_mat_setup(&tmp6, h, i_n, false);
    gru_mat_setup(&tmp7, h, i_n, false);
    gru_mat_setup(&tmp8, h, i_n, false);
    gru_mat_setup(&tmp9, h, i_n, false);
    gru_mat_setup(&mid1, h, i_n, false);
    gru_mat_setup(&mid2, h, i_n, false);
    gru_mat_setup(&mid3, h, i_n, false);
    gru_mat_setup(&mid4, h, i_n, false);

    // auto tmp1 = std::move(this->Wzx.mul(x));
    // auto tmp2 = std::move(this->Wrx.mul(x));
    cuMul(Wzx.data(), x.data(), tmp1, h, d, i_n);
    cuMul(Wrx.data(), x.data(), tmp2, h, d, i_n);
    cudaDeviceSynchronize();

    // auto tmp3 = std::move(this->z_act.forward(this->Wzh.mul(h).add(tmp1)));
    // auto tmp4 = std::move(this->r_act.forward(this->Wrh.mul(h).add(tmp2)));
    cuMul(Wzh.data(), hid.data(), mid1, h, h, i_n);    
    cuMul(Wzh.data(), hid.data(), mid2, h, h, i_n); 
    cuAdd(mid1, tmp1, mid3, h, i_n);
    cuAdd(mid2, tmp2, mid4, h, i_n);
    cuSigmoid(mid3, tmp3, h * i_n);
    cuSigmoid(mid4, tmp4, h * i_n);

    // auto tmp5 = std::move(tmp4.dot(h));
    cuDot(tmp4, hid.data(), tmp5, h, i_n);
    // auto tmp6 = std::move(this->Wx.mul(x));
    cuMul(tmp6, x.data(), tmp6, h, d, i_n);
    // auto tmp7 = std::move(this->h_act.forward(this->Wh.mul(tmp5).add(tmp6)));
    cuMul(Wh.data(), tmp5, mid1, h, h, i_n);
    cuAdd(mid1, tmp6, mid2, h, i_n);
    cuSigmoid(mid2, tmp7, h * i_n);

    // auto tmp8 = std::move(tmp3.dot(tmp7));
    cuDot(tmp3, tmp7, tmp8, h, i_n);

    // return std::move((-tmp3 + 1.0).dot(h).add(tmp8));
    cuNumMinus(tmp3, mid3, h * i_n, 0.0);
    cuNumMinus(mid3, mid4, h * i_n, -1.0);
    cuDot(mid4, hid.data(), mid1, h, i_n);
    cuAdd(tmp8, mid1, tmp9, h, i_n);

    cudaFree(tmp1);
    cudaFree(tmp2);
    cudaFree(tmp3);
    cudaFree(tmp4);
    cudaFree(tmp5);
    cudaFree(tmp6);
    cudaFree(tmp7);
    cudaFree(tmp8);
    cudaFree(mid1);
    cudaFree(mid2);
    cudaFree(mid3);
    cudaFree(mid4);

    return Matrix(tmp9, h, i_n);
}