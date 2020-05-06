//
// Created by yuwang on 2020-04-22.
//

#ifndef RNN_DATA_H
#define RNN_DATA_H

#include <random>
#include <cassert>
#include <cstring>



#define THREAD_NUM 8

class Matrix {
public:
    Matrix();

    Matrix(int M, int N, bool zero = true);

    Matrix(float *data, int M, int N);

    Matrix(const Matrix &m);

    ~Matrix();

    Matrix operator-();

    Matrix operator-(const float &);

    Matrix operator+(const float &);

    Matrix &operator=(Matrix &&) noexcept;

    int size();

    int size(int dim);

    Matrix T();

    Matrix add(Matrix &);

    Matrix mul(Matrix &);

    Matrix dot(Matrix &);

    float *data();

private:
    float *_data;

    int _M;
    int _N;
    int _size;

    static std::mt19937 _rd;
};


#endif //RNN_DATA_H
