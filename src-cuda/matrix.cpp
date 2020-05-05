//
// Created by yuwang on 2020-04-22.
//

#include "matrix.h"
#include "cuda_operator.h"

#include <cstring>

Matrix &Matrix::operator=(const Matrix &m) {
    if (&m == this) {
        return *this;
    }
    this->_data = new float[m._size];
    memcpy(this->_data, m._data, m._size * sizeof(float));

    this->_M = m._M;
    this->_N = m._N;
    this->_size = m._size;

    return *this;
}


int Matrix::size() {
    return _size;
}

int Matrix::size(int dim) {
    switch (dim) {
        case 0:
            return _M;
        case 1:
            return _N;
        default:
            throw dim;
    }
}

Matrix Matrix::T() {
    if (_size == 0) {
        return Matrix();
    }
    auto *data = new float[this->_size];
    auto M = _N;
    auto N = _M;

    // for (int i = 0; i < _M; i++) {
    //     for (int j = 0; j < _N; j++) {
    //         data[j * _M + i] = this->_data[i * _N + j];
    //     }
    // }
    cuT(this->_data, data, M, N);
    return Matrix(data, M, N);
}

Matrix Matrix::add(Matrix &d) {
    assert(_M == d._M);
    assert(_N == d._N);

    // seq
    auto *data = new float[_size];
    // if (this->_dev == SEQ) {
    //     for (int i = 0; i < _M; i++) {
    //         for (int j = 0; j < _N; j++) {
    //             data[i * _N + j] = _data[i * _N + j] + d._data[i * _N + j];
    //         }
    //     }
    // }

    // cuda-parallel
    cuAdd(_data, d._data, data, _M, _N);

    return Matrix(data, _M, _N);
}

Matrix Matrix::dot(Matrix &d) {
    assert(_M == d._M);
    assert(_N == d._N);

    auto *data = new float[_size];
    // if (this->_dev == SEQ) {
    //     for (int i = 0; i < _M; i++) {
    //         for (int j = 0; j < _N; j++) {
    //             data[i * _N + j] = _data[i * _N + j] * d._data[i * _N + j];
    //         }
    //     }
    // }
    cuDot(_data, d._data, data, _M, _N);
    return Matrix(data, _M, _N);
}

Matrix Matrix::mul(Matrix &d) {
    /* assert dimension */
    assert(_N == d._M);

    /* do calculation */
    auto *data = new float[_M * d._N];
    // if (this->_dev == SEQ) {
    //     for (int i = 0; i < _M; i++) {
    //         for (int j = 0; j < d._N; j++) {
    //             for (int k = 0; k < _N; k++) {
    //                 data[i * d._N + j] +=
    //                         _data[i * _N + k] * d._data[k * d._N + j];
    //             }
    //         }
    //     }
    // }
    cuMul(_data, d._data, data, _M, d._N);
    /* allocate new data */
    return Matrix(data, _M, d._N);
}

float *Matrix::data() {
    return _data;
}

Matrix Matrix::operator-() {
    auto *data = new float[_size];
    for (int i = 0; i < _size; i++) {
        data[i] = -_data[i];
    }
    // cuUniMinus(_data, data, _M * _N);
    return Matrix(data, _M, _N);
}

Matrix Matrix::operator-(const float &num) {
    auto *data = new float[_size];
    for (int i = 0; i < _size; i++) {
        data[i] = _data[i] - num;
    }
    // cuMinus(_data, data, _M * _N, num);
    return Matrix(data, _M, _N);
}

Matrix Matrix::operator+(const float &num) {
    auto *data = new float[_size];
    for (int i = 0; i < _size; i++) {
        data[i] = _data[i] + num;
    }
    // cuAddition(_data, data, _M * _N, num);
    return Matrix(data, _M, _N);
}

std::mt19937 Matrix::_rd = std::mt19937(0);