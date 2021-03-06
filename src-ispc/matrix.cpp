//
// Created by yuwang on 2020-04-22.
//
#include "matrix.h"
#include "matrix_ispc.h"

#include <cstring>

using namespace ispc;

Matrix::Matrix() : _data(nullptr), _M(0), _N(0), _size(0) {};

Matrix::Matrix(int M, int N, bool zero) : _M(M), _N(N), _size(M * N) {
    assert(M > 0);
    assert(N > 0);
    _data = new float[_size]();
    if (!zero) {
        for (int i = 0; i < _size; i++) {
            _data[i] = static_cast<float>(_rd());
        }
    }
}

Matrix::Matrix(float *data, int M, int N) : _data(data), _M(M), _N(N), _size(M * N) {
    assert(data != nullptr);
    assert(M > 0);
    assert(N > 0);
};

Matrix::Matrix(const Matrix &m) {
    _M = m._M;
    _N = m._N;
    _size = m._size;
    _data = new float[m._size];
    memcpy(_data, m._data, m._size * sizeof(float));
}

Matrix::~Matrix() {
    delete[] _data;
};

Matrix &Matrix::operator=(Matrix &&m) noexcept {
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

//    for (int i = 0; i < _M; i++) {
//        for (int j = 0; j < _N; j++) {
//            data[j * _M + i] = this->_data[i * _N + j];
//        }
//    }

    matT_ispc(this->_data, _M, _N, data);

    return Matrix(data, M, N);
}

Matrix Matrix::add(Matrix &d) {
    assert(_M == d._M);
    assert(_N == d._N);

    auto *data = new float[_size];
//    for (int i = 0; i < _M; i++) {
//        for (int j = 0; j < _N; j++) {
//            data[i * _N + j] = _data[i * _N + j] + d._data[i * _N + j];
//        }
//    }

    vecadd_ispc(_data, d._data, _size, data);

    return Matrix(data, _M, _N);
}

Matrix Matrix::dot(Matrix &d) {
    assert(_M == d._M);
    assert(_N == d._N);

    auto *data = new float[_size];
//    for (int i = 0; i < _M; i++) {
//        for (int j = 0; j < _N; j++) {
//            data[i * _N + j] = _data[i * _N + j] * d._data[i * _N + j];
//        }
//    }

    vecdot_ispc(_data, d._data, _size, data);

    return Matrix(data, _M, _N);
}

Matrix Matrix::mul(Matrix &d) {
    /* assert dimension */
    assert(_N == d._M);

    /* do calculation */
    auto *data = new float[_M * d._N]();

//    for (int i = 0; i < _M; i++) {
//        for (int j = 0; j < d._N; j++) {
//            for (int k = 0; k < _N; k++) {
//                data[i * d._N + j] +=
//                        _data[i * _N + k] * d._data[k * d._N + j];
//            }
//        }
//    }

    matmul_ispc(_data, d._data, _M, _N, d._N, data);

    /* allocate new data */
    return Matrix(data, _M, d._N);
}

float *Matrix::data() {
    return _data;
}

Matrix Matrix::operator-() {
    auto *data = new float[_size];
//    for (int i = 0; i < _size; i++) {
//        data[i] = -_data[i];
//    }

    veclinear_ispc(_data, -1.0, 0.0, _size, data);

    return Matrix(data, _M, _N);
}

Matrix Matrix::operator-(const float &num) {
    auto *data = new float[_size];
//    for (int i = 0; i < _size; i++) {
//        data[i] = _data[i] - num;
//    }

    veclinear_ispc(_data, 1.0, -num, _size, data);

    return Matrix(data, _M, _N);
}

Matrix Matrix::operator+(const float &num) {
    auto *data = new float[_size];
//    for (int i = 0; i < _size; i++) {
//        data[i] = _data[i] + num;
//    }

    veclinear_ispc(_data, 1.0, num, _size, data);

    return Matrix(data, _M, _N);
}

std::mt19937 Matrix::_rd = std::mt19937(0);