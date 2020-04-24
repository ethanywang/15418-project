//
// Created by yuwang on 2020-04-22.
//

#include <matrix.h>
#include <iostream>

Matrix &Matrix::operator=(const Matrix &m) {
    cout << "Here" << endl;
    if (&m == this) {
        return *this;
    }
    this->_data = m._data;
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
    auto *data = new double[this->_size];
    auto M = _N;
    auto N = _M;

    for (int i = 0; i < _M; i++) {
        for (int j = 0; j < _N; j++) {
            data[j * _M + i] = this->_data[i * _N + j];
        }
    }

    return Matrix(data, M, N);
}

Matrix Matrix::add(Matrix &d) {
    assert(_M == d._M);
    assert(_N == d._N);

    auto *data = new double[_size];
    for (int i = 0; i < _M; i++) {
        for (int j = 0; j < _N; j++) {
            data[i * _N + j] = _data[i * _N + j] + d._data[i * _N + j];
        }
    }

    return Matrix(data, _M, _N);
}

Matrix Matrix::dot(Matrix &d) {
    cout << _M << " " << d._M << " " << _N << " " << d._N << endl;
    assert(_M == d._M);
    assert(_N == d._N);

    auto *data = new double[_size];
    for (int i = 0; i < _M; i++) {
        for (int j = 0; j < _N; j++) {
            data[i * _N + j] = _data[i * _N + j] * d._data[i * _N + j];
        }
    }

    return Matrix(data, _M, _N);
}

Matrix Matrix::mul(Matrix &d) {
    /* assert dimension */
    assert(_N == d._M);

    /* do calculation */
    auto *data = new double[_M * d._N]();
    for (int i = 0; i < _M; i++) {
        for (int j = 0; j < d._N; j++) {
            for (int k = 0; k < _N; k++) {
                data[i * d._N + j] +=
                        _data[i * _N + k] * d._data[k * d._N + j];
            }
        }
    }

    /* allocate new data */
    return Matrix(data, _M, d._N);
}

double *Matrix::data() {
    return _data;
}

