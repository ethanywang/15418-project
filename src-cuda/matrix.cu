//
// Created by yuwang on 2020-04-22.
//

#include "matrix.h"
#include "cuda_operator.h"

#include <cstring>

#include <cuda.h>
#include <cuda_runtime.h>
#include <driver_functions.h>

Matrix::Matrix() : _data(nullptr), _M(0), _N(0), _size(0) {};

Matrix::Matrix(int M, int N, bool zero) : _M(M), _N(N), _size(M * N) {
    assert(M > 0);
    assert(N > 0);

    // _data = new float[m._size];
    cudaMallocManaged((void**)&_data, _size * sizeof(float));
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
    // _data = new float[m._size];
    cudaMallocManaged((void**)&_data, _size * sizeof(float));
    memcpy(_data, m._data, m._size * sizeof(float));
}

Matrix::~Matrix() {
    if (_data != nullptr)
        cudaFree(_data);
};

Matrix &Matrix::operator=(Matrix &&m) noexcept {
    if (&m == this) {
        return *this;
    }
    // this->_data = new float[m._size];
    // memcpy(this->_data, m._data, m._size * sizeof(float));
    this->_data = m._data;
    this->_M = m._M;
    this->_N = m._N;
    this->_size = m._size;
    
    m._data = nullptr;
    
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
    // auto *data = new float[this->_size];
    float *data;
    cudaMallocManaged((void**)&data, _size * sizeof(float));
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

    // seq
    // auto *data = new float[_size];
    float *data;
    cudaMallocManaged((void**)&data, _size * sizeof(float));
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

    // auto *data = new float[_size];
    float *data;
    cudaMallocManaged((void**)&data, _size * sizeof(float));
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
    // auto *data = new float[_M * d._N];
    float *data;
    cudaMallocManaged((void**)&data, _size * sizeof(float));
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
    cuMul(_data, d._data, data, _M, d._N, _N);
    /* allocate new data */
    return Matrix(data, _M, d._N);
}

float *Matrix::data() {
    return _data;
}

Matrix Matrix::operator-() {
    float *data;
    cudaMallocManaged((void**)&data, _size * sizeof(float));

    // for (int i = 0; i < _size; i++) {
    //     data[i] = -_data[i];
    // }
    cuNumMinus(_data, data, _size, 0);

    return Matrix(data, _M, _N);
}

Matrix Matrix::operator-(const float &num) {
    float *data;
    cudaMallocManaged((void**)&data, _size * sizeof(float));

    // for (int i = 0; i < _size; i++) {
    //     data[i] = _data[i] - num;
    // }
    cuNumMinus(_data, data, _size, num);

    return Matrix(data, _M, _N);
}

Matrix Matrix::operator+(const float &num) {
    float *data;
    cudaMallocManaged((void**)&data, _size * sizeof(float));

    // for (int i = 0; i < _size; i++) {
    //     data[i] = data[i] + num;
    // }
    cuNumAdd(_data, data, _size, num);

    return Matrix(data, _M, _N);
}

std::mt19937 Matrix::_rd = std::mt19937(0);