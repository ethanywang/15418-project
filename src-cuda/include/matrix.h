//
// Created by yuwang on 2020-04-22.
//

#ifndef RNN_DATA_H
#define RNN_DATA_H

#include "cudaOperator.h"
#include "device.h"

#include <random>
#include <cassert>
#include <cstring>
#include <string>
#include <iostream>

using namespace std;

class Matrix {
public:
    Matrix() : _data(nullptr), _M(0), _N(0), _size(0), _dev(SEQ), _cu() {
        _dev = SEQ;
    };

    Matrix(int M, int N, bool zero = true) : _M(M), _N(N), _size(M * N), _cu() {
        assert(M > 0);
        assert(N > 0);
        _data = new float[_size]();
        if (!zero) {
            for (int i = 0; i < _size; i++) {
                _data[i] = static_cast<float>(_rd());
            }
        }
    }

    Matrix(int M, int N, Device device) : _M(M), _N(N), _size(M * N), _dev(device), _cu() {
        assert(M > 0);
        assert(N > 0);
        _data = new float[_size]();
        std::cout << "constructor cuda\n";
        if (device == GPU) {
            _cu = new CudaOperator();
        }
    }

    Matrix(float *data, int M, int N) : _data(data), _M(M), _N(N), _size(M * N), _dev(SEQ), _cu() {
        assert(data != nullptr);
        assert(M > 0);
        assert(N > 0);
    };

    Matrix(const Matrix &m) {
        _M = m._M;
        _N = m._N;
        _size = m._size;
        _data = new float[m._size];
        memcpy(_data, m._data, m._size * sizeof(float));
    }

    ~Matrix() {
        delete[] _data;
        delete _cu;
    };

    Matrix operator-() const;

    Matrix operator-(const float &) const;

    Matrix operator+(const float &) const;

    Matrix &operator=(const Matrix &);

    int size();

    int size(int dim);

    Matrix T();

    Matrix add(Matrix &);

    Matrix mul(Matrix &);

    Matrix dot(Matrix &);

    float *data();

    // CUDA
    CudaOperator *_cu;
    Device _dev;

private:
    float *_data;
    int _M;
    int _N;
    int _size;

    static mt19937 _rd;
};


#endif //RNN_DATA_H
