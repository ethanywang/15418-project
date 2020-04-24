//
// Created by yuwang on 2020-04-22.
//

#ifndef RNN_DATA_H
#define RNN_DATA_H

#include <vector>
#include <random>
#include <cassert>
#include <cstring>

using namespace std;

class Matrix {
public:
    Matrix() : _data(nullptr), _M(0), _N(0), _size(0) {};

    Matrix(int M, int N, bool zero = true) : _M(M), _N(N), _size(M * N) {
        assert(M > 0);
        assert(N > 0);
        _data = new double[_size]();
        if (!zero) {
            random_device rd;
            for (int i = 0; i < _size; i++) {
                _data[i] = static_cast<double>(rd());
            }
        }
    }

    Matrix(double *data, int M, int N) : _data(data), _M(M), _N(N), _size(M * N) {
        assert(data != nullptr);
        assert(M > 0);
        assert(N > 0);
    };

    Matrix(const Matrix &m) {
        _M = m._M;
        _N = m._N;
        _size = m._size;
        _data = new double[m._size];
        memcpy(_data, m._data, m._size * sizeof(double));
    }

    ~Matrix() {
        delete[] _data;
    };

    Matrix &operator=(const Matrix &);

    int size();

    int size(int dim);

    Matrix T();

    Matrix add(Matrix &);

    Matrix mul(Matrix &);

    Matrix dot(Matrix &);

    double *data();

private:
    int _M;
    int _N;
    int _size;
    double *_data;
};


#endif //RNN_DATA_H
