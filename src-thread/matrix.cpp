//
// Created by yuwang on 2020-04-22.
//

#include "matrix.h"

#include <cstring>
#include <thread>
#include <vector>

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

void T_worker(Matrix &m, Matrix &_m, int start, int end) {
    int _M = _m.size(0);
    int _N = _m.size(1);
    for (int i = start; i < end; i++)
        for (int j = 0; j < _N; j++)
            m.data()[j * _M + i] = _m.data()[i * _N + j];
}

Matrix Matrix::T() {
    if (_size == 0) {
        return Matrix();
    }
    auto *data = new float[this->_size];
    auto M = _N;
    auto N = _M;
    Matrix m(data, M, N);

    // for (int i = 0; i < _M; i++) {
    //     for (int j = 0; j < _N; j++) {
    //         data[j * _M + i] = this->_data[i * _N + j];
    //     }
    // }
    std::vector<std::thread> threads;
    for (int i = 0; i < THREAD_NUM; i++) {
        int strip = (_M + THREAD_NUM - 1)/THREAD_NUM;
        int start = i * strip;
        int end = _M < (i+1) *strip ? _M : (i + 1) * strip;
        threads.push_back(
            std::thread(T_worker, std::ref(m), std::ref(*this), start, end)
        );
    }

    for (auto& th : threads) th.join();

    return m;
}

void add_worker(Matrix &m, Matrix &_m1, Matrix &_m2, int start, int end) {
    for (int i = start; i < end; i++)
        m.data()[i] = _m1.data()[i] + _m2.data()[i];
}

Matrix Matrix::add(Matrix &d) {
    assert(_M == d._M);
    assert(_N == d._N);

    auto *data = new float[_size];
    Matrix m(data, _M, _N);
    // for (int i = 0; i < _M; i++) {
    //     for (int j = 0; j < _N; j++) {
    //         data[i * _N + j] = _data[i * _N + j] + d._data[i * _N + j];
    //     }
    // }
    std::vector<std::thread> threads;
    for (int i = 0; i < THREAD_NUM; i++) {
        int strip = (_size + THREAD_NUM - 1)/THREAD_NUM;
        int start = i * strip;
        int end = _size < (i+1) *strip ? _size : (i + 1) * strip;
        threads.push_back(
            std::thread(add_worker, std::ref(m), std::ref(*this), std::ref(d), start, end)
        );
    }

    for (auto& th : threads) th.join();

    return m;
}

void dot_worker(Matrix &m, Matrix &_m1, Matrix &_m2, int start, int end) {
    for (int i = start; i < end; i++)
        m.data()[i] = _m1.data()[i] * _m2.data()[i];
}

Matrix Matrix::dot(Matrix &d) {
    assert(_M == d._M);
    assert(_N == d._N);

    auto *data = new float[_size];
    Matrix m(data, _M, _N);
    // for (int i = 0; i < _M; i++) {
    //     for (int j = 0; j < _N; j++) {
    //         data[i * _N + j] = _data[i * _N + j] * d._data[i * _N + j];
    //     }
    // }
    std::vector<std::thread> threads;
    for (int i = 0; i < THREAD_NUM; i++) {
        int strip = (_size + THREAD_NUM - 1)/THREAD_NUM;
        int start = i * strip;
        int end = _size < (i+1) *strip ? _size : (i + 1) * strip;
        threads.push_back(
            std::thread(dot_worker, std::ref(m), std::ref(*this), std::ref(d), start, end)
        );
    }

    for (auto& th : threads) th.join();

    return m;
}

void mul_worker(Matrix &m, Matrix &_m1, Matrix &_m2, int start, int end) {
    int _K = _m1.size(1);
    int _N = _m2.size(1);
    for (int i = start; i < end; i++)
        for (int j = 0; j < _N; j++) {
            for (int k = 0; k < _K; k++) {
                m.data()[i * _N + j] +=
                        _m1.data()[i * _K + k] * _m2.data()[k * _N + j];
            }
        }

}

Matrix Matrix::mul(Matrix &d) {
    /* assert dimension */
    assert(_N == d._M);

    /* do calculation */
    auto *data = new float[_M * d._N]();
    Matrix m(data, _M, d._N);
    // for (int i = 0; i < _M; i++) {
    //     for (int j = 0; j < d._N; j++) {
    //         for (int k = 0; k < _N; k++) {
    //             data[i * d._N + j] +=
    //                     _data[i * _N + k] * d._data[k * d._N + j];
    //         }
    //     }
    // }

    std::vector<std::thread> threads;
    for (int i = 0; i < THREAD_NUM; i++) {
        int strip = (_M + THREAD_NUM - 1)/THREAD_NUM;
        int start = i * strip;
        int end = _M < (i+1) *strip ? _M : (i + 1) * strip;
        threads.push_back(
            std::thread(mul_worker, std::ref(m), std::ref(*this), std::ref(d), start, end)
        );
    }

    for (auto& th : threads) th.join();

    /* allocate new data */
    return m;
}

float *Matrix::data() {
    return _data;
}

void minus_void_worker(Matrix &m, Matrix &_m, int start, int end) {
    for (int i = start; i < end; i++)
        m.data()[i] = -_m.data()[i];
}

Matrix Matrix::operator-() {
    auto *data = new float[_size];
    Matrix m(data, _M, _N);
    // for (int i = 0; i < _size; i++) {
    //     data[i] = -_data[i];
    // }

    std::vector<std::thread> threads;
    for (int i = 0; i < THREAD_NUM; i++) {
        int strip = (_size + THREAD_NUM - 1)/THREAD_NUM;
        int start = i * strip;
        int end = _size < (i+1) *strip ? _size : (i + 1) * strip;
        threads.push_back(
            std::thread(minus_void_worker, std::ref(m), std::ref(*this), start, end)
        );
    }

    for (auto& th : threads) th.join();

    return m;
}

void minus_float_worker(Matrix &m, Matrix &_m, const float &num, int start, int end) {
    for (int i = start; i < end; i++)
        m.data()[i] = _m.data()[i] - num;
}

Matrix Matrix::operator-(const float &num) {
    auto *data = new float[_size];
    Matrix m(data, _M, _N);
    // for (int i = 0; i < _size; i++) {
    //     data[i] = _data[i] - num;
    // }

    std::vector<std::thread> threads;
    for (int i = 0; i < THREAD_NUM; i++) {
        int strip = (_size + THREAD_NUM - 1)/THREAD_NUM;
        int start = i * strip;
        int end = _size < (i+1) *strip ? _size : (i + 1) * strip;
        threads.push_back(
            std::thread(minus_float_worker, std::ref(m), std::ref(*this), std::ref(num), start, end)
        );
    }

    for (auto& th : threads) th.join();

    return m;
}

void add_float_worker(Matrix &m, Matrix &_m, float num, int start, int end) {
    for (int i = start; i < end; i++)
        m.data()[i] = _m.data()[i] + num;
}

Matrix Matrix::operator+(const float &num) {
    auto *data = new float[_size];
    Matrix m(data, _M, _N);
    // for (int i = 0; i < _size; i++) {
    //     data[i] = _data[i] + num;
    // }

    std::vector<std::thread> threads;
    for (int i = 0; i < THREAD_NUM; i++) {
        int strip = (_size + THREAD_NUM - 1)/THREAD_NUM;
        int start = i * strip;
        int end = _size < (i+1) *strip ? _size : (i + 1) * strip;
        threads.push_back(
            std::thread(add_float_worker, std::ref(m), std::ref(*this), num, start, end)
        );
    }

    for (auto& th : threads) th.join();

    return m;
}

std::mt19937 Matrix::_rd = std::mt19937(0);