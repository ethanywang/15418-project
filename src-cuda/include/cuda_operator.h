#ifndef RNN_CUDA_OPERATOR_H
#define RNN_CUDA_OPERATOR_H

// Matrix
void cuAdd(float *src1, float *src2, float *dst, int M, int N);

void cuMul(float *A, float *B, float *C, int M, int N);

void cuDot(float *A, float *B, float *C, int M, int N);

void cuT(float *A, float *B, int M, int N);

// Operators
void cuSigmoid(float *src, float *dst, int length);

void cuTanh(float *src, float *dst, int length);

void cuUniMinus(float *src, float *dst, int length);

void cuMinus(float *src, float *dst, int length, const float &num);

void cuAddition(float *src, float *dst, int length, const float &num);
#endif // RNN_CUDA_OPERATOR_H
