#ifndef RNN_CUDA_OPERATOR_H
#define RNN_CUDA_OPERATOR_H

// Matrix
void cuAdd(float *src1, float *src2, float *dst, int M, int N);

void cuMul(float *A, float *B, float *C, int M, int N);

void cuDot(float *A, float *B, float *C, int M, int N);

// Operators
void cuSigmoid(float *src, float *dst, int length);

void cuTanh(float *src, float *dst, int length);

#endif // RNN_CUDA_OPERATOR_H
