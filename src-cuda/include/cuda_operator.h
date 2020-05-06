#ifndef RNN_CUDA_OPERATOR_H
#define RNN_CUDA_OPERATOR_H

// Matrix
void cuAdd(float *src1, float *src2, float *dst, int M, int N);

void cuMul(float *A, float *B, float *C, int M, int N, int R);

void cuDot(float *A, float *B, float *C, int M, int N);

void cuT(float *A, float *B, int M, int N);

// Operators
void cuSigmoid(float *src, float *dst, int length);

void cuTanh(float *src, float *dst, int length);

void cuUniMinus(float *src, float *dst, int length);

void cuMinus(float *src, float *dst, int length, const float &num);

void cuAddition(float *src, float *dst, int length, const float &num);

// Network
/* gru variables */

void cu_gru_forward(float *input, float *h, float *output, int input_m, int input_n, int h_m, int h_n);

void gru_forward_setup(int d, int h);

void gru_forward_clear();

void cu_mat_mul(float* A, float* B, float* C, int M, int N, int R);

void cu_mat_dot(float *A, float *B, float *C, int M, int N);

void cu_mat_add(float *src1, float *src2, float *dst, int M, int N);

void cu_mat_sigmoid(float *src, float *dst, int length);

void cu_mat_tanh(float *src, float *dst, int length);

void cu_minus(float* arr, int size, const float &num);

void gru_forward_setup(int i_m, int i_n, int h_m, int h_n);

void gru_mat_setup(float** ptr, int m, int n, bool init);

void gru_mat_clean(float *ptr);

void gru_forward_clear();
#endif // RNN_CUDA_OPERATOR_H
