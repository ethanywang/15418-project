#ifndef uint
#define uint unsigned int
#endif

class CudaOperator {
    public:
        // Matrix
        void cuAdd(double* src1, double *src2, double *dst, int M, int N);
        void cuMul(double* A, double* B, double *C, int M, int N);
        void cuDot(double* A, double *B, double *C, int M, int N);

        // Operators
        void cuSigmoid(double* src, double* dst, int length);
        void cuTanh(double *src, double* dst, int length);
        void setup(int size, double* data);
};