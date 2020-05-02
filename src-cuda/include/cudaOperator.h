#ifndef uint
#define uint unsigned int
#endif

class CudaOperator {
    public:
        void cuAdd(double* src1, double *src2, double *dst, int M, int N);
        void cuMul(double* A, double* B, double *C, int M, int N);
        void cuDot(double* A, double *B, double *C, int M, int N);
        void setup(int size, double* data);
};