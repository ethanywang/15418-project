#ifndef uint
#define uint unsigned int
#endif

class CudaOperator {
    public:
        double *add(double* data);
        double *dot(double* data);
        double *mul(double* data);
        void setup(int size, double* data);
};