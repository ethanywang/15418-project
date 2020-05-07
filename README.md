# 15418/618 Final Project - Parallel Recurrent Neural Network

This is the code for the CMU course 15418/618 final projects, which aims to explore different parallelization technologies on the RNN problem scope.

### Requirement

- CMake >= 3.8
- CUDA >= 6.0
- ISPC
- OpenMP
- g++/clang with CXX 11 support

### Environment
CMU Andrew GHC machines
```
export PATH=/usr/local/depot/cuda/bin:${PATH}
export PATH=/usr/lib64/openmpi/bin:${PATH}
export PATH=/usr/local/depot/ispc-v1.9.1-linux/:${PATH}
export PARH=/usr/local/depot/ispc/bin/:${PATH}
export LD_LIBRARY_PATH=/usr/local/depot/cuda/lib64/:${LD_LIBRARY_PATH}
```

### Build
```
mkdir build
cd build
cmake ..
make
```

Note on GHC machines, cmake with version 3.+ is named `cmake3`.

### Benchmarking
```
cd build
make bench
```
Run `make bench` command will directly start benchmarking and printing out results to the stdout.
