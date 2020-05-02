export void vecdot_ispc(uniform float input1[], uniform float input2[],
                        uniform int length, uniform float output[]) {
    foreach(i = 0 ... length) {
        output[i] = input1[i] * input2[i];
    }
}

export void vecadd_ispc(uniform float input1[], uniform float input2[],
                        uniform int length, uniform float output[]) {
    foreach(i = 0 ... length) {
        output[i] = input1[i] + input2[i];
    }
}

export void vecsub_ispc(uniform float input1[], uniform float input2[],
                        uniform int length, uniform float output[]) {
    foreach(i = 0 ... length) {
        output[i] = input1[i] - input2[i];
    }
}

export void veclinear_ispc(uniform float input[], uniform float a, uniform float b,
                           uniform int length, uniform float output[]) {
    foreach(i = 0 ... length) {
        output[i] = a * input[i] + b;
    }
}

export void matT_ispc(uniform float input[], uniform int M,
                      uniform int N, uniform float output[]) {
    foreach(j = 0 ... N, i = 0 ... M) {
        output[j * M + i] = input[i * N + j];
    }
}

export void matmul_ispc(uniform float input1[], uniform float input2[],
                        uniform int M, uniform int K, uniform int N,
                        uniform float output[]) {
    foreach(k = 0 ... K, j = 0 ... N, i = 0 ... M) {
        output[i * N + j] += input1[i * K + k] * input2[k * N + j];
    }
}
