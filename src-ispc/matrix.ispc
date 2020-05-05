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

#define BLOCK_WIDTH 100
#define BLOCK_HEIGHT 100

task void matT_ispc_task(uniform float input[], uniform int M,
                      uniform int N, uniform float output[]) {
    uniform int bstride = (N + BLOCK_WIDTH - 1) / BLOCK_WIDTH;

    uniform int ystart = (taskIndex % bstride) * BLOCK_WIDTH;
    uniform int yend = min(ystart + BLOCK_WIDTH, N);
    uniform int xstart = (taskIndex / bstride) * BLOCK_HEIGHT;
    uniform int xend = min(xstart + BLOCK_HEIGHT, M);

    foreach (j = ystart... yend, i = xstart... xend) {
       output[j * M + i] = input[i * N + j];
    }
}

export void matT_ispc(uniform float input[], uniform int M,
                      uniform int N, uniform float output[]) {
    // foreach(j = 0 ... N, i = 0 ... M) {
    //    output[j * M + i] = input[i * N + j];
    // }
    uniform int bWidthstride = (N + BLOCK_WIDTH - 1) / BLOCK_WIDTH;
    uniform int bHeightstride = (M + BLOCK_HEIGHT - 1) / BLOCK_HEIGHT;
    uniform int taskCount = bWidthstride * bHeightstride;

    launch[taskCount] matT_ispc_task(input, M, N, output);
}

export void matmul_ispc(uniform float input1[], uniform float input2[],
                        uniform int M, uniform int K, uniform int N,
                        uniform float output[]) {
    foreach(k = 0 ... K, j = 0 ... N, i = 0 ... M) {
        output[i * N + j] += input1[i * K + k] * input2[k * N + j];
    }
}
