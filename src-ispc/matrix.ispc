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

#define BLOCK_SIZE 100

task void matT_ispc_task(uniform float input[], uniform int M,
                      uniform int N, uniform float output[]) {
    uniform int bstride = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uniform int ystart = (taskIndex % bstride) * BLOCK_SIZE;
    uniform int yend = min(ystart + BLOCK_SIZE, N);
    uniform int xstart = (taskIndex / bstride) * BLOCK_SIZE;
    uniform int xend = min(xstart + BLOCK_SIZE, M);

    foreach (j = ystart... yend, i = xstart... xend) {
       output[j * M + i] = input[i * N + j];
    }
}

export void matT_ispc(uniform float input[], uniform int M,
                      uniform int N, uniform float output[]) {
    // foreach(j = 0 ... N, i = 0 ... M) {
    //    output[j * M + i] = input[i * N + j];
    // }
    uniform int bWidthstride = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uniform int bHeightstride = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uniform int taskCount = bWidthstride * bHeightstride;

    launch[taskCount] matT_ispc_task(input, M, N, output);
}


task void matmul_ispc_task(uniform float input1[], uniform float input2[],
                           uniform int M, uniform int K, uniform int N, 
                           uniform float output[]) {
    uniform int bstride = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;

    uniform int ystart = (taskIndex % bstride) * BLOCK_SIZE;
    uniform int yend = min(ystart + BLOCK_SIZE, N);
    uniform int xstart = (taskIndex / bstride) * BLOCK_SIZE;
    uniform int xend = min(xstart + BLOCK_SIZE, M);
 
    foreach (i = xstart... xend, j = ystart... yend) {
        float sum = 0.0;
        for (int k = 0; k < K; k++) {
            sum += input1[i * K + k] * input2[j * K + k];
        }
        output[i * N + j] = sum;
    }
}

export void matmul_ispc(uniform float input1[], uniform float input2[],
                        uniform int M, uniform int K, uniform int N,
                        uniform float output[]) {
    //foreach(k = 0 ... K, j = 0 ... N, i = 0 ... M) {
    //    output[i * N + j] += input1[i * K + k] * input2[k * N + j];
    //}
    uniform int bWidthstride = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uniform int bHeightstride = (K + BLOCK_SIZE - 1) / BLOCK_SIZE;
    uniform int taskCount = bWidthstride * bHeightstride;

    // Transpose the matrix first
    uniform float t_input2[1000*1000];
    launch[taskCount] matT_ispc_task(input2, K, N, t_input2);

    // Launch Multiply
    bHeightstride = (M + BLOCK_SIZE - 1) / BLOCK_SIZE;
    taskCount = bWidthstride * bHeightstride;
    launch[taskCount] matmul_ispc_task(input1, t_input2, M, K, N, output);
}
