
export void sigmoid_ispc(uniform float input[], uniform int length, uniform float output[]) {
    foreach(i = 0 ... length) {
        output[i] = 1 / (1 + exp(input[i]));
    }
}