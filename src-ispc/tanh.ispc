
export void tanh_ispc(uniform float input[], uniform int length, uniform float output[]) {
    foreach(i = 0 ... length) {
        float pos_e = exp(input[i]);
        float neg_e = exp(-input[i]);
        output[i] = (pos_e - neg_e) / (pos_e + neg_e);
    }
}