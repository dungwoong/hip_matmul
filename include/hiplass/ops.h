#pragma once
#include "hiplass/utils.h"

namespace hiplass {

template <int M, int N> // K is assumed 1
HOSTDEVICE void thread_matmul(float* A, float* B, float* C) {
    for (int m = 0; m < M; ++m) {
        for (int n = 0; n < N; ++n) {
            C[m * N + n] += A[m] * B[n];
        }
    }
}
}