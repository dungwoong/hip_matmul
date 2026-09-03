// Standalone benchmark: no Python/pybind/torch involved at all, just HIP
// (+ rocBLAS, if it's usable on this GPU). Runs the naive kernel and, if
// available, rocblas_sgemm on the same inputs, times both with hipEvents,
// checks they agree, and reports TFLOPS.
//
// Usage: bench_matmul [M] [K] [N] [iters]   (all default to 4096 / 10)

#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <cstdio>
#include <cstdlib>
#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <algorithm>
#include "naive_kernel.h"

#define HIP_CHECK(expr)                                                          \
    do {                                                                         \
        hipError_t _err = (expr);                                                \
        if (_err != hipSuccess) {                                                \
            fprintf(stderr, "[HIP ERROR] %s:%d: %s\n", __FILE__, __LINE__, hipGetErrorString(_err)); \
            exit(1);                                                             \
        }                                                                        \
    } while (0)

static float time_hip_call(const std::function<void()> &fn, int iters)
{
    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    fn(); // warmup
    HIP_CHECK(hipDeviceSynchronize());

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i)
        fn();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));

    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    return total_ms / iters;
}

int main(int argc, char **argv)
{
    int M = argc > 1 ? atoi(argv[1]) : 4096;
    int K = argc > 2 ? atoi(argv[2]) : 4096;
    int N = argc > 3 ? atoi(argv[3]) : 4096;
    int iters = argc > 4 ? atoi(argv[4]) : 10;

    printf("M=%d K=%d N=%d iters=%d\n", M, K, N, iters);

    std::vector<float> hA(M * K), hB(K * N), hC_naive(M * N), hC_rocblas(M * N);
    std::mt19937 rng(0);
    std::normal_distribution<float> dist(0.0f, 1.0f);
    for (auto &v : hA) v = dist(rng);
    for (auto &v : hB) v = dist(rng);

    float *dA, *dB, *dC;
    HIP_CHECK(hipMalloc(&dA, hA.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dB, hB.size() * sizeof(float)));
    HIP_CHECK(hipMalloc(&dC, hC_naive.size() * sizeof(float)));
    HIP_CHECK(hipMemcpy(dA, hA.data(), hA.size() * sizeof(float), hipMemcpyHostToDevice));
    HIP_CHECK(hipMemcpy(dB, hB.data(), hB.size() * sizeof(float), hipMemcpyHostToDevice));

    double flops = 2.0 * M * N * K;

    // --- naive kernel --------------------------------------------------
    float naive_ms = time_hip_call([&]() { launch_naive_matmul(dA, dB, dC, M, K, N); }, iters);
    HIP_CHECK(hipMemcpy(hC_naive.data(), dC, hC_naive.size() * sizeof(float), hipMemcpyDeviceToHost));
    double naive_tflops = flops / (naive_ms / 1e3) / 1e12;
    printf("\nnaive:    %8.3f ms  ->  %7.3f TFLOPS\n", naive_ms, naive_tflops);

    // --- rocBLAS (optional) -----------------------------------------------
    // A, B, C are row-major (M,K)/(K,N)/(M,N). rocBLAS/BLAS is column-major,
    // so instead of transposing anything we swap the operand order and
    // dimensions: interpreting our row-major buffers as column-major data
    // computes C^T = B^T * A^T, which is exactly the row-major C we want.
    //
    // rocBLAS ships precompiled kernels per GPU architecture (via Tensile);
    // not every arch has one bundled (notably several RDNA2 SKUs on the
    // Windows ROCm build). Rather than hard-failing the whole benchmark
    // when that's missing, try one rocBLAS call first and skip the
    // comparison if it doesn't come back rocblas_status_success.
    rocblas_handle handle = nullptr;
    bool handle_created = (rocblas_create_handle(&handle) == rocblas_status_success);
    bool rocblas_ok = handle_created;
    const float alpha = 1.0f, beta = 0.0f;

    if (rocblas_ok) {
        rocblas_status st = rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                                           N, M, K,
                                           &alpha,
                                           dB, N,
                                           dA, K,
                                           &beta,
                                           dC, N);
        HIP_CHECK(hipDeviceSynchronize());
        rocblas_ok = (st == rocblas_status_success);
    }

    if (!rocblas_ok) {
        printf("\nrocBLAS is not usable on this GPU/ROCm install (see stderr above for its error) -- skipping the comparison.\n");
    } else {
        float rocblas_ms = time_hip_call([&]() {
            rocblas_sgemm(handle, rocblas_operation_none, rocblas_operation_none,
                          N, M, K,
                          &alpha,
                          dB, N,
                          dA, K,
                          &beta,
                          dC, N);
        }, iters);
        HIP_CHECK(hipMemcpy(hC_rocblas.data(), dC, hC_rocblas.size() * sizeof(float), hipMemcpyDeviceToHost));
        double rocblas_tflops = flops / (rocblas_ms / 1e3) / 1e12;

        float max_diff = 0.0f;
        for (size_t i = 0; i < hC_naive.size(); ++i)
            max_diff = std::max(max_diff, std::fabs(hC_naive[i] - hC_rocblas[i]));

        printf("rocBLAS:  %8.3f ms  ->  %7.3f TFLOPS  (%.1fx faster than naive)\n",
               rocblas_ms, rocblas_tflops, naive_ms / rocblas_ms);
        printf("max abs diff naive vs rocBLAS: %.6f\n", max_diff);
    }

    if (handle_created)
        rocblas_destroy_handle(handle);
    HIP_CHECK(hipFree(dA));
    HIP_CHECK(hipFree(dB));
    HIP_CHECK(hipFree(dC));

    return 0;
}
