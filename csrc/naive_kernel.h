#pragma once
#include <hip/hip_runtime.h>

// Naive row-major matmul: C[MxN] = A[MxK] @ B[KxN]
// Shared between the pybind extension (matmul0.hip) and the standalone
// benchmark (bench_matmul.hip) so there's exactly one copy of the kernel.
__global__ void kernel1_naive(const float *A, const float *B, float *C, int M, int K, int N)
{
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;
    if (row < M && col < N)
    {
        float acc_c = 0.0f;
        for (int k = 0; k < K; ++k)
        {
            acc_c += A[row * K + k] * B[k * N + col];
        }
        C[row * N + col] = acc_c;
    }
}

inline void launch_naive_matmul(const float *A, const float *B, float *C, int M, int K, int N)
{
    dim3 block(16, 16);
    dim3 grid((N + block.x - 1) / block.x, (M + block.y - 1) / block.y);
    hipLaunchKernelGGL(kernel1_naive, grid, block, 0, 0, A, B, C, M, K, N);
}
