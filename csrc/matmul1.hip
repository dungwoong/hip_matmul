#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include "naive_kernel.h"
#include "hiplass/hiplass.h"


namespace kernels {

using namespace hiplass;

template<typename A, typename B>
using hmul = elementwise_multiply_t<A, B>;

// --- tiling layout, shared by kernel2 and matmul1_launcher below -----------
using THREAD_LAYOUT = IntTuple<4, 4>;

using WAVE_LAYOUT = IntTuple<4, 8>; // wave = warp
using WAVE_SERIAL = IntTuple<2, 2>; // waves loop over stuff

using WAVE_PARALLEL = IntTuple<4, 2>; // use multiple waves
constexpr int nWaves = WAVE_PARALLEL::size(); // wave = warp
constexpr int BLOCK_SIZE = nWaves * 32;

using WAVE_TILE = hmul<THREAD_LAYOUT, WAVE_LAYOUT>;
using WAVE_S_TILE = hmul<WAVE_TILE, WAVE_SERIAL>;
using WAVE_P_TILE = hmul<WAVE_S_TILE, WAVE_PARALLEL>;
constexpr int TK = 8;
constexpr int TM = WAVE_P_TILE::get<0>();
constexpr int TN = WAVE_P_TILE::get<1>();

constexpr int THREAD_M = WAVE_SERIAL::get<0>() * THREAD_LAYOUT::get<0>();
constexpr int THREAD_N = WAVE_SERIAL::get<1>() * THREAD_LAYOUT::get<1>();

// Perform C = A @ B, fp32 everything for now.
// A is [M,K] row-major; B is [N,K] row-major (i.e. B^T vs. the usual [K,N]
// layout -- see matmul1_launcher below, which documents/enforces this on
// the host side); C is [M,N] row-major.
__global__ void kernel2(float *A, float *B, float *C, int M, int N, int K) {

    __shared__ float As_ptr[TM * TK];
    __shared__ float Bs_ptr[TN * TK];

    gt gA(A, 1, 1, M, K);
    gt gB(B, 1, 1, N, K);
    gt gC(C, 1, 1, M, N);
    st<TM, TK> sA(As_ptr);
    st<TN, TK> sB(Bs_ptr);

    // For registers
    float A_col[THREAD_M]; // column of A
    float B_row[THREAD_N]; // row of B

    float C_regs[THREAD_M * THREAD_N] = {0.0};
    int laneid = threadIdx.x % 32;
    int waveid = threadIdx.x / 32;
    using HierLoad = HierarchicalLoad<THREAD_LAYOUT, WAVE_LAYOUT, WAVE_SERIAL, WAVE_PARALLEL>;
    using HierStore = HierarchicalStore<THREAD_LAYOUT, WAVE_LAYOUT, WAVE_SERIAL, WAVE_PARALLEL>;

    int mBase = blockIdx.x * TM;
    int nBase = blockIdx.y * TN;

    for (int lK = 0; lK < K; lK += TK) {
        // populate shared memory
        // read TM * TK for A
        load<BLOCK_SIZE>(gA, sA, threadIdx.x, 0, 0, mBase, lK);
        load<BLOCK_SIZE>(gB, sB, threadIdx.x, 0, 0, nBase, lK);
        __syncthreads();

        // computation
        // populate A_col and B_row from As and Bs
        
        // load stuff to registers
        for (int rK = 0; rK < TK; ++rK) {

            // load A to registers
            #pragma unroll
            for (int waveS = 0; waveS < WAVE_SERIAL::get<0>(); ++waveS) {
                #pragma unroll
                for (int threadS = 0; threadS < THREAD_LAYOUT::get<0>(); ++threadS) {
                    int rowA = HierLoad::rowA(waveid, laneid, waveS, threadS);
                    A_col[waveS * THREAD_LAYOUT::get<0>() + threadS] = sA(rowA, rK);
                }
            }

            #pragma unroll
            for (int waveS = 0; waveS < WAVE_SERIAL::get<1>(); ++waveS) {
                #pragma unroll
                for (int threadS = 0; threadS < THREAD_LAYOUT::get<1>(); ++threadS) {
                    int rowA = HierLoad::rowB(waveid, laneid, waveS, threadS);
                    B_row[waveS * THREAD_LAYOUT::get<1>() + threadS] = sB(rowA, rK);
                }
            }

            // matmul
            thread_matmul<THREAD_M, THREAD_N>(A_col, B_row, C_regs); 
        }

        __syncthreads();
    }
    
    // store back
    #pragma unroll
    for (int wave_s_idx = 0; wave_s_idx < WAVE_SERIAL::size(); ++wave_s_idx) {
        std::array<int, 2> waveS = WAVE_SERIAL::coord(wave_s_idx);   // {waveS_M, waveS_N}

        #pragma unroll
        for (int thread_s_idx = 0; thread_s_idx < THREAD_LAYOUT::size(); ++thread_s_idx) {
            std::array<int, 2> threadS = THREAD_LAYOUT::coord(thread_s_idx);   // {threadS_M, threadS_N}

            int mIdx = waveS[0] * THREAD_LAYOUT::get<0>() + threadS[0];
            int nIdx = waveS[1] * THREAD_LAYOUT::get<1>() + threadS[1];

            std::array<int, 2> rc = HierStore::crds(waveid, laneid, wave_s_idx, thread_s_idx);
            gC(0, 0, rc[0] + mBase, rc[1] + nBase) = C_regs[mIdx * THREAD_N + nIdx];
        }
    }
}

// --- host launcher / dispatcher entry point --------------------------------
// C[MxN] = A[MxK] @ B[NxK]^T. Note the B layout: this kernel reads B as
// [N,K] row-major (see kernel2's comment above), NOT the [K,N] layout
// matmul0's naive kernel expects -- callers must transpose B on the host
// before staging it, or this silently computes the wrong thing. A/B/C are
// HIP buffer addresses, same contract as matmul0.matmul (allocate/copy them
// with matmul0.alloc/to_hip/to_host -- this module only adds the kernel).
void matmul1_launcher(uintptr_t A, uintptr_t B, uintptr_t C, int M, int K, int N)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    hipLaunchKernelGGL(kernel2, grid, block, 0, 0,
        reinterpret_cast<float *>(A), reinterpret_cast<float *>(B), reinterpret_cast<float *>(C),
        M, N, K);
    hipDeviceSynchronize();
}

} // namespace kernels

PYBIND11_MODULE(matmul1, m)
{
    m.def("matmul", &kernels::matmul1_launcher,
          "Tiled HIP matmul: C[MxN] = A[MxK] @ B[NxK]^T (B stored [N,K], see csrc/matmul1.hip), "
          "all args are HIP buffer addresses (allocate/copy via matmul0's to_hip/alloc/to_host)");
}