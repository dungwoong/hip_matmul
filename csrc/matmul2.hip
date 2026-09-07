#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include "naive_kernel.h"
#include "hiplass/hiplass.h"


namespace kernels {
namespace kernel2 {

using namespace hiplass;

template<typename A, typename B>
using hmul = elementwise_multiply_t<A, B>;

// --- tiling layout, shared by kernel2 and matmul1_launcher below -----------
using THREAD_LAYOUT = IntTuple<8, 8>;

using WAVE_LAYOUT = IntTuple<4, 8>; // wave = warp

using WAVE_PARALLEL = IntTuple<4, 2>; // use multiple waves
constexpr int nWaves = WAVE_PARALLEL::size(); // wave = warp
constexpr int BLOCK_SIZE = nWaves * 32;

using WAVE_TILE = hmul<THREAD_LAYOUT, WAVE_LAYOUT>;
using WAVE_P_TILE = hmul<WAVE_TILE, WAVE_PARALLEL>;
constexpr int TK = 16;
constexpr int TM = WAVE_P_TILE::get<0>();
constexpr int TN = WAVE_P_TILE::get<1>();

constexpr int THREAD_M = THREAD_LAYOUT::get<0>();
constexpr int THREAD_N = THREAD_LAYOUT::get<1>();
constexpr int SMEM_PAD = 1;

using HierLoad = HierarchicalLoad<THREAD_LAYOUT, WAVE_LAYOUT, WAVE_PARALLEL>;
using HierStore = HierarchicalStore<THREAD_LAYOUT, WAVE_LAYOUT, WAVE_PARALLEL>;

// Consume one already-loaded+synced K-tile out of shared buffer BUF and
// accumulate it into C_regs. BUF is a *template* (compile-time) parameter,
// not a runtime index -- every sAs[BUF]/sBs[BUF] access below therefore
// compiles down to a static LDS offset (ds_read2_b32, same as the
// single-buffered kernel) instead of the flat-addressed, spill-heavy code
// a runtime buffer index (e.g. `sAs[sidx]`) forced the compiler into.
template <int BUF>
__device__ __forceinline__ void computeFromBuffer(
    st<TM, TK, SMEM_PAD> (&sAs)[2],
    st<TN, TK, SMEM_PAD> (&sBs)[2],
    int waveid, int laneid,
    float (&A_col)[THREAD_M],
    float (&B_row)[THREAD_N],
    float (&C_regs)[THREAD_M * THREAD_N])
{
    static_assert(BUF == 0 || BUF == 1, "double-buffered kernel only has 2 buffers");
    for (int rK = 0; rK < TK; ++rK) {
        // load A to registers
        #pragma unroll
        for (int threadS = 0; threadS < THREAD_LAYOUT::get<0>(); ++threadS) {
            int rowA = HierLoad::rowA(waveid, laneid, threadS);
            A_col[threadS] = sAs[BUF](rowA, rK);
        }

        #pragma unroll
        for (int threadS = 0; threadS < THREAD_LAYOUT::get<1>(); ++threadS) {
            int rowB = HierLoad::rowB(waveid, laneid, threadS);
            B_row[threadS] = sBs[BUF](rowB, rK);
        }

        // matmul
        thread_matmul<THREAD_M, THREAD_N>(A_col, B_row, C_regs);
    }
}

// Perform C = A @ B, fp32 everything for now.
// A is [M,K] row-major; B is [N,K] row-major (i.e. B^T vs. the usual [K,N]
// layout -- see matmul1_launcher below, which documents/enforces this on
// the host side); C is [M,N] row-major.
__global__ void
__launch_bounds__(BLOCK_SIZE) kernel2(float *A, float *B, float *C, int M, int N, int K) {

    // __shared__ float As_ptr[TM * TK];
    // __shared__ float Bs_ptr[TN * TK];
    __shared__ float As_ptr[2][TM][TK+SMEM_PAD];
    __shared__ float Bs_ptr[2][TN][TK+SMEM_PAD];

    gt gA(A, 1, 1, M, K);
    gt gB(B, 1, 1, N, K);
    gt gC(C, 1, 1, M, N);
    st<TM, TK, SMEM_PAD> sAs[2] = {st<TM, TK, SMEM_PAD>(As_ptr[0]), st<TM, TK, SMEM_PAD>(As_ptr[1])};
    st<TN, TK, SMEM_PAD> sBs[2] = {st<TN, TK, SMEM_PAD>(Bs_ptr[0]), st<TN, TK, SMEM_PAD>(Bs_ptr[1])};

    // For registers
    float A_col[THREAD_M]; // column of A
    float B_row[THREAD_N]; // row of B

    float C_regs[THREAD_M * THREAD_N] = {0.0};
    int laneid = threadIdx.x % 32;
    int waveid = threadIdx.x / 32;

    int mBase = blockIdx.x * TM;
    int nBase = blockIdx.y * TN;

    // pre fetch tile 0 into buffer 0
    load<BLOCK_SIZE>(gA, sAs[0], threadIdx.x, 0, 0, mBase, 0);
    load<BLOCK_SIZE>(gB, sBs[0], threadIdx.x, 0, 0, nBase, 0);
    __syncthreads();

    // Main loop, unrolled by 2 K-tiles per trip so every buffer index is a
    // compile-time 0 or 1 (see computeFromBuffer above) instead of a
    // runtime `sidx` that used to flip each iteration. Each half below is
    // exactly one "load next tile into the other buffer, compute the
    // current tile, sync" step -- same shape as the single-buffered
    // kernel's loop body, just naming which buffer plays "current" vs
    // "next" explicitly instead of toggling a variable.
    int lK = TK;
    for (; lK + TK < K; lK += 2 * TK) {
        // buffer 0 holds the tile we're about to consume; load the tile
        // after it into buffer 1 while that compute runs.
        load<BLOCK_SIZE>(gA, sAs[1], threadIdx.x, 0, 0, mBase, lK);
        load<BLOCK_SIZE>(gB, sBs[1], threadIdx.x, 0, 0, nBase, lK);
        computeFromBuffer<0>(sAs, sBs, waveid, laneid, A_col, B_row, C_regs);
        __syncthreads();

        // symmetric other half: buffer 1 holds the tile we're about to
        // consume; load the tile after that into buffer 0.
        load<BLOCK_SIZE>(gA, sAs[0], threadIdx.x, 0, 0, mBase, lK + TK);
        load<BLOCK_SIZE>(gB, sBs[0], threadIdx.x, 0, 0, nBase, lK + TK);
        computeFromBuffer<1>(sAs, sBs, waveid, laneid, A_col, B_row, C_regs);
        __syncthreads();
    }

    // Drain whatever's left: buffer 0 always holds one loaded-but-not-yet-
    // computed tile at this point. If K/TK is odd there's exactly one more
    // tile beyond it to load into buffer 1; if even, buffer 0's tile is the
    // very last one.
    if (lK < K) {
        load<BLOCK_SIZE>(gA, sAs[1], threadIdx.x, 0, 0, mBase, lK);
        load<BLOCK_SIZE>(gB, sBs[1], threadIdx.x, 0, 0, nBase, lK);
        computeFromBuffer<0>(sAs, sBs, waveid, laneid, A_col, B_row, C_regs);
        __syncthreads();
        computeFromBuffer<1>(sAs, sBs, waveid, laneid, A_col, B_row, C_regs);
    } else {
        computeFromBuffer<0>(sAs, sBs, waveid, laneid, A_col, B_row, C_regs);
    }

    // store back, 4 elements at a time (float4) since C is row-major and the
    // N dimension is contiguous in both C_regs and global memory
    static_assert(THREAD_N % 4 == 0, "float4 store requires THREAD_N divisible by 4");
    #pragma unroll
    for (int mIdx = 0; mIdx < THREAD_M; ++mIdx) {
        #pragma unroll
        for (int nIdx = 0; nIdx < THREAD_N; nIdx += 4) {
            int thread_s_idx = mIdx + THREAD_M * nIdx;   // inverse of THREAD_LAYOUT::coord
            std::array<int, 2> rc = HierStore::crds(waveid, laneid, thread_s_idx);

            float4 vals = make_float4(
                C_regs[mIdx * THREAD_N + nIdx + 0],
                C_regs[mIdx * THREAD_N + nIdx + 1],
                C_regs[mIdx * THREAD_N + nIdx + 2],
                C_regs[mIdx * THREAD_N + nIdx + 3]);

            *reinterpret_cast<float4*>(&gC(0, 0, rc[0] + mBase, rc[1] + nBase)) = vals;
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
void matmul2_launcher(uintptr_t A, uintptr_t B, uintptr_t C, int M, int K, int N)
{
    dim3 block(BLOCK_SIZE);
    dim3 grid((M + TM - 1) / TM, (N + TN - 1) / TN);
    hipLaunchKernelGGL(kernel2, grid, block, 0, 0,
        reinterpret_cast<float *>(A), reinterpret_cast<float *>(B), reinterpret_cast<float *>(C),
        M, N, K);
    hipDeviceSynchronize();
}

} //namespace kernel2
} // namespace kernels

PYBIND11_MODULE(matmul2, m)
{
    m.def("matmul", &kernels::kernel2::matmul2_launcher,
          "Tiled HIP matmul: C[MxN] = A[MxK] @ B[NxK]^T (B stored [N,K], see csrc/matmul1.hip), "
          "all args are HIP buffer addresses (allocate/copy via matmul0's to_hip/alloc/to_host)");
}
