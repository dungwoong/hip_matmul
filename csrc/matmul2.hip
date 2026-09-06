#include <pybind11/pybind11.h>
#include <hip/hip_runtime.h>
#include <rocblas/rocblas.h>
#include <stdexcept>
#include <string>

// --- rocBLAS-backed matmul, same address-in/address-out contract as
// matmul0/matmul1: C[MxN] = A[MxK] @ B[NxK]^T, where B is stored [N,K]
// row-major (pre-transposed -- see matmul.py's to_device_transposed()),
// same as every other kernel in this repo. A/B/C are HIP buffer
// addresses, allocated via matmul0.alloc/to_hip/to_host.
//
// rocBLAS is column-major (Fortran-style), so getting a row-major result
// out of it needs the standard "swap and transpose" trick. Both A_buf and
// B_buf are row-major on the host side; reinterpreting a row-major (R,Cc)
// buffer as column-major gives you its transpose with shape (Cc,R), same
// memory, no copy:
//
//   A_buf is row-major [M,K]      -> viewed col-major: A^T, shape (K,M), ld=K
//   B_buf is row-major [N,K]      -> viewed col-major: B  , shape (K,N), ld=K
//      (B_buf already stores B transposed, so viewing it col-major
//       un-transposes it back to plain B)
//   C_buf is row-major [M,N]      -> viewed col-major: C^T, shape (N,M), ld=N
//
// We want C^T = B^T @ A^T. From the above, B^T = transpose(B_buf-as-col),
// and A^T = A_buf-as-col with no further transpose needed.
// So: rocblas_sgemm(transA=T, transB=N, m=N, n=M, k=K,
//                    A=B_buf, lda=K, B=A_buf, ldb=K, C=C_buf, ldc=N)

static rocblas_handle g_handle = nullptr;

static void ensure_handle()
{
    if (!g_handle)
    {
        rocblas_status st = rocblas_create_handle(&g_handle);
        if (st != rocblas_status_success)
            throw std::runtime_error("rocblas_create_handle failed: " + std::to_string(st));
    }
}

void rocblas_matmul_launcher(uintptr_t A, uintptr_t B, uintptr_t C, int M, int K, int N)
{
    ensure_handle();

    const float alpha = 1.0f;
    const float beta = 0.0f;

    rocblas_status st = rocblas_sgemm(
        g_handle,
        rocblas_operation_transpose,   // transA, applied to the "A" arg below (our B_buf)
        rocblas_operation_none,        // transB, applied to the "B" arg below (our A_buf)
        N, M, K,
        &alpha,
        reinterpret_cast<const float *>(B), K,   // rocblas "A" = our B_buf, lda=K
        reinterpret_cast<const float *>(A), K,   // rocblas "B" = our A_buf, ldb=K
        &beta,
        reinterpret_cast<float *>(C), N);        // ldc=N

    if (st != rocblas_status_success)
        throw std::runtime_error("rocblas_sgemm failed with status " + std::to_string(st));

    hipDeviceSynchronize();
}

PYBIND11_MODULE(matmul2, m)
{
    m.def("matmul", &rocblas_matmul_launcher,
          "rocBLAS-backed matmul: C[MxN] = A[MxK] @ B[NxK]^T (B stored [N,K], "
          "same convention as matmul0/matmul1), all args are HIP buffer "
          "addresses (allocate/copy via matmul0's to_hip/alloc/to_host)");
}
