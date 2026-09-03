import time
import torch
from matmul import matmul, gpu_timer, timer_start, timer_stop

torch.manual_seed(0)
M, K, N = 4096, 4096, 4096
ITERS = 10
FLOPS = 2 * M * N * K  # multiply-add = 2 flops per output element per k-step

A = torch.randn(M, K, dtype=torch.float32)
B = torch.randn(K, N, dtype=torch.float32)

# --- correctness -----------------------------------------------------------
C = matmul(A, B)
C_ref = A @ B

# atol is loosened vs. a smaller-size test: with K=4096, entries of C have
# std ~sqrt(K) and float32 accumulation order differs from torch's BLAS, so
# some rounding drift is expected even for a correct kernel.
max_err = (C - C_ref).abs().max().item()
print(f"max abs error vs torch matmul: {max_err:.6f}")
assert torch.allclose(C, C_ref, atol=1.0, rtol=1e-2), "mismatch vs reference"
print("OK")

# --- timing ------------------------------------------------------------
# Kernel-only time via the general HIP-event timer (excludes host<->device
# copies and Python/pybind call overhead). We avoid torch.utils.benchmark /
# torch.cuda timing here: this torch build hard-blocks extension work on
# Windows+ROCm through torch.utils.cpp_extension, so timing has to stay
# independent of it.
matmul(A, B)  # warmup: first HIP call pays for context/module init

kernel_times_ms = []
for _ in range(ITERS):
    with gpu_timer() as t:
        matmul(A, B)
    kernel_times_ms.append(t.elapsed_ms)
kernel_ms = sum(kernel_times_ms) / len(kernel_times_ms)

# Equivalent without the context manager, if you'd rather bookend by hand:
#   timer_start()
#   matmul(A, B)
#   kernel_ms = timer_stop()

# Wall-clock time for the whole round trip (H2D copies + kernel + D2H copy),
# i.e. what a caller actually pays per matmul() call.
start = time.perf_counter()
for _ in range(ITERS):
    matmul(A, B)
wall_s = (time.perf_counter() - start) / ITERS

print(f"kernel time:      {kernel_ms:.3f} ms  ->  {FLOPS / (kernel_ms / 1e3) / 1e12:.3f} TFLOPS")
print(f"wall-clock time:  {wall_s * 1e3:.3f} ms  ->  {FLOPS / wall_s / 1e12:.3f} TFLOPS")
