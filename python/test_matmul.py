import time
import numpy as np
from matmul import matmul, to_device, gpu_timer, timer_start, timer_stop

np.random.seed(0)
M, K, N = 4096, 4096, 4096
ITERS = 10
FLOPS = 2 * M * N * K  # multiply-add = 2 flops per output element per k-step
KERNEL='tiled'

A = np.random.randn(M, K).astype(np.float32)
B = np.random.randn(N, K).astype(np.float32)

# Move to device once, like torch's A.to('cuda') -- everything below reuses
# these HipArrays, so no host<->device copy happens inside the timed loops.
A_gpu = to_device(A)
B_gpu = to_device(B)

# --- correctness -----------------------------------------------------------
C = matmul(A_gpu, B_gpu, kernel=KERNEL).cpu()
C_ref = A @ B.T

# atol is loosened vs. a smaller-size test: with K=4096, entries of C have
# std ~sqrt(K) and float32 accumulation order differs from numpy's BLAS, so
# some rounding drift is expected even for a correct kernel.
max_err = np.abs(C - C_ref).max()
print(f"max abs error vs numpy matmul: {max_err:.6f}")
assert np.allclose(C, C_ref, atol=1.0, rtol=1e-2), f"mismatch vs reference\n{C}\nC_ref"
print("OK")

# --- timing ------------------------------------------------------------
# Kernel-only time via the general HIP-event timer (excludes host<->device
# copies and Python/pybind call overhead -- and since A_gpu/B_gpu are
# already on the device, there's no copy to exclude here anyway).
matmul(A_gpu, B_gpu, kernel=KERNEL)  # warmup: first HIP call pays for context/module init

kernel_times_ms = []
for _ in range(ITERS):
    with gpu_timer() as t:
        matmul(A_gpu, B_gpu, kernel=KERNEL)
    kernel_times_ms.append(t.elapsed_ms)
kernel_ms = sum(kernel_times_ms) / len(kernel_times_ms)

# Equivalent without the context manager, if you'd rather bookend by hand:
#   timer_start()
#   matmul(A_gpu, B_gpu)
#   kernel_ms = timer_stop()

# Wall-clock time per matmul() call, now that A/B are already device-
# resident this is just kernel launch + the output alloc, no H2D copies.
start = time.perf_counter()
for _ in range(ITERS):
    matmul(A_gpu, B_gpu, kernel=KERNEL)
wall_s = (time.perf_counter() - start) / ITERS

print(f"kernel time:      {kernel_ms:.3f} ms  ->  {FLOPS / (kernel_ms / 1e3) / 1e12:.3f} TFLOPS")
print(f"wall-clock time:  {wall_s * 1e3:.3f} ms  ->  {FLOPS / wall_s / 1e12:.3f} TFLOPS")
