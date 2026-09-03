import os
import sys
import torch

# Import the compiled extension directly from python/build, rather than as
# `build.matmul0`: if the `build` PyPI package (the PEP 517 frontend) is
# installed, it shadows a local `build/` directory of the same name and
# `import build.matmul0` fails with "No module named 'build.matmul0'".
_build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)
import matmul0


def _check_inputs(A: torch.Tensor, B: torch.Tensor):
    assert not A.is_cuda and not B.is_cuda, "A and B must be CPU tensors"
    assert A.dtype == torch.float32 and B.dtype == torch.float32, "A and B must be float32"
    assert A.dim() == 2 and B.dim() == 2, "A and B must be 2D"
    assert A.shape[1] == B.shape[0], "inner dimensions must match: A is [M,K], B is [K,N]"


# matmul(A, B) -> C, computing C = A @ B for 2D CPU float32 tensors.
# A and B are staged into HIP device memory, the naive kernel runs there,
# and the result is copied back to a CPU tensor (see csrc/matmul0.hip for why
# this doesn't just take torch.cuda tensors: HIP-tensor dispatch isn't
# supported when building PyTorch extensions with ROCm on Windows).
def matmul(A: torch.Tensor, B: torch.Tensor) -> torch.Tensor:
    _check_inputs(A, B)
    M, K = A.shape
    N = B.shape[1]

    a_addr = matmul0.to_hip(A)
    b_addr = matmul0.to_hip(B)
    c_addr = matmul0.alloc(M * N)

    matmul0.matmul(a_addr, b_addr, c_addr, M, K, N)

    C = torch.empty((M, N), dtype=torch.float32)
    matmul0.to_torch(c_addr, C)

    matmul0.free_hip(a_addr)
    matmul0.free_hip(b_addr)
    matmul0.free_hip(c_addr)

    return C


class gpu_timer:
    """Context manager around matmul0's hipEvent-based timer.

    Wrap any code that issues HIP kernel launches (matmul() calls, or your
    own future kernels) and read .elapsed_ms afterwards:

        with gpu_timer() as t:
            matmul(A, B)
        print(t.elapsed_ms)

    Everything launched on the default stream between entering and exiting
    the block is what gets timed; exiting blocks until the GPU work is done.
    """

    def __enter__(self):
        matmul0.timer_start()
        self.elapsed_ms = None
        return self

    def __exit__(self, exc_type, exc, tb):
        if exc_type is None:
            self.elapsed_ms = matmul0.timer_stop()
        return False


# Thin re-exports of the raw start/stop calls, for when the context manager
# doesn't fit (e.g. timing spans multiple statements you don't want nested
# in a `with` block).
def timer_start():
    matmul0.timer_start()


def timer_stop() -> float:
    """Returns elapsed ms since the last timer_start()."""
    return matmul0.timer_stop()
