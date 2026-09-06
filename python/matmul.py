import os
import sys
import numpy as np

# Since Python 3.8, Windows extension modules (.pyd) no longer search PATH
# for their dependent DLLs -- only System32, the .pyd's own directory, and
# whatever's been added via os.add_dll_directory(). ROCm's installer drops
# amdhip64_*.dll into System32 (which is why matmul0/matmul1 "just work"),
# but rocblas.dll lives in ROCm's own bin/ dir and is never copied there,
# so matmul2 (rocBLAS) needs this explicit opt-in or it fails to import
# with "DLL load failed: the specified module could not be found" no
# matter what's on PATH.
if sys.platform == "win32" and hasattr(os, "add_dll_directory"):
    _rocm_root = os.environ.get("HIP_PATH") or r"C:\Program Files\AMD\ROCm\6.2"
    _rocm_bin = os.path.join(_rocm_root, "bin")
    if os.path.isdir(_rocm_bin):
        os.add_dll_directory(_rocm_bin)

# Import the compiled extension directly from python/build, rather than as
# `build.matmul0`: if the `build` PyPI package (the PEP 517 frontend) is
# installed, it shadows a local `build/` directory of the same name and
# `import build.matmul0` fails with "No module named 'build.matmul0'".
_build_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "build")
if _build_dir not in sys.path:
    sys.path.insert(0, _build_dir)
import matmul0

# matmul1 is optional: it's only importable once its ninja target has been
# built (add_hip_pyext(matmul1) in CMakeLists.txt). Fall back to naive-only
# if it's not there yet, rather than making every import of this module
# fail.
try:
    import matmul1
    _HAVE_MATMUL1 = True
except ImportError:
    _HAVE_MATMUL1 = False

# matmul2 wraps rocBLAS's rocblas_sgemm; same optional-import treatment as
# matmul1 above -- only present once its CMake target has been built.
try:
    import matmul2
    _HAVE_MATMUL2 = True
except ImportError:
    _HAVE_MATMUL2 = False
import matmul2


class HipArray:
    """A 2D float32 array living in HIP device memory.

    This is the `.to('cuda')`-equivalent handle: create one with
    `to_device(np_array)`, pass it into `matmul()` as many times as you
    like (no host<->device copy happens on those calls), and pull the
    result back with `.cpu()` only when you actually need it on the host.
    """

    def __init__(self, addr: int, shape: tuple[int, int]):
        self.addr = addr
        self.shape = shape

    def cpu(self) -> np.ndarray:
        """Copy this array back to a host numpy array."""
        out = np.empty(self.shape, dtype=np.float32)
        matmul0.to_host(self.addr, out)
        return out

    def free(self):
        if self.addr is not None:
            matmul0.free_hip(self.addr)
            self.addr = None

    def __del__(self):
        self.free()


def to_device(A: np.ndarray) -> HipArray:
    """Stage a 2D float32 numpy array into HIP device memory."""
    assert isinstance(A, np.ndarray), "A must be a numpy array"
    assert A.dtype == np.float32, "A must be float32"
    assert A.ndim == 2, "A must be 2D"

    addr = matmul0.to_hip(np.ascontiguousarray(A, dtype=np.float32))
    return HipArray(addr, A.shape)


def to_device_transposed(B: np.ndarray) -> HipArray:
    """Like to_device(), but stores B pre-transposed to [N,K] instead of
    the usual [K,N] -- what matmul() expects for B regardless of kernel
    (both matmul0's naive kernel and matmul1's tiled kernel read B as
    [N,K] row-major, not [K,N] -- see csrc/naive_kernel.h / csrc/matmul1.hip).
    """
    return to_device(np.ascontiguousarray(B.T, dtype=np.float32))


# Kernel name -> module.matmul. Both take HIP buffer addresses and
# (A_addr, B_addr, C_addr, M, K, N), and both expect B as [N,K] (see
# to_device_transposed() above).
_KERNELS = {
    "naive": matmul0.matmul,
}
if _HAVE_MATMUL1:
    _KERNELS["tiled"] = matmul1.matmul
if _HAVE_MATMUL2:
    _KERNELS["rocblas"] = matmul2.matmul


# matmul(A, B, kernel="naive") -> C, computing C = A @ B for HipArrays
# already resident on the device (see to_device()/to_device_transposed()
# above) -- no host<->device copy happens here, same as calling a torch op
# on tensors that are already on .to('cuda').
#
# kernel picks which HIP kernel runs the multiply ("naive", the default, or
# "tiled" -- matmul1's wave-hierarchy kernel, only available once its CMake
# target is built). Either way, B must come from to_device_transposed(B)
# ([N,K] layout), not to_device(B) -- passing a plain to_device(B) HipArray
# here will silently compute the wrong answer.
def matmul(A: HipArray, B: HipArray, kernel: str = "naive") -> HipArray:
    if kernel not in _KERNELS:
        raise ValueError(f"unknown kernel {kernel!r}, must be one of {list(_KERNELS)}")
    launcher = _KERNELS[kernel]

    M, K = A.shape
    N, K2 = B.shape
    assert K == K2, "inner dimensions must match: A is [M,K], B is [N,K] (pre-transposed, see to_device_transposed())"

    c_addr = matmul0.alloc(M * N)
    launcher(A.addr, B.addr, c_addr, M, K, N)

    return HipArray(c_addr, (M, N))


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
