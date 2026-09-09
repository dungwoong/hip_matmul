import numpy as np
from dataclasses import dataclass

HIP_DEVICE = 'hip'
CPU_DEVICE = 'cpu'

@dataclass
class HipArray:
    """A 2D float32 array living in HIP device memory.

    This is the `.to('cuda')`-equivalent handle: create one with
    `to_device(np_array)`, pass it into `matmul()` as many times as you
    like (no host<->device copy happens on those calls), and pull the
    result back with `.cpu()` only when you actually need it on the host.
    """
    addr: int
    shape: tuple
    stride: tuple

    def cpu(self) -> np.ndarray:
        """Copy this array back to a host numpy array."""
        pass

    def hip(self):
        pass

    def device(self) -> str:
        ... # need to create hip array class