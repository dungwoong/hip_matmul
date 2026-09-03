#include <torch/extension.h>
#include <hip/hip_runtime.h>
#include <stdexcept>
#include "naive_kernel.h"

void matmul_launcher(uintptr_t A, uintptr_t B, uintptr_t C, int M, int K, int N)
{
    launch_naive_matmul(reinterpret_cast<const float *>(A), reinterpret_cast<const float *>(B), reinterpret_cast<float *>(C), M, K, N);
    hipDeviceSynchronize();
}

// --- General-purpose GPU timer ---------------------------------------------
// A pair of hipEvents you record from Python around whatever sequence of
// kernel launches you want to profile: timer_start(), ..., timer_stop().
// Everything launched on the default stream between the two calls gets
// timed on-device; timer_stop() blocks until it's done and returns the
// elapsed time in milliseconds. This is deliberately independent of torch's
// own timing/profiling utilities (torch.utils.benchmark, torch.cuda.Event):
// those go through torch.utils.cpp_extension under the hood, which
// hard-blocks on Windows+ROCm regardless of whether the underlying binaries
// actually work.
static hipEvent_t g_start_event = nullptr;
static hipEvent_t g_stop_event = nullptr;

static void ensure_events()
{
    if (!g_start_event)
        hipEventCreate(&g_start_event);
    if (!g_stop_event)
        hipEventCreate(&g_stop_event);
}

void timer_start()
{
    ensure_events();
    hipEventRecord(g_start_event);
}

float timer_stop()
{
    if (!g_start_event)
        throw std::runtime_error("timer_stop() called before timer_start()");

    ensure_events();
    hipEventRecord(g_stop_event);
    hipEventSynchronize(g_stop_event);

    float elapsed_ms = 0.0f;
    hipEventElapsedTime(&elapsed_ms, g_start_event, g_stop_event);
    return elapsed_ms;
}

// --- CPU tensor <-> HIP device buffer helpers -----------------------------
// Windows + ROCm doesn't support running kernels directly against a
// torch.cuda tensor's dispatcher, so tensors stay on CPU in Python and we
// manually stage them into HIP device memory, run the kernel, then copy back.

uintptr_t allocate(int64_t numel)
{
    float *device_ptr;
    hipMalloc(&device_ptr, numel * sizeof(float));
    return reinterpret_cast<uintptr_t>(device_ptr);
}

uintptr_t move_to_hip(const at::Tensor &x)
{
    TORCH_CHECK(!x.is_cuda(), "Input must be on CPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    int64_t numel = x.numel();
    float *device_ptr;
    hipMalloc(&device_ptr, numel * sizeof(float));
    hipMemcpy(device_ptr, x.contiguous().data_ptr<float>(), numel * sizeof(float), hipMemcpyHostToDevice);

    return reinterpret_cast<uintptr_t>(device_ptr);
}

void move_to_torch(uintptr_t addr, const at::Tensor &x)
{
    TORCH_CHECK(!x.is_cuda(), "Output must be on CPU");
    TORCH_CHECK(x.dtype() == torch::kFloat32, "Only float32 supported");

    int64_t numel = x.numel();
    float *data = reinterpret_cast<float *>(addr);
    hipMemcpy(x.data_ptr<float>(), data, numel * sizeof(float), hipMemcpyDeviceToHost);
}

void free_hip(uintptr_t addr)
{
    void *ptr = reinterpret_cast<void *>(addr);
    hipFree(ptr);
}

PYBIND11_MODULE(matmul0, m)
{
    m.def("to_hip", &move_to_hip, "Copy a CPU float32 tensor to a HIP buffer, returns its address");
    m.def("to_torch", &move_to_torch, "Copy a HIP buffer back into a CPU float32 tensor");
    m.def("alloc", &allocate, "Allocate a HIP buffer of `numel` floats, returns its address");
    m.def("free_hip", &free_hip, "Free a HIP buffer");
    m.def("matmul", &matmul_launcher, "Naive HIP matmul: C[MxN] = A[MxK] @ B[KxN], all args are HIP buffer addresses");
    m.def("timer_start", &timer_start, "Record a HIP event marking the start of a profiling window");
    m.def("timer_stop", &timer_stop, "Record a HIP event marking the end of a profiling window, sync, and return elapsed ms since timer_start()");
}
