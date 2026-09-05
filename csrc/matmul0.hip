#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
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
// elapsed time in milliseconds. Kept independent of any framework's own
// timing/profiling utilities so this has no dependency beyond plain HIP.
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

// --- numpy array <-> HIP device buffer helpers ----------------------------
// Windows + ROCm doesn't support running kernels directly against a
// torch.cuda-style tensor's dispatcher, so arrays stay on the host in
// Python and we manually stage them into HIP device memory, run the
// kernel, then copy back.
namespace py = pybind11;

// py::array::c_style forces a C-contiguous layout, py::array::forcecast
// lets pybind11 convert non-float32 input rather than rejecting it outright
// -- mirrors what `.contiguous()` used to do on the torch side.
using float_array = py::array_t<float, py::array::c_style | py::array::forcecast>;

uintptr_t allocate(int64_t numel)
{
    float *device_ptr;
    hipMalloc(&device_ptr, numel * sizeof(float));
    return reinterpret_cast<uintptr_t>(device_ptr);
}

uintptr_t move_to_hip(float_array x)
{
    py::buffer_info info = x.request();

    int64_t numel = info.size;
    float *device_ptr;
    hipMalloc(&device_ptr, numel * sizeof(float));
    hipMemcpy(device_ptr, static_cast<const float *>(info.ptr), numel * sizeof(float), hipMemcpyHostToDevice);

    return reinterpret_cast<uintptr_t>(device_ptr);
}

// Writes into an already-allocated output array (e.g. np.empty(...)),
// same contract the old at::Tensor version had: caller owns/sizes it.
void move_to_host(uintptr_t addr, py::array_t<float> x)
{
    py::buffer_info info = x.request();

    int64_t numel = info.size;
    float *data = reinterpret_cast<float *>(addr);
    hipMemcpy(static_cast<float *>(info.ptr), data, numel * sizeof(float), hipMemcpyDeviceToHost);
}

void free_hip(uintptr_t addr)
{
    void *ptr = reinterpret_cast<void *>(addr);
    hipFree(ptr);
}

PYBIND11_MODULE(matmul0, m)
{
    m.def("to_hip", &move_to_hip, "Copy a CPU float32 numpy array to a HIP buffer, returns its address");
    m.def("to_host", &move_to_host, "Copy a HIP buffer back into an existing CPU float32 numpy array");
    m.def("alloc", &allocate, "Allocate a HIP buffer of `numel` floats, returns its address");
    m.def("free_hip", &free_hip, "Free a HIP buffer");
    m.def("matmul", &matmul_launcher, "Naive HIP matmul: C[MxN] = A[MxK] @ B[NxK]^T (B stored [N,K], see csrc/naive_kernel.h), all args are HIP buffer addresses");
    m.def("timer_start", &timer_start, "Record a HIP event marking the start of a profiling window");
    m.def("timer_stop", &timer_stop, "Record a HIP event marking the end of a profiling window, sync, and return elapsed ms since timer_start()");
}
