# Build steps (PowerShell, from the repo root):
#   1. .\add_paths.ps1
#      (sets PYBIND11_INCLUDE_DIR / PYTHON_INCLUDE_DIR / PYTHON_LIB / TORCH_DIR
#      for this shell session by asking your Python interpreter for them)
#   2. cmake -S . -B build `
#        -DCMAKE_C_COMPILER="C:/Program Files/AMD/ROCm/6.2/bin/clang.exe" `
#        -DCMAKE_CXX_COMPILER="C:/Program Files/AMD/ROCm/6.2/bin/clang++.exe" `
#        -G "Ninja" -DCMAKE_BUILD_TYPE=Release `
#        -DCMAKE_HIP_COMPILER_ROCM_ROOT="C:/Program Files/AMD/ROCm/6.2" `
#        -DCMAKE_HIP_ARCHITECTURES=gfx1031
#   3. ninja -C build
#
# Adjust the ROCm install path and CMAKE_HIP_ARCHITECTURES to match your
# machine/GPU. This builds two targets:
#   - matmul0: the Python extension, written to python/build/matmul0.pyd
#   - bench_matmul: a standalone C++/HIP executable (no Python involved) that
#     times the naive kernel against rocBLAS, written to build/bench_matmul.exe

cmake_minimum_required(VERSION 3.14)
project(PybindHIP LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_HIP_STANDARD 17)

find_package(HIP REQUIRED)

function(require_env_var var_name)
    if(NOT DEFINED ENV{${var_name}})
        message(FATAL_ERROR "Environment variable ${var_name} is not set. Run add_paths.ps1 in this shell first (see the comment at the top of CMakeLists.txt).")
    endif()
    set(${var_name} "$ENV{${var_name}}" PARENT_SCOPE)
endfunction()

require_env_var(PYBIND11_INCLUDE_DIR)
require_env_var(PYTHON_INCLUDE_DIR)
require_env_var(PYTHON_LIB)
require_env_var(TORCH_DIR)

include_directories(
    "${PYBIND11_INCLUDE_DIR}"
    "${PYTHON_INCLUDE_DIR}"
    "${CMAKE_HIP_COMPILER_ROCM_ROOT}/include"
    "${TORCH_DIR}"
    "${TORCH_DIR}/include/torch/csrc/api/include"
)
link_directories("${TORCH_DIR}/lib" "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib")
find_package(Torch REQUIRED PATHS "${TORCH_DIR}")

# --- Python extension: matmul0.pyd -----------------------------------------

add_library(matmul0 MODULE csrc/matmul0.hip)

set_target_properties(matmul0 PROPERTIES
    PREFIX ""
    SUFFIX ".pyd"
    RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/build"
    LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/build"
)

target_link_libraries(matmul0
    "${PYTHON_LIB}"
    torch
    torch_python
)

target_include_directories(matmul0 PRIVATE ${TORCH_INCLUDE_DIRS})

# --- Standalone benchmark: bench_matmul.exe --------------------------------
# No Python/torch dependency -- just HIP + rocBLAS.

add_executable(bench_matmul csrc/bench_matmul.hip)
target_link_libraries(bench_matmul rocblas)
