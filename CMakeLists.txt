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
# machine/GPU.
#
# Each Python extension (matmul0, and any future matmulN) is declared with
# add_hip_pyext() below, which sets up three ninja targets per name:
#   - <name>: the normal build, written to python/build/<name>.pyd
#   - <name>_asm: compiles csrc/<name>.hip down to *just* the device-side
#     GCN/RDNA assembly, written to build_asm/<name>/<name>.s
#   - <name>_from_asm: resumes compilation from that (possibly hand-edited)
#     .s file -- assemble -> link device object -> bundle into a HIP fat
#     binary -> recompile the host side with that fat binary embedded ->
#     link -- and installs the result over python/build/<name>.pyd
#
# To hand-edit a kernel's compiled assembly:
#   ninja -C build matmul0_asm
#   ...edit build_asm\matmul0\matmul0.s...
#   ninja -C build matmul0_from_asm
#   cd python && python test_matmul.py
#
# To add a new kernel version later: drop csrc/matmul1.hip (with its own
# PYBIND11_MODULE(matmul1, m)) and add one line: add_hip_pyext(matmul1).
#
# The asm/from_asm commands were built from the actual `clang -v` sub-command
# log on this machine (ROCm 6.2, clang 19, ninja on Windows): device
# preprocess -> device LLVM IR -> device .s -> assemble -> lld link into a
# loadable device object -> clang-offload-bundler packs that into a HIP fat
# binary -> host side recompiled with -fcuda-include-gpubinary pointing at
# that fat binary (instead of one clang would normally regenerate itself
# from source) -> link. If a ROCm/clang update changes these flags, re-run
# `ninja -C build -t commands <name>` (with add_paths.ps1's environment set)
# to see the current real command line and compare against add_hip_pyext().

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

set(EXT_INCLUDES
    "-I${PYBIND11_INCLUDE_DIR}"
    "-I${PYTHON_INCLUDE_DIR}"
    "-isystem${CMAKE_HIP_COMPILER_ROCM_ROOT}/include"
    "-isystem${TORCH_DIR}"
    "-isystem${TORCH_DIR}/include"
    "-isystem${TORCH_DIR}/include/torch/csrc/api/include"
)

# The normal matmul0 target gets these from CMAKE_CXX_STANDARD/MSVC-compat
# handling automatically; our manual add_hip_pyext() commands below invoke
# the compiler directly, so they need to be passed explicitly. Without
# -std=c++17, torch's headers (which use std::optional etc.) fail to parse.
set(EXT_FLAGS
    -std=c++17
    -fms-extensions
    -fms-compatibility
    -D_DLL
    -D_MT
    -Xclang --dependent-lib=msvcrt
)

# add_hip_pyext(<name> [extra header deps...])
#
# Declares a pybind11/HIP Python extension built from csrc/<name>.hip
# (which must contain PYBIND11_MODULE(<name>, m)), plus the <name>_asm /
# <name>_from_asm targets described above. Pass any headers the source
# includes (e.g. naive_kernel.h) as extra args so edits to them trigger a
# rebuild.
function(add_hip_pyext name)
    set(extra_deps ${ARGN})
    set(SRC "${CMAKE_SOURCE_DIR}/csrc/${name}.hip")

    # --- normal build: python/build/<name>.pyd -----------------------------
    add_library(${name} MODULE "${SRC}")
    set_target_properties(${name} PROPERTIES
        PREFIX ""
        SUFFIX ".pyd"
        RUNTIME_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/build"
        LIBRARY_OUTPUT_DIRECTORY "${CMAKE_SOURCE_DIR}/python/build"
    )
    target_link_libraries(${name} "${PYTHON_LIB}" torch torch_python)
    target_include_directories(${name} PRIVATE ${TORCH_INCLUDE_DIRS})

    # --- <name>_asm / <name>_from_asm ---------------------------------------
    set(ASM_DIR "${CMAKE_SOURCE_DIR}/build_asm/${name}")
    file(MAKE_DIRECTORY "${ASM_DIR}")

    set(DEVICE_S "${ASM_DIR}/${name}.s")
    set(DEVICE_OBJ "${ASM_DIR}/${name}-dev.o")
    set(DEVICE_SO "${ASM_DIR}/${name}-dev.out")
    set(DEVICE_HIPFB "${ASM_DIR}/${name}.hipfb")
    set(HOST_OBJ "${ASM_DIR}/${name}-host.obj")
    # NOTE: this is deliberately NOT python/build/<name>.pyd -- that path is
    # already claimed as an OUTPUT by the add_library() rule above, and
    # ninja refuses to build a graph where two rules produce the same file.
    set(PATCHED_PYD "${ASM_DIR}/${name}.pyd")
    set(INSTALL_STAMP "${ASM_DIR}/${name}.installed")

    add_custom_command(
        OUTPUT "${DEVICE_S}"
        COMMAND "${CMAKE_HIP_COMPILER}" --offload-arch=${CMAKE_HIP_ARCHITECTURES}
                --cuda-device-only -x hip -S -O3
                ${EXT_INCLUDES}
                ${EXT_FLAGS}
                "-D__HIP_ROCclr__=1" -DNDEBUG
                -o "${DEVICE_S}" "${SRC}"
        DEPENDS "${SRC}" ${extra_deps}
        COMMENT "${name}_asm: compiling csrc/${name}.hip to device assembly (${DEVICE_S})"
        VERBATIM
    )
    add_custom_target(${name}_asm DEPENDS "${DEVICE_S}")

    add_custom_command(
        OUTPUT "${DEVICE_OBJ}"
        COMMAND "${CMAKE_HIP_COMPILER}" -target amdgcn-amd-amdhsa
                -mcpu=${CMAKE_HIP_ARCHITECTURES}
                -c "${DEVICE_S}" -o "${DEVICE_OBJ}"
        DEPENDS "${DEVICE_S}"
        COMMENT "${name}_from_asm: assembling ${DEVICE_S}"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${DEVICE_SO}"
        COMMAND "${CMAKE_HIP_COMPILER_ROCM_ROOT}/bin/lld" -flavor gnu -m elf64_amdgpu
                --no-undefined -shared
                "-plugin-opt=mcpu=${CMAKE_HIP_ARCHITECTURES}" -plugin-opt=O3
                --whole-archive -o "${DEVICE_SO}" "${DEVICE_OBJ}" --no-whole-archive
        DEPENDS "${DEVICE_OBJ}"
        COMMENT "${name}_from_asm: linking device code object"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${DEVICE_HIPFB}"
        COMMAND "${CMAKE_HIP_COMPILER_ROCM_ROOT}/bin/clang-offload-bundler"
                -type=o -bundle-align=4096
                "-targets=host-x86_64-unknown-linux,hipv4-amdgcn-amd-amdhsa--${CMAKE_HIP_ARCHITECTURES}"
                -input=nul "-input=${DEVICE_SO}" "-output=${DEVICE_HIPFB}" -verbose
        DEPENDS "${DEVICE_SO}"
        COMMENT "${name}_from_asm: bundling into HIP fat binary"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${HOST_OBJ}"
        COMMAND "${CMAKE_HIP_COMPILER}" --cuda-host-only -x hip -c -O3
                -Xclang -fcuda-include-gpubinary -Xclang "${DEVICE_HIPFB}"
                ${EXT_INCLUDES}
                ${EXT_FLAGS}
                "-D${name}_EXPORTS" -DNDEBUG
                -o "${HOST_OBJ}" "${SRC}"
        DEPENDS "${DEVICE_HIPFB}" "${SRC}" ${extra_deps}
        COMMENT "${name}_from_asm: recompiling host side with the patched fat binary embedded"
        VERBATIM
    )

    add_custom_command(
        OUTPUT "${PATCHED_PYD}"
        COMMAND "${CMAKE_HIP_COMPILER}"
                -nostartfiles -nostdlib -O3 -DNDEBUG -D_DLL -D_MT
                -Xclang --dependent-lib=msvcrt
                --offload-arch=${CMAKE_HIP_ARCHITECTURES}
                --hip-link --rtlib=compiler-rt -fuse-ld=lld-link
                -shared -o "${PATCHED_PYD}"
                -Xlinker /MANIFEST:EMBED
                "-Xlinker" "/implib:${ASM_DIR}/${name}.lib"
                "-Xlinker" "/pdb:${ASM_DIR}/${name}.pdb"
                -Xlinker /version:0.0
                "${HOST_OBJ}"
                "-L${TORCH_DIR}/lib"
                "-L${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib"
                "${PYTHON_LIB}"
                "${TORCH_DIR}/lib/torch.lib"
                -ltorch_python.lib
                "${TORCH_DIR}/lib/torch_cpu.lib"
                "${TORCH_DIR}/lib/c10.lib"
                "${CMAKE_HIP_COMPILER_ROCM_ROOT}/lib/amdhip64.lib"
                -lkernel32 -luser32 -lgdi32 -lwinspool -lshell32 -lole32
                -loleaut32 -luuid -lcomdlg32 -ladvapi32 -loldnames
        DEPENDS "${HOST_OBJ}"
        COMMENT "${name}_from_asm: linking ${name}.pyd from patched assembly"
        VERBATIM
    )

    # Install the patched build over python/build/<name>.pyd. This is a
    # plain COMMAND, not a declared OUTPUT for that path, so it doesn't
    # collide with add_library()'s rule -- it just overwrites the file at
    # build time, same as you copying it by hand.
    add_custom_command(
        OUTPUT "${INSTALL_STAMP}"
        COMMAND "${CMAKE_COMMAND}" -E copy "${PATCHED_PYD}" "${CMAKE_SOURCE_DIR}/python/build/${name}.pyd"
        COMMAND "${CMAKE_COMMAND}" -E touch "${INSTALL_STAMP}"
        DEPENDS "${PATCHED_PYD}"
        COMMENT "${name}_from_asm: installing patched ${name}.pyd into python/build/"
        VERBATIM
    )
    add_custom_target(${name}_from_asm DEPENDS "${INSTALL_STAMP}")
endfunction()

add_hip_pyext(matmul0 "${CMAKE_SOURCE_DIR}/csrc/naive_kernel.h")

