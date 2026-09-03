# Sets the environment variables CMakeLists.txt needs (PYBIND11_INCLUDE_DIR,
# PYTHON_INCLUDE_DIR, PYTHON_LIB, TORCH_DIR) by asking a Python interpreter
# for them, so nobody has to hunt down or hardcode these paths by hand.
#
# Run this in the SAME PowerShell window you'll run cmake/ninja from -- it
# sets environment variables on the current process, which cmake picks up
# when you configure.
#
# If `python` on PATH isn't the interpreter with torch/pybind11 installed
# (e.g. you use a venv or a specific Python311 install), pass it explicitly:
#   .\add_paths.ps1 -Python "C:\Users\you\AppData\Local\Programs\Python\Python311\python.exe"

param(
    [string]$Python = "python"
)

function Get-PyValue([string]$code) {
    $result = & $Python -c $code 2>&1
    if ($LASTEXITCODE -ne 0 -or [string]::IsNullOrWhiteSpace($result)) {
        Write-Error "Failed running with '$Python':`n  $code`n$result"
        exit 1
    }
    return $result.Trim()
}

Write-Host "Using interpreter: $Python"

$env:PYBIND11_INCLUDE_DIR = Get-PyValue "import pybind11; print(pybind11.get_include())"
$env:PYTHON_INCLUDE_DIR   = Get-PyValue "import sysconfig; print(sysconfig.get_path('include'))"
$env:PYTHON_LIB           = Get-PyValue "import sys, os; print(os.path.join(sys.base_prefix, 'libs', f'python{sys.version_info.major}{sys.version_info.minor}.lib'))"
$env:TORCH_DIR            = Get-PyValue "import torch, os; print(os.path.dirname(torch.__file__))"

Write-Host ""
Write-Host "PYBIND11_INCLUDE_DIR = $env:PYBIND11_INCLUDE_DIR"
Write-Host "PYTHON_INCLUDE_DIR   = $env:PYTHON_INCLUDE_DIR"
Write-Host "PYTHON_LIB           = $env:PYTHON_LIB"
Write-Host "TORCH_DIR            = $env:TORCH_DIR"

if (-not (Test-Path $env:PYTHON_LIB)) {
    Write-Warning "PYTHON_LIB does not point to an existing file -- double check your Python install layout: $env:PYTHON_LIB"
}
if (-not (Test-Path $env:TORCH_DIR)) {
    Write-Warning "TORCH_DIR does not exist -- is torch installed for '$Python'? $env:TORCH_DIR"
}

Write-Host ""
Write-Host "Environment variables set for this PowerShell session. You can now run the cmake configure command."
