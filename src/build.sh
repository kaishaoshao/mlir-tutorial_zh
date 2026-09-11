#!/bin/bash
set -e

# 1. 获取 LLVM/MLIR 构建目录的绝对路径
LLVM_BUILD_DIR="$(cd ../../llvm-project/build && pwd)"

# 2. 清理旧缓存（防止缓存污染）
rm -rf build

# 3. 执行 CMake 配置（使用标准的 MLIR_DIR 和 LLVM_DIR）
cmake -B build -G Ninja \
  -DMLIR_DIR="${LLVM_BUILD_DIR}/lib/cmake/mlir" \
  -DLLVM_DIR="${LLVM_BUILD_DIR}/lib/cmake/llvm"

# 4. 执行构建
cmake --build build