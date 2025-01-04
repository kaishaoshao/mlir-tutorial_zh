cd llvm-project
# git checkout 186a4b3b657878ae2aea23caf684b6e103901162  && git switch -c mlir # 本教程使用的版本
# mkdir build && 
cd build
cmake -G Ninja ../llvm \
  -DCMAKE_INSTALL_PREFIX=${PWD}/install \
  -DLLVM_ENABLE_PROJECTS=mlir \
  -DLLVM_BUILD_EXAMPLES=ON \
  -DLLVM_TARGETS_TO_BUILD="Native;NVPTX;AMDGPU" \
  -DCMAKE_BUILD_TYPE=Release \
  -DLLVM_ENABLE_ASSERTIONS=ON && ninja -j7
