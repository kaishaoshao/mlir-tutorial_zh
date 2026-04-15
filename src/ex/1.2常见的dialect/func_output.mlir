module {
  func.func @main(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) -> memref<4x4xf32> {
    %0 = bufferization.to_tensor %arg1 : memref<4x4xf32>
    %1 = bufferization.to_tensor %arg0 : memref<4x4xf32>
    %2 = arith.addf %1, %0 : tensor<4x4xf32>
    %3 = bufferization.to_memref %2 : memref<4x4xf32>
    return %3 : memref<4x4xf32>
  }
}

