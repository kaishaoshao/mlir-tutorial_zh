#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) -> memref<4x4xf32> {
    %0 = bufferization.to_tensor %arg1 : memref<4x4xf32>
    %1 = bufferization.to_tensor %arg0 : memref<4x4xf32>
    %2 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%1, %0 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%1 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %4 = arith.addf %in, %in_0 : f32
      linalg.yield %4 : f32
    } -> tensor<4x4xf32>
    %3 = bufferization.to_memref %2 : memref<4x4xf32>
    return %3 : memref<4x4xf32>
  }
}

