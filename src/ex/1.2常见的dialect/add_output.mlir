#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %1 = arith.addf %in, %in_0 : f32
      linalg.yield %1 : f32
    } -> tensor<4x4xf32>
    return %0 : tensor<4x4xf32>
  }
}

