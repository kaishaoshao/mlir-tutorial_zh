func.func @main(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %result = arith.addf %A, %B : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}
