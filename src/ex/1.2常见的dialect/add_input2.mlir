func.func @main(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %temp = arith.addf %A, %B : tensor<4x4xf32>
    %result = arith.mulf %temp, %C : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}