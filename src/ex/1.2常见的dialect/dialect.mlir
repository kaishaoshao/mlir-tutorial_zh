func.func @foo(%a: tensor<16x64xf64>, %b: tensor<16x64xf64>) -> tensor<16x64xf64> {
  %c = arith.addf %a, %b : tensor<16x64xf64>
  func.return %c : tensor<16x64xf64>
}