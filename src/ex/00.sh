# tensor运算转换为循环

#  1.全面废弃旧版 Dialect-specific Bufferization（方言独立 Bufferize）：
#   -linalg-bufferize、-func-bufferize、-tensor-bufferize 
#   等旧的 Pass 已经被统一的 One-Shot Bufferize (-one-shot-bufferize) 取代。
#   旧 Pass 不仅已被标记弃用/移除，而且无法处理复杂的张量别名分析与全局函数签名转换。
# 2.Pass 重命名与合并： -convert-elementwise-to-linalg  在新版中已被移至 LinalgTransforms
#   并重命名为 -linalg-generalize-named-ops 或者是更具体的 
#   -convert-elementwise-to-linalg 搭配 -empty-tensor-to-alloc-tensor。
# 3.Bufferize 机制要求： 现代 Bufferize 要求张量创建（tensor.empty）
#   必须显式转换为内存分配（bufferization.alloc_tensor），
#   并且需要一次性（One-Shot）对整个 Module 进行内存逃逸和原地修改（in-place）分析。

# ./mlir-opt -convert-elementwise-to-linalg \
#   -empty-tensor-to-alloc-tensor \
#   -one-shot-bufferize="bufferize-function-boundaries=1" \
#   -convert-linalg-to-affine-loops ./00.mlir


# --empty-tensor-to-alloc-tensor 
./mlir-opt \
  --convert-elementwise-to-linalg \
  --one-shot-bufferize="bufferize-function-boundaries=1 \
  function-boundary-type-conversion=identity-layout-map copy-before-write=1" \
  --convert-linalg-to-affine-loops \
  00.mlir

# ./mlir-opt --pass-pipeline="                              \
#     builtin.module(convert-elementwise-to-linalg,         \
#     empty-tensor-to-alloc-tensor,                         \
#     one-shot-bufferize{bufferize-function-boundaries=1,   \
#     function-boundary-type-conversion=identity-layout-map, \ 
#     copy-before-write=1},                                 \
#     convert-linalg-to-affine-loops)" 00.mlir

# Pass-by-Pass 拆解
# 阶段1： --convert-elementwise-to-linalg
# 将arith.addf 泛化为结构的linalg.generic. 此时引入了tensor.empty 作为输出张量的纯数学容器(任处于Tensor概念空间)
##map = affine_map<(d0, d1) -> (d0, d1)>
# module {
#   func.func @foo(%arg0: tensor<16x64xf64>, %arg1: tensor<16x64xf64>) -> tensor<16x64xf64> {
#     %0 = linalg.generic {
#           indexing_maps = [#map, #map, #map], 
#           iterator_types = ["parallel", "parallel"]
#        } ins(%arg0, %arg1 : tensor<16x64xf64>, tensor<16x64xf64>)
#          outs(%arg0 : tensor<16x64xf64>) {
#     ^bb0(%in: f64, %in_0: f64, %out: f64):
#       %1 = arith.addf %in, %in_0 : f64
#       linalg.yield %1 : f64
#     } -> tensor<16x64xf64>
#     return %0 : tensor<16x64xf64>
#   }
# }
### 问题1：linalg.generic是什么东西? indexing_maps和iterator_types是什么？
### 问题2：tensor.empty是否有用
### 问题3：linalg.yield是什么东西

# 阶段2：  --one-shot-bufferize="bufferize-function-boundaries=1"
# 执行全局内存分配分析（tensor张量转显式内存指针memref）
# #map = affine_map<(d0, d1) -> (d0, d1)>
# module {
#   func.func @foo(%arg0: memref<16x64xf64, strided<[?, ?], offset: ?>>, %arg1: memref<16x64xf64, strided<[?, ?], offset: ?>>) -> memref<16x64xf64, strided<[?, ?], offset: ?>> {
#     linalg.generic {
#       indexing_maps = [#map, #map, #map],
#       iterator_types = ["parallel", "parallel"]} 
#       ins(%arg0, %arg1 : memref<16x64xf64, strided<[?, ?], offset: ?>>, memref<16x64xf64, strided<[?, ?], offset: ?>>) 
#       outs(%arg0 : memref<16x64xf64, strided<[?, ?], offset: ?>>) {
#     ^bb0(%in: f64, %in_0: f64, %out: f64):
#       %0 = arith.addf %in, %in_0 : f64
#       linalg.yield %0 : f64
#     }
#     return %arg0 : memref<16x64xf64, strided<[?, ?], offset: ?>>
#   }
### 问题1：stried 和 offset是什么

# 阶段3：--one-shot-bufferize="unknown-type-conversion=identity-layout-map"
# 告诉编译器，输入函数参数假定为标准连续的内存布局(Identity Layout), 去掉那些恼人的 strided<[?, ?], offset: ?>
# #map = affine_map<(d0, d1) -> (d0, d1)>
# module {
#   func.func @foo(%arg0: memref<16x64xf64>, %arg1: memref<16x64xf64>) -> memref<16x64xf64> {
#     linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<16x64xf64>, memref<16x64xf64>) outs(%arg0 : memref<16x64xf64>) {
#     ^bb0(%in: f64, %in_0: f64, %out: f64):
#       %0 = arith.addf %in, %in_0 : f64
#       linalg.yield %0 : f64
#     }
#     return %arg0 : memref<16x64xf64>
#   }
# }
### 问题1: 什么时候需要去除strided和offset什么时候不用去除
### 问题2：什么时候不是标准连续的内存布局


# 阶段4：--one-shot-bufferize="copy-before-write=1" 或者 显示的使用bufferization.clone / copy
# 告诉Bufferizer不要做侵入式的原地复写
# #map = affine_map<(d0, d1) -> (d0, d1)>
# module {
#   func.func @foo(%arg0: memref<16x64xf64>, %arg1: memref<16x64xf64>) -> memref<16x64xf64> {
#     linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : memref<16x64xf64>, memref<16x64xf64>) outs(%arg0 : memref<16x64xf64>) {
#     ^bb0(%in: f64, %in_0: f64, %out: f64):
#       %0 = arith.addf %in, %in_0 : f64
#       linalg.yield %0 : f64
#     }
#     return %arg0 : memref<16x64xf64>
#   }
# }

#阶段5： --convert-linalg-to-affine-loops
# 将作用在 memref 上的 linalg.generic 展开为嵌套的多维 affine.for 循环，
# 并在循环内部生成 affine.load 和 affine.store
# module {
#   func.func @foo(%arg0: memref<16x64xf64>, %arg1: memref<16x64xf64>) -> memref<16x64xf64> {
#     %alloc = memref.alloc() {alignment = 64 : i64} : memref<16x64xf64>
#     affine.for %arg2 = 0 to 16 {
#       affine.for %arg3 = 0 to 64 {
#         %0 = affine.load %arg0[%arg2, %arg3] : memref<16x64xf64>
#         %1 = affine.load %arg1[%arg2, %arg3] : memref<16x64xf64>
#         %2 = arith.addf %0, %1 : f64
#         affine.store %2, %alloc[%arg2, %arg3] : memref<16x64xf64>
#       }
#     }
#     return %alloc : memref<16x64xf64>
#   }
# }


