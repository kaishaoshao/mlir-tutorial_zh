# Dialect

## --convert-elementwise-to-linalg

`--convert-elementwise-to-linalg` 是 MLIR（Multi-Level Intermediate Representation）中的一个 **转换 Pass**，用于将 **逐元素操作（Elementwise Operations）** 转换为 **Linalg 操作**。这个 Pass 是 MLIR 优化管道中的一部分，主要用于将低层次的逐元素操作提升为高层次的 Linalg 操作，以便后续的优化和代码生成。

### 1. **什么是逐元素操作（Elementwise Operations）？**

逐元素操作是指对张量（Tensor）或数组中的每个元素独立执行的操作。例如：

- 加法：`C[i, j] = A[i, j] + B[i, j]`
- 乘法：`C[i, j] = A[i, j] * B[i, j]`
- 激活函数（如 ReLU）：`C[i, j] = max(0, A[i, j])`

这些操作的特点是：

- 每个元素的计算是独立的。
- 操作可以并行化。

在 MLIR 中，逐元素操作通常以简单的算术操作（如 `arith.addf`、`arith.mulf`）表示。

---

### 2. **什么是 Linalg 操作？**

Linalg（Linear Algebra）是 MLIR 中的一个 **Dialect（方言）**，专门用于表示线性代数操作。Linalg 提供了高层次的操作抽象，例如矩阵乘法、卷积、点积等。

Linalg 的特点：

- 支持多面体编译（Polyhedral Compilation），可以自动生成高效的循环嵌套。
- 提供了丰富的优化机会，例如循环融合、并行化、内存优化等。

---

### 3. **`--convert-elementwise-to-linalg` 的作用**

`--convert-elementwise-to-linalg` Pass 的作用是将 **逐元素操作** 转换为 **Linalg 操作**。具体来说：

- 将简单的逐元素操作（如 `arith.addf`、`arith.mulf`）转换为 `linalg.generic` 操作。
- 通过 `linalg.generic`，逐元素操作可以被统一表示为高层次的线性代数操作，便于后续优化。

---

### 4. **为什么需要这个转换？**

将逐元素操作转换为 Linalg 操作的主要目的是：

1. **统一表示**：
   - 将不同的逐元素操作统一表示为 `linalg.generic`，便于后续优化和代码生成。
2. **利用 Linalg 的优化能力**：
   - Linalg 提供了强大的优化框架，可以自动优化循环嵌套、并行化等。
3. **目标代码生成**：
   - Linalg 可以生成高效的 CPU 或 GPU 代码。

------

### 5. **转换过程**

假设有一个逐元素加法操作：

```mlir
%result = arith.addf %A, %B : tensor<4x4xf32>
```

通过 `--convert-elementwise-to-linalg` Pass 后，该操作会被转换为 `linalg.generic` 操作：

```mlir
%result = linalg.generic {
    indexing_maps = [
      affine_map<(i, j) -> (i, j)>,  // 输入 A 的索引映射
      affine_map<(i, j) -> (i, j)>,  // 输入 B 的索引映射
      affine_map<(i, j) -> (i, j)>   // 输出 C 的索引映射
    ],
    iterator_types = ["parallel", "parallel"]  // 迭代器类型（并行）
} ins(%A, %B : tensor<4x4xf32>, tensor<4x4xf32>)  // 输入张量
outs(%C : tensor<4x4xf32>) {  // 输出张量
    ^bb0(%a: f32, %b: f32, %c: f32):  // 基本块参数
        %sum = arith.addf %a, %b : f32  // 逐元素加法
        linalg.yield %sum : f32  // 返回结果
} -> tensor<4x4xf32>
```

在 MLIR 中，可以通过 `mlir-opt` 工具运行 `--convert-elementwise-to-linalg` Pass。例如：

```bash
mlir-opt --convert-elementwise-to-linalg input.mlir -o output.mlir
```

#### 输入示例（`add_input.mlir`）：

```mlir
func.func @main(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %result = arith.addf %A, %B : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}
```

#### 输出示例（`add_output.mlir`）：

```mlir
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
```

假设我们有两个逐元素操作：

```mlir
func.func @main(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>, %C: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %temp = arith.addf %A, %B : tensor<4x4xf32>
    %result = arith.mulf %temp, %C : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}
```

转换为 Linalg 操作后：

```mlir
#map = affine_map<(d0, d1) -> (d0, d1)>
module {
  func.func @main(%arg0: tensor<4x4xf32>, %arg1: tensor<4x4xf32>, %arg2: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %0 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%arg0, %arg1 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%arg0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.addf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<4x4xf32>
    %1 = linalg.generic {indexing_maps = [#map, #map, #map], iterator_types = ["parallel", "parallel"]} ins(%0, %arg2 : tensor<4x4xf32>, tensor<4x4xf32>) outs(%0 : tensor<4x4xf32>) {
    ^bb0(%in: f32, %in_0: f32, %out: f32):
      %2 = arith.mulf %in, %in_0 : f32
      linalg.yield %2 : f32
    } -> tensor<4x4xf32>
    return %1 : tensor<4x4xf32>
  }
}
```

---

### 7. **优化机会**

通过 `--convert-elementwise-to-linalg` 转换为 Linalg 操作后，可以进一步优化：

- **循环融合**：将多个逐元素操作融合到一个循环中，减少内存访问。
- **并行化**：利用 `iterator_types` 的并行性，生成并行代码。
- **内存优化**：通过 `indexing_maps` 优化内存布局。

---

通过 `--convert-elementwise-to-linalg`，可以将低层次的逐元素操作提升为高层次的 Linalg 操作，从而充分利用 Linalg 的优化能力，生成高效的代码。



## -func-buffersize

`-func-bufferize` 是 MLIR 中的一个 **转换 Pass**，用于将 **函数中的张量（Tensor）类型** 转换为 **缓冲区（Buffer）类型**。这个 Pass 是 MLIR 优化管道中的一部分，主要用于将高层次的张量操作转换为低层次的缓冲区操作，以便后续的代码生成和优化。

---

### 1. **什么是张量（Tensor）和缓冲区（Buffer）？**
- **张量（Tensor）**：
  - 张量是 MLIR 中的一种高层次数据类型，表示多维数组。
  - 张量操作（如逐元素加法、矩阵乘法）通常以声明式的方式表示，便于优化和分析。
- **缓冲区（Buffer）**：
  - 缓冲区是 MLIR 中的一种低层次数据类型，表示连续的内存区域。
  - 缓冲区操作（如内存加载、存储）更接近硬件，适合代码生成。

---

### 2. **`-func-bufferize` 的作用**
`-func-bufferize` Pass 的作用是将函数中的张量类型转换为缓冲区类型。具体来说：
- 将函数参数和返回值中的张量类型转换为缓冲区类型。
- 将函数内部的张量操作（如 `linalg.generic`）转换为缓冲区操作（如 `memref.load` 和 `memref.store`）。

---

### 3. **转换过程**
在 MLIR 中，可以通过 `mlir-opt` 工具运行 `-func-bufferize` Pass。例如：

```bash
mlir-opt -func-bufferize func_input.mlir -o func_output.mlir
```

#### 输入示例（`func_input.mlir`）：

一个使用张量的函数：

```mlir
func.func @main(%A: tensor<4x4xf32>, %B: tensor<4x4xf32>) -> tensor<4x4xf32> {
    %result = arith.addf %A, %B : tensor<4x4xf32>
    return %result : tensor<4x4xf32>
}
```

#### 输出示例（`func_output.mlir`）：

通过 `-func-bufferize` Pass 后，函数会被转换为使用缓冲区的形式：

```mlir
module {
  func.func @main(%arg0: memref<4x4xf32>, %arg1: memref<4x4xf32>) -> memref<4x4xf32> {
    %0 = bufferization.to_tensor %arg1 : memref<4x4xf32>
    %1 = bufferization.to_tensor %arg0 : memref<4x4xf32>
    %2 = arith.addf %1, %0 : tensor<4x4xf32>
    %3 = bufferization.to_memref %2 : memref<4x4xf32>
    return %3 : memref<4x4xf32>
  }
}
```

---

### 4. **关键变化**
- **函数签名**：
  - 输入和输出张量（`tensor<4x4xf32>`）被转换为缓冲区（`memref<4x4xf32>`）。
  - 返回值被移除，改为通过输出缓冲区传递结果。
- **操作类型**：
  - 张量操作（如 `arith.addf`）被转换为缓冲区操作（如 `linalg.generic`）。

---

### 5. **为什么需要 `-func-bufferize`？**
将张量转换为缓冲区的主要目的是：
1. **降低层次**：
   - 将高层次的张量操作转换为低层次的缓冲区操作，便于代码生成。
2. **内存优化**：
   - 缓冲区操作更接近硬件，可以更好地优化内存访问。
3. **目标代码生成**：
   - 缓冲区操作可以直接映射到目标硬件的内存模型。

---

### 6. **优化机会**
通过 `-func-bufferize`，可以进一步优化：
- **内存布局**：
  优化缓冲区的内存布局，减少缓存未命中。
- **循环融合**：
  将多个缓冲区操作融合到一个循环中，减少内存访问。
- **并行化**：
  利用缓冲区的并行性，生成并行代码。

---

通过 `-func-bufferize`，可以将高层次的张量操作转换为低层次的缓冲区操作，从而更好地优化和生成目标代码。



## -convert-linalg-to-affine-loops

`-convert-linalg-to-affine-loops` 是 MLIR 中的一个转换 pass，用于将 `linalg` 操作（如 `linalg.generic`）转换为显式的 `affine` 循环嵌套。这个 pass 通常用于将高级的线性代数操作（如矩阵乘法、逐元素操作等）降低为更底层的循环结构，以便进一步优化或生成目标代码。

---

### 1. 使用 `-convert-linalg-to-affine-loops`

假设你有一个包含 `linalg.generic` 操作的 MLIR 文件（例如 `/tmp/test.mlir`），你可以运行以下命令将其转换为 `affine` 循环：

```bash
mlir-opt -convert-linalg-to-affine-loops /tmp/test.mlir
```

---

### 2. 示例输入

以下是一个包含 `linalg.generic` 的示例输入文件 `func_output2.mlir`：

```mlir
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
```

---

### 3. 转换后的输出

运行 `mlir-opt -convert-linalg-to-affine-loops func_output2.mlir` 后，输出将如下所示`func_output3.mlir`：

```mlir
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
```

---

### 4. 总结

`-convert-linalg-to-affine-loops` 是一个非常有用的 pass，它将高级的 `linalg` 操作转换为显式的 `affine` 循环嵌套。通过结合其他优化 pass，你可以进一步优化生成的循环结构，以便更好地适应目标硬件或运行时环境。





## **参考文档**

- [MLIR Linalg Dialect](https://mlir.llvm.org/docs/Dialects/Linalg/)
- [MLIR Conversion Passes](https://mlir.llvm.org/docs/Passes/)

