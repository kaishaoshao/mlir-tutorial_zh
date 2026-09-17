## 创建函数流程

步骤 1：初始化上下文并注册方言（Dialect）
MLIR 是多方言架构，使用某些 Op（如 func.func, arith.addi）前必须显式注册并加载对应的 Dialect。

步骤 2：创建 Builder 与顶层容器（Module）
OpBuilder 是用来创建 Op、管理插入点光标的核心工具。
ModuleOp（即 builtin.module）是容纳所有函数、全局变量的顶层容器。

步骤 3：定义函数签名并创建 FuncOp
将光标定位到 mod 内部的 body 块末尾。
构造数据类型和函数类型 (i32, i32) -> i32。
创建名为 "test" 的函数操作。

步骤 4：添加入口基本块（Entry Block）并获取函数形参
调用 func.addEntryBlock()，MLIR 会根据函数类型自动创建同等类型和数量的块参数。
通过 entry->getArguments() 获取这些形参（对应 SSA 变量 %arg0, %arg1）。

步骤 6：创建终结指令（Terminator / ReturnOp）
MLIR 的基本块必须以终结指令结尾。将加法计算产生的结果封装为 ValueRange 作为返回值。

```
MLIRContext 加载方言 
  └─> OpBuilder
        └─> ModuleOp (设置插入点到 mod.getBody())
              └─> FuncOp (设置插入点到 func.addEntryBlock())
                    ├─> 从 entry 获取形参: args = entry->getArguments()
                    ├─> 计算指令: builder.create<arith::AddIOp>(..., args[0], args[1])
                    └─> 终结指令: builder.create<func::ReturnOp>(..., addi)
```

## getUnknownLoc() 等函数作用

在构建每个Operation(如ModuleOp、 FuncOp、 AddIOp、 ReturnOp)时，第一个参数通常时builder.getUnknownLoc()

1. 为什么创建每个Op都必须传入Location?

* 与传统的LLVM IR(调试信息知识可选的metadata)不同，在MLIR的设计原则中，源码位置信息(Location)时IR的"一等公民"强制约束： MLIR要求每个Operation在构造时都必须绑定一个mlir::Location

  核心目的：

  * 诊断与报错(Diagnostisc)： 当Pass验证失败、Verifier报错或执行emitError()时，编译器能够直接定位到源程序对应的位置
  * 优化溯源(Tracebility)： 经过多次编译Pass(如算子融合、代码下沉、 Lowering等)后，依然能追踪生成的指令是由哪些上层代码变换来的

2. builder.getUnknownLoc() 是做什么的?
   * builder.getUnKnownLoc()返回类型为mlir::Location的对象(底层是mlir：：UnknownLoc::get(context))
   * 它表示未知源码的位置
   * 适用场景：第一个参数必须传入一个 `Location`。当你不需要或无法提供具体的源文件行列号时
