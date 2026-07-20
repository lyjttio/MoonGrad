# MoonGrad: MoonBit 原生张量与自动微分（Autograd）库

[English](#english) | [中文](#中文)

---

## 中文

`MoonGrad` 是一个完全用 **MoonBit** 语言实现的原生、轻量级、且极具扩展性的张量计算与自动微分（Autograd）库。它旨在为 MoonBit 生态提供深度的机器学习计算基础，可编译为 WebAssembly、JavaScript 以及 Native，实现全平台的高性能机器学习基础设施。

### 🌟 核心特性

- **纯 MoonBit 实现**：100% 纯 MoonBit 编写，不依赖任何外部 C/C++ 动态库或运行时，提供极佳的跨平台特性（Wasm、JS、Native）。
- **动态计算图与 Autograd 引擎**：支持与 PyTorch 类似的动态计算图（Dynamic Computation Graph），通过深度优先搜索（DFS）进行拓扑排序，并在其上执行反向传播算法。
- **多维张量（N-Dimensional Tensor）**：支持基于步长（Strides）的多维张量寻址及形状广播（Broadcasting）机制，可跨维度自动对齐计算。
- **丰富的数学与形状算子**：重载了 `+`、`-`、`*`、`/` 运算符，并提供了 `matmul`（矩阵乘法）、`transpose`（转置）、`reshape`（形状重塑）、`slice`（单维切片）以及 `slice_multi`（多维范围切片）。
- **内置神经网络组件**：
  - `Linear` 层（带有 Xavier 权重初始化和偏置项）。
  - 激活函数：`ReLU`、`Sigmoid`、`Tanh`（基于 Taylor 级数与 Newton-Raphson 的高精度纯算法实现）。
  - 损失函数：`MSELoss`。
- **优化器支持**：内置 `SGD` 优化器，支持权重更新及梯度清零。
- **无编译警告与完整自测**：通过 100% 单元测试覆盖与数值梯度校验（Numerical Gradient Checking），保证数学解析梯度的准确性。

---

### 📦 安装指南 (Installation)

你可以通过 MoonBit 包管理器 `moon` 将 `MoonGrad` 直接添加为项目依赖：

```bash
moon add lyjttio/moongrad
```

或者在你的 package 依赖文件 `moon.pkg.json` 中配置：

```json
{
  "import": [
    "lyjttio/moongrad"
  ]
}
```

---

### 💡 核心 API 用例 (API Usage Examples)

#### 1. 张量创建与算子重载 (Tensor Creation & Operations)

```moonbit
// 创建张量
let a = @moongrad.from_array([1.0, 2.0, 3.0, 4.0], [2, 2], requires_grad=true)
let b = @moongrad.from_array([5.0, 6.0, 7.0, 8.0], [2, 2], requires_grad=true)

// 算子重载 + 矩阵乘法
let c = (a + b) * a
let d = c.matmul(b)

// 自动微分反向传播
d.backward()
```

#### 2. 形状变换与切片 (Reshape, Transpose & Slicing)

```moonbit
let x = @moongrad.from_array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0], [2, 3], requires_grad=true)

// 1. Reshape 重塑形状
let y = x.reshape([3, 2])

// 2. Transpose 矩阵转置
let z = x.t() // 转置 2D 矩阵为 [3, 2]

// 3. Slice 单维与多维切片
let s1 = x.slice(1, 1, 3)                // 沿第 1 维截取 [1..3]
let s2 = x.slice_multi([(0, 2), (1, 3)])  // 多维切片

s1.backward()
```

#### 3. 神经网络与 SGD 优化器训练流程 (Linear Layer & SGD Optimizer)

```moonbit
// 定义输入与目标
let x = @moongrad.from_array([0.0, 0.0, 0.0, 1.0, 1.0, 0.0, 1.0, 1.0], [4, 2])
let y = @moongrad.from_array([0.0, 1.0, 1.0, 0.0], [4, 1])

// 定义网络层 (Linear 2 -> 4 -> 1)
let fc1 = @moongrad.Linear::new(2, 4, requires_grad=true)
let fc2 = @moongrad.Linear::new(4, 1, requires_grad=true)

// 收集参数与构建 SGD 优化器
let params = fc1.parameters()
let p2 = fc2.parameters()
for i in 0..<p2.length() { params.push(p2[i]) }

let optimizer = @moongrad.SGD::new(params, 1.0)

// 训练循环
for epoch in 0..<1000 {
  let h1 = fc1.forward(x).sigmoid()
  let pred = fc2.forward(h1).sigmoid()
  let loss = @moongrad.mse_loss(pred, y)

  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
}
```

---

### 🚀 运行演示

你可以直接运行内置的 XOR 多层感知机（MLP）训练演示：

```bash
# 在 JavaScript 目标后端上运行演示
moon run cmd/main --target js
```

---

### 🏆 CCF OSC 2026 大赛说明与开源参考

本项目已提交参加 **2026 MoonBit 国产基础软件开源生态大赛**（CCF OSC 2026）。

#### 🔗 开源项目参考与许可范围说明 (Open Source Reference Scope)

本项目在开发过程中借鉴并移植了业界优秀的开源自动微分及张量库设计理念：

1. **[PyTorch](https://github.com/pytorch/pytorch)**
   - **开源许可证**：BSD-3-Clause License
   - **参考范围**：借鉴了 PyTorch 的张量步长寻址（Strided Tensor Layout）、动态计算图（Dynamic Computation Graph）设计模型以及 PyTorch 风格的 API 接口形式（`forward()`, `backward()`, `zero_grad()`, `step()`）。
2. **[micrograd](https://github.com/karpathy/micrograd)**
   - **开源许可证**：MIT License
   - **参考范围**：借鉴了基于有向无环图（DAG）的深度优先拓扑排序标量与张量反向传播算法核心逻辑。

#### 💡 原创性与架构重构声明
本项目 100% 使用 **MoonBit 语言原生重构**，充分结合了 MoonBit 的模式匹配、代数数据类型（Enum）、类型推导与内存安全管理机制，摒弃了传统 C/C++ 动态链接库依赖，实现了轻量化、高性能的全原生张量自动微分引擎。

---

## English

`MoonGrad` is a lightweight, pure-MoonBit tensor computation and automatic differentiation (autograd) library. It brings dynamic computation graphs, broadcasting arithmetic, strided tensor manipulation, and neural network primitives directly to the MoonBit ecosystem, targeting Wasm, JavaScript, and Native backends.

### 🌟 Key Features

- **Pure MoonBit Implementation**: 100% MoonBit code with zero external C FFI dependencies. Fully portable and ready to compile for Wasm-GC, Node.js, and Native binaries.
- **Dynamic Autograd Engine**: PyTorch-style automatic differentiation using dynamic computation graphs, topological sorting, and reverse-mode accumulation.
- **Broadcasting & Strided Tensors**: Complete N-dimensional float tensor support with automatic shape alignment and strided memory access.
- **Shape Operators & Slicing**: Optimized `matmul`, `transpose`, `reshape`, `slice`, and `slice_multi` operations.
- **Built-in NN Components**:
  - `Linear` layer with Glorot/Xavier initialization.
  - Activations: `ReLU`, `Sigmoid`, and `Tanh` (analytically approximated with high-precision Taylor and Newton-Raphson methods).
  - Loss functions: `MSELoss`.
- **SGD Optimizer**: Robust optimization loop with parameter grouping, weight update, and gradient resetting.

---

### 📦 Installation

Add `MoonGrad` to your MoonBit project:

```bash
moon add lyjttio/moongrad
```

---

### 🏆 CCF OSC 2026 & Open Source Reference

This project is submitted to the **2026 MoonBit Open Source Ecosystem Competition** (CCF OSC 2026).

#### 🔗 Open Source References & License Scope

- **[PyTorch](https://github.com/pytorch/pytorch)** (License: BSD-3-Clause)
  - *Reference Scope*: Inspired PyTorch's strided multi-dimensional tensor layout, dynamic computation graph structure, and PyTorch-style API design (`backward()`, `zero_grad()`, `step()`).
- **[micrograd](https://github.com/karpathy/micrograd)** (License: MIT)
  - *Reference Scope*: Inspired the topological sorting algorithm for DAG-based reverse-mode automatic differentiation.

---

## 📜 许可证 (License)

本项目采用 **Apache License 2.0** 授权协议。  
This project is licensed under the Apache License 2.0.