# MoonGrad: MoonBit 原生张量与自动微分（Autograd）库

[English](#english) | [中文](#中文)

---

## 中文

`MoonGrad` 是一个完全用 **MoonBit** 语言实现的原生、轻量级、且极具扩展性的张量计算与自动微分（Autograd）库。它旨在为 MoonBit 生态提供深度的机器学习计算基础，可编译为 WebAssembly、JavaScript 以及 Native，实现全平台的高性能机器学习基础设施。

### 🌟 核心特性

- **纯 MoonBit 实现**：100% 纯 MoonBit 编写，不依赖任何外部 C/C++ 动态库或运行时，提供极佳的跨平台特性（Wasm、JS、Native）。
- **动态计算图与 Autograd 引擎**：支持与 PyTorch 类似的动态计算图（Dynamic Computation Graph），通过深度优先搜索（DFS）进行拓扑排序，并在其上执行反向传播算法。
- **多维张量（N-Dimensional Tensor）**：支持基于步长（Strides）的多维张量寻址及形状广播（Broadcasting）机制，可跨维度自动对齐计算。
- **丰富的数学算子**：重载了 `+`、`-`、`*`、`/` 运算符（实现 `Add`、`Sub`、`Mul`、`Div` 特性），并提供了 `matmul`（矩阵乘法）、`transpose`（转置）及 `reshape`（重塑形状）。
- **内置神经网络组件**：
  - `Linear` 层（带有 Xavier 权重初始化和偏置项）。
  - 激活函数：`ReLU`、`Sigmoid`、`Tanh`（基于 Taylor 级数与 Newton-Raphson 的高精度纯算法实现）。
  - 损失函数：`MSELoss`。
- **优化器支持**：内置 `SGD` 优化器，支持权重更新及梯度清零。
- **无编译警告与完整自测**：通过 100% 单元测试覆盖与数值梯度校验（Numerical Gradient Checking），保证数学解析梯度的准确性。

### 🚀 快速上手

你可以直接在克隆本仓库后运行内置的 XOR 多层感知机（MLP）训练演示：

```bash
# 在 JavaScript 目标后端上运行演示
moon run cmd/main --target js
```

在 3000 轮训练后，你将看到 Loss 收敛至接近 `0.0005`，且网络完美拟合了 XOR 的四个真值表：

```
==================================================
     MoonGrad: Pure MoonBit Autograd Demo         
==================================================
Training a Multi-Layer Perceptron to solve XOR...
Epoch: 0, Loss: 0.2678483079023928
Epoch: 600, Loss: 0.06691294727678392
Epoch: 1200, Loss: 0.003242972156014352
Epoch: 2100, Loss: 0.001032620967370885
Epoch: 2999, Loss: 0.0005896057563273055

Final Predictions:
Input: [0, 0] -> Output: 0.021237 (Target: 0.0)
Input: [0, 1] -> Output: 0.977768 (Target: 1.0)
Input: [1, 0] -> Output: 0.974921 (Target: 1.0)
Input: [1, 1] -> Output: 0.027983 (Target: 0.0)
==================================================
```

### 🏆 CCF OSC 2026 大赛说明与开源参考

本项目已提交参加 **2026 MoonBit 国产基础软件开源生态大赛**（CCF OSC 2026）。
- **移植与设计参考**：本项目的自动微分（Autograd）引擎设计参考了 PyTorch 的动态计算图（Dynamic Computation Graph）和 PyTorch/micrograd 自动微分的拓扑排序反向传播机制。本项目在移植与重新实现时，针对 MoonBit 的强类型系统、代数数据类型（Enum）与模式匹配进行了原生重构与精简设计。
- **原创声明**：本项目代码为 100% 纯 MoonBit 实现，没有任何外部 C FFI 依赖，支持编译至 Wasm、JS 和 Native 目标。

---

## English

`MoonGrad` is a lightweight, pure-MoonBit tensor computation and automatic differentiation (autograd) library. It brings dynamic computation graphs, broadcasting arithmetic, and neural network primitives directly to the MoonBit ecosystem, targeting Wasm, JavaScript, and Native backends.

### 🌟 Key Features

- **Pure MoonBit Implementation**: 100% MoonBit code with zero external C FFI dependencies. Fully portable and ready to compile for Wasm-GC or Node.js.
- **Dynamic Autograd Engine**: PyTorch-style automatic differentiation using dynamic computation graphs, topological sorting, and reverse-mode accumulation.
- **Broadcasting & Strided Tensors**: Complete N-dimensional float tensor support with automatic shape alignment and strided memory access.
- **Operator Overloading**: Native support for `+`, `-`, `*`, and `/` via MoonBit's compiler traits, plus optimized `matmul`, `transpose`, and `reshape` functions.
- **Built-in NN Components**:
  - `Linear` layer with Glorot/Xavier initialization.
  - Activations: `ReLU`, `Sigmoid`, and `Tanh` (analytically approximated with high-precision Taylor and Newton-Raphson methods).
  - Loss functions: `MSELoss`.
- **SGD Optimizer**: Robust optimization loop with parameter grouping, weight update, and gradient resetting.
- **Fully Verified**: Covered by analytical and numerical gradient checking to guarantee backpropagation correctness.

### 🚀 Quick Start

Run the built-in XOR MLP training program out-of-the-box:

```bash
# Run the demo on the JS target
moon run cmd/main --target js
```

### 🏆 CCF OSC 2026 & Open Source Reference

This project is submitted to the **2026 MoonBit Open Source Ecosystem Competition** (CCF OSC 2026).
- **Design & Porting Reference**: The design of MoonGrad's autograd engine is inspired by PyTorch's dynamic computation graph and the topological sorting-based backpropagation in micrograd/PyTorch. We adapted and simplified the architecture natively using MoonBit's robust type system, algebraic data types (Enums), and pattern matching.
- **Originality**: The entire library is written in 100% pure MoonBit with zero external C FFI dependencies, fully compatible with WebAssembly, JavaScript, and Native backends.

---

## 📜 许可证 (License)

本项目采用 **Apache License 2.0** 授权协议。
This project is licensed under the Apache License 2.0.