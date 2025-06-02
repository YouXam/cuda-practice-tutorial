# CUDA 实践教程

[English](./README.md) | 简体中文

这是一个 CUDA 编程的实践教程，旨在帮助读者通过实际编程练习，逐步掌握 CUDA 并行计算的基础概念和常用操作。内容覆盖向量加法、矩阵运算、卷积、并行归约等基本操作，并在实践中加深对 GPU 并行加速的理解。

## 环境配置

首先，确保你的电脑/服务器上有可用的 Nvidia GPU，然后在 [Nvidia 官网](https://developer.nvidia.com/cuda-downloads)下载并安装 CUDA Toolkit 和相关驱动。安装说明见 [CUDA 快速入门指南](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html)。

Clone 本仓库：

```bash
git clone https://github.com/youxam/cuda-practice-tutorial.git
cd cuda-practice-tutorial
```

生成工作目录：

```bash
python3 generate.py [path]
```

例如：

```bash
python3 generate.py ~/cuda-practice-projects
```

生成的目录中包含 9 道题目，每个题目都有对应的 `README.md` 包含完整的教程、题目描述和说明，你也可以在线阅读。

1. [第 1 题：向量加法](./problems/zh/problem01_vector_add.md)
2. [第 2 题：SAXPY](./problems/zh/problem02_saxpy.md)
3. [第 3 题：一维 stencil](./problems/zh/problem03_1d_stencil.md)
4. [第 4 题：矩阵转置](./problems/zh/problem04_matrix_transpose.md)
5. [第 5 题：并行归约求和](./problems/zh/problem05_parallel_reduction_sum.md)
6. [第 6 题：二维卷积](./problems/zh/problem06_2d_convolution.md)
7. [第 7 题：分块矩阵乘法](./problems/zh/problem07_matrix_mul_tiled.md)
8. [第 8 题：直方图](./problems/zh/problem08_histogram.md)
9. [第 9 题：K-means 聚类](./problems/zh/problem09_kmeans.md)

你应当按照以下流程完成每道题目：

1. 阅读题目描述和要求，理解需要实现的功能；
2. 根据题目要求，在 `student.cu` 文件中实现 `// TODO` 部分的代码；
3. 使用 `make list` 查看测试点列表，使用 `make run TC=<test_case_prefix>` 命令编译并运行单个测试点，使用 `make test` 编译并运行所有测试点。
4. 如果遇到困难，查看 `answer.cu` 中的参考实现，理解其思路和实现方式。

如果你需要配置 LSP，可以参考本项目的 `.clangd` 文件。

## 学习资源

- [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
- 《CUDA编程基础与实践》（樊哲勇）(中文)