# 第 1 题：向量加法

欢迎来到 GPU 编程世界！我们从简单的内容开始，帮你熟悉基本的 CUDA 概念。

## 🎯 本题学习目标

- CUDA kernel 和 `__global__` 关键字
- thread indexing
- thread 和并行思维
- kernel 函数的调用

## 理解 CUDA Kernel

CUDA kernel 是一种特殊的函数，在 GPU 上运行，由成千上万个 thread 同时执行。每个 thread 运行相同的代码，但处理不同的数据。

在函数声明前添加 `__global__` 关键字，它告诉 CUDA 这个函数应该在 GPU 上运行：

```cpp
__global__ void my_kernel() {
    // 这段代码在 GPU 中运行
}
```

## Thread Indexing：并行计算的核心

假设你有 1000 个元素要处理，每个 block 有 256 个 thread。CUDA 将这些 thread 组织成网格结构：

```
Grid of Blocks:
Block 0: Thread 0-255    → 处理元素 0-255
Block 1: Thread 0-255    → 处理元素 256-511  
Block 2: Thread 0-255    → 处理元素 512-767
Block 3: Thread 0-255    → 处理元素 768-999
```

每个 thread 需要一个唯一的索引来确定它处理哪个元素。这个索引是通过以下公式计算的：

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

这个公式给每个 thread 分配一个唯一的全局索引：
- `blockIdx.x` 告诉你这个 thread 属于哪个 block
- `blockDim.x` 告诉你每个 block 中有多少个 thread  
- `threadIdx.x` 告诉你这个 thread 在其 block 内的位置

## 题目：向量加法

你的任务是实现向量加法，让每个 thread 计算一个元素。不再是一个 CPU 核心按顺序完成所有加法，而是数百个 GPU thread 同时各自处理一个加法运算。

向量加法在图形学、机器学习和科学计算等领域都非常常见，它是 GPU 编程的经典入门问题。

## 你的任务：完成 Kernel

我们已经提供了基本的代码框架，你需要专注于核心的并行逻辑：

### 实现 Kernel 逻辑

```cpp
__global__ void vectorAdd_kernel(int* d_A, int* d_B, int* d_C, int N) {
    int idx = /* TODO: 计算这个 thread 应该处理哪个数组元素 */;

    if (/* TODO: 边界检查：这个 thread 是否要运行？ */) {
        /* TODO: 执行这个元素的向量加法 */
    }
}
```

第一步是使用上面的公式计算你的 thread 的全局索引。第二步很关键：检查你的 thread 是否有有效工作。由于我们经常启动比数据元素更多的 thread，有些 thread 可能没有任何事情要做。

### 配置 Kernel Launch

```cpp
void vectorAdd_host(const std::vector<int> &h_A, const std::vector<int> &h_B, std::vector<int> &h_C) {
    // 内存分配和数据拷贝的代码
    
    int threadsPerBlock = /* TODO: 选择每个 block 有多少个 thread */;
    
    int numBlocks = /* TODO: 计算需要多少个 block */;
    
    vectorAdd_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // 内存拷贝和清理的代码
}
```

如果你不确定如何选择每个 block 的 thread 数量，可以从 256 开始。这个数字在 GPU 编程中非常常见，通常能提供良好的性能。在计算 block 的数量时，你需要保证 block 数量足以覆盖所有 N 个元素。你可以使用整数向上取整除法 `(N + threadsPerBlock - 1) / threadsPerBlock` 确保你有足够的 block。

## 理解边界检查

为什么我们需要边界检查？假设你有 1000 个元素，每个 block 256 个 thread。你需要 4 个 block，总共 1024 个 thread。最后 24 个 thread（索引 1000-1023）没有有效数据要处理，所以需要告诉它们什么都不做。

没有这个检查，那些额外的 thread 可能会访问数组之外的内存，可能导致崩溃或数据损坏。

## 测试

从小的测试用例开始。先试试只有 4 个元素的数组：

```cpp
int test_A[4] = {1, 2, 3, 4};
int test_B[4] = {5, 6, 7, 8};
// 预期结果: {6, 8, 10, 12}
```

你可以在 kernel 内使用 `printf` 进行调试（不过不能使用 std::cout）：

```cpp
if (idx < 4) {
    printf("Thread %d: C[%d] = %d + %d = %d\n", 
           idx, idx, d_A[idx], d_B[idx], d_C[idx]);
}
```

最后，你可以使用我们提供的测试系统来验证你的实现。

```bash
# 运行测试
make test
```

测试系统会自动编译你的代码并运行多个测试用例，将你的输出与正确结果进行比较。如果没有额外说明，后面的所有题目也会使用相同的测试命令。

**文件：**
- `student.cu`: 你需要编辑的文件（注意 TODO 部分）
- `answer.cu`: 参考答案文件
- `Makefile`: 构建系统
- `test_data/`: 测试用的输入输出文件

## 总结

完成这个问题后，你理解了 CUDA kernel 的基本概念，学会了如何计算 thread 索引，并且掌握了向量加法的并行实现。你迈出了第一步，进入了一个单个程序在数千个核心上同时运行的世界，在这里每个核心都在处理数据的一个小部分。

在第2题中，你将学到如何管理 GPU 内存、处理错误，也就是 CUDA 程序除了 kernel 之外的部分。