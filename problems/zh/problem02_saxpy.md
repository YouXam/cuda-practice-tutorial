# 第 2 题：SAXPY

完成了第一个题目之后，你已经大致了解了编写 CUDA kercel，但 CUDA 程序不只是 kernel 函数。你需要了解如何管理 GPU 内存、处理错误，这些也是 CUDA 编程的关键部分。

## 🎯 本题学习目标

- 完整的 GPU 内存管理生命周期
- 健壮的错误处理
- 标准的 CUDA 工作流模式
- 线性代数中的 SAXPY 操作

## 双内存系统

你的计算机实际上有两个完全独立的内存系统。CPU 有自己的 RAM，GPU 有自己的显存（VRAM）。这些内存空间相互隔离，只通过 PCIe 总线连接。

这种分离意味着 GPU kernel 只能访问 GPU 内存，CPU 代码只能直接访问 CPU 内存。当你想在 GPU 上处理数据时，必须先显式地将数据从 CPU 内存复制到 GPU 内存。处理完成后，必须将结果复制回来。

可以想象成在不同城市有两个工厂，每个工厂有自己的仓库。在 A 工厂的工作不能直接使用 B 工厂的仓库里的材料。你必须先将材料从 B 工厂的仓库运过来。

## 理解 CUDA 内存函数

CUDA 提供了管理 GPU 内存的专门函数：

```cpp
// 分配 GPU 内存
cudaMalloc((void**)&d_ptr, size_in_bytes);

// CPU → GPU 复制
cudaMemcpy(d_ptr, h_ptr, size_in_bytes, cudaMemcpyHostToDevice);

// GPU → CPU 复制
cudaMemcpy(h_ptr, d_ptr, size_in_bytes, cudaMemcpyDeviceToHost);

// 释放 GPU 内存
cudaFree(d_ptr);
```

`cudaMalloc` 函数类似于 `malloc`，但它在 GPU 上分配内存。需要双指针 `(void**)&d_ptr` 是因为 `cudaMalloc` 需要修改你的指针变量，使其指向新分配的内存。

在 64 位系统上，不需要显式指定内存复制方向，可以使用 `cudaMemcpy(a, b, size, cudaMemcpyDefault)`。

## SAXPY

SAXPY 代表"Single-precision A times X Plus Y"，计算 `C[i] = alpha * A[i] + B[i]`。这个操作在线性代数中很基础，在数值计算中随处可见。

与简单的向量加法不同，SAXPY 结合了标量乘法和向量加法。该操作涉及标量（alpha）和向量（A 和 B），展示了如何向 GPU kernel 传递不同类型的数据。

## 你的任务

这次，你将实现 kernel 和完整 CUDA 应用程序所需的所有其他代码。

### 第1部分：SAXPY Kernel

基于你的向量加法知识，实现 SAXPY 计算：

```cpp
__global__ void saxpy_kernel(int alpha, int* d_A, int* d_B, int* d_C, int N) {
    // TODO: 参考第一题，实现 C[i] = alpha * A[i] + B[i] 的计算
}
```

注意标量 `alpha` 是作为常规参数传递的。CUDA 自动将标量值复制到每个 thread，所以每个 thread 都得到自己的 `alpha` 副本。

### 第2部分：完整内存管理

现在实现 host 端代码：

```cpp
void saxpy_host(int alpha, const std::vector<int> &h_A, const std::vector<int> &h_B, std::vector<int> &h_C) {
    // 声明 device 指针
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;
    
    // 计算字节大小
    size_t bytes = N * sizeof(int);
    
    // TODO: 为所有三个数组分配 GPU 内存
    
    // TODO: 将输入数据从 host 复制到 device
    
    // TODO: 调用 kernel
    
    // TODO: 将结果复制回 host
    
    // TODO: 释放所有 GPU 内存
}
```

## 标准 CUDA 工作流

每个 CUDA 程序都遵循相同的五步模式：

**步骤1：分配 GPU 内存**：
你需要在 GPU 上为所有数组分配空间。如果 GPU 内存不足，内存分配可能失败，我们将马上讨论如何处理错误。

**步骤2：将输入数据复制到 GPU**：
将输入数组（A 和 B）从 CPU 内存传输到你刚分配的 GPU 内存。注意你不需要复制输出数组 C。

**步骤3：启动 Kernel**：
执行并行计算，这里你决定 block 和 thread 的数量。

**步骤4：复制回结果**：
kernel 完成后，将结果数组（C）从 GPU 内存复制回 CPU 内存，这样你的程序就可以使用它。

**步骤5：释放 GPU 内存**：
清理你分配的所有 GPU 内存。

## 错误处理

CUDA 操作可能以多种方式失败：GPU 内存不足，kernel 启动配置无效，或者 kernel 在执行期间崩溃。

```cpp
// 检查 kernel 启动的错误
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
}

// 等待 kernel 完成并检查执行错误
if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
}
```

`cudaDeviceSynchronize()` 函数等待所有挂起的 GPU 操作完成。如果没有运行该函数，你的 CPU 代码可能在 GPU 完成计算之前就尝试复制结果。

我们可以编写一个宏来简化错误检查：

```cpp
#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)
```

使用的时候，用 `CUDA_CHECK` 宏包裹 CUDA 函数：

```cpp
CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

// ...

CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

// ...
```

## 性能考虑

CPU 和 GPU 之间的内存传输比实际计算昂贵。对于向量加法和 SAXPY，你程序的大部分时间花在复制数据而不是计算结果上。

在实际应用中，你应当尽量将数据保持在 GPU 上，在复制结果回来之前执行多个操作。这将传输成本分摊到许多计算上，从而提高整体性能。

## 总结

完成这个问题后，你现在理解了 CUDA 编程的完整基础。你可以分配 GPU 内存、安全传输数据、启动带有适当错误检查的 kernel，并正确清理资源。

这个五步工作流模式出现在每个 CUDA 应用程序中，从简单计算到训练大规模神经网络。计算 kernel 变得更复杂，但基本流程不变。

在第3题中，我们将介绍 shared memory - 一种强大的优化，通过实现 thread 之间的高效协作来显著提高性能。