---
layout: center
highlighter: shiki
css: unocss
colorSchema: dark
transition: fade-out
title: CUDA 开发入门：从硬件到编程实践
exportFilename: CUDA 开发入门：从硬件到编程实践 - YouXam - 2025 周行 HPC 训练营
lineNumbers: false
drawings:
  persist: false
mdc: true
clicks: 0
preload: false
glowSeed: 100
---

<h1 class="text-2xl font-bold text-center" style="font-size: 3em">
  CUDA 开发入门：从硬件到编程实践
</h1>

<h3 class="text-center text-gray-500 mt-10">
  YouXam
</h3>

---
layout: center
---

## Agenda

<div class="grid grid-cols-2 gap-8 pt-4">
<div>

<h2 class="text-blue-300">

### Part 1: The Dawn of Parallel Computing
</h2>

- 从图形到通用计算的演进
- CPU vs. GPU

<h2 class="text-blue-300">

### Part 2: GPU Hardware
</h2>

- 现代 NVIDIA GPU 剖析
- 核心引擎：流式多处理器


</div>
<div>


<h2 class="text-blue-300">

### Part 3: CUDA Programming Model
</h2>

- 异构计算：主机与设备
- 并行抽象：线程、块、网格
- 内存模型：分层与权衡



<h2 class="text-blue-300">

### Part 4: CUDA 开发实践
</h2>

- 环境搭建
- 向量加法
</div>
</div>

---
layout: intro
---

<div>
<h2 style="font-size: 30pt; line-height: 30pt" class="my-4">
Part 1:
</h2>
<h1  style="font-size: 40pt; line-height: 50pt">
The Dawn of Parallel Computing
</h1>
</div>

---
layout: center
---

## GPU 的演进：从像素到 PetaFLOPS

<span></span>

- **早期 (1970s-1990s):** 专用图形芯片，负责将 CPU 处理好的信息转换为视频信号。
- **3D 革命 (1995-2006):** 游戏需求推动了3D图形加速卡的诞生。NVIDIA 在1999年发布 GeForce 256，首次将变换 (Transform)、光照 (Lighting) 集成到单芯片上，并推广了 "GPU" 这一术语。
- **GPGPU 时代 (2007-至今):** NVIDIA 发布 CUDA (Compute Unified Device Architecture)。开发者首次可以使用 C/C++ 等高级语言直接在 GPU 上进行通用目的计算 (GPGPU)，释放了其强大的并行计算能力。


---
layout: center
---

<h1 class="text-center">
CPU vs. GPU
</h1>

<div class="grid grid-cols-2 gap-8 pt-4">

<div>

### CPU (Latency-Optimized)

- **目标:** 最小化**单个任务**的完成时间。
- **架构:**
    - **少量、强大的核心**：集成了复杂控制逻辑（如分支预测、乱序执行）。
    - **大型多级缓存**：快速访问常用数据，避免等待。

</div>
<div>

### GPU (Throughput-Optimized)

- **目标:** 最大化**单位时间内**的总任务量。
- **架构:**
    - **海量、简单的核心**：专注于高效执行算术运算。
    - **高带宽内存**：通过极高的内存带宽提高数据吞吐量。

</div>
</div>

---

## CPU vs. GPU 架构示意图

![](/cpugpu.png)

---

# Part 2: GPU Hardware

<div class="abs-br m-6 text-xs text-gray-500">
NVIDIA Hopper 架构图，展示了 GPC (大框) 和 SM (小框) 的层次结构
</div>


![](/hopper.png)


---
layout: center
---

## GPU 硬件架构

一个现代 GPU 芯片是一个高度分层的并行处理器系统。

- **GPU Die (芯片):** 整个系统的物理基础。
- **Graphics Processing Clusters (GPC):** 可看作微型 GPU，包含所有关键部件。
- **Streaming Multiprocessors (SM):**
    - **GPU 架构中最重要的核心计算单元。**
    - 我们编写的 CUDA 程序，最终被分解成任务块，交由 SM 执行。
- **Memory Controllers:** 连接到高速显存 (GDDR/HBM)，为海量 SM 提供数据流。

---
layout: center
---

## 核心引擎：流式多处理器 (SM)

如果 GPU 是工厂，SM 就是高效运转的生产车间。我们写的并行代码，都在 SM 中执行。

一个 SM 内部集成了多种功能单元：

- **CUDA Cores:** 基本的算术逻辑单元 (ALU)，执行浮点和整数运算。
- **Warp Schedulers:** 负责指令调度，选择一个就绪的线程束 (Warp) 并分派指令。
- **Register File:** 海量寄存器，动态分配给在 SM 上运行的所有线程，用于存储私有变量。
- **L1 Cache / Shared Memory:** 高速片上内存，可配置为 L1 缓存和程序员可控的共享内存，是实现高性能的关键。
- **Specialized Cores:**
    - **Tensor Cores:** 加速 AI 中的混合精度矩阵运算。
    - **RT Cores:** 加速图形学中的光线追踪计算。

---
layout: center
---

## GPU 核心设计：为高并行度而生

GPU 的核心设计思想是通过大规模并行来隐藏内存访问延迟。

- **问题:** 访问主内存 (DRAM) 速度很慢，会导致计算单元长时间等待，造成资源浪费。
- **CPU 的策略:** 用巨大的缓存和复杂的预测逻辑来**避免延迟**。
- **GPU 的策略:** 接受延迟，并通过快速上下文切换来**隐藏延迟**。

当一个线程束 (Warp) 因等待数据而停顿时，SM 的调度器会**立即**将计算资源切换给另一个已就绪的 Warp，让计算核心始终保持忙碌。

这种低开销的上下文切换能力，是 GPU 能够爆发出惊人计算吞吐量的根本原因。它要求 SM 必须配备海量的寄存器来保存所有线程的状态。

---
layout: image-right
image: /model.svg
---

<div class="h-full flex items-center justify-center">

<div>
<h2 style="font-size: 40pt; line-height: 40pt" class="my-4">
Part 3:
</h2>
<h1  style="font-size: 50pt; line-height: 70pt">
CUDA Programming Model
</h1>
</div>

</div>

---
layout: center
---

## 异构计算：主机 (Host) 与 设备 (Device)

CUDA 程序在一个**异构计算 (Heterogeneous Computing)** 环境中运行，同时利用 CPU 和 GPU。


<div class="grid grid-cols-2 gap-8 pt-4">

<div>

- **主机 (Host):**
    - 指 **CPU** 及其主内存。
    - 负责程序的整体流程控制、I/O 等串行任务。
    - 负责启动 GPU 上的计算任务。

</div>
<div>

- **设备 (Device):**
    - 指 **GPU** 及其专用显存。
    - 负责执行大规模并行、计算密集型的任务。
    - 这些并行任务被称为**内核 (Kernel)**。

</div>
</div>

主机和设备拥有**各自独立的内存空间**。数据必须通过 PCIe 总线在两者之间显式拷贝。

---
layout: image-right
image: /model.svg
---

## 线程组织：Grid, Block, Thread

为了有效管理数以万计的并行任务，CUDA 提供了简洁而强大的三级线程层次结构。

- 线程 (Thread): 执行计算的**最小单位**，每个线程通过唯一的 `threadIdx` 识别自己。

- 线程块 (Block): 协作的单位。同一个块内的线程可以通过**共享内存**交换数据，并通过 `__syncthreads()` 进行同步。每个块通过唯一的 `blockIdx` 识别自己。  

- 网格 (Grid): 独立的单位。一次内核启动对应一个网格。不同块中的线程被认为是相互独立的，不能直接通信。这种独立性是**可扩展性** 的关键。

<!-- 这个三级层次结构可以被定义为一维、二维或三维，方便处理向量、矩阵或三维体数据。 -->

---
layout: default
---

## 编程模型到硬件的映射

CUDA 的线程层次结构与 GPU 的物理硬件结构具有直接的对应关系。

<div class="grid grid-cols-5 items-center pt-8">

<div class="p-4 rounded-lg col-span-2">
<h3 class="text-2xl text-center mb-4">编程模型 (Software)</h3>
<div class="space-y-4 text-center">
  <p class="p-2 border rounded"><b>Grid (网格)</b></p>
  <p class="p-2 border rounded"><b>Block (线程块)</b></p>
  <p class="p-2 border rounded"><b>Thread (线程)</b></p>
</div>
</div>

<div class="flex justify-center items-center h-full">
<img src="/arrow.svg" alt="Arrow" class="w-16 h-16 mx-auto">
</div>

<div class="p-4 col-span-2">
<h3 class="text-2xl text-center mb-4">硬件结构 (Hardware)</h3>
<div class="space-y-4 text-center">
  <p class="p-2 border rounded"><b>Device (设备)</b></p>
  <p class="p-2 border rounded"><b>SM (流式多处理器)</b></p>
  <p class="p-2 border rounded"><b>CUDA Core (核心)</b></p>
</div>
</div>
</div>

**一个完整的线程块 (Block) 会被调度到一个 SM 上执行。** 块内的所有线程都在这个 SM 上运行，共享该 SM 的资源（如共享内存和寄存器）。


---
layout: center
---

## 可扩展性

Grid/Block/Thread 的层次结构是程序员与硬件之间的约定。

- **程序员的责任:** 将问题分解成许多**相互独立的线程块 (Blocks)**。
- **硬件/运行时的承诺:** CUDA 运行时会根据当前 GPU 的 SM 数量，自动地、动态地将这些 Blocks 调度到不同的 SM 上去执行。

这意味着，你编写的 CUDA 程序**无需修改一行代码**，就可以在从笔记本 GPU 到数据中心级 GPU 的不同硬件上运行，并自动利用可用的硬件资源获得性能提升。


---
layout: image-right
image: /warp.png
---

## Warp: 并发执行的基本单位

硬件层面，SM 调度和执行的基本单位是一个由 **32 个线程**组成的集合，称为 **Warp (线程束)**。

- **SIMT:** 这是 CUDA 的执行模型。在一个 Warp 中的 32 个线程，在任意一个时间点，都在**执行完全相同的指令**。

- **Warp Divergence (线程束分化):** 如果一个 Warp 内的不同线程因 `if-else` 等条件判断走向了不同的代码分支，就会发生分化。此时，硬件会串行化执行这些分支：先执行 `if` 分支（另一部分线程被禁用），再执行 `else` 分支（之前执行 `if` 的线程被禁用）。这会大幅降低并行效率，是 CUDA 编程中需要重点关注和避免的性能陷阱。

---
layout: default
---

## 调度与资源最大化

CUDA 运行时系统负责将成千上万的线程块调度到 GPU 上的 SM 中，以最大化硬件资源的利用率。

- **块调度 (Block Scheduling):**
    
    运行时维护一个待执行的 Block 列表。当一个 SM 有空闲资源时，调度器会从列表中取出一个或多个 Block 分配给它。一个 Block 一旦被分配给某个 SM，就会一直在该 SM 上运行直到执行完毕。

- **Warp 调度 (Warp Scheduling):**

    SM 内部的 Warp 调度器负责管理在该 SM 上运行的所有 Warp。它不断地在所有**就绪 (ready)** 的 Warp 之间切换，以隐藏访存等操作带来的延迟。

- **占用率 (Occupancy):**
    
    指一个 SM 上活跃的 Warp 数量与该 SM 所能支持的最大 Warp 数量的比例。较高的占用率通常意味着 SM 有更多的 Warp 可供选择来隐藏延迟，从而可能带来更高的性能。但占用率并非越高越好，它只是众多性能指标之一。


---
layout: center
---

## CUDA 内存模型

理解并正确利用分层的内存模型，是榨干 GPU 性能的关键。

- **寄存器 (Register):** **最快**，线程私有。
- **共享内存 (Shared Memory):** **非常快**，块内线程共享，是实现高效协作的核心。
- **L1/L2 缓存 (Cache):** 硬件自动管理，对程序员不完全透明。
- **全局内存 (Global Memory):** **最慢但容量最大**，所有线程可见，是 Host/Device 数据交换的场所。
- **常量/纹理内存 (Constant/Texture):** 带专用只读缓存的全局内存，适用于特定访问模式。

性能优化的本质，就是最大化地利用高速内存（寄存器和共享内存），最小化地访问慢速的全局内存。

---
layout: intro
---

# Part 4: CUDA 开发实践

<span></span>

## 环境搭建

开始编码前，请确保已正确配置开发环境。

1.  **[安装 NVIDIA 驱动](https://www.nvidia.com/Download/index.aspx)**，确保操作系统能与 GPU 硬件通信。

2.  **[安装 CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit)**，包含 NVCC 编译器、核心库、调试和性能分析工具。

---
layout: default
---

## Kernel 启动：`<<<grid, block>>>`

CUDA 使用一种特殊的 `<<<...>>>` 语法来从主机启动设备上的 Kernel 函数，并定义其执行配置。

```cpp
// 定义一个内核函数
__global__ void myKernel(float* data) {
    // ... GPU 并行代码 ...
}

// 主机代码
int main() {
    dim3 gridSize(16, 16);    // 16x16 的网格，共 256 个块
    dim3 blockSize(256);      // 每个块包含 256 个线程

    // 启动内核
    myKernel<<<gridSize, blockSize>>>(d_data);
}
```

- **Grid (网格):** 定义了本次内核启动包含多少个**线程块 (Block)**。
- **Block (块):** 定义了每个线程块包含多少个**线程 (Thread)**。

这次启动总共会创建 `16 * 16 * 256 = 65,536` 个并行线程。

---
layout: center
---

## 设备代码（Kernel）

定义了每个 GPU 线程需要执行的具体工作。

````md magic-move {lines: true}
```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)  {
    // 运行在 GPU 上的代码
}
```

```cpp {*|6}
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)  {
    // 计算当前线程的全局唯一索引
    //   blockDim.x      每个线程块的线程数
    //   blockIdx.x      当前线程块的索引
    //   threadIdx.x     当前线程在块内的索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;
}
```

```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)  {
    // 计算当前线程的全局唯一索引
    //   blockDim.x      每个线程块的线程数
    //   blockIdx.x      当前线程块的索引
    //   threadIdx.x     当前线程在块内的索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // 确保索引不越界
    if (i < n) {
        // 计算
    }
}
```

```cpp
__global__ void vectorAdd(const float *A, const float *B, float *C, int n)  {
    // 计算当前线程的全局唯一索引
    //   blockDim.x      每个线程块的线程数
    //   blockIdx.x      当前线程块的索引
    //   threadIdx.x     当前线程在块内的索引
    int i = blockDim.x * blockIdx.x + threadIdx.x;

    // 确保索引不越界
    if (i < n) {
        C[i] = A[i] + B[i];
    }
}
```
````

---

## 主机代码

<span></span>

负责流程编排：内存管理、数据传输和内核启动。


````md magic-move {lines: true}
```cpp
int main() {
    int n = 1 << 20;

    // ...

    // 配置并启动内核
    vectorAdd<<</* ? */, /* ? */>>>(/* ? */);
    
    // ...
}
```


```cpp {7,8}
int main() {
    int n = 1 << 20;

    // ...

    // 配置并启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(/* ? */);

    // ...
}
```

```cpp {5,10}
int main() {
    int n = 1 << 20;
    // ...

    float *d_A, *d_B, *d_C;

    // 配置并启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // ...
}
```

```cpp {6-9}
int main() {
    int n = 1 << 20;
    // ...

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    // 配置并启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // ...
}
```

```cpp {10-11}
int main() {
    int n = 1 << 20;
    // ...

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 配置并启动内核
    int threadsPerBlock = 256;
    int blocksPerGrid = (n + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, n);

    // ...
}
```


```cpp {3}
int main() {
    int n = 1 << 20;
    float *h_A, *h_B, *h_C;

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));

    cudaMemcpy(d_A, h_A, size, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B, size, cudaMemcpyHostToDevice);

    // 配置并启动内核 ...

    // ...
}
```


```cpp {4-6}
int main() {
    int n = 1 << 20;
    float *h_A, *h_B, *h_C;
    malloc(h_A, n * sizeof(float));
    malloc(h_B, n * sizeof(float));
    malloc(h_C, n * sizeof(float));

    float *d_A, *d_B, *d_C;
    cudaMalloc(&d_A, n * sizeof(float));
    cudaMalloc(&d_B, n * sizeof(float));
    cudaMalloc(&d_C, n * sizeof(float));
    // ... copy data to device ...

    // 配置并启动内核 ...

    // ...
}
```


```cpp {7-10}
int main() {
    int n = 1 << 20;
    float *h_A, *h_B, *h_C;
    malloc(h_A, n * sizeof(float));
    malloc(h_B, n * sizeof(float));
    malloc(h_C, n * sizeof(float));
    for (auto i = 0; i < n; i++) {
        h_A[i] = static_cast<float>(i);
        h_B[i] = static_cast<float>(i * 2);
    }

    float *d_A, *d_B, *d_C;
    // ... cudaMalloc and copy data to device ...


    // 配置并启动内核 ...

    // ...
}
```

```cpp {11}
int main() {
    int n = 1 << 20;
    float *h_A, *h_B, *h_C;
    // ... malloc and initialize h_A, h_B ...

    float *d_A, *d_B, *d_C;
    // ... cudaMalloc and copy data to device ...

    // 配置并启动内核 ...

    cudaMemcpy(h_C, d_C, size, cudaMemcpyDeviceToHost);
}
```


````

<span></span>

<v-switch at="0">
  <template #1>
    根据每个 block 的线程数可计算出需要的 block 数量；
  </template>
  <template #2>
    核函数传入参数时，基本类型（例如 int）可以直接传递，复杂类型（例如数组）需要使用指针传递；
  </template>
  <template #3> 
    核函数不能使用主机内存， 所以需要使用 cudaMalloc 在设备上分配内存， 然后在核函数中使用；
  </template>
  <template #4> 
    在主机代码中，使用 cudaMemcpy 将数据从主机内存复制到设备内存；
  </template>
  <template #5> 
    定义主机内存的指针；
  </template>
  <template #6> 
    在主机内存上分配空间；
  </template>
  <template #7> 
    初始化两个向量；
  </template>
  <template #8> 
    核函数运行后，将结果从设备内存复制回主机内存。
  </template>
</v-switch>


<v-click at="9"> 

## 编译与运行

<span></span>

```bash
nvcc vector_add.cu -o vector_add
./vector_add
```

</v-click>

---
layout: intro
---


## 实践

[https://github.com/YouXam/cuda-practice-tutorial/](https://github.com/YouXam/cuda-practice-tutorial/blob/main/README.zh.md)

大家可以通过这个项目来通过实际编写代码熟悉 CUDA 编程。

---
layout: center
class: text-center
---

# Q & A

**Thank You!**

