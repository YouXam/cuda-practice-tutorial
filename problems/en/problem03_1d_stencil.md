# Problem 3: 1D Stencil - Using Shared Memory

We will explore a key concept in GPU programming: shared memory and collaborative computing. You'll master how to leverage shared memory to significantly boost program performance by implementing a 1D averaging filter.

## 🎯 Learning Objectives

- Understanding shared memory and GPU memory hierarchy
- Mastering thread collaboration and synchronization mechanisms
- Implementing optimized 1D Stencil algorithms

## GPU Memory Architecture Analysis

GPUs contain multi-level memory structures, each with distinct characteristics:

**Global Memory**: Main memory space accessible to all threads. It offers large capacity (GB-scale) but is relatively slow, requiring hundreds of clock cycles for access.

**Shared Memory**: High-speed memory shared among threads within a block. It's small (a few KB per block) but extremely fast, requiring only a few clock cycles for access.

**Registers**: Thread-private storage that's fastest but limited in quantity.

Shared memory access is hundreds of times faster than global memory, and proper utilization can bring dramatic performance improvements.

## 1D Averaging Filter Principles

Stencil algorithms use neighboring elements to compute output values. In this experiment, each output element is the average of itself and its left and right neighbors:

Taking the array `[4, 7, 2, 9, 1, 8, 3, 6]` as an example, the output for the third element would be:

```
[4, 7, 2, 9, 1, 8, 3, 6]
    ^  ^  ^
(7 + 2 + 9) / 3 = 6
```

## Performance Bottleneck Analysis

In naive implementations, each thread needs to read data multiple times from global memory:

```cpp
// Each thread reads 3 values from slow global memory
C[idx] = (A[idx - 1] + A[idx] + A[idx + 1]) / 3;
```

The thread processing element 5 reads A[4], A[5], and A[6]. The thread processing element 6 reads A[5], A[6], and A[7]. Both need to read A[5] and A[6], leading to massive redundant global memory accesses that severely impact performance.

## Shared Memory Optimization Strategy

We can optimize this problem using the following approach:

1. Threads collaborate to load data blocks into shared memory
2. Synchronize to wait for data loading completion
3. Read data from high-speed shared memory to compute results

This method transforms multiple slow global memory accesses into a single load followed by multiple high-speed shared memory accesses.

During data loading, each thread needs to access its left and right neighboring elements. For example, a 16-thread block processing elements 100-115 actually needs to load elements 99-116. The additional elements 99 and 116 are called "halo" or "ghost" cells.

```
Global Memory: [... 99 100 101 102 ... 114 115 116 ...]
Shared Memory: [99][100 101 102 ... 114 115][116]
                ↑        ← Main data region →         ↑
              Left halo                         Right halo
```

## Task

Please implement a 1D averaging filter using shared memory optimization:

### Step 1: Declare Shared Memory

```cpp
__global__ void stencil1D_kernel(int* d_A, int* d_C, int N) {
    extern __shared__ int s_data[];  // Dynamic shared memory
    
    const int radius = 1;
    int tid = threadIdx.x;
    int gid = blockIdx.x * blockDim.x + threadIdx.x;
```

The `extern __shared__` declaration creates memory space shared among threads within a block, with size determined at kernel launch time.

### Step 2: Collaborative Data Loading

The first and last threads are responsible for loading the front and back halo elements respectively, while other elements are loaded collaboratively by threads.

When loading halo elements, special attention must be paid to handling boundary conditions to ensure no out-of-bounds access occurs. We employ an "edge clamping" approach:

- For the first element, the left neighbor A[-1] doesn't exist, so we use A[0].
- For the last element, the right neighbor A[N] doesn't exist, so we use A[N-1].

### Step 3: Synchronization

```cpp
    __syncthreads();  // Wait for all threads to complete loading
```

Synchronization is crucial, ensuring all data is ready before computation begins. The `__syncthreads()` call creates a barrier - ensuring all threads reach the synchronization point before continuing execution.

Synchronization serves two purposes here: correctness (ensuring data is loaded before use) and performance (ensuring all threads proceed together for maximum efficiency).

### Step 4: Shared Memory Computation

```cpp
    if (gid < N) {
        int left = s_data[tid + radius - 1];
        int center = s_data[tid + radius];
        int right = s_data[tid + radius + 1];
        
        d_C[gid] = (left + center + right) / 3;
    }
```

Now all data accesses are completed in high-speed shared memory.

## Host-side Configuration

The kernel uses `extern __shared__` to declare dynamic shared memory, and when calling the kernel, you need to specify the size of the shared memory.

```cpp
int threadsPerBlock = 256;
int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

// Calculate shared memory size
int radius = 1;
int sharedMemSize = (threadsPerBlock + 2 * radius) * sizeof(int);

// Launch kernel
stencil1D_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_C, N);
```

The advantage of dynamic shared memory is that it can flexibly adjust the size according to the problem scale without being fixed at compile time. However, in this problem, the shared memory size is actually fixed, so we can also use static shared memory declaration:

```cpp
// Use static shared memory in kernel function
__shared__ int s_data[256 + 2];
```

Static shared memory does not require specifying the size when calling, and you only need to pass two parameters when launching the kernel function.

## Performance Improvement Analysis

After shared memory optimization, global memory accesses are reduced from "three times per thread" to "once per block." For a 256-thread block, this theoretically reduces global memory traffic by a factor of 256.

## Summary

You now understand collaborative parallel algorithms, thread synchronization, and shared memory optimization. These skills apply to virtually every high-performance GPU algorithm.

In subsequent experiments, we'll extend these concepts to 2D, where they become even more powerful for processing images and matrices.

Congratulations on mastering one of CUDA's most powerful optimization techniques!