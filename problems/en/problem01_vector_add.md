# Problem 1: Welcome to CUDA - Vector Addition

Welcome to the exciting world of GPU programming! 🚀 You're about to write your first CUDA kernel - the foundation of all GPU computing. We'll start with something simple to help you get familiar with the basic CUDA concepts.

## 🎯 Learning Objectives

- Understanding CUDA kernels and the `__global__` keyword
- Mastering thread indexing
- Developing parallel thinking with threads
- Learning kernel function invocation

## Understanding CUDA Kernels

A CUDA kernel is a special function that runs on the GPU, executed simultaneously by thousands of threads. Think of it as giving the same instructions to an army of workers, where each worker handles a small portion of the total workload.

Adding the `__global__` keyword before a function declaration tells CUDA that this function should run on the GPU:

```cpp
__global__ void my_kernel() {
    // This code runs on the GPU
}
```

## Thread Indexing: The Heart of Parallel Computing

Imagine you have 1,000 elements to process, with 256 threads per block. CUDA organizes these threads into a grid structure:

```
Grid of Blocks:
Block 0: Thread 0-255    → processes elements 0-255
Block 1: Thread 0-255    → processes elements 256-511  
Block 2: Thread 0-255    → processes elements 512-767
Block 3: Thread 0-255    → processes elements 768-999
```

Each thread needs a unique index to determine which element it should process. This index is calculated using the following formula:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
```

This formula assigns each thread a unique global index:
- `blockIdx.x` tells you which block this thread belongs to
- `blockDim.x` tells you how many threads are in each block  
- `threadIdx.x` tells you this thread's position within its block

## The Problem: Vector Addition

Your task is to implement vector addition where each thread computes one element. Instead of having a single CPU core sequentially perform all additions, you'll have hundreds of GPU threads simultaneously handling one addition operation each.

Vector addition is fundamental in graphics, machine learning, and scientific computing - making it a classic introduction to GPU programming.

## Your Mission: Complete the Kernel

We've provided the basic code framework, so you can focus on the core parallel logic:

### Implement the Kernel Logic

```cpp
__global__ void vectorAdd_kernel(int* d_A, int* d_B, int* d_C, int N) {
    int idx = /* TODO: Calculate which array element this thread should process */;

    if (/* TODO: Boundary check: Should this thread run? */) {
        /* TODO: Perform vector addition for this element */
    }
}
```

The first step is calculating your thread's global index using the formula above. The second step is crucial: checking whether your thread has valid work to do. Since we often launch more threads than data elements, some threads might have nothing to process.

### Configure the Kernel Launch

```cpp
void vectorAdd_host(const std::vector<int> &h_A, const std::vector<int> &h_B, std::vector<int> &h_C) {
    // Memory allocation and data copying code
    
    int threadsPerBlock = /* TODO: Choose how many threads per block */;
    
    int numBlocks = /* TODO: Calculate how many blocks are needed */;
    
    vectorAdd_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, N);
    
    // Memory copying and cleanup code
}
```

If you're unsure about the number of threads per block, start with 256. This number is very common in GPU programming and typically provides good performance. When calculating the number of blocks, ensure you have enough blocks to cover all N elements. You can use ceiling division `(N + threadsPerBlock - 1) / threadsPerBlock` to guarantee sufficient blocks.

## Understanding Boundary Checks

Why do we need boundary checks? Suppose you have 1,000 elements with 256 threads per block. You need 4 blocks, totaling 1,024 threads. The last 24 threads (indices 1000-1023) have no valid data to process, so they need to be told to do nothing.

Without this check, those extra threads might access memory beyond the array bounds, potentially causing crashes or data corruption.

## Testing

Start with small test cases. Try an array with just 4 elements:

```cpp
int test_A[4] = {1, 2, 3, 4};
int test_B[4] = {5, 6, 7, 8};
// Expected result: {6, 8, 10, 12}
```

You can use `printf` for debugging inside the kernel (though you can't use std::cout):

```cpp
if (idx < 4) {
    printf("Thread %d: C[%d] = %d + %d = %d\n", 
           idx, idx, d_A[idx], d_B[idx], d_C[idx]);
}
```

Finally, you can use our provided testing system to validate your implementation:

```bash
# Run tests
make test
```

The testing system will automatically compile your code and run multiple test cases, comparing your output against the correct results. Unless otherwise specified, all subsequent problems will use the same test command.

**Files:**
- `student.cu`: The file you need to edit (note the TODO sections)
- `answer.cu`: Reference solution file
- `Makefile`: Build system
- `test_data/`: Input and output files for testing

## Summary

After completing this problem, you'll understand the basic concepts of CUDA kernels, know how to calculate thread indices, and have mastered the parallel implementation of vector addition. You've taken your first step into a world where a single program runs simultaneously on thousands of cores, each processing a small piece of the data.

In Problem 2, you'll learn how to manage GPU memory and handle errors - the parts of CUDA programming beyond just kernels.
