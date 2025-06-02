# Problem 5: Parallel Reduction Algorithm

Welcome to one of the most elegant and essential algorithms in parallel computing—parallel reduction! This algorithm transforms what would otherwise be a sequential multi-value combination task into an efficient parallel operation. It’s a perfect entry point for learning about sophisticated thread coordination patterns.

## 🎯 Learning Objectives

- Master tree-based parallel algorithms and their logarithmic complexity properties
- Learn advanced thread collaboration and synchronization patterns
- Implement multi-level computational strategies leveraging both GPU and CPU
- Understand the foundational reduction pattern in parallel computing

## The Core Idea of Parallel Reduction

A reduction operation combines multiple input values into a single output via associative operators such as addition, finding the maximum, or multiplication. The traditional sequential approach processes elements one by one, failing to fully exploit parallel hardware capabilities. In contrast, the parallel approach allows multiple associative operations to run concurrently. Consider the example of summing the numbers 1 through 8:

```
Sequential Execution:
1+2=3 → 3+3=6 → 6+4=10 → 10+5=15 → 15+6=21 → 21+7=28 → 28+8=36
Time: 7 steps (proportional to N)

Parallel Tree Reduction:
Round 1: [1+2] [3+4] [5+6] [7+8] → [3] [7] [11] [15] (4 parallel ops)
Round 2: [3+7] [11+15] → [10] [26] (2 parallel ops)  
Round 3: [10+26] → [36] (1 op)
Time: 3 steps (proportional to log N)
```

By structuring computation in layers, this reduction algorithm lowers the operational complexity from O(N) to O(log N), vastly improving efficiency.

## Task: Parallel Array Summation

Your objective is to implement a parallel tree reduction algorithm to compute the sum of an integer array.

## Tree Reduction Algorithm

The elegance of the tree reduction algorithm lies in its simplicity and efficiency. Here’s how 8 threads collaborate in shared memory:

```
Initial shared memory: [5] [3] [8] [1] [7] [2] [9] [4]
Thread IDs:            0   1   2   3   4   5   6   7

Step size = 4:
Threads 0-3 each add the element 4 positions ahead:
  [5+7] [3+2] [8+9] [1+4] [ ] [ ] [ ] [ ]
→ [12]  [5]   [17]  [5]   [ ] [ ] [ ] [ ]

Step size = 2:  
Threads 0-1 each add the element 2 positions ahead:
  [12+17] [5+5] [ ] [ ] [ ] [ ] [ ] [ ]
→ [29]    [10]  [ ] [ ] [ ] [ ] [ ] [ ]

Step size = 1:
Thread 0 adds the element 1 position ahead:
  [29+10] [ ] [ ] [ ] [ ] [ ] [ ] [ ]
→ [39]    [ ] [ ] [ ] [ ] [ ] [ ] [ ]
```

After three rounds, the sum of all 8 values is stored at the shared memory location for thread 0.

### Step 1: Grid-Stride Loading

```cpp
__global__ void reduce_sum_kernel(int* d_A_global, int* d_block_sums, int N) {
    extern __shared__ int s_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    // Each thread accumulates multiple elements
    int temp_sum = 0;
    for (int i = idx; i < N; i += stride) {
        temp_sum += d_A_global[i];
    }
    s_data[tid] = temp_sum;
    
    __syncthreads();  // Ensure all data is ready before reduction
```

The kernel uses dynamic shared memory to store the reduction data. Each block processes multiple elements through a grid-stride loop.

For example, with 4 threads processing 8 elements, thread 0 sums up elements 0 and 4, thread 1 sums elements 1 and 5, and so on. Once these partial sums are calculated, the tree reduction proceeds.

Grid-stride loops are a classic GPU programming pattern. Instead of launching N threads, we launch a reasonable number of blocks and have each thread process multiple elements, maximizing GPU utilization.

### Step 2: Tree Reduction

```cpp
    // Perform tree-based reduction
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();  // Synchronization point
    }
```

This is the core of the algorithm. In each iteration, the step size halves, and active threads add values from a fixed offset.

Synchronization is crucial. Omitting it can cause threads to access inconsistent data, producing incorrect results.

### Step 3: Storing Block Results

```cpp
    // Thread 0 stores the partial sum for this block
    if (tid == 0) {
        d_block_sums[blockIdx.x] = s_data[0];
    }
```

After the reduction, thread 0 in each block writes the partial sum to global memory for further processing on the host.

## Step 4: Reducing Results from Different Blocks

Now that we have the partial sum from each block, we need to reduce these into a final result. There are three ways to do this:

1. Accumulate the partial sums on the CPU using a loop;
2. Use atomic operations in a kernel to accumulate into a single variable (atomic operations will be discussed in a later problem);
3. Apply the tree reduction algorithm again.

We’ll choose the third approach and run the tree reduction algorithm on the block partial sums. Since the number of partial sums equals the number of blocks, we simply run the reduction kernel with a single block:

```cpp
reduce_sum_kernel<<<num_blocks, BLOCK_SIZE, shared_memory_size>>>(d_A_global, d_block_sums, N);
reduce_sum_kernel<<<1, BLOCK_SIZE, shared_memory_size>>>(d_block_sums, result, num_blocks);
```

## Summary

With this problem, you not only learn about complex thread collaboration and barrier synchronization, but also master multi-level parallel algorithm design. These skills empower you to fully leverage the combined computational power of the GPU and CPU.

It’s important to note that the reduction pattern is not limited to summation. The same tree structure can be applied to finding maximum values, computing dot products, convergence checks, and more.

In the next problem, we’ll extend these collaborative concepts to 2D image processing, exploring how threads can efficiently handle spatial neighborhood data.
