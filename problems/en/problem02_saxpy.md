# Problem 2: SAXPY

After completing the first problem, you now have a general understanding of writing CUDA kernels, but CUDA programs are more than just kernel functions. You need to learn how to manage GPU memory and handle errors - these are also crucial parts of CUDA programming.

## 🎯 Learning Objectives

- Complete GPU memory management lifecycle
- Robust error handling
- Standard CUDA workflow patterns
- The SAXPY operation in linear algebra

## The Dual Memory System

Your computer actually has two completely separate memory systems. The CPU has its own RAM, and the GPU has its own video memory (VRAM). These memory spaces are isolated from each other, connected only through the PCIe bus.

This separation means that GPU kernels can only access GPU memory, while CPU code can only directly access CPU memory. When you want to process data on the GPU, you must first explicitly copy the data from CPU memory to GPU memory. After processing is complete, you must copy the results back.

Think of it like having two factories in different cities, each with its own warehouse. Work in Factory A cannot directly use materials from Factory B's warehouse. You must first transport the materials from Factory B's warehouse.

## Understanding CUDA Memory Functions

CUDA provides specialized functions for managing GPU memory:

```cpp
// Allocate GPU memory
cudaMalloc((void**)&d_ptr, size_in_bytes);

// CPU → GPU copy
cudaMemcpy(d_ptr, h_ptr, size_in_bytes, cudaMemcpyHostToDevice);

// GPU → CPU copy
cudaMemcpy(h_ptr, d_ptr, size_in_bytes, cudaMemcpyDeviceToHost);

// Free GPU memory
cudaFree(d_ptr);
```

The `cudaMalloc` function is similar to `malloc`, but it allocates memory on the GPU. The double pointer `(void**)&d_ptr` is needed because `cudaMalloc` needs to modify your pointer variable to point to the newly allocated memory.

On 64-bit systems, you don't need to explicitly specify the memory copy direction - you can use `cudaMemcpy(a, b, size, cudaMemcpyDefault)`.

## SAXPY

SAXPY stands for "Single-precision A times X Plus Y" and computes `C[i] = alpha * A[i] + B[i]`. This operation is fundamental in linear algebra and appears everywhere in numerical computing.

Unlike simple vector addition, SAXPY combines scalar multiplication with vector addition. The operation involves both a scalar (alpha) and vectors (A and B), demonstrating how to pass different types of data to GPU kernels.

## Your Task

This time, you'll implement both the kernel and all the other code needed for a complete CUDA application.

### Part 1: SAXPY Kernel

Building on your vector addition knowledge, implement the SAXPY computation:

```cpp
__global__ void saxpy_kernel(int alpha, int* d_A, int* d_B, int* d_C, int N) {
    // TODO: Following Problem 1, implement C[i] = alpha * A[i] + B[i]
}
```

Note that the scalar `alpha` is passed as a regular parameter. CUDA automatically copies scalar values to each thread, so every thread gets its own copy of `alpha`.

### Part 2: Complete Memory Management

Now implement the host-side code:

```cpp
void saxpy_host(int alpha, const std::vector<int> &h_A, const std::vector<int> &h_B, std::vector<int> &h_C) {
    // Declare device pointers
    int* d_A = nullptr;
    int* d_B = nullptr;
    int* d_C = nullptr;
    
    // Calculate byte size
    size_t bytes = N * sizeof(int);
    
    // TODO: Allocate GPU memory for all three arrays
    
    // TODO: Copy input data from host to device
    
    // TODO: Launch kernel
    
    // TODO: Copy results back to host
    
    // TODO: Free all GPU memory
}
```

## Standard CUDA Workflow

Every CUDA program follows the same five-step pattern, and you'll implement each step:

**Step 1: Allocate GPU Memory**:
You need to allocate space on the GPU for all arrays. Memory allocation can fail if the GPU runs out of memory, and we'll discuss how to handle errors shortly.

**Step 2: Copy Input Data to GPU**:
Transfer the input arrays (A and B) from CPU memory to the GPU memory you just allocated. Note that you don't need to copy the output array C.

**Step 3: Launch Kernel**:
Execute the parallel computation. Here you choose how many threads per block and how many blocks to use.

**Step 4: Copy Results Back**:
After the kernel completes, copy the result array (C) from GPU memory back to CPU memory so your program can use it.

**Step 5: Free GPU Memory**:
Clean up all the GPU memory you allocated.

## Error Handling

CUDA operations can fail in various ways: insufficient GPU memory, invalid kernel launch configuration, or kernel crashes during execution.

```cpp
saxpy_kernel<<<num_blocks, threads_per_block>>>(alpha, d_A, d_B, d_C, N);
// Check for kernel launch errors
cudaError_t err = cudaGetLastError();
if (err != cudaSuccess) {
    printf("Kernel launch failed: %s\n", cudaGetErrorString(err));
}

// Wait for kernel completion and check execution errors
if (cudaDeviceSynchronize() != cudaSuccess) {
    printf("Kernel execution failed: %s\n", cudaGetErrorString(err));
}
```

The `cudaDeviceSynchronize()` function waits for all pending GPU operations to complete. Without this, your CPU code might try to copy results before the GPU has finished computing them.

We can write a macro to simplify error checking:

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

When using it, wrap CUDA functions with the `CUDA_CHECK` macro:

```cpp
CUDA_CHECK(cudaMalloc((void**)&d_A, bytes));
CUDA_CHECK(cudaMemcpy(d_A, h_A, bytes, cudaMemcpyHostToDevice));

// ...

saxpy_kernel<<<num_blocks, threads_per_block>>>(alpha, d_A, d_B, d_C, N);
CUDA_CHECK(cudaGetLastError());
CUDA_CHECK(cudaDeviceSynchronize());

// ...
```

## Performance Considerations

Memory transfers between CPU and GPU are more expensive than the actual computation. For vector addition and SAXPY, your program spends most of its time copying data rather than computing results.

In real applications, you should try to keep data on the GPU and perform multiple operations before copying results back. This amortizes the transfer cost across many computations, improving overall performance.

## Summary

After completing this problem, you now understand the complete foundation of CUDA programming. You can allocate GPU memory, safely transfer data, launch kernels with proper error checking, and correctly clean up resources.

This five-step workflow pattern appears in every CUDA application, from simple computations to training large-scale neural networks. The computational kernels become more complex, but the basic flow remains the same.

In Problem 3, we'll introduce shared memory - a powerful optimization that can significantly improve performance by enabling efficient collaboration between threads.