# Problem 4: Matrix Transpose

In modern computing, two-dimensional parallelism is ubiquitous: every pixel operation in image processing, every transformation in computer graphics, and every finite difference calculation in scientific simulation relies on two-dimensional parallel processing.

CUDA natively supports a two-dimensional thread model, allowing you to organize threads according to the inherent two-dimensional structure of your problem, rather than forcing a two-dimensional problem into a one-dimensional arrangement.

## 🎯 Learning Objectives

- Understand the organization and indexing of two-dimensional threads
- Master configuration strategies for two-dimensional grids and thread blocks
- Grasp row-major memory layout of matrices and their access patterns
- Appreciate the fundamental principles of coordinate transformations in graphics and linear algebra

## CUDA's Two-Dimensional Thread Model

CUDA's two-dimensional thread model allows you to organize threads into two-dimensional grids and blocks, making the processing of two-dimensional data (such as matrices) more intuitive. Specifically, each grid consists of multiple blocks, and each block contains multiple threads, all organized in a two-dimensional fashion.

```
Grid:               Block B00:          Thread T00:
┌─────┬─────┐       ┌───┬───┬───┐       ┌─────────┐
│ B00 │ B01 │       │T00│T01│T02│       │ Thread  │
├─────┼─────┤       ├───┼───┼───┤       └─────────┘
│ B10 │ B11 │       │T10│T11│T12│
└─────┴─────┘       └───┴───┴───┘
```

You can locate each thread using the following built-in variables:

* `blockIdx.x/y` represents the index of the block containing the current thread
* `threadIdx.x/y` represents the index of the current thread within its block

By combining these coordinates with dimension information, you can calculate each thread's row and column indices in the global matrix.

## Matrix Transpose

Matrix transpose flips a matrix along its diagonal, converting rows into columns.

```
Original Matrix A (3×4):        Transposed Matrix B (4×3):
┌─────────────────┐             ┌─────────────┐
│  1   2   3   4  │             │  1   5   9  │
│  5   6   7   8  │   →         │  2   6  10  │
│  9  10  11  12  │             │  3   7  11  │
└─────────────────┘             │  4   8  12  │
                                └─────────────┘
```

While conceptually straightforward, it helps you deeply understand two-dimensional memory access, coordinate mapping, and layout transformations.

## Row-Major Memory Layout

Matrices in computer memory are stored as 1D arrays using row-major order. Understanding this layout is crucial for efficient GPU programming:

```
Matrix A[3][4]:      Memory Layout:
┌─ 1  2  3  4 ─┐    [1][2][3][4][5][6][7][8][9][10][11][12]
├─ 5  6  7  8 ─┤    ↑   Row 0    ↑   Row 1    ↑   Row 2    ↑
└─ 9 10 11 12 ─┘
```

The one-dimensional index for element `A[row][col]`: row * num_cols + col

During transposition, element A[i][j] at index `i * K + j` is written to position `j * M + i`.

## Two-Dimensional Thread Indexing

Host-side code uses `dim3` to configure two-dimensional grids and thread blocks, with two parameters representing the sizes of the first and second dimensions.

We use `n` for the number of matrix rows and `m` for the number of matrix columns.

```cpp
void transpose_matrix_host(const std::vector<int> &h_A, std::vector<int> &h_B, int n, int m) {
    // Memory allocation and copying...
    
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    
    // Calculate 2D grid size to cover the entire matrix
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);
    
    transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, n, m);
    
    // Copy back and cleanup...
}
```

You may notice that when setting `dim3` dimensions, the first dimension represents the column direction size, and the second dimension represents the row direction size. This is because CUDA's thread indexing dimensions are arranged from smallest to largest, and at the memory level, elements in the first dimension are contiguous.

When writing kernel functions, remember that `blockIdx.x/threadIdx.x` (first dimension) handles the column direction, while `blockIdx.y/threadIdx.y` (second dimension) handles the row direction.

```cpp
__global__ void transpose_kernel(const int* d_A, int* d_B, int n, int m) {
    // Calculate coordinates
    int col_A = blockIdx.x * blockDim.x + threadIdx.x;
    int row_A = blockIdx.y * blockDim.y + threadIdx.y;
    
    // Check input matrix boundaries
    if (row_A < n && col_A < m) {
        // Each thread processes one matrix element
    }
}
```

## Performance Optimization

Global memory has two access patterns: coalesced and uncoalesced. In a single data transfer, the starting address of a memory chunk transferred from global memory to L2 cache must be a multiple of the minimum granularity. If threads in the same thread block access consecutive memory addresses, they can directly utilize cached memory from other accesses.

In transpose operations, reading matrix A is sequential. If the first thread accesses A[0][0], the memory read might load A[0][0] through A[0][7] into cache simultaneously, allowing threads 2-8 to read directly from cache, achieving coalesced memory access. However, writing to matrix B is more scattered, as each thread writes to different rows with addresses far apart, causing uncoalesced access.

To optimize performance, shared memory can be used to improve memory access patterns. The basic idea is for each thread block to first load a sub-matrix into shared memory, then transpose and write back to global memory from shared memory, ensuring threads in the same block access consecutive memory addresses when accessing global memory.

Specific implementation can be found in `answer.cu`.

## Summary

You have now mastered the fundamentals of two-dimensional CUDA programming: configuring two-dimensional grids and thread blocks, computing global two-dimensional coordinates, performing boundary checks, and executing coordinate transformations. In the next problem, we will explore Parallel Reduction, investigating how to efficiently aggregate results in CUDA.