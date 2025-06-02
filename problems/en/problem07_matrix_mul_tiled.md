# Problem 7: Matrix Multiplication

In this problem, we will delve deep into high-performance implementations of matrix multiplication, particularly using tiling techniques on GPUs to optimize computation. This problem is not only a cornerstone of deep learning and scientific computing but also one of the core algorithms driving the modern artificial intelligence revolution.

## 🎯 Learning Objectives

- Understand the fundamental concepts and applications of matrix multiplication
- Master tiled matrix multiplication algorithms on GPUs

## Matrix Multiplication

Let's begin with the basic concepts of matrix multiplication.

Matrix multiplication is a fundamental operation in linear algebra that combines two matrices to produce a new matrix. Suppose we have two matrices $A$ and $B$, where $A$ has dimensions $M \times K$ and $B$ has dimensions $K \times N$. Their product $C$ will be an $M \times N$ matrix.

The computation rule for matrix multiplication is as follows: the element in the $i$-th row and $j$-th column of $C$ is the sum of products of corresponding elements from the $i$-th row of $A$ and the $j$-th column of $B$:

$$
C_{i,j} = \sum_{k=0}^{K-1} A_{i,k} \cdot B_{k,j}
$$

For example, consider a $2 \times 3$ matrix $A$ and a $3 \times 2$ matrix $B$:

$$
A = \begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6
\end{bmatrix}, \quad
B = \begin{bmatrix}
7 & 8 \\
9 & 10 \\
11 & 12
\end{bmatrix}
$$

When computing the element in the first row and first column of the result matrix $C$, we take the first row of $A$ $[1, 2, 3]$ and the first column of $B$ $[7, 9, 11]$, multiply corresponding elements and sum them: $1 \times 7 + 2 \times 9 + 3 \times 11 = 58$.

Continuing this process, we can compute all elements. The first row of $A$ multiplied with the second column of $B$ gives 64, the second row of $A$ multiplied with the first column of $B$ gives 139, and so forth. Finally, we obtain a $2 \times 2$ result matrix $C$:

$$
C = \begin{bmatrix}
58 & 64 \\
139 & 154
\end{bmatrix}
$$

This seemingly simple operation harbors tremendous practical value. Every 3D object you see in games undergoes matrix transformations to determine its position and orientation on screen. When ChatGPT generates a response, it performs millions of such matrix operations internally to understand your question and organize language. Weather forecasting, drug discovery, financial risk analysis—these seemingly unrelated fields all rely heavily on matrix multiplication.

## Matrix Multiplication Tiling

When processing two $1024 \times 1024$ matrices, the number of required operations is $1024 \times 1024 \times 1024$, which exceeds one billion multiplication operations. If we use the most naive approach to write code, it would look like this:

```cpp
for (int i = 0; i < M; i++) {           // Iterate through each row of A
    for (int j = 0; j < N; j++) {       // Iterate through each column of B
        float sum = 0;
        for (int k = 0; k < K; k++) {   // Compute dot product
            sum += A[i][k] * B[k][j];
        }
        C[i][j] = sum;
    }
}
```

While this algorithm appears clear, it has a fatal flaw: extremely poor memory access efficiency. When computing each column of the result matrix $C$, we must re-read all rows of matrix $A$. This means every element of $A$ must be read from memory $N$ times. Similarly, when computing each row of $C$, we must re-read all columns of matrix $B$, so every element of $B$ is also read $M$ times.

Such repeated reading is enormously wasteful. In modern computers, the speed of reading data from memory is far slower than the processor's computation speed. Although GPUs have powerful computational capabilities, if most time is spent waiting for memory data, these computational resources are wasted.

The key to solving this problem lies in matrix tiling. Instead of processing element by element, we divide large matrices into blocks of smaller matrices and process pairs of small blocks at a time. The advantage is that we can load these small blocks into the GPU's shared memory and repeatedly use them for computation, avoiding repeated global memory accesses.

Let's understand how tiling works with a concrete example. Suppose we want to compute the multiplication of two $4 \times 4$ matrices. We can divide each into four $2 \times 2$ blocks:

```
Matrix A:                    Matrix B:
┌─────────┬─────────┐      ┌─────────┬─────────┐
│   A00   │   A01   │      │   B00   │   B01   │
│  (2×2)  │  (2×2)  │      │  (2×2)  │  (2×2)  │
├─────────┼─────────┤  ×   ├─────────┼─────────┤
│   A10   │   A11   │      │   B10   │   B11   │
│  (2×2)  │  (2×2)  │      │  (2×2)  │  (2×2)  │
└─────────┴─────────┘      └─────────┴─────────┘
```

We can treat these blocks like individual numbers. The top-left block $C_{0,0}$ of the result matrix $C$ equals $A_{0,0} \times B_{0,0}$ plus $A_{0,1} \times B_{1,0}$.

For instance,

$$A=\begin{bmatrix} 1 & 2 & 5 & 6 \\ 3 & 4 & 7 & 8 \\ 9 & 10 & 13 & 14 \\ 11 & 12 & 15 & 16 \end{bmatrix}, \quad B=\begin{bmatrix} 1 & 0 & 1 & 1 \\ 0 & 1 & 0 & 0 \\ 1 & 1 & 0 & 1 \\ 0 & 0 & 1 & 0 \end{bmatrix}$$

Then

$$C = A \times B = \begin{bmatrix} 6 & 7 & 7 & 6 \\ 10 & 11 & 11 & 10 \\ 22 & 23 & 23 & 22 \\ 26 & 27 & 27 & 26 \end{bmatrix} $$

Now let's compute $C_{0,0}$ using the tiling approach:

$A_{00} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$, $B_{00} = \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix}$, so $$A_{00} \times B_{00} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} \times \begin{bmatrix} 1 & 0 \\ 0 & 1 \end{bmatrix} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix}$$

$A_{01} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix}$, $B_{10} = \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix}$, so $$A_{01} \times B_{10} = \begin{bmatrix} 5 & 6 \\ 7 & 8 \end{bmatrix} \times \begin{bmatrix} 1 & 1 \\ 0 & 0 \end{bmatrix} = \begin{bmatrix} 5 & 5 \\ 7 & 7 \end{bmatrix}$$

Finally:

$$C_{00} = A_{00} \times B_{00} + A_{01} \times B_{10} = \begin{bmatrix} 1 & 2 \\ 3 & 4 \end{bmatrix} + \begin{bmatrix} 5 & 5 \\ 7 & 7 \end{bmatrix} = \begin{bmatrix} 6 & 7 \\ 10 & 11 \end{bmatrix}$$

This is not a coincidence but a beautiful property of matrix multiplication: it maintains consistency at any level.

The tiling algorithm doesn't directly reduce the time complexity of the algorithm, but it provides better cache locality. By dividing matrices into blocks, multiplication can process small matrix blocks locally, significantly improving cache hit rates and reducing memory access latency.

Using $16 \times 16$ blocks as an example, when we load a block of $A$ into shared memory, each element in this block will be used 16 times—corresponding to the 16 columns of the $B$ block, each column requiring this element from the $A$ block. Similarly, each element in the $B$ block will also be used 16 times, corresponding to the 16 rows of the $A$ block. This means we exchange one memory access for 16 computations. This 16-fold reuse factor greatly improves computational efficiency.

More importantly, this reuse pattern transforms an algorithm originally limited by memory bandwidth into one primarily limited by computational capacity. Combined with the tiling algorithm's friendliness to parallelization, we can fully leverage GPU's massive parallel computing advantages.

Now, please implement the tiled matrix multiplication algorithm in `student.cu`. You need to use CUDA's shared memory to store matrix blocks and compute the corresponding results in each thread block.

## Summary

Through this problem, you have not only learned the specific algorithm of matrix multiplication but also gained deeper understanding of how to analyze algorithm bottlenecks, design memory-friendly data access patterns, and leverage GPU architectural features to maximize performance.

In the upcoming Problem 8, we will explore a completely different but equally important parallel computing challenge: atomic operations and histogram computation. There, you will learn how to elegantly handle race conditions that arise when multiple threads need to simultaneously update the same memory location.