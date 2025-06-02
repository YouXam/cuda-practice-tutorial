# CUDA Practice Tutorial

English | [简体中文](./README.zh.md)

This is a practical CUDA programming tutorial designed to help readers master the basic concepts and common operations of CUDA parallel computing through hands-on exercises. The content covers fundamental operations such as vector addition, matrix operations, convolution, and parallel reduction, deepening the understanding of GPU parallel acceleration through practice.

## Environment Setup

First, ensure your computer/server has an available Nvidia GPU, then download and install the CUDA Toolkit and the corresponding driver from the [Nvidia official website](https://developer.nvidia.com/cuda-downloads). For installation instructions, refer to the [CUDA Quick Start Guide](https://docs.nvidia.com/cuda/cuda-quick-start-guide/index.html).

Clone this repository:

```bash
git clone https://github.com/youxam/cuda-practice-tutorial.git
cd cuda-practice-tutorial
```

Generate a working directory:

```bash
python3 generate.py [path]
```

For example:

```bash
python3 generate.py ~/cuda-practice-projects
```

The generated directory contains 9 exercises, each with a corresponding `README.md` that includes a complete tutorial, problem description, and explanations. You can also read them online.

1. [Problem 1: Vector Addition](./problems/en/problem01_vector_add.md)
2. [Problem 2: SAXPY](./problems/en/problem02_saxpy.md)
3. [Problem 3: 1D Stencil](./problems/en/problem03_1d_stencil.md)
4. [Problem 4: Matrix Transposition](./problems/en/problem04_matrix_transpose.md)
5. [Problem 5: Parallel Reduction Sum](./problems/en/problem05_parallel_reduction_sum.md)
6. [Problem 6: 2D Convolution](./problems/en/problem06_2d_convolution.md)
7. [Problem 7: Tiled Matrix Multiplication](./problems/en/problem07_matrix_mul_tiled.md)
8. [Problem 8: Histogram](./problems/en/problem08_histogram.md)
9. [Problem 9: K-means Clustering](./problems/en/problem09_kmeans.md)

You should complete each problem by following these steps:

1. Read the problem description and requirements to understand the functionality you need to implement.
2. Based on the requirements, implement the `// TODO` sections of the `student.cu` file.
3. Use `make list` to view the list of test cases. Use `make run TC=<test_case_prefix>` to compile and run a specific test case. Use `make test` to compile and run all test cases.
4. If you encounter difficulties, refer to the `answer.cu` file for a reference implementation to understand the approach and how it is implemented.

If you need to configure LSP, you can refer to the `.clangd` file in this project.

## Learning Resources

* [CUDA C Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
* *CUDA Programming Basics and Practice* by Zheyong Fan (Chinese)