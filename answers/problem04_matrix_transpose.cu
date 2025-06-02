#include <iostream>
#include <string>
#include <vector>

#include "common.cuh"

constexpr int TILE_SIZE = 16;

__global__ void transpose_kernel(const int *d_A, int *d_B, int m, int n) {
    // STUDENT_START
    __shared__ int tile[TILE_SIZE][TILE_SIZE + 1]; // +1 to avoid bank conflicts

    int x = blockIdx.x * TILE_SIZE + threadIdx.x;
    int y = blockIdx.y * TILE_SIZE + threadIdx.y;

    if (x < m && y < n) {
        tile[threadIdx.y][threadIdx.x] = d_A[y * m + x];
    }
    __syncthreads();

    x = blockIdx.y * TILE_SIZE + threadIdx.x;
    y = blockIdx.x * TILE_SIZE + threadIdx.y;
    
    if (x < n && y < m) {
        d_B[y * n + x] = tile[threadIdx.x][threadIdx.y];
    }
    // STUDENT_END
}

void transpose_matrix_host(const std::vector<int> &h_A, std::vector<int> &h_B, int n, int m) {
    if (n == 0 || m == 0) {
        h_B.clear();
        return;
    }
    h_B.resize((size_t)m * n);

    int *d_A, *d_B;
    size_t byteSize_A = (size_t)n * m * sizeof(int);
    size_t byteSize_B = (size_t)m * n * sizeof(int);

    CUDA_CHECK(cudaMalloc((void **)&d_A, byteSize_A));
    CUDA_CHECK(cudaMalloc((void **)&d_B, byteSize_B));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), byteSize_A, cudaMemcpyHostToDevice));

    // STUDENT_START
    dim3 threadsPerBlock(TILE_SIZE, TILE_SIZE);
    dim3 numBlocks((m + threadsPerBlock.x - 1) / threadsPerBlock.x,
                   (n + threadsPerBlock.y - 1) / threadsPerBlock.y);

    transpose_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, m, n);
    // STUDENT_END

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_B.data(), d_B, byteSize_B, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_A_file> <output_B_file>" << std::endl;
        return 1;
    }
    std::string inputA_path = argv[1];
    std::string outputB_path = argv[2];

    int n, m;
    bool success_A;
    std::vector<int> h_A = readMatrixFromFile(inputA_path, n, m, success_A);
    std::vector<int> h_B;

    if (!success_A) {
        writeMatrixToFile({}, 0, 0, outputB_path);
        return 1;
    }

    transpose_matrix_host(h_A, h_B, n, m);
    writeMatrixToFile(h_B, m, n, outputB_path);
    return 0;
}