#include <iostream>
#include <string>
#include <vector>

#include "common.cuh"

#define TILE_DIM 16

__global__ void matrixMulTiled_kernel(const float *d_A, const float *d_B, float *d_C, int M, int K, int N) {
    // STUDENT_START
    __shared__ float s_A[TILE_DIM][TILE_DIM];
    __shared__ float s_B[TILE_DIM][TILE_DIM];

    int tx = threadIdx.x;
    int ty = threadIdx.y;

    // Global row and column of C this thread's block is computing
    int block_row_C = blockIdx.y;
    int block_col_C = blockIdx.x;

    // Global row and column of C this thread is responsible for (within its tile)
    int row_C_global = block_row_C * TILE_DIM + ty;
    int col_C_global = block_col_C * TILE_DIM + tx;

    float sum = 0.0f;

    // Loop over the tiles of A and B required to compute the C tile
    for (int t = 0; t < (K + TILE_DIM - 1) / TILE_DIM; ++t) {
        // Load tile of A into s_A
        // s_A[ty][tx] corresponds to A[row_C_global][t*TILE_DIM + tx]
        int row_A_load = row_C_global;
        int col_A_load = t * TILE_DIM + tx;
        if (row_A_load < M && col_A_load < K) {
            s_A[ty][tx] = d_A[row_A_load * K + col_A_load];
        } else {
            s_A[ty][tx] = 0.0f;
        }

        // Load tile of B into s_B
        // s_B[ty][tx] corresponds to B[t*TILE_DIM + ty][col_C_global]
        int row_B_load = t * TILE_DIM + ty;
        int col_B_load = col_C_global;
        if (row_B_load < K && col_B_load < N) {
            s_B[ty][tx] = d_B[row_B_load * N + col_B_load];
        } else {
            s_B[ty][tx] = 0.0f;
        }
        __syncthreads();

        // Multiply tiles from shared memory
        for (int i = 0; i < TILE_DIM; ++i) {
            sum += s_A[ty][i] * s_B[i][tx];
        }
        __syncthreads();  // Sync before loading next pair of tiles
    }

    // Write the result to d_C
    if (row_C_global < M && col_C_global < N) {
        d_C[row_C_global * N + col_C_global] = sum;
    }
    // STUDENT_END
}

void matrixMulTiled_host(const std::vector<float> &h_A, const std::vector<float> &h_B, std::vector<float> &h_C, int M, int K, int N) {
    if (M == 0 || N == 0) {  // If output C has 0 rows or 0 cols
        h_C.clear();
        return;
    }
    if (K == 0 && M > 0 && N > 0) {  // A is M*0, B is 0*N, result C is M*N of zeros
        h_C.assign((size_t)M * N, 0.0f);
        return;
    }
    if (M * K == 0 || K * N == 0) {  // One of input matrices is empty due to M or N being 0, but K might not be.
        // This case should be covered by M==0 or N==0 for C. If K=0, handled above.
        // If M=0, A is empty, C is empty. If N=0, B is empty, C is empty.
        h_C.clear();  // Should be covered by M==0 or N==0 for C.
        return;
    }

    h_C.assign((size_t)M * N, 0.0f);

    float *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // STUDENT_START
    size_t byteSize_A = (size_t)M * K * sizeof(float);
    size_t byteSize_B = (size_t)K * N * sizeof(float);
    size_t byteSize_C = (size_t)M * N * sizeof(float);

    CUDA_CHECK(cudaMalloc((void **)&d_A, byteSize_A));
    CUDA_CHECK(cudaMalloc((void **)&d_B, byteSize_B));
    CUDA_CHECK(cudaMalloc((void **)&d_C, byteSize_C));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), byteSize_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), byteSize_B, cudaMemcpyHostToDevice));
    // d_C is initialized by kernel, or could be cudaMemset to 0 if sum starts from d_C value.

    dim3 threadsPerBlock(TILE_DIM, TILE_DIM);
    dim3 numBlocks((N + TILE_DIM - 1) / TILE_DIM, (M + TILE_DIM - 1) / TILE_DIM);

    matrixMulTiled_kernel<<<numBlocks, threadsPerBlock>>>(d_A, d_B, d_C, M, K, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, byteSize_C, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    // STUDENT_END
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_A_file> <input_B_file> <output_C_file>" << std::endl;
        return 1;
    }
    std::string fileA = argv[1];
    std::string fileB = argv[2];
    std::string fileC = argv[3];

    int M, K_A, K_B, N;
    bool successA, successB;
    auto h_A = readMatrixFromFile<float>(fileA, M, K_A, successA);
    auto h_B = readMatrixFromFile<float>(fileB, K_B, N, successB);
    std::vector<float> h_C;

    if (!successA || !successB) {
        writeMatrixToFile({}, 0, 0, fileC);
        return 1;
    }
    if (K_A != K_B && !(M * K_A == 0 || K_B * N == 0)) {
        if (!((M == 0 || K_A == 0) || (K_B == 0 || N == 0))) {  // Only error if both matrices are non-empty in relevant dim
            std::cerr << "Matrix dimensions mismatch: A_cols (" << K_A << ") != B_rows (" << K_B << ")" << std::endl;
            writeMatrixToFile({}, 0, 0, fileC);
            return 1;
        }
    }
    int K = K_A;

    matrixMulTiled_host(h_A, h_B, h_C, M, K, N);
    writeMatrixToFile(h_C, M, N, fileC);

    return 0;
}
