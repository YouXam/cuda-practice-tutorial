#include <iostream>
#include <string>
#include <vector>

#include "common.cuh"

__global__ void saxpy_kernel(int alpha, int *d_A, int *d_B, int *d_C, int N) {
    // STUDENT_START
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_C[idx] = alpha * d_A[idx] + d_B[idx];
    }
    // STUDENT_END
}

void saxpy_host(int alpha, const std::vector<int> &h_A, const std::vector<int> &h_B, std::vector<int> &h_C) {
    int N = h_A.size();
    h_C.resize(N);
    if (N == 0) return;

    int *d_A = nullptr, *d_B = nullptr, *d_C = nullptr;

    // STUDENT_START
    size_t byteSize = (size_t)N * sizeof(int);

    CUDA_CHECK(cudaMalloc((void **)&d_A, byteSize));
    CUDA_CHECK(cudaMalloc((void **)&d_B, byteSize));
    CUDA_CHECK(cudaMalloc((void **)&d_C, byteSize));

    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), byteSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_B, h_B.data(), byteSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;

    saxpy_kernel<<<blocksPerGrid, threadsPerBlock>>>(alpha, d_A, d_B, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, byteSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_B));
    CUDA_CHECK(cudaFree(d_C));
    // STUDENT_END
}

int main(int argc, char *argv[]) {
    if (argc != 5) {
        std::cerr << "Usage: " << argv[0] << " <alpha_file> <input_A_file> <input_B_file> <output_C_file>" << std::endl;
        return 1;
    }
    std::string alpha_path = argv[1];
    std::string inputA_path = argv[2];
    std::string inputB_path = argv[3];
    std::string outputC_path = argv[4];

    bool success_alpha, success_A, success_B;
    int alpha = readScalarFromFile(alpha_path, success_alpha);
    std::vector<int> h_A = readVectorFromFile(inputA_path, success_A);
    std::vector<int> h_B = readVectorFromFile(inputB_path, success_B);
    std::vector<int> h_C;

    if (!success_alpha || !success_A || !success_B) {
        writeVectorToFile({}, outputC_path);
        return 1;
    }
    if (h_A.size() != h_B.size() && !(h_A.empty() && h_B.empty())) {
        std::cerr << "Input vectors A and B must have the same size or both be empty." << std::endl;
        writeVectorToFile({}, outputC_path);
        return 1;
    }
    if (h_A.empty() && h_B.empty()) {
        writeVectorToFile({}, outputC_path);
        return 0;
    }

    saxpy_host(alpha, h_A, h_B, h_C);
    writeVectorToFile(h_C, outputC_path);

    return 0;
}
