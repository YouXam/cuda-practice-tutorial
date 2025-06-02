#include <iostream>
#include <cuda_runtime.h>
#include <string>
#include <vector>

#include "common.cuh"

__global__ void vectorAdd_kernel(int *d_A, int *d_B, int *d_C, int N) {
    // STUDENT_START
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < N) {
        d_C[idx] = d_A[idx] + d_B[idx];
    }
    // STUDENT_END
}

void vectorAdd_host(const std::vector<int> &h_A, const std::vector<int> &h_B, std::vector<int> &h_C) {
    int N = h_A.size();
    h_C.resize(N);
    if (N == 0) return;

    int *d_A, *d_B, *d_C;
    size_t byteSize = (size_t)N * sizeof(int);

    cudaMalloc((void **)&d_A, byteSize);
    cudaMalloc((void **)&d_B, byteSize);
    cudaMalloc((void **)&d_C, byteSize);
    
    cudaMemcpy(d_A, h_A.data(), byteSize, cudaMemcpyHostToDevice);
    cudaMemcpy(d_B, h_B.data(), byteSize, cudaMemcpyHostToDevice);
    
    // STUDENT_START
    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    vectorAdd_kernel<<<blocksPerGrid, threadsPerBlock>>>(d_A, d_B, d_C, N);
    // STUDENT_END

    cudaDeviceSynchronize();

    cudaMemcpy(h_C.data(), d_C, byteSize, cudaMemcpyDeviceToHost);

    cudaFree(d_A);
    cudaFree(d_B);
    cudaFree(d_C);
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0] << " <input_A_file> <input_B_file> <output_C_file>" << std::endl;
        return 1;
    }

    std::string inputA_path = argv[1];
    std::string inputB_path = argv[2];
    std::string outputC_path = argv[3];

    bool successA, successB;
    std::vector<int> h_A = readVectorFromFile(inputA_path, successA);
    std::vector<int> h_B = readVectorFromFile(inputB_path, successB);
    std::vector<int> h_C;

    if (!successA || !successB) {
        writeVectorToFile({}, outputC_path);
        return 1;
    }
    if (h_A.size() != h_B.size() && !(h_A.empty() && h_B.empty())) {
        std::cerr << "Input vectors must have the same size or both be empty." << std::endl;
        writeVectorToFile({}, outputC_path);
        return 1;
    }
    if (h_A.empty() && h_B.empty()) {
        writeVectorToFile({}, outputC_path);
        return 0;
    }

    vectorAdd_host(h_A, h_B, h_C);
    writeVectorToFile(h_C, outputC_path);

    return 0;
}
