#include <iostream>
#include <string>
#include <vector>

#include "common.cuh"

__global__ void reduce_sum_kernel(const long long *d_A_global, long long *d_block_sums, int N) {
    // STUDENT_START
    extern __shared__ int s_data[];
    
    int tid = threadIdx.x;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    long long sum = 0;
    for (int i = idx; i < N; i += stride) {
        sum += d_A_global[i];
    }
    s_data[tid] = sum;
    __syncthreads();

    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_data[tid] += s_data[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        d_block_sums[blockIdx.x] = s_data[0];
    }
    // STUDENT_END
}

constexpr int BLOCK_SIZE = 16;
long long reduce_sum_host(const std::vector<long long> &h_A) {
    int N = h_A.size();
    if (N == 0) return 0LL;

    // STUDENT_START
    long long *d_A_global = nullptr;
    long long *d_block_sums = nullptr;
    long long *result = nullptr;
    size_t size_A = N * sizeof(long long);
    size_t size_block_sums = ((N + BLOCK_SIZE - 1) / BLOCK_SIZE) * sizeof(long long);
    size_t shared_memory_size = BLOCK_SIZE * sizeof(long long);
    int num_blocks = (N + BLOCK_SIZE - 1) / BLOCK_SIZE;
    CUDA_CHECK(cudaMalloc((void**)&d_A_global, size_A));
    CUDA_CHECK(cudaMalloc((void**)&d_block_sums, size_block_sums));
    CUDA_CHECK(cudaMallocManaged((void**)&result, sizeof(long long)));
    CUDA_CHECK(cudaMemcpy(d_A_global, h_A.data(), size_A, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_block_sums, 0, size_block_sums));
    CUDA_CHECK(cudaMemset(result, 0, sizeof(long long)));

    reduce_sum_kernel<<<num_blocks, BLOCK_SIZE, shared_memory_size>>>(d_A_global, d_block_sums, N);
    CUDA_CHECK(cudaGetLastError());
    reduce_sum_kernel<<<1, BLOCK_SIZE, shared_memory_size>>>(d_block_sums, result, num_blocks);
    CUDA_CHECK(cudaGetLastError());

    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaFree(d_A_global));
    CUDA_CHECK(cudaFree(d_block_sums));
    return *result;
    // STUDENT_END
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_A_file> <output_sum_file>" << std::endl;
        return 1;
    }
    std::string inputA_path = argv[1];
    std::string outputSum_path = argv[2];

    bool success_A;
    auto h_A = readVectorFromFile<long long>(inputA_path, success_A);

    if (!success_A) {
        writeScalarToFile(0, outputSum_path);
        return 1;
    }

    long long total_sum = reduce_sum_host(h_A);
    writeScalarToFile(total_sum, outputSum_path);
    return 0;
}