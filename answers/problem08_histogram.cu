#include <iostream>
#include <string>
#include <vector>

#include "common.cuh"

#define NUM_BINS 256

__global__ void histogram_kernel(const int *d_data, int *d_histogram, int N) {
    // STUDENT_START
    __shared__ int s_hist[NUM_BINS];

    // Initialize shared memory histogram
    // Each thread initializes a portion of s_hist
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        s_hist[i] = 0;
    }
    __syncthreads();

    // Each thread processes a stride of the input data
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int stride = gridDim.x * blockDim.x;

    for (int i = idx; i < N; i += stride) {
        int value = d_data[i];
        if (value >= 0 && value < NUM_BINS) {
            atomicAdd(&s_hist[value], 1);
        }
    }
    __syncthreads();

    // Each thread helps write a portion of the shared histogram to global memory
    for (int i = threadIdx.x; i < NUM_BINS; i += blockDim.x) {
        if (s_hist[i] > 0) {  // Optimization: only add if there's something to add
            atomicAdd(&d_histogram[i], s_hist[i]);
        }
    }
    // STUDENT_END
}

void histogram_host(const std::vector<int> &h_data, std::vector<int> &h_histogram) {
    int N = h_data.size();
    h_histogram.assign(NUM_BINS, 0);

    if (N == 0) return;

    int *d_data = nullptr;
    int *d_histogram = nullptr;

    // STUDENT_START
    size_t byteSize_data = (size_t)N * sizeof(int);
    size_t byteSize_hist = (size_t)NUM_BINS * sizeof(int);

    CUDA_CHECK(cudaMalloc((void **)&d_data, byteSize_data));
    CUDA_CHECK(cudaMalloc((void **)&d_histogram, byteSize_hist));

    CUDA_CHECK(cudaMemcpy(d_data, h_data.data(), byteSize_data, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemset(d_histogram, 0, byteSize_hist));  // Initialize global histogram to 0

    int threadsPerBlock = 256;
    // Number of blocks can be tuned. More blocks can help with occupancy if N is large.
    // Fewer blocks if N is small to avoid too little work per block.
    // Let's choose a moderate number of blocks, e.g., 64, or enough to cover N.
    int numBlocks = std::min(64, (N + threadsPerBlock - 1) / threadsPerBlock);
    if (numBlocks == 0 && N > 0) numBlocks = 1;

    if (numBlocks > 0) {  // Only launch kernel if there's work
        histogram_kernel<<<numBlocks, threadsPerBlock>>>(d_data, d_histogram, N);
        CUDA_CHECK(cudaGetLastError());
        CUDA_CHECK(cudaDeviceSynchronize());
    }

    CUDA_CHECK(cudaMemcpy(h_histogram.data(), d_histogram, byteSize_hist, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_data));
    CUDA_CHECK(cudaFree(d_histogram));
    // STUDENT_END
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_data_file> <output_histogram_file>" << std::endl;
        return 1;
    }
    std::string dataFile = argv[1];
    std::string histFile = argv[2];

    bool success;
    auto h_data = readVectorFromFile(dataFile, success);
    std::vector<int> h_histogram(NUM_BINS);

    if (!success) {
        writeVectorToFile({}, histFile);
        return 1;
    }

    histogram_host(h_data, h_histogram);
    writeVectorToFile(h_histogram, histFile);

    return 0;
}
