#include <iostream>
#include <string>
#include <vector>

#include "common.cuh"

__global__ void stencil1D_kernel(const int *d_A, int *d_C, int N) {
    // STUDENT_START
    extern __shared__ int s_data[]; // Dynamically allocated shared memory

    const int radius = 1;
    int tid = threadIdx.x; // Thread ID within the block (0 to blockDim.x - 1)
    int gid = blockIdx.x * blockDim.x + tid; // Global thread ID

    // Step 2: Collaborate data loading from global memory to shared memory
    // Each thread loads its corresponding element d_A[gid] into s_data[tid + radius]
    // This covers the "main data region" in shared memory: s_data[radius] to s_data[blockDim.x + radius - 1]
    
    // Current global index for this thread to load into its main shared memory slot
    int current_g_idx_for_main_load = blockIdx.x * blockDim.x + tid;

    if (current_g_idx_for_main_load < N) {
        s_data[tid + radius] = d_A[current_g_idx_for_main_load];
    } else {
        // This thread is a "padding" thread if the block overshoots N.
        // It won't compute an output for d_C.
        // However, its s_data[tid+radius] slot might be needed as a right halo
        // for an earlier active thread in the same block.
        // Load the last valid element from d_A to avoid out-of-bounds access
        // and provide correct "edge clamping" for computations relying on this halo.
        if (N > 0) { // Ensure N is not 0 to prevent d_A[-1] access
          s_data[tid + radius] = d_A[N - 1];
        }
    }

    // Thread 0 of the block loads the left halo element
    // s_data[0] needs d_A[block_start_gid - radius]
    if (tid == 0) {
        int left_halo_global_idx = blockIdx.x * blockDim.x - radius;
        if (left_halo_global_idx < 0) {
            // Edge clamping: A[-1] (conceptual) is treated as A[0]
            if (N > 0) { // Ensure N is not 0
                s_data[0] = d_A[0];
            }
        } else {
            // This check is technically not needed if N > 0 due to previous check,
            // but good for clarity or if N could be small relative to radius
            if (left_halo_global_idx < N) {
                 s_data[0] = d_A[left_halo_global_idx];
            } else if (N > 0) { // Should not happen if left_halo_global_idx >=0 and < N logic is sound
                 s_data[0] = d_A[N-1];
            }
        }
    }

    // The last thread of the block (tid == blockDim.x - 1) loads the right halo element
    // s_data[blockDim.x + 2*radius - 1] (which is s_data[blockDim.x + 1] for radius=1)
    // needs d_A[gid_of_last_thread_in_block + radius]
    if (tid == blockDim.x - 1) {
        int right_halo_global_idx = (blockIdx.x * blockDim.x + (blockDim.x - 1)) + radius;
        // Shared memory index for the rightmost halo element: blockDim.x + 2*radius - 1
        int right_halo_sm_idx = blockDim.x + 2*radius - 1; 
        if (right_halo_global_idx >= N) {
            // Edge clamping: A[N] (conceptual) is treated as A[N-1]
            if (N > 0) { // Ensure N is not 0
                s_data[right_halo_sm_idx] = d_A[N - 1];
            }
        } else {
             if (right_halo_global_idx >= 0) { // Ensure it's not negative (shouldn't be for right halo)
                s_data[right_halo_sm_idx] = d_A[right_halo_global_idx];
             } else if (N > 0) { // Should not happen
                s_data[right_halo_sm_idx] = d_A[0];
             }
        }
    }

    // Step 3: Synchronize to ensure all data is loaded into shared memory
    __syncthreads();

    // Step 4: Compute using data from shared memory
    if (gid < N) { // Ensure we only compute and write for valid output elements
        // Access pattern for radius = 1:
        // s_data[tid + radius - 1]  (left element for current thread's computation)
        // s_data[tid + radius]      (center element)
        // s_data[tid + radius + 1]  (right element)
        
        int left_val   = s_data[tid + radius - 1];
        int center_val = s_data[tid + radius];
        int right_val  = s_data[tid + radius + 1];
        
        d_C[gid] = (left_val + center_val + right_val) / 3;
    }
    // STUDENT_END
}

void stencil1D_host(const std::vector<int> &h_A, std::vector<int> &h_C) {
    int N = h_A.size();
    h_C.resize(N);
    if (N == 0) return;

    int *d_A, *d_C;

    // STUDENT_START
    size_t byteSize = (size_t)N * sizeof(int);

    CUDA_CHECK(cudaMalloc((void **)&d_A, byteSize));
    CUDA_CHECK(cudaMalloc((void **)&d_C, byteSize));
    CUDA_CHECK(cudaMemcpy(d_A, h_A.data(), byteSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int blocksPerGrid = (N + threadsPerBlock - 1) / threadsPerBlock;
    int radius = 1;
    size_t sharedMemSize = (threadsPerBlock + 2 * radius) * sizeof(int);

    stencil1D_kernel<<<blocksPerGrid, threadsPerBlock, sharedMemSize>>>(d_A, d_C, N);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    CUDA_CHECK(cudaMemcpy(h_C.data(), d_C, byteSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaFree(d_A));
    CUDA_CHECK(cudaFree(d_C));
    // STUDENT_END
}

int main(int argc, char *argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_A_file> <output_C_file>" << std::endl;
        return 1;
    }
    std::string inputA_path = argv[1];
    std::string outputC_path = argv[2];

    bool success_A;
    std::vector<int> h_A = readVectorFromFile(inputA_path, success_A);
    std::vector<int> h_C;

    if (!success_A) {
        writeVectorToFile({}, outputC_path);
        return 1;
    }
    if (h_A.empty()) {
        writeVectorToFile({}, outputC_path);
        return 0;
    }

    stencil1D_host(h_A, h_C);
    writeVectorToFile(h_C, outputC_path);
    return 0;
}
