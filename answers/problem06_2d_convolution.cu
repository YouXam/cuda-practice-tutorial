#include <iostream>
#include <string>
#include <vector>

#include "common.cuh"

constexpr int TILE_SIZE = 16;

__global__ void convolution2D_kernel(const float *__restrict__ d_inputImage,
                                     const float *__restrict__ d_kernel,
                                     float *d_outputImage, int height,
                                     int width, int kernelHeight,
                                     int kernelWidth) {
    // STUDENT_START
    const int radiusY = kernelHeight >> 1;
    const int radiusX = kernelWidth >> 1;

    const int sharedH = TILE_SIZE + 2 * radiusY;  // rows in shared tile
    const int sharedW = TILE_SIZE + 2 * radiusX;  // cols in shared tile

    extern __shared__ float s_data[];  // tile + halo

    // ---- Load tile (including halo) into shared memory ----
    for (int y = threadIdx.y; y < sharedH; y += blockDim.y) {
        const int globalY = blockIdx.y * TILE_SIZE + y - radiusY;
        for (int x = threadIdx.x; x < sharedW; x += blockDim.x) {
            const int globalX = blockIdx.x * TILE_SIZE + x - radiusX;
            if (globalY >= 0 && globalY < height && globalX >= 0 &&
                globalX < width)
                s_data[y * sharedW + x] =
                    d_inputImage[globalY * width + globalX];
            else
                s_data[y * sharedW + x] = 0.0f;  // zero‑pad out‑of‑range
        }
    }
    __syncthreads();

    // ---- Compute one output pixel per thread ----
    const int outY = blockIdx.y * TILE_SIZE + threadIdx.y;
    const int outX = blockIdx.x * TILE_SIZE + threadIdx.x;
    if (outY >= height || outX >= width) return;

    float sum = 0.0f;
    const int sY = threadIdx.y + radiusY;
    const int sX = threadIdx.x + radiusX;

    for (int ky = 0; ky < kernelHeight; ++ky) {
        const int sRow =
            (sY - radiusY + ky) * sharedW;  // shared mem row offset
        const int kRow = ky * kernelWidth;  // kernel row offset
        for (int kx = 0; kx < kernelWidth; ++kx) {
            sum += s_data[sRow + (sX - radiusX + kx)] * d_kernel[kRow + kx];
        }
    }
    d_outputImage[outY * width + outX] = sum;
    // STUDENT_END
}

void convolution2D_host(const std::vector<float> &h_inputImage,
                        const std::vector<float> &h_kernel,
                        std::vector<float> &h_outputImage, int height,
                        int width, int kernelHeight, int kernelWidth) {
    if (height == 0 || width == 0) {
        h_outputImage.clear();
        return;
    }
    h_outputImage.resize((size_t)height * width);

    // STUDENT_START
    const size_t imgBytes = static_cast<size_t>(height) * width * sizeof(float);
    const size_t kerBytes =
        static_cast<size_t>(kernelHeight) * kernelWidth * sizeof(float);

    float *d_inputImage = nullptr, *d_kernel = nullptr,
          *d_outputImage = nullptr;
    CUDA_CHECK(cudaMalloc(&d_inputImage, imgBytes));
    CUDA_CHECK(cudaMalloc(&d_kernel, kerBytes));
    CUDA_CHECK(cudaMalloc(&d_outputImage, imgBytes));

    CUDA_CHECK(cudaMemcpy(d_inputImage, h_inputImage.data(), imgBytes,
                          cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_kernel, h_kernel.data(), kerBytes,
                          cudaMemcpyHostToDevice));

    dim3 block(TILE_SIZE, TILE_SIZE);
    dim3 grid((width + TILE_SIZE - 1) / TILE_SIZE,
              (height + TILE_SIZE - 1) / TILE_SIZE);

    const size_t sharedBytes =
        static_cast<size_t>(TILE_SIZE + kernelWidth - 1) *
        static_cast<size_t>(TILE_SIZE + kernelHeight - 1) * sizeof(float);

    convolution2D_kernel<<<grid, block, sharedBytes>>>(
        d_inputImage, d_kernel, d_outputImage, height, width, kernelHeight,
        kernelWidth);
    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());
    CUDA_CHECK(cudaMemcpy(h_outputImage.data(), d_outputImage, imgBytes,
                          cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_inputImage));
    CUDA_CHECK(cudaFree(d_kernel));
    CUDA_CHECK(cudaFree(d_outputImage));
    // STUDENT_END
}

int main(int argc, char *argv[]) {
    if (argc != 4) {
        std::cerr << "Usage: " << argv[0]
                  << " <input_image_file> <kernel_file> <output_image_file>"
                  << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string kernelFile = argv[2];
    std::string outputFile = argv[3];

    int height, width, kernelHeight, kernelWidth;
    bool success;

    auto h_inputImage =
        readMatrixFromFile<float>(inputFile, height, width, success);
    if (!success) {
        writeMatrixToFile({}, 0, 0, outputFile);
        return 1;
    }

    auto h_kernel = readMatrixFromFile<float>(kernelFile, kernelHeight,
                                              kernelWidth, success);
    if (!success) {
        writeMatrixToFile({}, 0, 0, outputFile);
        return 1;
    }

    std::vector<float> h_outputImage;
    convolution2D_host(h_inputImage, h_kernel, h_outputImage, height, width,
                       kernelHeight, kernelWidth);
    writeMatrixToFile(h_outputImage, height, width, outputFile);

    return 0;
}