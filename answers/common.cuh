#include <iostream>
#include <vector>
#include <fstream>
#include <cstdio>
#include <cstdlib>

#define CUDA_CHECK(call) do { \
    cudaError_t err = call; \
    if (err != cudaSuccess) { \
        printf("CUDA error at %s:%d - %s\n", __FILE__, __LINE__, \
               cudaGetErrorString(err)); \
        exit(1); \
    } \
} while(0)

// Function to read vector from file
template <typename T=int>
std::vector<T> readVectorFromFile(const std::string& filename, bool& success) {
    std::ifstream inFile(filename);
    std::vector<T> vec;
    success = true;
    if (!inFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        success = false;
        return vec;
    }
    T val;
    while (inFile >> val) {
        vec.push_back(val);
    }
    if (inFile.bad()) { // I/O error during read
        std::cerr << "Error reading data from file: " << filename << std::endl;
        success = false;
        vec.clear(); // Clear partial data
    } else if (!inFile.eof() && inFile.fail()) { // Format error (e.g. non-integer)
         std::cerr << "Invalid data format in file: " << filename << std::endl;
         success = false;
         vec.clear();
    }
    inFile.close();
    return vec;
}

// Function to write vector to file
template <typename T=int>
void writeVectorToFile(const std::vector<T>& vec, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    for (size_t i = 0; i < vec.size(); ++i) {
        outFile << vec[i] << (i == vec.size() - 1 ? "" : " ");
    }
    outFile << std::endl;
    outFile.close();
}


// Function to read a single scalar from file
template <typename T=int>
int readScalarFromFile(const std::string& filename, bool& success) {
    std::ifstream inFile(filename);
    T val = 0;
    success = true;
    if (!inFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        success = false;
        return 0;
    }
    if (!(inFile >> val)) {
        std::cerr << "Error reading scalar from file: " << filename << std::endl;
        success = false;
        return 0;
    }
    if (!inFile.eof()){ // Check if there's more data than expected
        char ch;
        if (inFile >> ch) { // Try to read more, if successful it's an error
             std::cerr << "Extra data found in scalar file: " << filename << std::endl;
             success = false;
             return 0;
        }
    }
    inFile.close();
    return val;
}

// Function to write a single scalar to file
template <typename T=int>
void writeScalarToFile(T val, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    outFile << val << std::endl;
    outFile.close();
}

// Function to read matrix from file
template <typename T=int>
std::vector<T> readMatrixFromFile(const std::string& filename, int& rows, int& cols, bool& success) {
    std::ifstream inFile(filename);
    std::vector<T> mat;
    success = true;
    rows = 0;
    cols = 0;
    if (!inFile.is_open()) {
        std::cerr << "Error opening file: " << filename << std::endl;
        success = false;
        return mat;
    }
    if (!(inFile >> rows >> cols)) {
        std::cerr << "Error reading dimensions from file: " << filename << std::endl;
        success = false;
        return mat;
    }
    if (rows < 0 || cols < 0) {
        std::cerr << "Invalid matrix dimensions (negative) in file: " << filename << std::endl;
        success = false;
        return mat;
    }
    if (rows == 0 || cols == 0) { // Valid empty matrix
        if (rows * cols > 0) { // e.g. 0 rows, 5 cols is invalid if not fully 0
             success = false; return mat;
        }
        return mat;
    }
    mat.resize(rows * cols);
    for (int i = 0; i < rows * cols; ++i) {
        if (!(inFile >> mat[i])) {
            std::cerr << "Error reading matrix data from file: " << filename << " (expected " << rows*cols << " elements, got " << i << ")" << std::endl;
            success = false;
            mat.clear();
            return mat;
        }
    }
    inFile.close();
    return mat;
}

// Function to write matrix to file
template <typename T=int>
void writeMatrixToFile(const std::vector<T>& mat, int rows, int cols, const std::string& filename) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return;
    }
    outFile << rows << " " << cols << std::endl;
    if (rows == 0 || cols == 0) { // Handle empty matrix data part
        outFile.close();
        return;
    }
    for (int i = 0; i < rows; ++i) {
        for (int j = 0; j < cols; ++j) {
            outFile << mat[i * cols + j] << (j == cols - 1 ? "" : " ");
        }
        outFile << std::endl;
    }
    outFile.close();
}

