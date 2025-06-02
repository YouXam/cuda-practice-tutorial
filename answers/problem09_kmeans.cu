#include <iomanip>
#include <iostream>
#include <string>
#include <vector>
#include <limits>

#include "common.cuh"

// Point data structure (ensure this matches in student/answer.cu)
struct Point {
    float x, y;
    int cluster_id;  // To store result
};

struct Centroid {
    float x, y;
};

bool readKMeansDataFromFile(const std::string& filename,
                            int& num_points_out,
                            int& num_clusters_out,  // K
                            int& max_iterations_out,
                            std::vector<Point>& points_out);
void writeKMeansResultsToFile(const std::string& filename,
                              const std::vector<Point>& points_in,
                              const std::vector<Centroid>& centroids_in,
                              int num_clusters_k);

// STUDENT_START
__global__ void assign_clusters_kernel(Point* points, Centroid* centroids, int N, int K, int* d_changed_flag_int) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float min_dist_sq = std::numeric_limits<float>::max();
    int new_cluster_id = -1;

    for (int k = 0; k < K; ++k) {
        float dx = points[idx].x - centroids[k].x;
        float dy = points[idx].y - centroids[k].y;
        float dist_sq = dx * dx + dy * dy;
        if (dist_sq < min_dist_sq) {
            min_dist_sq = dist_sq;
            new_cluster_id = k;
        }
    }

    if (points[idx].cluster_id != new_cluster_id && new_cluster_id != -1) {
        points[idx].cluster_id = new_cluster_id;
        atomicOr(d_changed_flag_int, 1);  // Set flag to true (1) if any change
    }
}

__global__ void update_centroids_stage1_kernel(Point* points, int N, int K, float* d_sum_x, float* d_sum_y, int* d_counts) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    int cluster = points[idx].cluster_id;
    if (cluster != -1 && cluster < K) {  // Ensure cluster ID is valid
        atomicAdd(&d_sum_x[cluster], points[idx].x);
        atomicAdd(&d_sum_y[cluster], points[idx].y);
        atomicAdd(&d_counts[cluster], 1);
    }
}

__global__ void update_centroids_stage2_kernel(Centroid* new_centroids, int K, const float* d_sum_x, const float* d_sum_y, const int* d_counts, Point* d_points_for_empty, int N_points) {
    int kdx = blockIdx.x * blockDim.x + threadIdx.x;
    if (kdx >= K) return;

    if (d_counts[kdx] > 0) {
        new_centroids[kdx].x = d_sum_x[kdx] / d_counts[kdx];
        new_centroids[kdx].y = d_sum_y[kdx] / d_counts[kdx];
    } else {
        // Handle empty cluster: re-initialize to a random point (or a fixed strategy)
        // This is a simple re-initialization strategy. A more robust one might be needed.
        if (N_points > 0) {  // Only if there are points to pick from
                             // Pick a point based on kdx to ensure different centroids pick different points if possible
            int point_idx_for_reinit = kdx % N_points;
            new_centroids[kdx].x = d_points_for_empty[point_idx_for_reinit].x + (kdx * 0.01f);  // Add small jitter
            new_centroids[kdx].y = d_points_for_empty[point_idx_for_reinit].y + (kdx * 0.01f);
        } else {                          // No points, cannot re-initialize, leave as is (e.g. 0,0 or previous)
            new_centroids[kdx].x = 0.0f;  // Or some default
            new_centroids[kdx].y = 0.0f;
        }
    }
}
// STUDENT_END

void run_kmeans_host(std::vector<Point>& h_points, std::vector<Centroid>& h_centroids, int K_param, int max_iterations) {
    int N = h_points.size();
    int K = K_param;

    if (N == 0) {
        h_centroids.clear();
        return;
    }
    if (K == 0) {
        h_centroids.clear();
        for (auto& p : h_points) p.cluster_id = -1;
        return;
    }
    K = std::min(K, N);  // K cannot be more than N
    h_centroids.resize(K);

    // STUDENT_START
    Point* d_points = nullptr;
    Centroid* d_centroids = nullptr;
    Centroid* d_new_centroids = nullptr;  // For updating centroids
    int* d_changed_flag_int = nullptr;

    float* d_sum_x = nullptr;
    float* d_sum_y = nullptr;
    int* d_counts = nullptr;

    size_t pointsByteSize = (size_t)N * sizeof(Point);
    size_t centroidsByteSize = (size_t)K * sizeof(Centroid);
    size_t sumsByteSize = (size_t)K * sizeof(float);
    size_t countsByteSize = (size_t)K * sizeof(int);

    CUDA_CHECK(cudaMalloc((void**)&d_points, pointsByteSize));
    CUDA_CHECK(cudaMalloc((void**)&d_centroids, centroidsByteSize));
    CUDA_CHECK(cudaMalloc((void**)&d_new_centroids, centroidsByteSize));  // Will store results of stage2
    CUDA_CHECK(cudaMalloc((void**)&d_changed_flag_int, sizeof(int)));
    CUDA_CHECK(cudaMalloc((void**)&d_sum_x, sumsByteSize));
    CUDA_CHECK(cudaMalloc((void**)&d_sum_y, sumsByteSize));
    CUDA_CHECK(cudaMalloc((void**)&d_counts, countsByteSize));

    CUDA_CHECK(cudaMemcpy(d_points, h_points.data(), pointsByteSize, cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(d_centroids, h_centroids.data(), centroidsByteSize, cudaMemcpyHostToDevice));

    int threadsPerBlock = 256;
    int numBlocks_assign = (N + threadsPerBlock - 1) / threadsPerBlock;
    int numBlocks_update_stage1 = numBlocks_assign;
    int numBlocks_update_stage2 = (K + threadsPerBlock - 1) / threadsPerBlock;
    if (numBlocks_assign == 0 && N > 0) numBlocks_assign = 1;
    if (numBlocks_update_stage1 == 0 && N > 0) numBlocks_update_stage1 = 1;
    if (numBlocks_update_stage2 == 0 && K > 0) numBlocks_update_stage2 = 1;

    int h_changed_val = 1;  // Start with changed = true
    for (int iter = 0; iter < max_iterations && h_changed_val != 0; ++iter) {
        h_changed_val = 0;  // Reset for current iteration
        CUDA_CHECK(cudaMemcpy(d_changed_flag_int, &h_changed_val, sizeof(int), cudaMemcpyHostToDevice));

        if (numBlocks_assign > 0 && K > 0) {
            assign_clusters_kernel<<<numBlocks_assign, threadsPerBlock>>>(d_points, d_centroids, N, K, d_changed_flag_int);
            CUDA_CHECK(cudaGetLastError());
            // cudaDeviceSynchronize(); // Not strictly needed before D2H copy of flag
        }

        CUDA_CHECK(cudaMemcpy(&h_changed_val, d_changed_flag_int, sizeof(int), cudaMemcpyDeviceToHost));
        if (h_changed_val == 0 && iter > 0) break;  // Converged (allow first iteration to always run update)

        if (K > 0) {  // Only update centroids if K > 0
            CUDA_CHECK(cudaMemset(d_sum_x, 0, sumsByteSize));
            CUDA_CHECK(cudaMemset(d_sum_y, 0, sumsByteSize));
            CUDA_CHECK(cudaMemset(d_counts, 0, countsByteSize));

            if (numBlocks_update_stage1 > 0) {
                update_centroids_stage1_kernel<<<numBlocks_update_stage1, threadsPerBlock>>>(d_points, N, K, d_sum_x, d_sum_y, d_counts);
                CUDA_CHECK(cudaGetLastError());
                // cudaDeviceSynchronize(); // Not strictly needed before next kernel on same stream
            }

            if (numBlocks_update_stage2 > 0) {
                update_centroids_stage2_kernel<<<numBlocks_update_stage2, threadsPerBlock>>>(d_new_centroids, K, d_sum_x, d_sum_y, d_counts, d_points, N);
                CUDA_CHECK(cudaGetLastError());
                // cudaDeviceSynchronize(); // Not strictly needed before D2D copy
            }
            CUDA_CHECK(cudaMemcpy(d_centroids, d_new_centroids, centroidsByteSize, cudaMemcpyDeviceToDevice));
        }
        // std::cout << "Iteration " << iter + 1 << " done. Changed: " << h_changed_val << std::endl;
    }

    CUDA_CHECK(cudaMemcpy(h_points.data(), d_points, pointsByteSize, cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(h_centroids.data(), d_centroids, centroidsByteSize, cudaMemcpyDeviceToHost));

    CUDA_CHECK(cudaFree(d_points));
    CUDA_CHECK(cudaFree(d_centroids));
    CUDA_CHECK(cudaFree(d_new_centroids));
    CUDA_CHECK(cudaFree(d_changed_flag_int));
    CUDA_CHECK(cudaFree(d_sum_x));
    CUDA_CHECK(cudaFree(d_sum_y));
    CUDA_CHECK(cudaFree(d_counts));
    // STUDENT_END
}

int main(int argc, char* argv[]) {
    if (argc != 3) {
        std::cerr << "Usage: " << argv[0] << " <input_kmeans_file> <output_kmeans_file>" << std::endl;
        return 1;
    }
    std::string inputFile = argv[1];
    std::string outputFile = argv[2];

    int N, K, max_iter;
    std::vector<Point> h_points;
    bool success = readKMeansDataFromFile(inputFile, N, K, max_iter, h_points);

    if (!success) {
        writeKMeansResultsToFile(outputFile, {}, {}, 0);
        return 1;
    }

    std::vector<Centroid> h_centroids(K);
    if (N > 0 && K > 0) {
        K = std::min(K, N);                 // Ensure K <= N
        h_centroids.resize(K);         // Resize if K was adjusted
        for (int i = 0; i < K; ++i) {  // Initialize centroids to first K points
            h_centroids[i].x = h_points[i].x;
            h_centroids[i].y = h_points[i].y;
        }
    } else if (K > 0) {       // N is 0, but K > 0
        h_centroids.clear();  // No centroids if no points
        K = 0;
    }

    run_kmeans_host(h_points, h_centroids, K, max_iter);
    writeKMeansResultsToFile(outputFile, h_points, h_centroids, K);

    return 0;
}

bool readKMeansDataFromFile(const std::string& filename,
                            int& num_points_out,
                            int& num_clusters_out,  // K
                            int& max_iterations_out,
                            std::vector<Point>& points_out) {
    std::ifstream inFile(filename);
    points_out.clear();
    if (!inFile.is_open()) {
        std::cerr << "Error opening K-Means input file: " << filename << std::endl;
        return false;
    }
    if (!(inFile >> num_points_out >> num_clusters_out >> max_iterations_out)) {
        std::cerr << "Error reading K-Means parameters (N, K, iter) from file: " << filename << std::endl;
        inFile.close();
        return false;
    }
    if (num_points_out < 0 || num_clusters_out < 0 || max_iterations_out < 0) {
        std::cerr << "Invalid K-Means parameters (negative values) in file: " << filename << std::endl;
        inFile.close();
        return false;
    }
    if (num_points_out == 0) {  // Valid case: no points
        inFile.close();
        return true;
    }
    if (num_clusters_out > num_points_out && num_points_out > 0) {
        std::cerr << "K (num_clusters) cannot be greater than num_points in file: " << filename << std::endl;
        inFile.close();
        return false;
    }

    points_out.resize(num_points_out);
    for (int i = 0; i < num_points_out; ++i) {
        if (!(inFile >> points_out[i].x >> points_out[i].y)) {
            std::cerr << "Error reading point data for point " << i << " from file: " << filename << std::endl;
            inFile.close();
            points_out.clear();
            return false;
        }
        points_out[i].cluster_id = -1;  // Initialize cluster ID
    }
    inFile.close();
    return true;
}

void writeKMeansResultsToFile(const std::string& filename,
                              const std::vector<Point>& points_in,
                              const std::vector<Centroid>& centroids_in,
                              int num_clusters_k) {
    std::ofstream outFile(filename);
    if (!outFile.is_open()) {
        std::cerr << "Error opening K-Means output file for writing: " << filename << std::endl;
        return;
    }
    outFile << std::fixed << std::setprecision(6);

    outFile << points_in.size() << " " << num_clusters_k << std::endl;  // N and K

    // Output cluster ID for each point
    for (size_t i = 0; i < points_in.size(); ++i) {
        outFile << points_in[i].cluster_id << std::endl;
    }

    // Output final centroid positions
    for (size_t i = 0; i < centroids_in.size(); ++i) {
        outFile << centroids_in[i].x << " " << centroids_in[i].y << std::endl;
    }
    outFile.close();
}
