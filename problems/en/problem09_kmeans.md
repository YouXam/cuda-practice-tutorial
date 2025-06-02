# Problem 9: K-Means Clustering

K-Means is a classical clustering algorithm in machine learning, captivating researchers with its elegant simplicity and remarkable versatility. Despite being decades old, K-Means remains an indispensable tool in the data science toolkit.

## The K-Means Clustering Algorithm

Imagine you have a pile of scattered beads that need to be sorted into groups based on some pattern. K-Means works like an intelligent sorter that first selects several representative points (called centroids) from the pile, then follows the principle of "birds of a feather flock together" by having each bead join the nearest representative point. The sorter then recalculates the center position of each group and adjusts the representative points accordingly. This process repeats until every bead finds its proper "home."

This iterative optimization philosophy embodies the wisdom of algorithmic design. Each iteration makes the clustering result more reasonable until reaching a relatively stable state. The algorithm achieves this by minimizing the Within-Cluster Sum of Squares (WCSS), ensuring data points within each cluster gather as tightly as possible around their centroid.

The workflow can be summarized in the following steps:

1. **Initialization**: Randomly select K data points as initial centroids.
2. **Assignment Phase**: Assign each data point to the cluster of its nearest centroid.
3. **Update Phase**: Calculate new centroid positions for each cluster.
4. **Convergence Detection**: Check if centroids have changed. If no changes occur, the algorithm converges and iteration ends; otherwise, return to step 2.

This process embodies profound mathematical principles. Each iteration reduces the objective function value, and while it may converge to a local optimum, reasonable initialization strategies typically yield satisfactory clustering results.

## CUDA Implementation of K-Means

### Parallel Assignment Strategy: Finding Nearest Neighbors

The assignment phase offers the strongest parallelism in the entire algorithm. Each GPU thread independently processes one data point, traversing all centroids to find the nearest one:

```cpp
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
```

Here's an important performance optimization: using squared distances instead of actual distances for comparison. Square root operations are relatively expensive on GPUs, while squared distance relationships perfectly match actual distance relationships, so this optimization doesn't affect algorithmic correctness.

Convergence detection represents another crucial aspect of the assignment phase. We need to efficiently detect whether any points have changed cluster assignments:

```cpp
if (points[idx].cluster_id != new_cluster_id && new_cluster_id != -1) {
    points[idx].cluster_id = new_cluster_id;
    atomicOr(d_changed_flag_int, 1);  // Set flag when changes occur
}
```

This design uses atomic OR operations to set a global change flag, providing an efficient signaling mechanism for convergence detection in the main loop.

### Centroid Updates: Two-Phase Parallel Reduction

Centroid updating represents the most complex part of the algorithm, requiring decomposition into two independent GPU kernels. The first phase handles data accumulation:

```cpp
int idx = blockIdx.x * blockDim.x + threadIdx.x;
if (idx >= N) return;

int cluster = points[idx].cluster_id;
if (cluster != -1 && cluster < K) {
    atomicAdd(&d_sum_x[cluster], points[idx].x);
    atomicAdd(&d_sum_y[cluster], points[idx].y);
    atomicAdd(&d_counts[cluster], 1);
}
```

The boundary check `cluster != -1 && cluster < K` ensures program robustness, preventing invalid cluster IDs from causing memory access errors. Atomic addition operations guarantee that multiple threads updating the same cluster's statistics simultaneously won't encounter data races.

The second phase calculates new centroid positions while handling the special case of empty clusters:

```cpp
int kdx = blockIdx.x * blockDim.x + threadIdx.x;
if (kdx >= K) return;

if (d_counts[kdx] > 0) {
    new_centroids[kdx].x = d_sum_x[kdx] / d_counts[kdx];
    new_centroids[kdx].y = d_sum_y[kdx] / d_counts[kdx];
} else {
    // Handle empty clusters: reinitialize to some data point
    int point_idx_for_reinit = kdx % N_points;
    new_centroids[kdx].x = d_points_for_empty[point_idx_for_reinit].x + (kdx * 0.01f);
    new_centroids[kdx].y = d_points_for_empty[point_idx_for_reinit].y + (kdx * 0.01f);
}
```

Empty cluster handling represents a crucial consideration in practical applications. When a cluster loses all data points during iteration, we need intelligent reinitialization. The strategy employed here sets empty cluster centroids to some data point's position, adding small random offsets to avoid duplication.

### Main Loop Coordination Mechanism

The main iterative loop requires careful coordination of various computational phases while handling convergence detection and memory management:

```cpp
int h_changed_val = 1;  // Initially assume changes occur
for (int iter = 0; iter < max_iterations && h_changed_val != 0; ++iter) {
    h_changed_val = 0;  // Reset current iteration flag
    cudaMemcpy(d_changed_flag_int, &h_changed_val, sizeof(int), cudaMemcpyHostToDevice);
    
    // Execute assignment phase
    assign_clusters_kernel<<<numBlocks_assign, threadsPerBlock>>>(
        d_points, d_centroids, N, K, d_changed_flag_int);
    
    // Check convergence
    cudaMemcpy(&h_changed_val, d_changed_flag_int, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_changed_val == 0 && iter > 0) break;
    
    // Execute centroid updates
    cudaMemset(d_sum_x, 0, sumsByteSize);
    cudaMemset(d_sum_y, 0, sumsByteSize);
    cudaMemset(d_counts, 0, countsByteSize);
    
    update_centroids_stage1_kernel<<<numBlocks_update_stage1, threadsPerBlock>>>(
        d_points, N, K, d_sum_x, d_sum_y, d_counts);
    
    update_centroids_stage2_kernel<<<numBlocks_update_stage2, threadsPerBlock>>>(
        d_new_centroids, K, d_sum_x, d_sum_y, d_counts, d_points, N);
    
    cudaMemcpy(d_centroids, d_new_centroids, centroidsByteSize, cudaMemcpyDeviceToDevice);
}
```

## Conclusion

You now possess fundamental CUDA programming capabilities: writing parallel computation kernels, managing device memory, utilizing thread blocks for task allocation, and implementing basic algorithm acceleration. These skills serve as your gateway into the GPU computing realm.

To become a truly professional developer requires continued deep learning and practice. I recommend:

1. Reading official CUDA documentation, such as the [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
2. Read "CUDA Programming Fundamentals and Practice" by Zheyong Fan (Chinese).
3. Learning multi-GPU collaborative computing techniques to distribute computational tasks across multiple devices
4. Mastering Nsight tools for performance analysis, identifying memory bandwidth bottlenecks or instruction throughput issues

Finally, practice makes perfect. Try applying this knowledge in real projects to solve real-world problems. As experience accumulates, you'll be able to write more efficient and sophisticated GPU-accelerated code.