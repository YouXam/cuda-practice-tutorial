# 第 9 题：K-Means 聚类

K-Means 算法是机器学习领域的一种经典聚类算法，它的魅力在于其简单易懂的设计和强大的实用性。尽管 K-Means 算法已经诞生了几十年，但它依然是数据科学领域中不可或缺的工具之一。

## K-Means 聚类算法

想象一下，你面前有一堆散乱的珠子，需要把它们按照某种规律分成几组。K-Means 算法就像一个聪明的分拣员，它会先在珠子堆中选择几个代表点（我们称之为质心），然后根据"物以类聚"的原则，让每颗珠子都投靠离自己最近的代表点。接下来，分拣员会重新计算每组的中心位置，调整代表点的位置，如此反复，直到所有珠子都找到了自己的"归宿"。

这种迭代优化的思想体现了算法设计的智慧。每一次迭代都让聚类结果变得更加合理，直到达到一个相对稳定的状态。算法通过最小化簇内平方和（Within-Cluster Sum of Squares，WCSS）来实现这一目标，让每个簇内的数据点尽可能紧密地聚集在质心周围。

它的工作流程可以概括为以下几个步骤：

1. **初始化**：随机选择 K 个数据点作为初始质心。
2. **分配阶段**：将每个数据点分配到距离最近的质心所在的簇。
3. **更新阶段**：计算每个簇的新的质心位置。
4. **收敛检测**：检查质心是否发生变化，如果没有变化则算法收敛，结束迭代；否则返回第 2 步。

这个过程背后蕴含着深刻的数学原理。每次迭代都在降低目标函数的值，虽然可能收敛到局部最优解，但通过合理的初始化策略，我们通常能够获得满意的聚类效果。

## K-Means 算法的 CUDA 实现

### 并行分配策略：寻找最近邻

分配阶段是整个算法中并行性最强的部分。每个GPU线程独立处理一个数据点，通过遍历所有质心来找到距离最近的那个：

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

这里有个重要的性能优化：使用平方距离而不是真实距离来比较。平方根运算在 GPU 上相对昂贵，而平方距离的大小关系与真实距离完全一致，因此这个优化不会影响算法的正确性。

收敛检测是分配阶段的另一个关键环节。我们需要高效地检测是否有任何点改变了簇分配：

```cpp
if (points[idx].cluster_id != new_cluster_id && new_cluster_id != -1) {
    points[idx].cluster_id = new_cluster_id;
    atomicOr(d_changed_flag_int, 1);  // 有变化时设置标志
}
```

这种设计使用原子 OR 操作来设置全局变化标志，为主循环的收敛判断提供了高效的信号机制。

### 质心更新：两阶段并行归约

质心更新是算法中最复杂的部分，需要将其分解为两个独立的GPU kernel来处理。第一阶段负责数据累积：

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

这里的边界检查 `cluster != -1 && cluster < K` 确保了程序的健壮性，防止无效的簇ID导致内存访问错误。原子加法操作保证了多个线程同时更新同一个簇的统计信息时不会发生数据竞争。

第二阶段负责计算新的质心位置，同时处理空簇的特殊情况：

```cpp
int kdx = blockIdx.x * blockDim.x + threadIdx.x;
if (kdx >= K) return;

if (d_counts[kdx] > 0) {
    new_centroids[kdx].x = d_sum_x[kdx] / d_counts[kdx];
    new_centroids[kdx].y = d_sum_y[kdx] / d_counts[kdx];
} else {
    // 处理空簇：重新初始化到某个数据点
    int point_idx_for_reinit = kdx % N_points;
    new_centroids[kdx].x = d_points_for_empty[point_idx_for_reinit].x + (kdx * 0.01f);
    new_centroids[kdx].y = d_points_for_empty[point_idx_for_reinit].y + (kdx * 0.01f);
}
```

空簇处理是实际应用中必须考虑的问题。当某个簇在迭代过程中失去所有数据点时，我们需要智能地重新初始化它。这里采用的策略是将空簇的质心设置为某个数据点的位置，并添加小的随机偏移以避免重复。

### 主控循环的协调机制

主迭代循环需要精心协调各个计算阶段，同时处理收敛检测和内存管理：

```cpp
int h_changed_val = 1;  // 开始时假设有变化
for (int iter = 0; iter < max_iterations && h_changed_val != 0; ++iter) {
    h_changed_val = 0;  // 重置当前迭代的标志
    cudaMemcpy(d_changed_flag_int, &h_changed_val, sizeof(int), cudaMemcpyHostToDevice);
    
    // 执行分配阶段
    assign_clusters_kernel<<<numBlocks_assign, threadsPerBlock>>>(
        d_points, d_centroids, N, K, d_changed_flag_int);
    
    // 检查是否收敛
    cudaMemcpy(&h_changed_val, d_changed_flag_int, sizeof(int), cudaMemcpyDeviceToHost);
    if (h_changed_val == 0 && iter > 0) break;
    
    // 执行质心更新
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

## 总结

你现在已掌握CUDA编程的基础能力：能够编写并行计算核心函数，管理设备内存，运用线程块进行任务分配，并实现简单的算法加速。这些技能是你进入GPU计算领域的起点。

要成为真正专业的开发者，还需要后续的深入学习和实践。建议：

1. 阅读 CUDA 官方文档，例如 [CUDA C++ Programming Guide](https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html)
2. 阅读 《CUDA编程基础与实践》（樊哲勇）
3. 学习多 GPU 协同计算技术，将计算任务分布到多个设备上
4. 学习使用 Nsight 工具进行性能分析，定位内存带宽瓶颈或指令吞吐问题。

最后，实践是最好的老师。尝试在实际项目中应用这些知识，解决真实世界的问题。随着经验的积累，你将能够编写出更高效、更复杂的 GPU 加速代码。