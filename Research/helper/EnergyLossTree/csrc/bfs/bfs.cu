// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).
// Rewritten as a deterministic per-tree CUDA traversal.  This avoids the
// divergent block barriers in the legacy kernel and supports small windows.

#include "bfs.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>


namespace {

__global__ void bfs_kernel(
    const int* edge_index,
    int* sorted_index,
    int* sorted_parent,
    int* sorted_child,
    int* adjacency,
    int* adjacency_length,
    int* parent_vertex,
    int vertex_count,
    int max_adj_per_vertex
) {
    const int batch = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const int edge_count = vertex_count - 1;
    edge_index += batch * edge_count * 2;
    sorted_index += batch * vertex_count;
    sorted_parent += batch * vertex_count;
    sorted_child += batch * vertex_count * max_adj_per_vertex;
    adjacency += batch * vertex_count * max_adj_per_vertex;
    adjacency_length += batch * vertex_count;
    parent_vertex += batch * vertex_count;

    for (int edge = 0; edge < edge_count; ++edge) {
        const int source = edge_index[2 * edge];
        const int target = edge_index[2 * edge + 1];
        const int source_offset = adjacency_length[source]++;
        const int target_offset = adjacency_length[target]++;
        if (source_offset < max_adj_per_vertex) {
            adjacency[source * max_adj_per_vertex + source_offset] = target;
        }
        if (target_offset < max_adj_per_vertex) {
            adjacency[target * max_adj_per_vertex + target_offset] = source;
        }
    }

    sorted_index[0] = 0;
    sorted_parent[0] = 0;
    parent_vertex[0] = 0;
    int queue_length = 1;
    for (int position = 0; position < queue_length; ++position) {
        const int current = sorted_index[position];
        int child_count = 0;
        const int degree = adjacency_length[current];
        for (int adjacent = 0; adjacent < degree; ++adjacent) {
            const int child = adjacency[current * max_adj_per_vertex + adjacent];
            if (child == parent_vertex[current]) {
                continue;
            }
            const int child_position = queue_length++;
            sorted_index[child_position] = child;
            sorted_parent[child_position] = position;
            sorted_child[position * max_adj_per_vertex + child_count] = child_position;
            parent_vertex[child] = current;
            child_count += 1;
        }
    }
}

}  // namespace


std::tuple<at::Tensor, at::Tensor, at::Tensor> bfs_forward(
    const at::Tensor& edge_index,
    int64_t max_adj_per_vertex
) {
    TORCH_CHECK(edge_index.is_cuda(), "edge_index must be a CUDA tensor.");
    TORCH_CHECK(edge_index.is_contiguous(), "edge_index must be contiguous.");
    TORCH_CHECK(edge_index.dim() == 3 && edge_index.size(2) == 2,
                "edge_index must have shape [B, V-1, 2].");
    TORCH_CHECK(edge_index.scalar_type() == at::kInt,
                "edge_index must use int32.");
    TORCH_CHECK(max_adj_per_vertex > 0,
                "max_adj_per_vertex must be positive.");

    c10::cuda::CUDAGuard device_guard(edge_index.device());
    const int64_t batch_size = edge_index.size(0);
    const int64_t vertex_count = edge_index.size(1) + 1;
    const auto options = edge_index.options();
    auto sorted_index = at::zeros({batch_size, vertex_count}, options);
    auto sorted_parent = at::zeros({batch_size, vertex_count}, options);
    auto sorted_child = at::zeros(
        {batch_size, vertex_count, max_adj_per_vertex},
        options
    );
    auto adjacency = at::zeros_like(sorted_child);
    auto adjacency_length = at::zeros({batch_size, vertex_count}, options);
    auto parent_vertex = at::zeros({batch_size, vertex_count}, options);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    bfs_kernel<<<static_cast<unsigned int>(batch_size), 1, 0, stream>>>(
        edge_index.data_ptr<int>(),
        sorted_index.data_ptr<int>(),
        sorted_parent.data_ptr<int>(),
        sorted_child.data_ptr<int>(),
        adjacency.data_ptr<int>(),
        adjacency_length.data_ptr<int>(),
        parent_vertex.data_ptr<int>(),
        static_cast<int>(vertex_count),
        static_cast<int>(max_adj_per_vertex)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();
    return std::make_tuple(sorted_index, sorted_parent, sorted_child);
}
