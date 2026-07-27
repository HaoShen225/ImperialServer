// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).
// Rewritten as forward-only deterministic tree recurrences.  Each
// batch/channel pair is independent and runs on CUDA.

#include "refine.hpp"

#include <ATen/cuda/CUDAContext.h>
#include <c10/cuda/CUDAGuard.h>
#include <c10/cuda/CUDAException.h>
#include <cuda_runtime.h>


namespace {

__global__ void feature_tree_filter_kernel(
    const float* feature_in,
    const float* edge_weight,
    const int* sorted_index,
    const int* sorted_parent,
    const int* sorted_child,
    float* upward,
    float* aggregate,
    int channel_count,
    int vertex_count,
    int max_adj_per_vertex
) {
    const int linear_block = blockIdx.x;
    const int batch = linear_block / channel_count;
    const int channel = linear_block % channel_count;
    if (threadIdx.x != 0) {
        return;
    }

    const int feature_offset = (batch * channel_count + channel) * vertex_count;
    const int tree_offset = batch * vertex_count;
    const int child_offset = batch * vertex_count * max_adj_per_vertex;
    feature_in += feature_offset;
    upward += feature_offset;
    aggregate += feature_offset;
    edge_weight += tree_offset;
    sorted_index += tree_offset;
    sorted_parent += tree_offset;
    sorted_child += child_offset;

    for (int position = vertex_count - 1; position >= 0; --position) {
        const int original_vertex = sorted_index[position];
        float value = feature_in[original_vertex];
        for (int slot = 0; slot < max_adj_per_vertex; ++slot) {
            const int child_position =
                sorted_child[position * max_adj_per_vertex + slot];
            if (child_position <= 0) {
                break;
            }
            value += upward[child_position] * edge_weight[child_position];
        }
        upward[position] = value;
    }

    for (int position = 0; position < vertex_count; ++position) {
        const int original_vertex = sorted_index[position];
        if (position == 0) {
            aggregate[original_vertex] = upward[position];
            continue;
        }
        const int parent_position = sorted_parent[position];
        const int parent_vertex = sorted_index[parent_position];
        const float weight = edge_weight[position];
        aggregate[original_vertex] =
            upward[position] * (1.0f - weight * weight) +
            aggregate[parent_vertex] * weight;
    }
}


__global__ void normalizer_tree_filter_kernel(
    const float* edge_weight,
    const int* sorted_index,
    const int* sorted_parent,
    const int* sorted_child,
    float* upward,
    float* aggregate,
    int vertex_count,
    int max_adj_per_vertex
) {
    const int batch = blockIdx.x;
    if (threadIdx.x != 0) {
        return;
    }

    const int tree_offset = batch * vertex_count;
    const int child_offset = batch * vertex_count * max_adj_per_vertex;
    edge_weight += tree_offset;
    sorted_index += tree_offset;
    sorted_parent += tree_offset;
    sorted_child += child_offset;
    upward += tree_offset;
    aggregate += tree_offset;

    for (int position = vertex_count - 1; position >= 0; --position) {
        float value = 1.0f;
        for (int slot = 0; slot < max_adj_per_vertex; ++slot) {
            const int child_position =
                sorted_child[position * max_adj_per_vertex + slot];
            if (child_position <= 0) {
                break;
            }
            value += upward[child_position] * edge_weight[child_position];
        }
        upward[position] = value;
    }

    for (int position = 0; position < vertex_count; ++position) {
        const int original_vertex = sorted_index[position];
        if (position == 0) {
            aggregate[original_vertex] = upward[position];
            continue;
        }
        const int parent_position = sorted_parent[position];
        const int parent_vertex = sorted_index[parent_position];
        const float weight = edge_weight[position];
        aggregate[original_vertex] =
            upward[position] * (1.0f - weight * weight) +
            aggregate[parent_vertex] * weight;
    }
}

}  // namespace


at::Tensor refine_forward(
    const at::Tensor& feature_in,
    const at::Tensor& edge_weight,
    const at::Tensor& sorted_index,
    const at::Tensor& sorted_parent,
    const at::Tensor& sorted_child
) {
    TORCH_CHECK(feature_in.is_cuda(), "feature_in must be a CUDA tensor.");
    TORCH_CHECK(edge_weight.is_cuda() && sorted_index.is_cuda() &&
                sorted_parent.is_cuda() && sorted_child.is_cuda(),
                "All tree-filter inputs must be CUDA tensors.");
    TORCH_CHECK(feature_in.is_contiguous() && edge_weight.is_contiguous() &&
                sorted_index.is_contiguous() && sorted_parent.is_contiguous() &&
                sorted_child.is_contiguous(),
                "All tree-filter inputs must be contiguous.");
    TORCH_CHECK(feature_in.scalar_type() == at::kFloat &&
                edge_weight.scalar_type() == at::kFloat,
                "feature_in and edge_weight must use float32.");
    TORCH_CHECK(sorted_index.scalar_type() == at::kInt &&
                sorted_parent.scalar_type() == at::kInt &&
                sorted_child.scalar_type() == at::kInt,
                "Tree indices must use int32.");
    TORCH_CHECK(feature_in.dim() == 3,
                "feature_in must have shape [B, C, V].");
    TORCH_CHECK(edge_weight.dim() == 2,
                "edge_weight must have shape [B, V].");
    TORCH_CHECK(sorted_index.dim() == 2 && sorted_parent.dim() == 2 &&
                sorted_child.dim() == 3,
                "Invalid tree-order tensor dimensions.");
    TORCH_CHECK(feature_in.size(0) == edge_weight.size(0) &&
                feature_in.size(2) == edge_weight.size(1),
                "feature_in and edge_weight dimensions do not match.");
    TORCH_CHECK(sorted_index.sizes() == edge_weight.sizes() &&
                sorted_parent.sizes() == edge_weight.sizes() &&
                sorted_child.size(0) == edge_weight.size(0) &&
                sorted_child.size(1) == edge_weight.size(1),
                "Tree-order tensors do not match edge_weight.");
    TORCH_CHECK(feature_in.device() == edge_weight.device() &&
                feature_in.device() == sorted_index.device() &&
                feature_in.device() == sorted_parent.device() &&
                feature_in.device() == sorted_child.device(),
                "All tree-filter inputs must be on the same CUDA device.");

    c10::cuda::CUDAGuard device_guard(feature_in.device());
    const int64_t batch_size = feature_in.size(0);
    const int64_t channel_count = feature_in.size(1);
    const int64_t vertex_count = feature_in.size(2);
    const int64_t max_adj_per_vertex = sorted_child.size(2);
    auto feature_upward = at::zeros_like(feature_in);
    auto feature_aggregate = at::zeros_like(feature_in);
    auto weight_upward = at::zeros(
        {batch_size, vertex_count},
        feature_in.options()
    );
    auto weight_aggregate = at::zeros_like(weight_upward);

    const cudaStream_t stream = at::cuda::getCurrentCUDAStream();
    feature_tree_filter_kernel<<<
        static_cast<unsigned int>(batch_size * channel_count), 1, 0, stream>>>(
        feature_in.data_ptr<float>(),
        edge_weight.data_ptr<float>(),
        sorted_index.data_ptr<int>(),
        sorted_parent.data_ptr<int>(),
        sorted_child.data_ptr<int>(),
        feature_upward.data_ptr<float>(),
        feature_aggregate.data_ptr<float>(),
        static_cast<int>(channel_count),
        static_cast<int>(vertex_count),
        static_cast<int>(max_adj_per_vertex)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    normalizer_tree_filter_kernel<<<
        static_cast<unsigned int>(batch_size), 1, 0, stream>>>(
        edge_weight.data_ptr<float>(),
        sorted_index.data_ptr<int>(),
        sorted_parent.data_ptr<int>(),
        sorted_child.data_ptr<int>(),
        weight_upward.data_ptr<float>(),
        weight_aggregate.data_ptr<float>(),
        static_cast<int>(vertex_count),
        static_cast<int>(max_adj_per_vertex)
    );
    C10_CUDA_KERNEL_LAUNCH_CHECK();

    return feature_aggregate / weight_aggregate.unsqueeze(1).clamp_min(1e-12);
}
