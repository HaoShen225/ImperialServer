// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).
// The reference implementation also constructs the MST on CPU.

#include "mst.hpp"

#include "boruvka.hpp"

#include <cstdint>
#include <memory>
#include <thread>
#include <vector>


namespace {

void build_one_tree(
    const int* edge_index,
    const float* edge_weight,
    int* edge_out,
    int vertex_count,
    int edge_count
) {
    std::unique_ptr<Graph> graph(create_graph(vertex_count, edge_count));
    for (int edge = 0; edge < edge_count; ++edge) {
        graph->edge[edge].src = edge_index[2 * edge];
        graph->edge[edge].dest = edge_index[2 * edge + 1];
        graph->edge[edge].weight = edge_weight[edge];
    }
    boruvka_mst(graph.get(), edge_out);
    delete[] graph->edge;
    graph->edge = nullptr;
}

}  // namespace


at::Tensor mst_forward(
    const at::Tensor& edge_index,
    const at::Tensor& edge_weight,
    int64_t vertex_count
) {
    TORCH_CHECK(edge_index.dim() == 3 && edge_index.size(2) == 2,
                "edge_index must have shape [B, E, 2].");
    TORCH_CHECK(edge_weight.dim() == 2,
                "edge_weight must have shape [B, E].");
    TORCH_CHECK(edge_index.size(0) == edge_weight.size(0) &&
                edge_index.size(1) == edge_weight.size(1),
                "edge_index and edge_weight dimensions do not match.");
    TORCH_CHECK(edge_index.scalar_type() == at::kInt,
                "edge_index must use int32.");
    TORCH_CHECK(edge_weight.scalar_type() == at::kFloat,
                "edge_weight must use float32.");
    TORCH_CHECK(vertex_count >= 1, "vertex_count must be positive.");
    TORCH_CHECK(edge_index.size(1) >= vertex_count - 1,
                "The graph has too few edges to be connected.");

    const auto edge_index_cpu = edge_index.to(at::kCPU).contiguous();
    const auto edge_weight_cpu = edge_weight.to(at::kCPU).contiguous();
    auto edge_out_cpu = at::empty(
        {edge_index.size(0), vertex_count - 1, 2},
        edge_index_cpu.options()
    );

    const int64_t batch_size = edge_index.size(0);
    const int edge_count = static_cast<int>(edge_index.size(1));
    const int vertices = static_cast<int>(vertex_count);
    const int* edge_index_ptr = edge_index_cpu.data_ptr<int>();
    const float* edge_weight_ptr = edge_weight_cpu.data_ptr<float>();
    int* edge_out_ptr = edge_out_cpu.data_ptr<int>();

    std::vector<std::thread> workers;
    workers.reserve(batch_size);
    for (int64_t batch = 0; batch < batch_size; ++batch) {
        workers.emplace_back(
            build_one_tree,
            edge_index_ptr + batch * edge_count * 2,
            edge_weight_ptr + batch * edge_count,
            edge_out_ptr + batch * (vertex_count - 1) * 2,
            vertices,
            edge_count
        );
    }
    for (std::thread& worker : workers) {
        worker.join();
    }
    return edge_out_cpu.to(edge_index.device());
}
