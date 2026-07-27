// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).

#pragma once

#include <torch/extension.h>


std::tuple<at::Tensor, at::Tensor, at::Tensor> bfs_forward(
    const at::Tensor& edge_index,
    int64_t max_adj_per_vertex
);
