// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).

#pragma once

#include <torch/extension.h>


at::Tensor mst_forward(
    const at::Tensor& edge_index,
    const at::Tensor& edge_weight,
    int64_t vertex_count
);
