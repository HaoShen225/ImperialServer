// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).

#pragma once

#include <torch/extension.h>


at::Tensor refine_forward(
    const at::Tensor& feature_in,
    const at::Tensor& edge_weight,
    const at::Tensor& sorted_index,
    const at::Tensor& sorted_parent,
    const at::Tensor& sorted_child
);
