// Adapted from megvii-research/TreeEnergyLoss (Apache-2.0).
// Modified for a forward-only PyTorch 2.7 CUDA extension.

#include <torch/extension.h>

#include "bfs/bfs.hpp"
#include "mst/mst.hpp"
#include "refine/refine.hpp"


PYBIND11_MODULE(TORCH_EXTENSION_NAME, module) {
    module.def("mst_forward", &mst_forward, "Minimum-spanning-tree forward");
    module.def("bfs_forward", &bfs_forward, "Tree breadth-first ordering");
    module.def("refine_forward", &refine_forward, "Normalized tree-filter forward");
}
