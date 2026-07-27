# Third-party notice

The CUDA minimum-spanning-tree and tree-filter implementation under `csrc/`
is adapted from:

- **TreeEnergyLoss**, Copyright (c) 2022 megvii-model
- <https://github.com/megvii-research/TreeEnergyLoss>

The upstream work is licensed under the Apache License, Version 2.0. This port
removes obsolete THC dependencies, updates tensor and CUDA APIs for PyTorch
2.7.1, adds validation and CUDA error checks, and retains only the forward
operations required to construct detached pseudo labels.

See `LICENSE` in this directory for the license text.
