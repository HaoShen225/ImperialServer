# EnergyLossTree CUDA extension

The propagation module uses a forward-only port of the TreeFilter extension
from the official Tree Energy Loss implementation. It targets the project's
PyTorch 2.7.1 and CUDA 11.8 environment.

Build it on a PBS node with a visible GPU so PyTorch can select the correct
compute capability:

```bash
module load GCC/11.3.0
module load CUDA/11.8.0
cd /rds/general/user/hs225/home/TTA_Project_runs/Research/helper/EnergyLossTree
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python setup.py build_ext --inplace
```

If compiling without a visible GPU, set `TORCH_CUDA_ARCH_LIST` explicitly to
the compute capability used by the target queue.

Run the tests from the project root:

```bash
/rds/general/user/hs225/home/miniforge3/envs/TTA/bin/python -m unittest discover \
  -s Research/helper/EnergyLossTree/tests -v
```

Compiled `.so`, `build/`, and egg-info outputs are local artifacts and should
not be committed.

The repository also includes `test_cuda.pbs`, which runs the complete test
suite on the `v1_gpu72` L40S queue.

The propagator expects source-model probabilities rather than logits:

```python
from Research.helper.EnergyLossTree import propagate_dual_tree_pseudo_labels

targets = propagate_dual_tree_pseudo_labels(
    source_logits.softmax(dim=1),
    shallow_guidance,
    deep_guidance,
)
shallow_target, deep_target = targets.shallow, targets.deep
```

This default performs full-image propagation and applies the same 16-pixel
spatial temperature to both trees. For every tree edge, the feature and
spatial terms are combined as
`exp(-feature_distance / sigma - spatial_distance / spatial_temperature)`.
Consequently, a source label's spatial influence decays as
`exp(-tree_path_length / spatial_temperature)`. Distances are measured in
pseudo-label pixels after guidance interpolation.

Local window propagation remains available as an explicit option:

```python
windowed_targets = propagate_dual_tree_pseudo_labels(
    source_logits.softmax(dim=1),
    shallow_guidance,
    deep_guidance,
    window_size=64,
    stride=32,
)
```

Pass `spatial_temperature=None` to disable geometric path decay and reproduce
the previous feature-only affinity. `window_size` and `stride` must either
both be omitted or both be supplied.

No pseudo-label weight map is passed in these examples, so both trees use
uniform all-one weights.
