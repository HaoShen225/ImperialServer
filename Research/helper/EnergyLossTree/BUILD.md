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
    window_size=64,
    stride=32,
)
shallow_target, deep_target = targets.shallow, targets.deep
```

No pseudo-label weight map is passed in this initial configuration, so both
trees use uniform all-one weights.
