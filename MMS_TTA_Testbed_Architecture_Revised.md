# M&Ms Medical Segmentation TTA Testbed Architecture

## Revised specification after architecture review

This document defines a compact, reproducible test-time adaptation (TTA) testbed for multi-vendor cardiac MRI segmentation on the M&Ms dataset. It incorporates the accepted findings of the architecture review while preserving the original design goal:

> Each TTA method lives in its own subpackage, while the outer experiment harness remains small, explicit, and method-agnostic.

Two labels are used throughout this specification:

- **VERIFY:** the exact mechanism or value must be checked against the paper and official implementation before final experiments.
- **SEG-MOD:** a segmentation-specific modification of a method originally designed for image classification. It must be described as a modification rather than an exact reproduction.

---

## 1. Experimental scope

- **Source domain:** Vendor A.
- **Target domains:** Vendors B, C, and D.
- **Backbone:** 2D ResUNet-34.
- **Encoder initialization:** ImageNet-pretrained ResNet-34.
- **Normalization:** `BatchNorm2d` in the encoder and decoder.
- **Input:** one-channel cardiac MRI slice.
- **Output classes:** background, LV, MYO, and RV.
- **Source-training seeds:** `2022`, `2023`, `2024`, `2025`, and `2026`.
- **Source optimizer:** AdamW.
- **Initial methods:** Source-only, TENT, EATA, SAR, CoTTA, RoTTA, RoID, and DeYO.

The fixed scientific invariants are:

1. All dataset splits are patient-level.
2. The Vendor A split is fixed across the five source-training seeds.
3. Target labels never enter any TTA method.
4. Every method receives the same source checkpoint, stream order, preprocessing, timing, and reset policy within a comparison.
5. Metrics are computed after prediction is restored to the native 3D grid.
6. Method-specific optimization scope and normalization behavior belong to the method, not to the global TTA protocol.

---

## 2. Accepted corrections from the review

The following changes are mandatory.

### 2.1 Method-owned semantics are removed from the global protocol

The following keys must not appear under the shared `tta:` configuration:

```text
parameter_scope
use_target_batch_statistics
accumulate_target_running_statistics
```

They are not universal protocol decisions:

- TENT and EATA normally adapt normalization affine parameters.
- SAR has a method-specific normalization scope that must be verified.
- CoTTA may update a much broader parameter set.
- RoTTA replaces ordinary BN behavior with robust normalization.

Each method configures its own trainable parameters and normalization behavior in `setup()`. The testbed records the actual state by introspection:

- trainable parameter names;
- normalization module types;
- BN flags and buffer presence;
- teacher/student prediction source;
- optimizer type and resolved hyperparameters.

### 2.2 RoTTA includes robust normalization

RoTTA is not represented by only a teacher and a memory bank. Its package gains:

```text
tta_methods/rotta/rbn.py
```

`rbn.py` implements the paper's robust normalization mechanism and registers its evolving statistics as module buffers so that checkpointing and reset work through `state_dict`.

The exact RBN update rule and coefficients are **VERIFY** items.

### 2.3 Surface metrics are evaluated on the native grid

HD95 and ASSD are physical-distance metrics. They are invalid if a prediction on a resized `256 x 256` grid is combined with native image spacing.

The required path is:

```text
native volume
-> preprocessing metadata
-> 256 x 256 network input
-> prediction
-> inverse spatial preprocessing
-> native-grid 3D label volume
-> Dice / HD95 / ASSD with native spacing
```

The inverse transform belongs in `data.py`. `metrics.py` accepts only native-grid predictions and native-grid labels.

### 2.4 Batch-statistic TTA is safe by construction

For methods that require current-batch BN statistics without running-statistic accumulation, every BN module is converted as follows during `setup()`:

```python
module.track_running_stats = False
module.running_mean = None
module.running_var = None
```

With absent running buffers, PyTorch BatchNorm uses batch statistics even if a future helper accidentally calls `model.eval()`. This makes BN behavior structurally robust rather than dependent only on contributor discipline.

The rule still remains that `predict()` must not change model mode.

### 2.5 Stateful randomness is method-local

CoTTA augmentation/restoration, RoTTA memory sampling, and DeYO transformation must not consume an uncontrolled global RNG. Every stochastic method owns a generator initialized from a recorded `method_seed`.

Reset must restore or deterministically reseed this generator.

### 2.6 Optimizers are rebuilt on reset

The initial optimizer state is empty. Instead of snapshotting and restoring it, each method implements `_build_optimizer()` and calls it from both `setup()` and `reset()`.

This avoids parameter-aliasing problems and correctly handles wrappers such as SAR's SAM optimizer.

### 2.7 Target stream JSON is authoritative

`target_streams.json` is loaded at runtime and is never regenerated by `run_tta.py`. It contains:

- schema version;
- generation seed;
- vendor order;
- patient order;
- phase order.
- excluded official data parts.

Protocol v3 excludes `Training/Unlabeled` from the complete arrival stream. Vendor C therefore contains only the 10 validation and 40 testing patients; the excluded 25 patients cannot alter continual adaptation state and are never scored.

The file hash is written into every result record.

### 2.8 `common.py` is narrowed

A function is shared only when at least two methods use exactly the same semantics. Similar names are insufficient.

The following are removed from `common.py`:

- `stochastic_restore`, which belongs to CoTTA;
- generic `entropy_selection`, because EATA, SAR, and DeYO use different gates;
- generic `masked_mean`, because their weighting semantics differ.

---

## 3. Revised repository structure

```text
mms_tta/
├── config.yaml
├── data.py
├── model.py
├── train_source.py
├── run_tta.py
├── metrics.py
├── utils.py
│
├── tta_methods/
│   ├── __init__.py
│   ├── source.py
│   ├── base.py
│   ├── common.py
│   │
│   ├── tent/
│   │   ├── __init__.py
│   │   └── method.py
│   │
│   ├── eata/
│   │   ├── __init__.py
│   │   ├── method.py
│   │   └── fisher.py
│   │
│   ├── sar/
│   │   ├── __init__.py
│   │   ├── method.py
│   │   └── sam.py
│   │
│   ├── cotta/
│   │   ├── __init__.py
│   │   ├── method.py
│   │   └── augment.py
│   │
│   ├── rotta/
│   │   ├── __init__.py
│   │   ├── method.py
│   │   ├── memory.py
│   │   └── rbn.py
│   │
│   ├── roid/
│   │   ├── __init__.py
│   │   └── method.py
│   │
│   └── deyo/
│       ├── __init__.py
│       ├── method.py
│       └── transform.py
│
├── tests/
│   ├── conftest.py
│   ├── test_data.py
│   ├── test_metrics.py
│   ├── test_protocol.py
│   ├── test_state.py
│   └── test_methods.py
│
├── splits/
│   ├── vendor_a_split.json
│   └── target_streams.json
│
├── checkpoints/
│   ├── seed2022_best.pt
│   ├── seed2023_best.pt
│   ├── seed2024_best.pt
│   ├── seed2025_best.pt
│   ├── seed2026_best.pt
│   ├── fisher_seed2022.pt
│   └── ...
│
└── results/
```

Only two structural additions are accepted:

1. `rotta/rbn.py`, because RBN is a defining RoTTA component and an independently testable module;
2. a small `tests/` directory, because the scientific invariants must be executable rather than aspirational.

No `scripts/`, per-method YAML directory, registry module, dependency-injection framework, or universal state manager is added.

---

## 4. Experimental protocol

### 4.1 Source and target split

- Vendor A is divided into fixed patient-level training and validation sets.
- Vendors B, C, and D are target-only.
- The same Vendor A split is used for all five source seeds.
- The source validation set may select the source checkpoint and support explicitly documented source-only derivations.
- Target labels cannot select thresholds, optimizers, update scopes, or method variants.

### 4.2 Stream unit and batching

The stream unit is one complete ED or ES patient volume. Contiguous slices are batched within that volume:

```text
Patient 001 / ED: [0..3], [4..7], [8..9]
Patient 001 / ES: [0..3], [4..7], ...
Patient 002 / ED: ...
```

Rules:

- TTA arrival batch size is four.
- The final partial batch is retained.
- A batch never crosses patient or phase boundaries.
- Slice order is preserved through reconstruction.
- Before volume expansion, each Vendor independently shuffles its eligible patient blocks
  with the matching source-checkpoint seed; no worker-side or global RNG state is used.

### 4.3 Timing

The harness supports:

- `adapt_then_predict`: update on the current batch, then perform a second forward pass for its evaluated prediction;
- `predict_then_adapt`: predict the current batch, then update for future batches.

The primary comparison may use `adapt_then_predict`, but it must be reported as a uniform medical-segmentation protocol rather than exact online evaluation from every original paper. Both supported modes perform distinct predict and adapt operations, and their computational costs must be logged.

Every method in the same result table uses the same timing.

### 4.4 Reset policy

The harness owns reset timing:

- `patient`: reset before each patient;
- `vendor`: reset before each independent vendor stream;
- `never`: one continual state across the complete stream.

Primary independent-domain results:

```text
A -> B
A -> C
A -> D
```

Secondary continual result:

```text
A -> B -> C -> D
```

No reset policy is embedded inside an individual method.

### 4.5 Native-grid evaluation

`data.py` stores sufficient preprocessing metadata to invert every spatial operation:

- original shape;
- original spacing;
- orientation/slice order;
- crop and padding offsets;
- resampling or resize factors.

For discrete prediction maps, inverse resize uses nearest-neighbor interpolation. If logits are inversely resized before argmax, bilinear interpolation is permitted but the choice must be fixed globally.

The selected convention must be tested on synthetic shapes with analytically known distances.

---

## 5. Outer-module responsibilities

### 5.1 `data.py`

Owns all M&Ms data semantics:

```python
class MMSSourceDataset(Dataset):
    ...


class MMSTargetVolumeDataset(Dataset):
    ...


def build_source_loaders(cfg):
    ...


def build_target_stream(vendor, cfg, order_seed):
    """Resolve one vendor-local patient order from the checkpoint seed."""
    ...


def split_volume_into_batches(images, batch_size):
    ...


def to_native_grid(prediction, spatial_metadata):
    ...
```

`build_target_stream` requires the source-checkpoint seed as `order_seed`. Each target
Vendor independently shuffles its eligible patient list with that seed. Patient blocks
remain atomic (ED then ES), and slices within each volume remain z-index ascending.
The resolved patient order and its SHA-256 are persisted in every run manifest.

### 5.2 `model.py`

Contains the complete ResUNet-34:

```python
class DecoderBlock(nn.Module):
    ...


class ResUNet34(nn.Module):
    ...


def build_model(cfg):
    ...


def load_source_checkpoint(model, path):
    ...
```

The RGB input kernel is converted to one channel by averaging over its input-channel dimension. Encoder and decoder use `BatchNorm2d`; the segmentation head has no normalization.

The forward method may optionally return the final decoder feature map:

```python
return {
    "logits": logits,
    "features": features if return_features else None,
}
```

### 5.3 `train_source.py`

Owns source training, validation, checkpoint selection, and the five source seeds.

Source optimizer:

```text
Optimizer:                    AdamW
Encoder learning rate:        1e-4
Decoder/head learning rate:   3e-4
Weight decay:                 1e-4
Betas:                        (0.9, 0.999)
Epsilon:                      1e-8
Physical batch size:          8
Warm-up:                      5 epochs
Scheduler:                    cosine to 1e-6
```

Biases and normalization affine parameters receive no weight decay.

After checkpoint selection, `train_source.py --fisher` may call:

```python
from tta_methods.eata.fisher import estimate_fisher
```

and save `checkpoints/fisher_seed{seed}.pt`. This is an artifact-preparation exception, not a runtime dependency. The function receives a source loader; the EATA package never imports `data.py`.

Fisher sample count, loss definition, and adapted parameter names are **VERIFY** items.

### 5.4 `run_tta.py`

Owns orchestration only:

- source checkpoint loading;
- method construction;
- authoritative stream loading;
- reset policy;
- common timing;
- slice batching and reconstruction;
- inverse preprocessing;
- evaluation and logging.

It never implements entropy, SAM, Fisher regularization, EMA, RBN, memory, prior correction, stochastic restoration, or PLPD.

The label-isolation boundary is the signature of `run_volume`:

```python
def run_volume(method, images, batch_size, device):
    """No mask or full volume dictionary is accepted."""
    predictions = []
    records = []

    for batch in split_volume_into_batches(images, batch_size):
        logits, info = method.process_batch(batch.to(device))
        predictions.append(logits.argmax(dim=1).cpu())
        records.append(info)

    return torch.cat(predictions), records
```

Only after `run_volume` returns may the evaluator receive a mask:

```python
prediction, infos = run_volume(
    method,
    volume["image"],
    cfg["tta"]["batch_size"],
    device,
)

native_prediction = to_native_grid(
    prediction,
    volume["spatial_metadata"],
)

scores = evaluate_volume(
    native_prediction,
    volume["native_mask"],
    volume["native_spacing"],
)
```

### 5.5 `metrics.py`

Contains native-grid 3D metrics only:

```python
def dice_score(...):
    ...


def hd95(...):
    ...


def assd(...):
    ...


def evaluate_volume(...):
    ...


def aggregate_results(...):
    ...
```

Absent-class behavior must be explicitly defined for:

- absent in both prediction and ground truth;
- present only in ground truth;
- present only in prediction.

The evaluator must not silently inherit an undocumented library convention.

### 5.6 `utils.py`

Contains only small infrastructure helpers:

```python
def load_config(...):
    ...


def set_seed(...):
    ...


def file_sha256(...):
    ...


def save_json(...):
    ...


def get_device(...):
    ...


def run_metadata(...):
    ...
```

No TTA algorithm belongs in `utils.py`.

---

## 6. Public TTA interface

### 6.1 `AdaptationResult`

```python
from dataclasses import dataclass, field


@dataclass
class AdaptationResult:
    loss: float | None = None
    n_seen: int = 0
    n_selected: int = 0
    updated: bool = False
    extras: dict[str, float] = field(default_factory=dict)
    probe_payload: dict | None = field(default=None, repr=False)
```

`extras` is logged opaquely by the harness. The harness never branches on a method-specific diagnostic.

`probe_payload` is an ephemeral, non-serialized method diagnostic. The harness may compare it with labels only after adaptation and prediction have completed; labels never cross the `run_volume(method, images, batch_size, device)` boundary and probe calculations cannot feed back into method state.

Each method documents whether `n_selected` counts slices or pixels.

### 6.2 State helpers

The reset snapshot is captured after method setup. Model snapshots live on CPU:

```python
def cpu_state_dict(module):
    return {
        key: value.detach().cpu().clone()
        for key, value in module.state_dict().items()
    }


def cpu_parameter_dict(module):
    return {
        name: parameter.detach().cpu().clone()
        for name, parameter in module.named_parameters()
    }
```

The distinction is intentional:

- `source_parameter_state` is captured before method-specific module surgery and is used for drift, recovery, and stochastic parameter restoration;
- `_initial_state["model"]` is captured after setup and matches the actual post-surgery module structure used by reset.

This avoids loading pre-surgery BN running-buffer keys into a post-surgery model whose buffers are `None`.

### 6.3 `BaseTTA`

```python
from abc import ABC, abstractmethod
import torch


class BaseTTA(ABC):
    """Common method lifecycle.

    Subclass rules:
      1. Do not override __init__.
      2. Create all method state in setup().
      3. Route all method randomness through self.generator.
      4. adapt() and predict() receive image tensors only.
      5. predict() may be overridden; process_batch() may not.
    """

    def __init__(self, model, cfg, protocol_cfg, device):
        self.model = model.to(device)
        self.cfg = cfg
        self.protocol_cfg = protocol_cfg
        self.device = device
        self.optimizer = None

        self.generator = torch.Generator(device="cpu")
        self.source_parameter_state = cpu_parameter_dict(self.model)

        self.setup()
        self._initial_state = self.capture_state()

    @abstractmethod
    def setup(self):
        """Configure modes, parameters, optimizer, RNG, and method state."""
        raise NotImplementedError

    @abstractmethod
    def adapt(self, images):
        """Perform one logical unlabeled adaptation update."""
        raise NotImplementedError

    @torch.no_grad()
    def predict(self, images):
        """Return logits without changing model mode or adaptation state."""
        return self.model(images)["logits"]

    def process_batch(self, images):
        timing = self.protocol_cfg["timing"]

        if timing == "adapt_then_predict":
            info = self.adapt(images)
            logits = self.predict(images)
        elif timing == "predict_then_adapt":
            logits = self.predict(images)
            info = self.adapt(images)
        else:
            raise ValueError(f"Unknown timing: {timing}")

        return logits, info

    def capture_state(self):
        return {
            "model": cpu_state_dict(self.model),
            "generator": self.generator.get_state(),
        }

    def reset(self):
        self.model.load_state_dict(self._initial_state["model"])
        self.generator.set_state(self._initial_state["generator"])
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self):
        return None
```

`process_batch` is treated as final by project convention. CoTTA and RoTTA override `predict`, `capture_state`, and `reset` where required.

`predict` must not update model parameters, teacher parameters, memory, or normalization buffers. A method whose official evaluated prediction uses stochastic augmentation may advance its method-local generator; this exception must be documented and remains reproducible through generator reset.

There is no universal `StateManager`, callback system, `finalize`, or cross-method intermediate base class.

### 6.4 Method registry

`tta_methods/__init__.py` contains the complete public registry:

```python
from .source import Source
from .tent.method import TENT
from .eata.method import EATA
from .sar.method import SAR
from .cotta.method import CoTTA
from .rotta.method import RoTTA
from .roid.method import RoID
from .deyo.method import DeYO


METHODS = {
    "source": Source,
    "tent": TENT,
    "eata": EATA,
    "sar": SAR,
    "cotta": CoTTA,
    "rotta": RoTTA,
    "roid": RoID,
    "deyo": DeYO,
}


def build_method(name, model, cfg, protocol_cfg, device):
    try:
        method_class = METHODS[name]
    except KeyError as error:
        raise ValueError(
            f"Unknown method {name!r}; available methods: {sorted(METHODS)}"
        ) from error

    return method_class(
        model=model,
        cfg=cfg,
        protocol_cfg=protocol_cfg,
        device=device,
    )
```

Every method validates unknown configuration keys when constructed.

---

## 7. Final boundary of `common.py`

Only the following operations are shared initially:

| Operation | Consumers | Reason |
|---|---|---|
| `pixel_entropy` | TENT, EATA, SAR, RoID, DeYO | Identical per-pixel entropy |
| `slice_entropy` | same | Explicit plain spatial mean |
| `configure_bn_for_batch_stats` | Batch-statistic methods | Identical BN buffer surgery |
| `collect_bn_affine` | Normalization-affine methods | Identical freeze/unfreeze rule |
| `build_optimizer` | Gradient methods | Plain Adam/SGD construction |
| `ema_update` | CoTTA and RoTTA | Shared only after exact buffer/parameter semantics are verified |
| `parameter_drift` | All methods | L2 over named parameters shared with the source snapshot |

Suggested signatures:

```python
def pixel_entropy(logits):
    ...


def slice_entropy(logits):
    """Plain mean of pixel entropy over H and W."""
    ...


def configure_bn_for_batch_stats(model):
    ...


def collect_bn_affine(model):
    """Return parameters and their names."""
    ...


def build_optimizer(parameters, cfg):
    ...


@torch.no_grad()
def ema_update(teacher, student, momentum):
    ...


def parameter_drift(model, source_parameter_state):
    ...
```

Method-specific selection and weighting stay local even if they all consume entropy.

---

## 8. Method package requirements

### 8.1 Source-only

`source.py` remains a single control module:

- complete model in evaluation mode;
- source BN running statistics;
- all parameters frozen;
- no optimizer;
- no adaptation.

### 8.2 TENT

```text
tent/{__init__.py, method.py}
```

- apply batch-statistics BN surgery;
- adapt the verified normalization-affine scope;
- minimize plain mean pixel entropy;
- log predicted foreground area by class to detect background expansion.

Any foreground-weighted entropy is a separate **SEG-MOD** variant, not exact TENT.

Optimizer, learning rate, and exact update scope are **VERIFY** items.

### 8.3 EATA

```text
eata/{__init__.py, method.py, fisher.py}
```

- Fisher is estimated only from Vendor A source training data;
- the Fisher artifact path and SHA-256 are recorded;
- reliability selection, redundancy filtering, and weighting remain in `method.py`;
- selected ratio and update rate are mandatory diagnostics.

The classification descriptor used for redundancy filtering has no automatic segmentation equivalent. A slice descriptor such as spatially averaged class probability is a **SEG-MOD**. Adjacent slices may be rejected as inherently redundant; this must be measured before interpreting EATA performance.

Threshold rule, Fisher objective/sample count, regularization strength, optimizer, and descriptor definition are **VERIFY/SEG-MOD** items.

### 8.4 SAR

```text
sar/{__init__.py, method.py, sam.py}
```

- both SAM passes use the same arrival batch;
- the two passes count as one logical update;
- recovery restores named parameters from `source_parameter_state`;
- EMA-loss state and recovery counters are reset;
- recovery triggering rate is logged.

The official norm-layer exclusions, entropy margin, recovery rule, SAM radius, optimizer, and learning rate are **VERIFY** items.

SAR's published motivation favors batch-agnostic normalization in wild streams. Running it on the shared ResUNet-34-BN backbone is a deliberate benchmark constraint and must be disclosed when interpreting results.

### 8.5 CoTTA

```text
cotta/{__init__.py, method.py, augment.py}
```

- student, EMA teacher, and frozen source anchor are method-owned;
- stochastic restoration stays in `cotta/`;
- all stochastic operations use the method generator;
- reset restores teacher state and generator and rebuilds the optimizer;
- the source parameter snapshot is on CPU and is reused for restoration;
- no extra source model copy is retained beyond the required frozen anchor.

Segmentation logits from spatially augmented views must be inverse-warped before averaging. Flips and 90-degree rotations are preferred initially because they invert exactly. Affine and scaling transforms introduce interpolation error and must be tested.

The official CoTTA segmentation configuration is the primary reference. Prediction source, confidence gate, update scope, optimizer, EMA momentum, restore probability, augmentation count, and augmentation set are **VERIFY** items.

### 8.6 RoTTA

```text
rotta/{__init__.py, method.py, memory.py, rbn.py}
```

- `rbn.py` implements robust normalization as a proper module with buffers;
- `memory.py` implements CSTU insertion, eviction, sampling, serialization, and reset;
- arrival batch size and memory replay batch size remain distinct;
- teacher state, memory contents, ages, counters, generator, and RBN buffers are reset;
- the evaluated prediction source is verified against the official code.

Classification-style category balance is not directly defined for a multi-class segmentation map. Dominant foreground class, class-presence vector, and foreground-fraction bins are possible **SEG-MOD** choices. One choice must be frozen using source-side reasoning and documented.

RBN update, memory score, category key, teacher momentum, replay objective, optimizer, and learning rate are **VERIFY/SEG-MOD** items.

### 8.7 RoID

```text
roid/{__init__.py, method.py}
```

Keep one method file until it becomes demonstrably too large.

Soft-likelihood-ratio loss, certainty weighting, diversity weighting, weight ensembling, and prior correction are **VERIFY** items. Contiguous slices may appear artificially non-diverse. Pixel-level background dominance also makes naive class-prior correction inappropriate; a foreground-restricted or disabled prior correction would be a **SEG-MOD**.

### 8.8 DeYO

```text
deyo/{__init__.py, method.py, transform.py}
```

Classification DeYO assumes one predicted label per image. Segmentation has a label map, and patch shuffling destroys spatial correspondence. Therefore, a segmentation analog must explicitly define:

1. the object-destructive transform;
2. per-pixel PLPD on the originally predicted class;
3. foreground-aware aggregation to a slice or pixel weight;
4. patch size relative to cardiac anatomy;
5. entropy and PLPD gate semantics.

This is a substantial **SEG-MOD**, not an exact reproduction. The original classification method and the segmentation adaptation must be reported separately.

---

## 9. BatchNorm and model-mode lifecycle

Mode is configured once in `setup()` and is not changed afterward.

| Method | Normalization behavior | Running accumulation | Dropout |
|---|---|---|---|
| Source | Source running statistics | None | Off |
| TENT/EATA/SAR/RoID/DeYO | Current target batch via removed BN buffers | None | Off |
| CoTTA student/teacher | **VERIFY**; apply the same structural policy if batch statistics are required | None unless method specifies otherwise | **VERIFY**, normally off |
| RoTTA | RBN test-time statistics | RBN buffers evolve by design | Off |

For ordinary batch-statistic methods:

1. freeze all parameters;
2. remove BN running buffers and disable tracking;
3. enable only the verified normalization affine parameters;
4. call `model.eval()` once if needed to disable dropout;
5. do not change mode inside `adapt` or `predict`.

Because BN buffers are absent, evaluation mode cannot silently restore source statistics.

The final partial batch may contain one slice. Convolutional BN remains mathematically valid because it aggregates over `B x H x W`, but this batch is noisier. Actual batch size is logged for every update.

---

## 10. Reset and deterministic replay

No universal state framework is needed. Three mechanisms are sufficient:

1. model parameters and registered buffers use a post-setup CPU `state_dict`;
2. optimizers and known-empty containers are rebuilt or cleared;
3. teacher/anchor companion models are restored by method overrides.

Required reset state:

| Method | Reset state |
|---|---|
| Source | Model only |
| TENT | Model, generator, fresh optimizer |
| EATA | Base state plus redundancy descriptor/history and counters |
| SAR | Base state plus EMA loss and recovery counters |
| CoTTA | Base state plus teacher; anchor remains immutable |
| RoTTA | Base state plus teacher, empty memory, age/counters; RBN buffers ride model state |
| RoID | Base state plus running prediction/prior and weight-ensemble state |
| DeYO | Base state plus transformation RNG state |

The defining behavioral guarantee is:

```text
run stream S
-> reset
-> run stream S again
-> bit-identical predictions and adaptation records
```

---

## 11. Revised configuration

The following is a schema, not a claim that every method hyperparameter is already verified.

```yaml
experiment:
  source_seeds: [2022, 2023, 2024, 2025, 2026]
  harness_seed: 3101
  target_vendors: [B, C, D]

data:
  source_vendor: A
  image_size: [256, 256]
  split_file: splits/vendor_a_split.json
  stream_file: splits/target_streams.json

model:
  name: resunet34
  pretrained_encoder: true
  in_channels: 1
  num_classes: 4
  encoder_norm: batch_norm
  decoder_norm: batch_norm

source:
  batch_size: 8
  epochs: 200
  optimizer: adamw
  encoder_lr: 0.0001
  decoder_lr: 0.0003
  weight_decay: 0.0001
  warmup_epochs: 5
  scheduler: cosine
  min_lr: 0.000001
  early_stopping_patience: 30

tta:
  batch_size: 4
  timing: adapt_then_predict
  reset: vendor

methods:
  source: {}

  tent:
    profile_verified: false
    method_seed: 4101
    steps: 1
    update_scope: bn_affine       # VERIFY
    bn_policy: batch_no_running   # VERIFY
    optimizer: null               # VERIFY
    lr: null                      # VERIFY

  eata:
    profile_verified: false
    method_seed: 4101
    steps: 1
    update_scope: bn_affine       # VERIFY
    bn_policy: batch_no_running   # VERIFY
    optimizer: null               # VERIFY
    lr: null                      # VERIFY
    entropy_rule: null            # VERIFY
    redundancy_rule: null         # VERIFY / SEG-MOD
    fisher_path: null
    fisher_alpha: null            # VERIFY

  sar:
    profile_verified: false
    method_seed: 4101
    steps: 1
    update_scope: null            # VERIFY
    bn_policy: batch_no_running   # Deliberate BN-backbone benchmark
    optimizer: null               # VERIFY
    lr: null                      # VERIFY
    entropy_rule: null            # VERIFY
    rho: null                     # VERIFY
    recovery_rule: null           # VERIFY

  cotta:
    profile_verified: false
    method_seed: 4101
    steps: 1
    update_scope: null            # VERIFY against segmentation code
    bn_policy: null               # VERIFY
    optimizer: null               # VERIFY
    lr: null                      # VERIFY
    teacher_momentum: null        # VERIFY
    confidence_gate: null         # VERIFY
    restore_probability: null     # VERIFY
    num_augmentations: null       # VERIFY
    augmentation_profile: null    # VERIFY / SEG-MOD

  rotta:
    profile_verified: false
    method_seed: 4101
    steps: 1
    update_scope: null            # VERIFY
    optimizer: null               # VERIFY
    lr: null                      # VERIFY
    teacher_momentum: null        # VERIFY
    rbn_alpha: null               # VERIFY
    memory_capacity: 64           # VERIFY
    memory_batch_size: 16         # VERIFY
    memory_category_key: null     # SEG-MOD
    timeliness_weight: null       # VERIFY
    uncertainty_weight: null      # VERIFY

  roid:
    profile_verified: false
    method_seed: 4101
    steps: 1
    update_scope: null            # VERIFY
    optimizer: null               # VERIFY
    lr: null                      # VERIFY
    diversity_rule: null          # VERIFY / SEG-MOD
    prior_correction: null        # VERIFY / SEG-MOD
    weight_ensemble_momentum: null # VERIFY

  deyo:
    profile_verified: false
    method_seed: 4101
    steps: 1
    update_scope: bn_affine       # VERIFY
    bn_policy: batch_no_running   # VERIFY
    optimizer: null               # VERIFY
    lr: null                      # VERIFY
    entropy_rule: null            # VERIFY
    plpd_rule: null               # VERIFY / SEG-MOD
    destruction_profile: null     # SEG-MOD
```

Final experiment execution should reject `profile_verified: false`. Development smoke tests may explicitly allow unverified profiles.

---

## 12. Reproducibility and provenance

Each result record contains:

- expanded resolved configuration;
- method and profile verification status;
- source seed and method seed;
- vendor, patient, and phase;
- timing and reset policy;
- source checkpoint SHA-256;
- Fisher artifact SHA-256 for EATA;
- source split and target stream JSON SHA-256;
- git commit;
- PyTorch, CUDA, and dependency versions;
- introspected trainable parameter names;
- introspected normalization summary;
- actual arrival batch size;
- prediction source, such as student or teacher;
- adaptation loss, update status, and selection counts;
- parameter drift;
- per-class predicted foreground area.

TTA runs should favor deterministic execution:

```python
torch.use_deterministic_algorithms(True)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False
```

Target patient orders are deterministically shuffled per checkpoint seed and the loaders
remain single-process. TTA should initially run in FP32; AMP can be introduced only after
deterministic equivalence is tested.

---

## 13. Leakage controls

Leakage prevention has four layers:

1. **Signature:** methods receive image tensors only.
2. **Imports:** `tta_methods/` cannot import `data.py`, `metrics.py`, or `run_tta.py`.
3. **Behavior:** permuting every target mask must leave every prediction bit-identical.
4. **Process:** method thresholds come only from paper formulas or documented Vendor A source-validation derivations.

Any threshold or segmentation design choice selected using B/C/D Dice or HD95 invalidates the locked-target interpretation.

---

## 14. Minimal automated tests

Tests use a tiny synthetic four-class BN segmentation network whenever real M&Ms data are unnecessary.

| Test | Required assertion |
|---|---|
| Patient split | No patient overlap across source train, source validation, B, C, and D |
| Label permutation | Predictions are bit-identical after arbitrary target-mask permutation |
| Trainable scope | Only introspected declared parameters change |
| BN mode | Batch-stat methods have absent running buffers; Source uses running buffers |
| Eval immunity | Calling `eval()` cannot switch surgically configured BN back to source statistics |
| Running accumulation | Ordinary batch-stat methods accumulate nothing; RoTTA RBN evolves and resets |
| Deterministic replay | Stream, reset, and replay produce bit-identical outputs and records |
| Volume batching | No cross-patient/phase batch; final partial batch retained; order preserved |
| Native-grid metrics | Known synthetic distances produce correct millimeter HD95/ASSD |
| Timing | First-batch behavior differs correctly between the two timing modes |
| Checkpoint identity | Different methods use the same checkpoint hash for the same source seed |
| Import hygiene | No upward or cross-method runtime imports |
| Registry/config | Every method constructs; unknown keys raise immediately |
| Augmentation inversion | Exactly invertible spatial transforms recover logits at original coordinates |

Absent-class metric conventions are included in the native-grid metric test.

---

## 15. Implementation order and gates

### Stage 0: data spine

Implement indexing, fixed splits, preprocessing metadata, inverse transform, target JSON, volume batching, and native-grid metrics.

**Gate:** split, batching, and native-grid metric tests pass.

### Stage 1: source models

Implement ResUNet-34-BN, five AdamW source runs, checkpoint selection, hashing, and Source-only A-to-B/C/D evaluation.

**Gate:** source BN behavior, checkpoint identity, volume reconstruction, and native-grid metrics pass.

### Stage 2: protocol core and TENT

Implement `BaseTTA`, narrowed `common.py`, BN surgery, state reset, timing, label isolation, and TENT.

**Gate:** complete deterministic replay and leakage suite passes for Source and TENT; duplicate A-to-B runs are bit-identical.

### Stage 3: EATA

Verify official mechanisms, generate Fisher artifacts from Vendor A, and implement selection/redundancy logic.

**Gate:** selected ratio and update rate are logged; contiguous-slice starvation is evaluated before target interpretation.

### Stage 4: SAR

Implement same-batch SAM and recovery.

**Gate:** recovery trigger rate is logged and BN-backbone deviation is documented.

### Stage 5: CoTTA

Use the official segmentation configuration, teacher prediction override, inverse-consistent augmentation, and stochastic restoration.

**Gate:** teacher/generator reset passes deterministic replay; exact augmentations pass inversion tests.

### Stage 6: RoTTA

Implement and test RBN before memory and teacher integration.

**Gate:** RBN buffers evolve and reset correctly; memory category definition is documented as a SEG-MOD.

### Stage 7: RoID

Verify all components and implement in one file.

**Gate:** diversity and prior-correction behavior are diagnosed on source validation without target tuning.

### Stage 8: DeYO

Design the segmentation PLPD and transformation only after the simpler baselines are stable.

**Gate:** the method is explicitly reported as a segmentation adaptation and its spatial correspondence assumptions are tested.

### Stage 9: continual experiments

Run the locked final matrix only after every profile is verified:

```text
8 methods x 5 source seeds x {B, C, D}
+ continual B -> C -> D
```

---

## 16. Deliberately rejected complexity

The following remain rejected:

- plugin or decorator-based method discovery;
- Hydra/OmegaConf or per-method YAML trees;
- universal state/snapshot frameworks;
- intermediate method base classes;
- dependency-injection containers;
- callback/hook systems;
- generic memory-bank abstractions;
- a standalone Fisher entry script;
- speculative splits such as `roid/prior.py`;
- a result database before JSON aggregation becomes inadequate;
- a third timing mode that forces every `adapt()` call to return evaluated logits.

The accepted additions solve current correctness or testability problems; the rejected additions solve hypothetical scale problems.

---

## 17. Updated prompt for Claude

```text
You are a senior PyTorch research engineer. Implement or refactor a medical-segmentation TTA testbed according to the attached revised specification.

Do not redesign the repository unless a concrete correctness defect requires it. Preserve these constraints:

1. The outer harness consists of config.yaml, data.py, model.py, train_source.py, run_tta.py, metrics.py, and utils.py.
2. Source-only, TENT, EATA, SAR, CoTTA, RoTTA, RoID, and DeYO use separate subpackages or modules under tta_methods/.
3. Every TTA method directly inherits BaseTTA; do not use cross-method inheritance.
4. Method-specific parameter scope and normalization policy remain inside each method's setup().
5. RoTTA must include its verified robust-normalization mechanism in rotta/rbn.py.
6. Batch-statistic methods remove BN running buffers so accidental eval() cannot restore source statistics.
7. Target labels are accepted only by native-grid evaluation after run_volume returns.
8. Surface metrics are computed in millimeters on the reconstructed native grid.
9. State snapshots live on CPU; optimizers are rebuilt on reset; stochastic methods own and reset a method-local generator.
10. common.py contains only byte-semantically shared operations. Selection, weighting, restoration, RBN, memory, and PLPD remain method-local when their semantics differ.
11. The target stream JSON is authoritative and hashed into results.
12. Final runs are prohibited until every method profile is verified against its paper and official repository.

Work in stages. Before implementing each method, produce a short verification table containing: official parameter-update scope, optimizer and learning rate, normalization behavior, prediction source, number of updates per arrival batch, reset/recovery behavior, and every classification-to-segmentation modification.

For each stage:

- show the files changed;
- explain the scientific invariant protected by each change;
- add or update the corresponding fast synthetic tests;
- run the tests;
- do not proceed while a gate is failing;
- never tune a method using Vendor B, C, or D labels.

Begin with an audit of the current repository against this specification. Return:

1. blocking correctness problems;
2. the minimal patch plan;
3. unresolved VERIFY and SEG-MOD decisions;
4. the first implementation stage and its tests.

Avoid factories beyond the existing dictionary registry, dependency injection, generic state managers, callbacks, auto-discovery, and speculative helper files.
```
