#!/usr/bin/env python3
"""实验一：冻结骨干，仅学习一个全局 BN 统计量插值参数。

实验目的
========
本实验使用 ``Research/backbone_params`` 中的 15 个 Mean Teacher teacher
backbone，在 ACDC -> MMS 的 3-shot × 5-seed × 4-domain 共 60 个任务上，
检验“最小化无标签预测熵所选择的 BN 插值幅度”是否与“最大化有标签 Dice
所选择的插值幅度”一致。

数学定义
========
所有 BatchNorm2d 层共享一个标量 ``a``，令 ``s = sigmoid(a)``。对第 l 层，
源域统计量为 ``mu_s, var_s``，当前目标切片统计量为 ``mu_t, var_t``。
目标统计量在纯 target-BN 捕获前向中估计，并在优化中 stop-gradient：

    mu_hat = mu_s + s * sg(mu_t - mu_s)

    log_sigma_hat
      = 0.5*log(var_s + eps)
      + s * sg(0.5*log(var_t + eps) - 0.5*log(var_s + eps))

    y = gamma_s * (x - mu_hat) / exp(log_sigma_hat) + beta_s

骨干权重和 BN affine 参数 gamma/beta 始终冻结，只有全局 ``a`` 可学习。
每张测试切片均从 ``s=0.5`` 独立重置，分别使用标准 TENT 像素熵和
前景/背景平衡熵优化；为避免 Adam 最后一步过冲，返回初始化及全部优化
迭代中熵最低的 ``a``。标签只用于事后 Dice oracle 和正式性能评估，
绝不参与 ``a`` 的梯度更新。

对齐检验
========
每张切片还会在 ``s=0,0.05,...,1`` 上进行固定网格扫描。对 GT 中实际
出现的 RV/MYO/LV 计算宏 Dice；完全无前景切片保留预测，但不进入
“熵最小点 vs Dice 最大点”的对齐统计。输出同时包含逐切片最优点、
任务级固定-s 曲线、两种可学习方法、两种熵网格选择方法和 Dice oracle
上界。Dice oracle 是使用标签的诊断上界，不是可部署的 TTA 方法。
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, MutableMapping, Sequence, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import spearmanr

from helper.Evaluator import PatientStreamEvaluator
from helper.Intent import balanced_fg_bg_entropy, categorical_entropy, model_logits
from helper.backbones.UNet import UNet
from helper.dataloader import TestLoader, _stack_images, _stack_masks


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_BACKBONES = RESEARCH_ROOT / "backbone_params"
DEFAULT_DATASET = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
DEFAULT_OUTPUT = RESEARCH_ROOT / "myExperiment1_global_learnable_interpolation_results"

PATIENT_SETTINGS = (1, 2, 3)
SEEDS = (0, 1, 2, 3, 4)
VENDORS = ("A", "B", "C", "D")
FOREGROUND_CLASSES = (1, 2, 3)
NUM_CLASSES = 4

LEARNED_TENT = "GlobalLearnableInterpolation-TENT"
LEARNED_BALANCED = "GlobalLearnableInterpolation-Balanced"
GRID_TENT = "GlobalGridSelection-TENT"
GRID_BALANCED = "GlobalGridSelection-Balanced"
DICE_ORACLE = "GlobalGridOracle-Dice"

METHOD_DIRECTORY_NAMES = {
    LEARNED_TENT: "learned_tent",
    LEARNED_BALANCED: "learned_balanced",
    GRID_TENT: "grid_tent_selected",
    GRID_BALANCED: "grid_balanced_selected",
    DICE_ORACLE: "dice_oracle",
}

RUN_SUMMARY_FIELDS = (
    "method",
    "method_type",
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "vendor_name",
    "domain",
    "checkpoint",
    "batch_size",
    "objective",
    "initial_s",
    "learning_rate",
    "adaptation_steps",
    "grid_step",
    "n_patients",
    "n_slices",
    "n_alignment_slices",
    "mean_final_s",
    "std_final_s",
    "mean_initial_loss",
    "mean_final_loss",
    "dice_rv",
    "dice_myo",
    "dice_lv",
    "dice_mean",
    "hd95_rv",
    "hd95_myo",
    "hd95_lv",
    "hd95_mean",
    "output_dir",
)

GRID_SUMMARY_FIELDS = (
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "domain",
    "grid_index",
    "s",
    "n_slices",
    "mean_tent_entropy",
    "mean_balanced_entropy",
    "n_patients",
    "dice_rv",
    "dice_myo",
    "dice_lv",
    "dice_mean",
    "hd95_rv",
    "hd95_myo",
    "hd95_lv",
    "hd95_mean",
)

ALIGNMENT_FIELDS = (
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "domain",
    "patient_id",
    "phase",
    "z_index",
    "slice_id",
    "has_foreground",
    "n_present_classes",
    "tent_grid_s",
    "balanced_grid_s",
    "dice_oracle_s",
    "tent_grid_dice",
    "balanced_grid_dice",
    "dice_oracle_value",
    "tent_grid_gap_to_oracle",
    "balanced_grid_gap_to_oracle",
    "spearman_neg_tent_entropy_dice",
    "spearman_neg_balanced_entropy_dice",
    "tent_learned_initial_s",
    "tent_learned_final_s",
    "tent_learned_initial_loss",
    "tent_learned_final_loss",
    "tent_learned_dice",
    "tent_learned_gap_to_oracle",
    "balanced_learned_initial_s",
    "balanced_learned_final_s",
    "balanced_learned_initial_loss",
    "balanced_learned_final_loss",
    "balanced_learned_dice",
    "balanced_learned_gap_to_oracle",
)

ALIGNMENT_SUMMARY_FIELDS = (
    "task_id",
    "patient_setting",
    "seed",
    "vendor",
    "domain",
    "n_slices",
    "n_alignment_slices",
    "tent_exact_match_rate",
    "balanced_exact_match_rate",
    "mean_tent_grid_gap_to_oracle",
    "median_tent_grid_gap_to_oracle",
    "mean_balanced_grid_gap_to_oracle",
    "median_balanced_grid_gap_to_oracle",
    "mean_spearman_neg_tent_entropy_dice",
    "mean_spearman_neg_balanced_entropy_dice",
    "mean_tent_learned_s",
    "std_tent_learned_s",
    "mean_tent_learned_gap_to_oracle",
    "mean_balanced_learned_s",
    "std_balanced_learned_s",
    "mean_balanced_learned_gap_to_oracle",
    "task_tent_entropy_min_s",
    "task_balanced_entropy_min_s",
    "task_dice_max_s",
    "task_tent_gap_to_dice",
    "task_balanced_gap_to_dice",
    "task_spearman_neg_tent_entropy_dice",
    "task_spearman_neg_balanced_entropy_dice",
)


def task_coordinates(task_id: int) -> Tuple[int, int, str]:
    """Map one of 60 tasks to shot, seed, and MMS vendor."""
    if not 0 <= int(task_id) < 60:
        raise ValueError("task-id must be in [0, 59].")
    patient_setting = int(task_id) // 20 + 1
    remainder = int(task_id) % 20
    seed = remainder // 4
    vendor = VENDORS[remainder % 4]
    return patient_setting, seed, vendor


def checkpoint_path(root: Path, patient_setting: int, seed: int) -> Path:
    return (
        root
        / f"Patient{int(patient_setting)}"
        / f"Seed{int(seed)}"
        / "baseline_model_with_metadata.pt"
    )


def resolve_device(name: str) -> torch.device:
    text = str(name).strip().lower()
    if text == "auto":
        text = "cuda" if torch.cuda.is_available() else "cpu"
    device = torch.device(text)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def set_seed(seed: int, cuda: bool) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if cuda:
        torch.cuda.manual_seed_all(int(seed))
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, indent=2, sort_keys=True)
        handle.write("\n")
    temporary.replace(path)


def write_csv(
    path: Path,
    rows: Sequence[Mapping[str, Any]],
    fields: Sequence[str],
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=list(fields),
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(dict(row) for row in rows)
    temporary.replace(path)


def read_csv(path: Path) -> List[Dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def finite_values(values: Iterable[Any]) -> List[float]:
    output: List[float] = []
    for value in values:
        if value is None or value == "":
            continue
        number = float(value)
        if math.isfinite(number):
            output.append(number)
    return output


def finite_mean(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(sum(xs) / len(xs)) if xs else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    if len(xs) <= 1:
        return 0.0 if xs else float("nan")
    return float(np.std(np.asarray(xs, dtype=np.float64), ddof=1))


def finite_median(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(np.median(np.asarray(xs, dtype=np.float64))) if xs else float("nan")


def safe_spearman(first: Sequence[float], second: Sequence[float]) -> float:
    pairs = [
        (float(a), float(b))
        for a, b in zip(first, second)
        if math.isfinite(float(a)) and math.isfinite(float(b))
    ]
    if len(pairs) < 2:
        return float("nan")
    a_values = [item[0] for item in pairs]
    b_values = [item[1] for item in pairs]
    if len(set(a_values)) <= 1 or len(set(b_values)) <= 1:
        return float("nan")
    result = spearmanr(a_values, b_values)
    return float(result.statistic)


def make_grid(step: float) -> Tuple[float, ...]:
    step_value = float(step)
    if not math.isfinite(step_value) or step_value <= 0.0 or step_value > 1.0:
        raise ValueError("grid-step must be finite and in (0, 1].")
    count_float = 1.0 / step_value
    count = int(round(count_float))
    if not math.isclose(count * step_value, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise ValueError("grid-step must divide 1 exactly.")
    return tuple(float(index / count) for index in range(count + 1))


def logit(value: float) -> float:
    number = float(value)
    if not 0.0 < number < 1.0:
        raise ValueError("initial-s must be strictly between 0 and 1.")
    return float(math.log(number / (1.0 - number)))


class GlobalInterpolationState:
    """Shared scalar and per-slice target statistics for every BN layer."""

    def __init__(self, device: torch.device, dtype: torch.dtype = torch.float32) -> None:
        self.a = nn.Parameter(torch.zeros((), device=device, dtype=dtype))
        self.mode = "fixed"
        self.fixed_s = 0.5
        self.target_statistics: Dict[str, Tuple[torch.Tensor, torch.Tensor]] = {}

    def reset_parameter(self, initial_s: float) -> None:
        with torch.no_grad():
            self.a.fill_(logit(initial_s))

    def use_capture(self) -> None:
        self.target_statistics.clear()
        self.mode = "capture"

    def use_fixed(self, value: float) -> None:
        number = float(value)
        if not 0.0 <= number <= 1.0:
            raise ValueError("Fixed interpolation fraction must be in [0, 1].")
        self.fixed_s = number
        self.mode = "fixed"

    def use_learned(self) -> None:
        self.mode = "learned"

    def interpolation_fraction(
        self,
        *,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        if self.mode == "learned":
            return torch.sigmoid(self.a).to(device=device, dtype=dtype)
        if self.mode == "fixed":
            return torch.as_tensor(self.fixed_s, device=device, dtype=dtype)
        raise RuntimeError("Interpolation fraction is unavailable in capture mode.")


class InterpolatedBatchNorm2d(nn.Module):
    """Frozen BatchNorm2d with differentiable source/target statistic mixing."""

    def __init__(
        self,
        original: nn.BatchNorm2d,
        *,
        layer_name: str,
        state: GlobalInterpolationState,
    ) -> None:
        super().__init__()
        if original.running_mean is None or original.running_var is None:
            raise ValueError(f"BatchNorm layer {layer_name!r} has no source statistics.")
        self.layer_name = str(layer_name)
        self.state = state
        self.eps = float(original.eps)
        self.register_buffer(
            "source_mean",
            original.running_mean.detach().clone(),
        )
        self.register_buffer(
            "source_var",
            original.running_var.detach().clone(),
        )
        if original.affine:
            self.weight = nn.Parameter(
                original.weight.detach().clone(),
                requires_grad=False,
            )
            self.bias = nn.Parameter(
                original.bias.detach().clone(),
                requires_grad=False,
            )
        else:
            self.register_parameter("weight", None)
            self.register_parameter("bias", None)

    @staticmethod
    def _moments(inputs: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        dimensions = (0, 2, 3)
        mean = inputs.mean(dim=dimensions)
        variance = inputs.var(dim=dimensions, unbiased=False)
        return mean, variance

    def _affine(self, normalized: torch.Tensor) -> torch.Tensor:
        output = normalized
        if self.weight is not None:
            output = output * self.weight.reshape(1, -1, 1, 1)
        if self.bias is not None:
            output = output + self.bias.reshape(1, -1, 1, 1)
        return output

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        if inputs.ndim != 4:
            raise ValueError(
                f"InterpolatedBatchNorm2d expects [B,C,H,W], got {tuple(inputs.shape)}"
            )
        if self.state.mode == "capture":
            target_mean, target_var = self._moments(inputs)
            if (
                not bool(torch.isfinite(target_mean).all())
                or not bool(torch.isfinite(target_var).all())
            ):
                raise RuntimeError(
                    f"Non-finite target BN statistics at layer {self.layer_name}"
                )
            self.state.target_statistics[self.layer_name] = (
                target_mean.detach().clone(),
                target_var.detach().clone(),
            )
            normalized = (
                inputs - target_mean.reshape(1, -1, 1, 1)
            ) * torch.rsqrt(
                target_var.reshape(1, -1, 1, 1) + self.eps
            )
            return self._affine(normalized)

        if self.layer_name not in self.state.target_statistics:
            raise RuntimeError(
                f"No target statistics captured for layer {self.layer_name}"
            )
        target_mean, target_var = self.state.target_statistics[self.layer_name]
        target_mean = target_mean.to(device=inputs.device, dtype=inputs.dtype)
        target_var = target_var.to(device=inputs.device, dtype=inputs.dtype)
        source_mean = self.source_mean.to(device=inputs.device, dtype=inputs.dtype)
        source_var = self.source_var.to(device=inputs.device, dtype=inputs.dtype)
        fraction = self.state.interpolation_fraction(
            device=inputs.device,
            dtype=inputs.dtype,
        )

        mixed_mean = source_mean + fraction * (target_mean - source_mean)
        source_log_sigma = 0.5 * torch.log(source_var.clamp_min(0.0) + self.eps)
        target_log_sigma = 0.5 * torch.log(target_var.clamp_min(0.0) + self.eps)
        mixed_log_sigma = source_log_sigma + fraction * (
            target_log_sigma - source_log_sigma
        )
        normalized = (
            inputs - mixed_mean.reshape(1, -1, 1, 1)
        ) * torch.exp(-mixed_log_sigma).reshape(1, -1, 1, 1)
        return self._affine(normalized)


def replace_batch_norms(
    module: nn.Module,
    state: GlobalInterpolationState,
    *,
    prefix: str = "",
) -> List[str]:
    """Replace every BatchNorm2d in place and return its fully qualified name."""
    names: List[str] = []
    for child_name, child in list(module.named_children()):
        full_name = f"{prefix}.{child_name}" if prefix else child_name
        if isinstance(child, nn.BatchNorm2d):
            setattr(
                module,
                child_name,
                InterpolatedBatchNorm2d(
                    child,
                    layer_name=full_name,
                    state=state,
                ),
            )
            names.append(full_name)
        else:
            names.extend(replace_batch_norms(child, state, prefix=full_name))
    return names


def capture_target_statistics(
    model: nn.Module,
    state: GlobalInterpolationState,
    image: torch.Tensor,
    expected_layers: Sequence[str],
) -> None:
    state.use_capture()
    with torch.no_grad():
        logits = model_logits(model(image))
    if not bool(torch.isfinite(logits).all()):
        raise RuntimeError("Non-finite logits during target-statistic capture.")
    captured = set(state.target_statistics)
    expected = set(str(name) for name in expected_layers)
    if captured != expected:
        raise RuntimeError(
            f"Target BN capture mismatch: missing={sorted(expected-captured)} "
            f"unexpected={sorted(captured-expected)}"
        )


def probabilities_for_current_state(
    model: nn.Module,
    image: torch.Tensor,
) -> torch.Tensor:
    logits = model_logits(model(image))
    if logits.ndim != 4 or logits.shape[0] != 1 or logits.shape[1] != NUM_CLASSES:
        raise ValueError(
            f"Expected logits [1,{NUM_CLASSES},H,W], got {tuple(logits.shape)}"
        )
    probabilities = F.softmax(logits, dim=1)
    if not bool(torch.isfinite(probabilities).all()):
        raise RuntimeError("Non-finite probabilities.")
    return probabilities


def entropy_loss(
    probabilities: torch.Tensor,
    objective: str,
    *,
    background_index: int,
    eps: float,
) -> torch.Tensor:
    if objective == "tent":
        return categorical_entropy(probabilities, eps=eps).mean()
    if objective == "balanced":
        return balanced_fg_bg_entropy(
            probabilities,
            background_index=background_index,
            eps=eps,
        ).mean()
    raise ValueError(f"Unsupported objective: {objective}")


@dataclass(frozen=True)
class LearnedSliceResult:
    objective: str
    initial_s: float
    final_s: float
    initial_loss: float
    final_loss: float
    probabilities: torch.Tensor


def optimize_slice(
    model: nn.Module,
    state: GlobalInterpolationState,
    image: torch.Tensor,
    *,
    objective: str,
    initial_s: float,
    learning_rate: float,
    adaptation_steps: int,
    background_index: int,
    entropy_eps: float,
) -> LearnedSliceResult:
    state.reset_parameter(initial_s)
    state.use_learned()
    optimizer = torch.optim.Adam([state.a], lr=float(learning_rate))
    initial_loss = float("nan")
    best_loss = float("inf")
    best_a = state.a.detach().clone()

    for step in range(int(adaptation_steps)):
        optimizer.zero_grad(set_to_none=True)
        probabilities = probabilities_for_current_state(model, image)
        loss = entropy_loss(
            probabilities,
            objective,
            background_index=background_index,
            eps=entropy_eps,
        )
        if not bool(torch.isfinite(loss)):
            raise RuntimeError(f"Non-finite {objective} loss at step {step}.")
        loss_value = float(loss.detach().item())
        if step == 0:
            initial_loss = loss_value
        if loss_value < best_loss:
            best_loss = loss_value
            best_a = state.a.detach().clone()
        loss.backward()
        if state.a.grad is None or not bool(torch.isfinite(state.a.grad)):
            raise RuntimeError(f"Invalid gradient for global a under {objective}.")
        optimizer.step()
        if not bool(torch.isfinite(state.a)):
            raise RuntimeError(f"Non-finite global a after {objective} update.")

    state.use_learned()
    with torch.no_grad():
        candidate_probabilities = probabilities_for_current_state(model, image)
        candidate_loss = entropy_loss(
            candidate_probabilities,
            objective,
            background_index=background_index,
            eps=entropy_eps,
        )
        if not bool(torch.isfinite(candidate_loss)):
            raise RuntimeError(f"Non-finite final {objective} loss.")
        if float(candidate_loss.item()) < best_loss:
            best_a = state.a.detach().clone()

        # Adam can overshoot in the one-dimensional problem.  Report the
        # lowest-entropy iterate (including the initialization and last step),
        # rather than an arbitrarily worse final optimizer iterate.
        state.a.copy_(best_a)
        final_probabilities = probabilities_for_current_state(model, image)
        final_loss_tensor = entropy_loss(
            final_probabilities,
            objective,
            background_index=background_index,
            eps=entropy_eps,
        )
        final_s = float(torch.sigmoid(state.a).item())
    return LearnedSliceResult(
        objective=objective,
        initial_s=float(initial_s),
        final_s=final_s,
        initial_loss=initial_loss,
        final_loss=float(final_loss_tensor.item()),
        probabilities=final_probabilities.detach(),
    )


def present_class_macro_dice(
    prediction: torch.Tensor,
    target: torch.Tensor,
    *,
    eps: float = 1e-6,
) -> Tuple[float, int]:
    pred = prediction.detach().cpu().long().squeeze()
    truth = target.detach().cpu().long().squeeze()
    values: List[float] = []
    for cls in FOREGROUND_CLASSES:
        truth_class = truth == int(cls)
        if not bool(truth_class.any()):
            continue
        pred_class = pred == int(cls)
        denominator = float(pred_class.sum().item() + truth_class.sum().item())
        intersection = float(torch.logical_and(pred_class, truth_class).sum().item())
        values.append((2.0 * intersection + eps) / (denominator + eps))
    return finite_mean(values), len(values)


def first_argmin(values: Sequence[float]) -> int:
    finite = [
        (index, float(value))
        for index, value in enumerate(values)
        if math.isfinite(float(value))
    ]
    if not finite:
        raise ValueError("No finite value for argmin.")
    return min(finite, key=lambda item: (item[1], item[0]))[0]


def first_argmax(values: Sequence[float]) -> int:
    finite = [
        (index, float(value))
        for index, value in enumerate(values)
        if math.isfinite(float(value))
    ]
    if not finite:
        raise ValueError("No finite value for argmax.")
    return max(finite, key=lambda item: (item[1], -item[0]))[0]


def load_backbone(
    path: Path,
    patient_setting: int,
    seed: int,
    device: torch.device,
) -> Tuple[UNet, Dict[str, Any]]:
    checkpoint = torch.load(path, map_location="cpu")
    if not isinstance(checkpoint, Mapping):
        raise TypeError(f"Checkpoint is not a mapping: {path}")
    metadata = dict(checkpoint.get("metadata", {}))
    if int(metadata.get("labeled_cases_per_class", patient_setting)) != int(
        patient_setting
    ):
        raise ValueError(f"Patient setting mismatch in {path}")
    if int(metadata.get("seed", seed)) != int(seed):
        raise ValueError(f"Seed mismatch in {path}")
    weights = checkpoint.get(
        "teacher_state_dict",
        checkpoint.get("model_state_dict"),
    )
    if weights is None:
        raise KeyError(f"No teacher/model state dict in {path}")
    model = UNet(
        n_channels=1,
        n_classes=NUM_CLASSES,
        only_feature=False,
        bilinear=False,
    )
    model.load_state_dict(weights, strict=True)
    model.to(device)
    model.eval()
    model.requires_grad_(False)
    return model, metadata


def group_by_patient(
    records: Sequence[Any],
    max_patients: int,
    max_slices: int,
) -> List[List[Any]]:
    groups: Dict[str, List[Any]] = {}
    order: List[str] = []
    for record in records:
        patient_id = str(record.patient_id)
        if patient_id not in groups:
            groups[patient_id] = []
            order.append(patient_id)
        groups[patient_id].append(record)
    selected = [groups[patient_id] for patient_id in order]
    if int(max_patients) > 0:
        selected = selected[: int(max_patients)]
    if int(max_slices) > 0:
        remaining = int(max_slices)
        limited: List[List[Any]] = []
        for group in selected:
            if remaining <= 0:
                break
            current = group[:remaining]
            if current:
                limited.append(current)
                remaining -= len(current)
        selected = limited
    return selected


def validate_inputs(backbone_root: Path, dataset_root: Path) -> None:
    expected = {
        checkpoint_path(backbone_root, patient, seed).resolve()
        for patient in PATIENT_SETTINGS
        for seed in SEEDS
    }
    discovered = {
        path.resolve()
        for path in backbone_root.glob(
            "Patient*/Seed*/baseline_model_with_metadata.pt"
        )
    }
    missing = sorted(str(path) for path in expected - discovered)
    unexpected = sorted(str(path) for path in discovered - expected)
    if len(discovered) != 15 or missing or unexpected:
        raise RuntimeError(
            "Expected exactly 15 backbones; "
            f"found={len(discovered)} missing={missing} unexpected={unexpected}"
        )
    if not dataset_root.is_dir():
        raise FileNotFoundError(f"MMS root does not exist: {dataset_root}")
    for vendor in VENDORS:
        TestLoader(vendor=vendor, batch_size=1, dataset_root=dataset_root)


def evaluator_summary_row(
    evaluator: PatientStreamEvaluator,
    *,
    method: str,
    method_type: str,
    objective: str,
    task_id: int,
    patient_setting: int,
    seed: int,
    loader: TestLoader,
    checkpoint: Path,
    task_dir: Path,
    n_slices: int,
    n_alignment_slices: int,
    args: argparse.Namespace,
    learned_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Any]:
    metrics = evaluator.seed_summary()
    output_dir = task_dir / "methods" / METHOD_DIRECTORY_NAMES[method]
    final_s_values: List[float] = []
    initial_losses: List[float] = []
    final_losses: List[float] = []
    if objective in ("tent", "balanced") and method.startswith(
        "GlobalLearnableInterpolation"
    ):
        prefix = f"{objective}_learned_"
        final_s_values = finite_values(
            row.get(prefix + "final_s") for row in learned_rows
        )
        initial_losses = finite_values(
            row.get(prefix + "initial_loss") for row in learned_rows
        )
        final_losses = finite_values(
            row.get(prefix + "final_loss") for row in learned_rows
        )
    return {
        "method": method,
        "method_type": method_type,
        "task_id": int(task_id),
        "patient_setting": int(patient_setting),
        "seed": int(seed),
        "vendor": loader.vendor,
        "vendor_name": loader.vendor_name,
        "domain": loader.domain,
        "checkpoint": str(checkpoint.resolve()),
        "batch_size": 1,
        "objective": objective,
        "initial_s": float(args.initial_s)
        if method.startswith("GlobalLearnableInterpolation")
        else "",
        "learning_rate": float(args.learning_rate)
        if method.startswith("GlobalLearnableInterpolation")
        else "",
        "adaptation_steps": int(args.adaptation_steps)
        if method.startswith("GlobalLearnableInterpolation")
        else 0,
        "grid_step": float(args.grid_step),
        "n_patients": int(metrics["n_patients"]),
        "n_slices": int(n_slices),
        "n_alignment_slices": int(n_alignment_slices),
        "mean_final_s": finite_mean(final_s_values),
        "std_final_s": finite_std(final_s_values),
        "mean_initial_loss": finite_mean(initial_losses),
        "mean_final_loss": finite_mean(final_losses),
        "dice_rv": metrics["dice_rv"],
        "dice_myo": metrics["dice_myo"],
        "dice_lv": metrics["dice_lv"],
        "dice_mean": metrics["dice_mean"],
        "hd95_rv": metrics["hd95_rv"],
        "hd95_myo": metrics["hd95_myo"],
        "hd95_lv": metrics["hd95_lv"],
        "hd95_mean": metrics["hd95_mean"],
        "output_dir": str(output_dir.resolve()),
    }


def make_grid_summary(
    *,
    grid: Sequence[float],
    grid_evaluators: Sequence[PatientStreamEvaluator],
    entropy_values: Mapping[str, Sequence[Sequence[float]]],
    task_id: int,
    patient_setting: int,
    seed: int,
    vendor: str,
    domain: str,
) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for index, (fraction, evaluator) in enumerate(zip(grid, grid_evaluators)):
        metrics = evaluator.seed_summary()
        rows.append(
            {
                "task_id": int(task_id),
                "patient_setting": int(patient_setting),
                "seed": int(seed),
                "vendor": str(vendor),
                "domain": str(domain),
                "grid_index": int(index),
                "s": float(fraction),
                "n_slices": len(entropy_values["tent"][index]),
                "mean_tent_entropy": finite_mean(
                    entropy_values["tent"][index]
                ),
                "mean_balanced_entropy": finite_mean(
                    entropy_values["balanced"][index]
                ),
                "n_patients": int(metrics["n_patients"]),
                "dice_rv": metrics["dice_rv"],
                "dice_myo": metrics["dice_myo"],
                "dice_lv": metrics["dice_lv"],
                "dice_mean": metrics["dice_mean"],
                "hd95_rv": metrics["hd95_rv"],
                "hd95_myo": metrics["hd95_myo"],
                "hd95_lv": metrics["hd95_lv"],
                "hd95_mean": metrics["hd95_mean"],
            }
        )
    return rows


def make_alignment_summary(
    rows: Sequence[Mapping[str, Any]],
    grid_rows: Sequence[Mapping[str, Any]],
    *,
    task_id: int,
    patient_setting: int,
    seed: int,
    vendor: str,
    domain: str,
) -> Dict[str, Any]:
    eligible = [
        row
        for row in rows
        if int(row.get("has_foreground", 0) or 0) == 1
    ]
    task_tent_index = first_argmin(
        [float(row["mean_tent_entropy"]) for row in grid_rows]
    )
    task_balanced_index = first_argmin(
        [float(row["mean_balanced_entropy"]) for row in grid_rows]
    )
    task_s_values = [float(row["s"]) for row in grid_rows]
    task_tent_s = task_s_values[task_tent_index]
    task_balanced_s = task_s_values[task_balanced_index]
    task_dice_values = [float(row["dice_mean"]) for row in grid_rows]
    if finite_values(task_dice_values):
        task_dice_index = first_argmax(task_dice_values)
        task_dice_s = task_s_values[task_dice_index]
    else:
        # A deliberately tiny smoke subset can contain only empty-background
        # slices.  Preserve entropy diagnostics while marking the supervised
        # task-level optimum as unavailable.
        task_dice_s = float("nan")
    return {
        "task_id": int(task_id),
        "patient_setting": int(patient_setting),
        "seed": int(seed),
        "vendor": str(vendor),
        "domain": str(domain),
        "n_slices": len(rows),
        "n_alignment_slices": len(eligible),
        "tent_exact_match_rate": finite_mean(
            math.isclose(
                float(row["tent_grid_s"]),
                float(row["dice_oracle_s"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for row in eligible
        ),
        "balanced_exact_match_rate": finite_mean(
            math.isclose(
                float(row["balanced_grid_s"]),
                float(row["dice_oracle_s"]),
                rel_tol=0.0,
                abs_tol=1e-9,
            )
            for row in eligible
        ),
        "mean_tent_grid_gap_to_oracle": finite_mean(
            row["tent_grid_gap_to_oracle"] for row in eligible
        ),
        "median_tent_grid_gap_to_oracle": finite_median(
            row["tent_grid_gap_to_oracle"] for row in eligible
        ),
        "mean_balanced_grid_gap_to_oracle": finite_mean(
            row["balanced_grid_gap_to_oracle"] for row in eligible
        ),
        "median_balanced_grid_gap_to_oracle": finite_median(
            row["balanced_grid_gap_to_oracle"] for row in eligible
        ),
        "mean_spearman_neg_tent_entropy_dice": finite_mean(
            row["spearman_neg_tent_entropy_dice"] for row in eligible
        ),
        "mean_spearman_neg_balanced_entropy_dice": finite_mean(
            row["spearman_neg_balanced_entropy_dice"] for row in eligible
        ),
        "mean_tent_learned_s": finite_mean(
            row["tent_learned_final_s"] for row in rows
        ),
        "std_tent_learned_s": finite_std(
            row["tent_learned_final_s"] for row in rows
        ),
        "mean_tent_learned_gap_to_oracle": finite_mean(
            row["tent_learned_gap_to_oracle"] for row in eligible
        ),
        "mean_balanced_learned_s": finite_mean(
            row["balanced_learned_final_s"] for row in rows
        ),
        "std_balanced_learned_s": finite_std(
            row["balanced_learned_final_s"] for row in rows
        ),
        "mean_balanced_learned_gap_to_oracle": finite_mean(
            row["balanced_learned_gap_to_oracle"] for row in eligible
        ),
        "task_tent_entropy_min_s": task_tent_s,
        "task_balanced_entropy_min_s": task_balanced_s,
        "task_dice_max_s": task_dice_s,
        "task_tent_gap_to_dice": abs(task_tent_s - task_dice_s)
        if math.isfinite(task_dice_s)
        else float("nan"),
        "task_balanced_gap_to_dice": abs(task_balanced_s - task_dice_s)
        if math.isfinite(task_dice_s)
        else float("nan"),
        "task_spearman_neg_tent_entropy_dice": safe_spearman(
            [-float(row["mean_tent_entropy"]) for row in grid_rows],
            [float(row["dice_mean"]) for row in grid_rows],
        ),
        "task_spearman_neg_balanced_entropy_dice": safe_spearman(
            [-float(row["mean_balanced_entropy"]) for row in grid_rows],
            [float(row["dice_mean"]) for row in grid_rows],
        ),
    }


def save_method_evaluators(
    task_dir: Path,
    evaluators: Mapping[str, PatientStreamEvaluator],
) -> None:
    for method, evaluator in evaluators.items():
        evaluator.save_csv(
            task_dir / "methods" / METHOD_DIRECTORY_NAMES[method]
        )


def run_task(args: argparse.Namespace) -> Dict[str, Any]:
    task_id = int(args.task_id)
    patient_setting, seed, vendor = task_coordinates(task_id)
    device = resolve_device(args.device)
    set_seed(seed, cuda=device.type == "cuda")
    grid = make_grid(args.grid_step)

    output_root = Path(args.output_root)
    task_dir = output_root / "shards" / f"task_{task_id}"
    completion_path = task_dir / "completion.json"
    if completion_path.is_file() and args.resume and not args.overwrite:
        with completion_path.open(encoding="utf-8") as handle:
            payload = json.load(handle)
        if payload.get("status") != "complete":
            raise ValueError(f"Invalid completion status in {completion_path}")
        print(f"[SKIP] complete task={task_id}", flush=True)
        return dict(payload)
    if task_dir.exists() and not args.resume and not args.overwrite:
        raise FileExistsError(
            f"Task output exists; use --resume or --overwrite: {task_dir}"
        )
    if task_dir.exists() and args.overwrite:
        shutil.rmtree(task_dir)

    backbone = checkpoint_path(
        Path(args.backbone_root),
        patient_setting,
        seed,
    )
    loader = TestLoader(
        vendor=vendor,
        batch_size=1,
        shuffle_all_slices=False,
        seed=seed,
        dataset_root=Path(args.dataset_root),
    )
    patient_groups = group_by_patient(
        loader.records,
        max_patients=int(args.max_patients),
        max_slices=int(args.max_slices),
    )
    if not patient_groups:
        raise RuntimeError("The selected test stream is empty.")

    model, metadata = load_backbone(
        backbone,
        patient_setting,
        seed,
        device,
    )
    state = GlobalInterpolationState(device=device)
    bn_layer_names = replace_batch_norms(model, state)
    model.eval()
    model.requires_grad_(False)
    if not bn_layer_names:
        raise RuntimeError("No BatchNorm2d layers found.")
    if state.a.requires_grad is not True:
        raise AssertionError("Global a must remain trainable.")

    grid_evaluators = [
        PatientStreamEvaluator(
            domain=loader.domain,
            seed=seed,
            backbone_id=(
                f"MeanTeacher_Patient{patient_setting}_Seed{seed}"
                f"_GlobalFixedS{fraction:.2f}"
            ),
        )
        for fraction in grid
    ]
    method_evaluators = {
        method: PatientStreamEvaluator(
            domain=loader.domain,
            seed=seed,
            backbone_id=(
                f"MeanTeacher_Patient{patient_setting}_Seed{seed}_{method}"
            ),
        )
        for method in METHOD_DIRECTORY_NAMES
    }
    entropy_values: Dict[str, List[List[float]]] = {
        "tent": [[] for _ in grid],
        "balanced": [[] for _ in grid],
    }
    alignment_rows: List[Dict[str, Any]] = []
    n_slices = 0

    print(
        f"[RUN] task={task_id} shot={patient_setting} seed={seed} "
        f"domain={loader.domain} patients={len(patient_groups)} "
        f"grid={len(grid)} objectives=tent|balanced "
        f"bn_layers={len(bn_layer_names)} device={device}",
        flush=True,
    )

    for patient_step, records in enumerate(patient_groups, start=1):
        grid_patient_predictions: List[List[torch.Tensor]] = [
            [] for _ in grid
        ]
        method_patient_predictions: Dict[str, List[torch.Tensor]] = {
            method: [] for method in METHOD_DIRECTORY_NAMES
        }

        for record in records:
            n_slices += 1
            image = _stack_images([record]).to(
                device,
                non_blocking=device.type == "cuda",
            )
            mask = _stack_masks([record])
            capture_target_statistics(
                model,
                state,
                image,
                bn_layer_names,
            )

            grid_probabilities: List[torch.Tensor] = []
            grid_tent_entropies: List[float] = []
            grid_balanced_entropies: List[float] = []
            grid_dice_values: List[float] = []
            present_count = 0

            with torch.no_grad():
                for grid_index, fraction in enumerate(grid):
                    state.use_fixed(fraction)
                    probabilities = probabilities_for_current_state(
                        model,
                        image,
                    ).detach()
                    grid_probabilities.append(probabilities)
                    tent_value = float(
                        entropy_loss(
                            probabilities,
                            "tent",
                            background_index=int(args.background_index),
                            eps=float(args.entropy_eps),
                        ).item()
                    )
                    balanced_value = float(
                        entropy_loss(
                            probabilities,
                            "balanced",
                            background_index=int(args.background_index),
                            eps=float(args.entropy_eps),
                        ).item()
                    )
                    prediction = torch.argmax(
                        probabilities,
                        dim=1,
                    ).detach().cpu().to(torch.uint8)
                    dice_value, current_present_count = (
                        present_class_macro_dice(prediction, mask)
                    )
                    present_count = current_present_count
                    grid_tent_entropies.append(tent_value)
                    grid_balanced_entropies.append(balanced_value)
                    grid_dice_values.append(dice_value)
                    entropy_values["tent"][grid_index].append(tent_value)
                    entropy_values["balanced"][grid_index].append(
                        balanced_value
                    )
                    grid_patient_predictions[grid_index].append(prediction[0])

            tent_grid_index = first_argmin(grid_tent_entropies)
            balanced_grid_index = first_argmin(grid_balanced_entropies)
            has_foreground = present_count > 0
            dice_oracle_index = (
                first_argmax(grid_dice_values)
                if has_foreground
                else tent_grid_index
            )

            learned_tent = optimize_slice(
                model,
                state,
                image,
                objective="tent",
                initial_s=float(args.initial_s),
                learning_rate=float(args.learning_rate),
                adaptation_steps=int(args.adaptation_steps),
                background_index=int(args.background_index),
                entropy_eps=float(args.entropy_eps),
            )
            learned_balanced = optimize_slice(
                model,
                state,
                image,
                objective="balanced",
                initial_s=float(args.initial_s),
                learning_rate=float(args.learning_rate),
                adaptation_steps=int(args.adaptation_steps),
                background_index=int(args.background_index),
                entropy_eps=float(args.entropy_eps),
            )

            learned_tent_prediction = torch.argmax(
                learned_tent.probabilities,
                dim=1,
            ).detach().cpu().to(torch.uint8)
            learned_balanced_prediction = torch.argmax(
                learned_balanced.probabilities,
                dim=1,
            ).detach().cpu().to(torch.uint8)
            learned_tent_dice, _ = present_class_macro_dice(
                learned_tent_prediction,
                mask,
            )
            learned_balanced_dice, _ = present_class_macro_dice(
                learned_balanced_prediction,
                mask,
            )

            method_patient_predictions[LEARNED_TENT].append(
                learned_tent_prediction[0]
            )
            method_patient_predictions[LEARNED_BALANCED].append(
                learned_balanced_prediction[0]
            )
            method_patient_predictions[GRID_TENT].append(
                torch.argmax(
                    grid_probabilities[tent_grid_index],
                    dim=1,
                ).detach().cpu().to(torch.uint8)[0]
            )
            method_patient_predictions[GRID_BALANCED].append(
                torch.argmax(
                    grid_probabilities[balanced_grid_index],
                    dim=1,
                ).detach().cpu().to(torch.uint8)[0]
            )
            method_patient_predictions[DICE_ORACLE].append(
                torch.argmax(
                    grid_probabilities[dice_oracle_index],
                    dim=1,
                ).detach().cpu().to(torch.uint8)[0]
            )

            oracle_s = float(grid[dice_oracle_index]) if has_foreground else float("nan")
            row: Dict[str, Any] = {
                "task_id": task_id,
                "patient_setting": patient_setting,
                "seed": seed,
                "vendor": loader.vendor,
                "domain": loader.domain,
                "patient_id": record.patient_id,
                "phase": record.phase,
                "z_index": int(record.z_index),
                "slice_id": record.slice_id,
                "has_foreground": int(has_foreground),
                "n_present_classes": int(present_count),
                "tent_grid_s": float(grid[tent_grid_index]),
                "balanced_grid_s": float(grid[balanced_grid_index]),
                "dice_oracle_s": oracle_s,
                "tent_grid_dice": grid_dice_values[tent_grid_index],
                "balanced_grid_dice": grid_dice_values[balanced_grid_index],
                "dice_oracle_value": grid_dice_values[dice_oracle_index]
                if has_foreground
                else float("nan"),
                "tent_grid_gap_to_oracle": abs(
                    float(grid[tent_grid_index]) - oracle_s
                )
                if has_foreground
                else float("nan"),
                "balanced_grid_gap_to_oracle": abs(
                    float(grid[balanced_grid_index]) - oracle_s
                )
                if has_foreground
                else float("nan"),
                "spearman_neg_tent_entropy_dice": safe_spearman(
                    [-value for value in grid_tent_entropies],
                    grid_dice_values,
                )
                if has_foreground
                else float("nan"),
                "spearman_neg_balanced_entropy_dice": safe_spearman(
                    [-value for value in grid_balanced_entropies],
                    grid_dice_values,
                )
                if has_foreground
                else float("nan"),
                "tent_learned_initial_s": learned_tent.initial_s,
                "tent_learned_final_s": learned_tent.final_s,
                "tent_learned_initial_loss": learned_tent.initial_loss,
                "tent_learned_final_loss": learned_tent.final_loss,
                "tent_learned_dice": learned_tent_dice,
                "tent_learned_gap_to_oracle": abs(
                    learned_tent.final_s - oracle_s
                )
                if has_foreground
                else float("nan"),
                "balanced_learned_initial_s": learned_balanced.initial_s,
                "balanced_learned_final_s": learned_balanced.final_s,
                "balanced_learned_initial_loss": learned_balanced.initial_loss,
                "balanced_learned_final_loss": learned_balanced.final_loss,
                "balanced_learned_dice": learned_balanced_dice,
                "balanced_learned_gap_to_oracle": abs(
                    learned_balanced.final_s - oracle_s
                )
                if has_foreground
                else float("nan"),
            }
            alignment_rows.append(row)

        masks = _stack_masks(records)
        meta = [record.meta(include_mask_path=True) for record in records]
        for grid_index, evaluator in enumerate(grid_evaluators):
            evaluator.update(
                torch.stack(grid_patient_predictions[grid_index]).long(),
                masks,
                meta,
                step=patient_step,
            )
        for method, evaluator in method_evaluators.items():
            evaluator.update(
                torch.stack(method_patient_predictions[method]).long(),
                masks,
                meta,
                step=patient_step,
            )

        partial_grid_rows = make_grid_summary(
            grid=grid,
            grid_evaluators=grid_evaluators,
            entropy_values=entropy_values,
            task_id=task_id,
            patient_setting=patient_setting,
            seed=seed,
            vendor=loader.vendor,
            domain=loader.domain,
        )
        write_csv(
            task_dir / "alignment_rows.csv",
            alignment_rows,
            ALIGNMENT_FIELDS,
        )
        write_csv(
            task_dir / "grid_summary.csv",
            partial_grid_rows,
            GRID_SUMMARY_FIELDS,
        )
        save_method_evaluators(task_dir, method_evaluators)
        print(
            f"[PATIENT] {patient_step}/{len(patient_groups)} "
            f"id={records[0].patient_id} slices={len(records)}",
            flush=True,
        )

    grid_rows = make_grid_summary(
        grid=grid,
        grid_evaluators=grid_evaluators,
        entropy_values=entropy_values,
        task_id=task_id,
        patient_setting=patient_setting,
        seed=seed,
        vendor=loader.vendor,
        domain=loader.domain,
    )
    alignment_summary = make_alignment_summary(
        alignment_rows,
        grid_rows,
        task_id=task_id,
        patient_setting=patient_setting,
        seed=seed,
        vendor=loader.vendor,
        domain=loader.domain,
    )
    n_alignment_slices = int(alignment_summary["n_alignment_slices"])
    method_metadata = {
        LEARNED_TENT: ("learned", "tent"),
        LEARNED_BALANCED: ("learned", "balanced"),
        GRID_TENT: ("grid_selection", "tent"),
        GRID_BALANCED: ("grid_selection", "balanced"),
        DICE_ORACLE: ("oracle", "dice"),
    }
    run_rows = [
        evaluator_summary_row(
            evaluator,
            method=method,
            method_type=method_metadata[method][0],
            objective=method_metadata[method][1],
            task_id=task_id,
            patient_setting=patient_setting,
            seed=seed,
            loader=loader,
            checkpoint=backbone,
            task_dir=task_dir,
            n_slices=n_slices,
            n_alignment_slices=n_alignment_slices,
            args=args,
            learned_rows=alignment_rows,
        )
        for method, evaluator in method_evaluators.items()
    ]

    write_csv(task_dir / "run_summary.csv", run_rows, RUN_SUMMARY_FIELDS)
    write_csv(task_dir / "grid_summary.csv", grid_rows, GRID_SUMMARY_FIELDS)
    write_csv(
        task_dir / "alignment_summary.csv",
        [alignment_summary],
        ALIGNMENT_SUMMARY_FIELDS,
    )
    save_method_evaluators(task_dir, method_evaluators)
    write_json(
        task_dir / "run_config.json",
        {
            "task_id": task_id,
            "patient_setting": patient_setting,
            "seed": seed,
            "vendor": loader.vendor,
            "domain": loader.domain,
            "checkpoint": str(backbone.resolve()),
            "dataset_root": str(Path(args.dataset_root).resolve()),
            "output_root": str(output_root.resolve()),
            "bn_layer_names": bn_layer_names,
            "bn_layer_count": len(bn_layer_names),
            "grid": list(grid),
            "initial_s": float(args.initial_s),
            "learning_rate": float(args.learning_rate),
            "adaptation_steps": int(args.adaptation_steps),
            "objectives": ["tent", "balanced"],
            "target_variance_unbiased": False,
            "target_statistics_detached": True,
            "metadata": metadata,
            "torch_version": torch.__version__,
            "cuda_version": torch.version.cuda,
            "device": str(device),
        },
    )
    payload = {
        "status": "complete",
        "task_id": task_id,
        "run_summaries": run_rows,
        "alignment_summary": alignment_summary,
    }
    write_json(completion_path, payload)
    print(
        f"[COMPLETE] task={task_id} slices={n_slices} "
        f"tent_dice={run_rows[0]['dice_mean']:.6f} "
        f"balanced_dice={run_rows[1]['dice_mean']:.6f}",
        flush=True,
    )
    return payload


def aggregate(output_root: Path) -> None:
    run_rows: List[Dict[str, Any]] = []
    grid_rows: List[Dict[str, Any]] = []
    alignment_rows: List[Dict[str, Any]] = []
    missing: List[int] = []
    failures: Dict[str, str] = {}
    for task_id in range(60):
        task_dir = output_root / "shards" / f"task_{task_id}"
        completion_path = task_dir / "completion.json"
        if not completion_path.is_file():
            missing.append(task_id)
            continue
        try:
            with completion_path.open(encoding="utf-8") as handle:
                payload = json.load(handle)
            if payload.get("status") != "complete":
                raise ValueError(f"status={payload.get('status')}")
            current_run_rows = read_csv(task_dir / "run_summary.csv")
            current_grid_rows = read_csv(task_dir / "grid_summary.csv")
            current_alignment_rows = read_csv(
                task_dir / "alignment_summary.csv"
            )
            if len(current_run_rows) != len(METHOD_DIRECTORY_NAMES):
                raise ValueError(
                    f"run_summary rows={len(current_run_rows)}"
                )
            run_rows.extend(current_run_rows)
            grid_rows.extend(current_grid_rows)
            alignment_rows.extend(current_alignment_rows)
        except Exception as error:
            failures[str(task_id)] = f"{type(error).__name__}: {error}"

    run_rows.sort(
        key=lambda row: (
            int(row["task_id"]),
            str(row["method"]),
        )
    )
    grid_rows.sort(
        key=lambda row: (
            int(row["task_id"]),
            int(row["grid_index"]),
        )
    )
    alignment_rows.sort(key=lambda row: int(row["task_id"]))
    write_csv(output_root / "run_summary.csv", run_rows, RUN_SUMMARY_FIELDS)
    write_csv(output_root / "grid_summary.csv", grid_rows, GRID_SUMMARY_FIELDS)
    write_csv(
        output_root / "alignment_summary.csv",
        alignment_rows,
        ALIGNMENT_SUMMARY_FIELDS,
    )
    write_json(
        output_root / "missing_tasks.json",
        {
            "expected": 60,
            "complete": 60 - len(missing) - len(failures),
            "missing": missing,
            "failures": failures,
        },
    )
    print(
        f"[AGGREGATE] complete={60-len(missing)-len(failures)}/60 "
        f"missing={missing} failures={failures}",
        flush=True,
    )
    if missing or failures:
        raise RuntimeError("Global interpolation matrix is incomplete.")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Learn one per-slice global BN statistic interpolation parameter "
            "and compare entropy minima with Dice optima."
        )
    )
    parser.add_argument("--task-id", type=int)
    parser.add_argument("--aggregate-only", action="store_true")
    parser.add_argument("--backbone-root", default=str(DEFAULT_BACKBONES))
    parser.add_argument("--dataset-root", default=str(DEFAULT_DATASET))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT))
    parser.add_argument("--grid-step", type=float, default=0.05)
    parser.add_argument("--initial-s", type=float, default=0.5)
    parser.add_argument("--learning-rate", type=float, default=0.1)
    parser.add_argument("--adaptation-steps", type=int, default=10)
    parser.add_argument("--background-index", type=int, default=0)
    parser.add_argument("--entropy-eps", type=float, default=1e-12)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--max-patients", type=int, default=0)
    parser.add_argument("--max-slices", type=int, default=0)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def validate_args(args: argparse.Namespace) -> None:
    if args.resume and args.overwrite:
        raise ValueError("--resume and --overwrite are mutually exclusive.")
    if args.aggregate_only:
        if args.resume or args.overwrite:
            raise ValueError(
                "--resume/--overwrite do not apply to --aggregate-only."
            )
        return
    if args.task_id is None:
        raise ValueError("--task-id is required unless --aggregate-only is used.")
    task_coordinates(args.task_id)
    make_grid(args.grid_step)
    logit(args.initial_s)
    if not math.isfinite(float(args.learning_rate)) or args.learning_rate <= 0:
        raise ValueError("learning-rate must be finite and positive.")
    if int(args.adaptation_steps) <= 0:
        raise ValueError("adaptation-steps must be positive.")
    if not 0 <= int(args.background_index) < NUM_CLASSES:
        raise ValueError(f"background-index must be in [0,{NUM_CLASSES-1}].")
    if not math.isfinite(float(args.entropy_eps)) or args.entropy_eps <= 0:
        raise ValueError("entropy-eps must be finite and positive.")
    if min(int(args.max_patients), int(args.max_slices)) < 0:
        raise ValueError("max-patients/max-slices cannot be negative.")


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    validate_args(args)
    if args.aggregate_only:
        aggregate(Path(args.output_root))
        return
    validate_inputs(
        Path(args.backbone_root),
        Path(args.dataset_root),
    )
    device = resolve_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] name={torch.cuda.get_device_name(device)} "
            f"torch={torch.__version__} cuda={torch.version.cuda} "
            f"cudnn={torch.backends.cudnn.version()}",
            flush=True,
        )
    else:
        print(f"[CPU] torch={torch.__version__}", flush=True)
    run_task(args)


if __name__ == "__main__":
    try:
        main()
    except Exception as error:
        print(
            f"[ERROR] {type(error).__name__}: {error}",
            file=sys.stderr,
            flush=True,
        )
        raise
