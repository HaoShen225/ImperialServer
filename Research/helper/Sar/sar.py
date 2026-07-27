"""SAR utilities for online multiclass medical image segmentation.

This module adapts the sample-level reliable entropy filtering from the SAR
reference implementation to segmentation by treating each 2-D slice as one
sample: pixel entropies are averaged spatially before reliability filtering.
"""

from __future__ import annotations

from copy import deepcopy
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from .sam import SAM


ModelState = Dict[str, torch.Tensor]
OptimizerState = Dict[str, Any]
NORM_TYPES = (nn.BatchNorm2d, nn.GroupNorm, nn.LayerNorm)


def model_logits(output: Any) -> torch.Tensor:
    """Extract logits from tensor, sequence, or mapping model outputs."""
    if isinstance(output, torch.Tensor):
        return output
    if isinstance(output, (tuple, list)):
        if not output:
            raise ValueError("The model returned an empty sequence.")
        logits = output[-1]
    elif isinstance(output, Mapping):
        if "logits" not in output:
            raise KeyError("A mapping model output must contain a 'logits' key.")
        logits = output["logits"]
    else:
        raise TypeError(f"Unsupported model output type: {type(output).__name__}")
    if not isinstance(logits, torch.Tensor):
        raise TypeError("Extracted logits must be a torch.Tensor.")
    return logits


def softmax_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Return categorical entropy, preserving all non-class dimensions."""
    if logits.ndim < 2:
        raise ValueError("Logits must have a class dimension at index 1.")
    return -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)


def slice_entropy(logits: torch.Tensor) -> torch.Tensor:
    """Reduce pixel entropy to one reliability value per batch item."""
    entropy = softmax_entropy(logits)
    if entropy.ndim == 1:
        return entropy
    return entropy.flatten(start_dim=1).mean(dim=1)


def update_ema(ema: Optional[float], value: float, decay: float = 0.9) -> float:
    """Update the scalar entropy exponential moving average."""
    if not 0.0 <= float(decay) < 1.0:
        raise ValueError("EMA decay must satisfy 0 <= decay < 1.")
    return float(value) if ema is None else float(decay) * float(ema) + (1.0 - float(decay)) * float(value)


def configure_model(model: nn.Module) -> nn.Module:
    """Freeze the model and enable affine parameters of supported norms."""
    model.train()
    model.requires_grad_(False)
    for module in model.modules():
        if isinstance(module, NORM_TYPES):
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name, None)
                if parameter is not None:
                    parameter.requires_grad_(True)
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
    return model


def collect_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect trainable scale and shift parameters from supported norms."""
    params: List[nn.Parameter] = []
    names: List[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, NORM_TYPES):
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name, None)
            if parameter is not None and parameter.requires_grad:
                params.append(parameter)
                names.append(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    if not params:
        raise RuntimeError("SAR found no trainable normalization affine parameters.")
    return params, names


def check_model(model: nn.Module) -> None:
    """Validate that only supported normalization affine parameters train."""
    if not model.training:
        raise AssertionError("SAR requires train mode.")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError("SAR requires trainable normalization parameters.")
    valid_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, NORM_TYPES)
        for parameter in (getattr(module, "weight", None), getattr(module, "bias", None))
        if parameter is not None
    }
    invalid = [name for name, parameter in trainable if id(parameter) not in valid_ids]
    if invalid:
        raise AssertionError(f"Only normalization affine parameters may train: {', '.join(invalid)}")
    adaptive_norms = [
        module
        for module in model.modules()
        if isinstance(module, NORM_TYPES)
        and any(
            parameter is not None and parameter.requires_grad
            for parameter in (getattr(module, "weight", None), getattr(module, "bias", None))
        )
    ]
    if not adaptive_norms:
        raise AssertionError("SAR requires at least one adaptive normalization layer.")
    if any(isinstance(module, nn.BatchNorm2d) and module.track_running_stats for module in adaptive_norms):
        raise AssertionError("Adaptive BatchNorm2d layers must use current-batch statistics.")


def copy_model_and_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[ModelState, OptimizerState]:
    """Copy model and optimizer states for recovery or episodic reset."""
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: ModelState,
    optimizer_state: OptimizerState,
) -> None:
    """Restore model and optimizer states in place."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


@torch.enable_grad()
def forward_and_adapt_sar(
    images: torch.Tensor,
    model: nn.Module,
    optimizer: SAM,
    margin_e0: float,
    ema: Optional[float] = None,
    ema_decay: float = 0.9,
    reset_constant_em: float = 0.2,
) -> Tuple[torch.Tensor, Optional[float], bool, Dict[str, Any]]:
    """Run one reliable-entropy SAM update and report recovery diagnostics."""
    optimizer.zero_grad(set_to_none=True)
    logits = model_logits(model(images))
    entropy_first = slice_entropy(logits)
    reliable_first = torch.isfinite(entropy_first) & (entropy_first < float(margin_e0))
    stats: Dict[str, Any] = {
        "entropy_first": float(entropy_first.detach().mean().cpu()),
        "entropy_second": None,
        "reliable_count_first": int(reliable_first.sum().item()),
        "reliable_count_second": 0,
        "ema": ema,
        "sam_grad_norm": None,
        "updated": False,
        "reset_triggered": False,
    }
    if not bool(reliable_first.any()):
        return logits, ema, False, stats

    loss_first = entropy_first[reliable_first].mean()
    if not torch.isfinite(loss_first):
        return logits, ema, False, stats
    loss_first.backward()
    if not optimizer.first_step(zero_grad=True):
        stats["sam_grad_norm"] = optimizer.last_grad_norm
        return logits, ema, False, stats
    stats["sam_grad_norm"] = optimizer.last_grad_norm

    logits_second = model_logits(model(images))
    entropy_second = slice_entropy(logits_second)
    reliable_second = reliable_first & torch.isfinite(entropy_second) & (entropy_second < float(margin_e0))
    stats["entropy_second"] = float(entropy_second.detach().mean().cpu())
    stats["reliable_count_second"] = int(reliable_second.sum().item())
    if not bool(reliable_second.any()):
        optimizer.cancel_step(zero_grad=True)
        return logits, ema, False, stats

    loss_second = entropy_second[reliable_second].mean()
    if not torch.isfinite(loss_second):
        optimizer.cancel_step(zero_grad=True)
        return logits, ema, False, stats
    loss_second.backward()
    optimizer.second_step(zero_grad=True)
    stats["updated"] = True

    ema = update_ema(ema, float(loss_second.detach().cpu()), decay=ema_decay)
    reset_triggered = bool(ema < float(reset_constant_em))
    stats["ema"] = ema
    stats["reset_triggered"] = reset_triggered
    return logits, ema, reset_triggered, stats


class SAR(nn.Module):
    """Online Sharpness-Aware and Reliable entropy minimization wrapper."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: SAM,
        steps: int = 1,
        episodic: bool = False,
        margin_e0: Optional[float] = None,
        num_classes: int = 4,
        reset_constant_em: float = 0.2,
        ema_decay: float = 0.9,
    ) -> None:
        super().__init__()
        if int(steps) < 1:
            raise ValueError("SAR steps must be at least 1.")
        if int(num_classes) < 2:
            raise ValueError("SAR num_classes must be at least 2.")
        if not 0.0 <= float(ema_decay) < 1.0:
            raise ValueError("EMA decay must satisfy 0 <= decay < 1.")
        check_model(model)
        if not isinstance(optimizer, SAM):
            raise TypeError("SAR requires a SAM optimizer.")

        self.model = model
        self.optimizer = optimizer
        self.steps = int(steps)
        self.episodic = bool(episodic)
        self.margin_e0 = float(0.4 * math.log(num_classes) if margin_e0 is None else margin_e0)
        self.reset_constant_em = float(reset_constant_em)
        self.ema_decay = float(ema_decay)
        self.ema: Optional[float] = None
        self.last_stats: Dict[str, Any] = {}
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
        logits: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            logits, next_ema, reset_requested, stats = forward_and_adapt_sar(
                images,
                self.model,
                self.optimizer,
                self.margin_e0,
                self.ema,
                self.ema_decay,
                self.reset_constant_em,
            )
            self.last_stats = stats
            if reset_requested:
                self.reset()
                self.last_stats = dict(stats)
                self.last_stats["reset_triggered"] = True
            else:
                self.ema = next_ema
        if logits is None:
            raise RuntimeError("SAR did not execute an adaptation step.")
        return logits

    def reset(self) -> None:
        """Restore the initial source model and optimizer and clear entropy EMA."""
        load_model_and_optimizer(self.model, self.optimizer, self.model_state, self.optimizer_state)
        self.ema = None
        self.last_stats = {}


__all__: Sequence[str] = (
    "SAR",
    "check_model",
    "collect_params",
    "configure_model",
    "copy_model_and_optimizer",
    "forward_and_adapt_sar",
    "load_model_and_optimizer",
    "model_logits",
    "slice_entropy",
    "softmax_entropy",
    "update_ema",
)
