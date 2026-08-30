"""Only operations whose semantics are identical across multiple methods."""

from __future__ import annotations

from typing import Any, Iterable

import torch
from torch import nn


def pixel_entropy(logits: torch.Tensor) -> torch.Tensor:
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def slice_entropy(logits: torch.Tensor) -> torch.Tensor:
    return pixel_entropy(logits).mean(dim=(-2, -1))


def remove_bn_running_buffers(model: nn.Module) -> None:
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None
            module.num_batches_tracked = None


def configure_bn_for_batch_stats(model: nn.Module) -> None:
    model.train()
    model.requires_grad_(False)
    remove_bn_running_buffers(model)
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.weight.requires_grad_(True)
            module.bias.requires_grad_(True)


def collect_bn_affine(model: nn.Module, excluded_prefixes: tuple[str, ...] = ()) -> tuple[list[nn.Parameter], list[str]]:
    parameters, names = [], []
    for module_name, module in model.named_modules():
        if excluded_prefixes and any(module_name.startswith(prefix) for prefix in excluded_prefixes):
            continue
        if isinstance(module, nn.BatchNorm2d):
            for parameter_name in ("weight", "bias"):
                parameter = getattr(module, parameter_name)
                if parameter.requires_grad:
                    parameters.append(parameter)
                    names.append(f"{module_name}.{parameter_name}")
    return parameters, names


def build_optimizer(parameters: Iterable[nn.Parameter] | list[dict[str, Any]], cfg: dict[str, Any]) -> torch.optim.Optimizer:
    name = cfg["optimizer"].lower()
    if name == "sgd":
        return torch.optim.SGD(
            parameters,
            lr=float(cfg["lr"]),
            momentum=float(cfg.get("momentum", 0.0)),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
    if name == "adam":
        return torch.optim.Adam(
            parameters,
            lr=float(cfg["lr"]),
            betas=(float(cfg.get("beta", 0.9)), 0.999),
            weight_decay=float(cfg.get("weight_decay", 0.0)),
        )
    raise ValueError(f"Unknown optimizer: {name}")


@torch.no_grad()
def ema_update(teacher: nn.Module, student: nn.Module, momentum: float) -> None:
    teacher_parameters = dict(teacher.named_parameters())
    for name, parameter in student.named_parameters():
        teacher_parameters[name].mul_(momentum).add_(parameter, alpha=1.0 - momentum)


def cpu_state_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {key: value.detach().cpu().clone() for key, value in module.state_dict().items()}


def cpu_parameter_dict(module: nn.Module) -> dict[str, torch.Tensor]:
    return {name: parameter.detach().cpu().clone() for name, parameter in module.named_parameters()}


@torch.no_grad()
def restore_named_parameters(module: nn.Module, source: dict[str, torch.Tensor]) -> None:
    for name, parameter in module.named_parameters():
        if name in source:
            parameter.copy_(source[name].to(parameter.device))


@torch.no_grad()
def parameter_drift(module: nn.Module, source: dict[str, torch.Tensor]) -> float:
    squared = torch.zeros((), device=next(module.parameters()).device)
    for name, parameter in module.named_parameters():
        if name in source:
            difference = parameter.detach() - source[name].to(parameter.device)
            squared += difference.square().sum()
    return float(squared.sqrt().cpu())


def predicted_foreground_area(logits: torch.Tensor) -> dict[str, float]:
    labels = logits.argmax(dim=1)
    return {f"foreground_pixels_class_{class_id}": float((labels == class_id).sum()) for class_id in (1, 2, 3)}
