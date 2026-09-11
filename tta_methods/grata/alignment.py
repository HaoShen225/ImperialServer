"""Small, testable primitives for GraTA's implicit gradient alignment."""

from __future__ import annotations

from collections.abc import Iterable

import torch
from torch import nn


@torch.no_grad()
def gradient_norm(gradients: Iterable[torch.Tensor | None]) -> torch.Tensor:
    """Return the L2 norm of a possibly sparse parameter-gradient vector."""
    squared: torch.Tensor | None = None
    for gradient in gradients:
        if gradient is None:
            continue
        value = gradient.detach().float().square().sum()
        squared = value if squared is None else squared + value
    if squared is None:
        return torch.zeros(())
    return squared.sqrt()


@torch.no_grad()
def gradient_cosine(
    first: Iterable[torch.Tensor | None],
    second: Iterable[torch.Tensor | None],
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Compute cosine similarity and both norms over matching gradients."""
    first_values = list(first)
    second_values = list(second)
    if len(first_values) != len(second_values):
        raise ValueError("GraTA gradient vectors must have the same length")

    inner: torch.Tensor | None = None
    paired_first: list[torch.Tensor | None] = []
    paired_second: list[torch.Tensor | None] = []
    for first_gradient, second_gradient in zip(first_values, second_values):
        if first_gradient is None or second_gradient is None:
            paired_first.append(None)
            paired_second.append(None)
            continue
        product = (first_gradient.detach().float() * second_gradient.detach().float()).sum()
        inner = product if inner is None else inner + product
        paired_first.append(first_gradient)
        paired_second.append(second_gradient)

    first_norm = gradient_norm(paired_first)
    second_norm = gradient_norm(paired_second)
    if inner is None:
        device = next(
            (
                value.device
                for value in first_values + second_values
                if value is not None
            ),
            torch.device("cpu"),
        )
        inner = torch.zeros((), device=device)
        first_norm = first_norm.to(device)
        second_norm = second_norm.to(device)
    cosine = inner / (first_norm * second_norm + float(epsilon))
    return cosine.clamp(-1.0, 1.0), first_norm, second_norm


def aligned_learning_rate(base_lr: float, cosine: torch.Tensor | float) -> float:
    """Map gradient cosine from [-1, 1] to GraTA's [0, base_lr] interval."""
    value = float(cosine.detach().cpu()) if isinstance(cosine, torch.Tensor) else float(cosine)
    value = max(-1.0, min(1.0, value))
    return float(base_lr) * 0.25 * (value + 1.0) ** 2


@torch.no_grad()
def perturb_parameters(
    parameters: Iterable[nn.Parameter],
    gradients: Iterable[torch.Tensor | None],
    scale: float,
) -> list[torch.Tensor]:
    """Save parameters and apply the paper's raw auxiliary-gradient step."""
    parameter_values = list(parameters)
    gradient_values = list(gradients)
    if len(parameter_values) != len(gradient_values):
        raise ValueError("GraTA parameters and gradients must have the same length")
    originals = [parameter.detach().clone() for parameter in parameter_values]
    for parameter, gradient in zip(parameter_values, gradient_values):
        if gradient is not None:
            parameter.sub_(gradient, alpha=float(scale))
    return originals


@torch.no_grad()
def restore_parameters(
    parameters: Iterable[nn.Parameter], originals: Iterable[torch.Tensor]
) -> None:
    parameter_values = list(parameters)
    original_values = list(originals)
    if len(parameter_values) != len(original_values):
        raise ValueError("GraTA parameters and saved values must have the same length")
    for parameter, original in zip(parameter_values, original_values):
        parameter.copy_(original)
