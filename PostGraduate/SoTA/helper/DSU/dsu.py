from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class DistributionUncertainty(nn.Module):
    """Distribution uncertainty layer from DSU.

    During training, DSU perturbs per-instance feature statistics by sampling
    channel-wise means and standard deviations from the uncertainty estimated
    across the current mini-batch. In eval mode it is an identity mapping.
    """

    def __init__(self, p: float = 0.5, eps: float = 1e-6, factor: float = 1.0):
        super().__init__()
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if float(factor) < 0.0:
            raise ValueError(f"factor must be non-negative, got {factor}")
        self.p = float(p)
        self.eps = float(eps)
        self.factor = float(factor)
        self._activated = True

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, eps={self.eps}, "
            f"factor={self.factor}, activated={self._activated})"
        )

    def set_activation_status(self, status: bool = True) -> None:
        self._activated = bool(status)

    def _sqrt_batch_var(self, x: torch.Tensor) -> torch.Tensor:
        return (x.var(dim=0, keepdim=True) + self.eps).sqrt().repeat(x.size(0), 1)

    def _reparameterize(self, mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        if self.factor == 0.0:
            return mean
        epsilon = torch.randn_like(std) * self.factor
        return mean + epsilon * std

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._activated:
            return x
        if self.p <= 0.0 or random.random() > self.p:
            return x
        if x.ndim != 4:
            raise ValueError(f"DistributionUncertainty expects [B,C,H,W], got shape {tuple(x.shape)}")

        batch_size = int(x.size(0))
        if batch_size <= 1:
            return x

        mean = x.mean(dim=(2, 3), keepdim=False)
        std = (x.var(dim=(2, 3), keepdim=False) + self.eps).sqrt()
        sqrtvar_mu = self._sqrt_batch_var(mean)
        sqrtvar_std = self._sqrt_batch_var(std)
        beta = self._reparameterize(mean, sqrtvar_mu)
        gamma = self._reparameterize(std, sqrtvar_std)

        mean = mean.view(batch_size, -1, 1, 1)
        std = std.view(batch_size, -1, 1, 1)
        beta = beta.view(batch_size, -1, 1, 1)
        gamma = gamma.view(batch_size, -1, 1, 1)
        return (x - mean) / std * gamma + beta


def _iter_dsu_modules(model: nn.Module) -> Iterator[DistributionUncertainty]:
    for module in model.modules():
        if isinstance(module, DistributionUncertainty):
            yield module


def deactivate_dsu(model: nn.Module) -> None:
    for module in _iter_dsu_modules(model):
        module.set_activation_status(False)


def activate_dsu(model: nn.Module) -> None:
    for module in _iter_dsu_modules(model):
        module.set_activation_status(True)


@contextmanager
def run_without_dsu(model: nn.Module):
    deactivate_dsu(model)
    try:
        yield
    finally:
        activate_dsu(model)


@contextmanager
def run_with_dsu(model: nn.Module):
    activate_dsu(model)
    try:
        yield
    finally:
        deactivate_dsu(model)
