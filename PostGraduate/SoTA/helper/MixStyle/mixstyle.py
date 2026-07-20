from __future__ import annotations

import random
from contextlib import contextmanager
from typing import Iterator

import torch
import torch.nn as nn


class MixStyle(nn.Module):
    """Mix feature statistics as in MixStyle.

    The implementation follows the public Dassl.pytorch MixStyle layer:
    feature content is normalized by its own channel statistics, then restored
    with statistics interpolated from another sample in the same mini-batch.
    """

    def __init__(self, p: float = 0.5, alpha: float = 0.1, eps: float = 1e-6, mix: str = "random"):
        super().__init__()
        if float(alpha) <= 0.0:
            raise ValueError(f"alpha must be positive, got {alpha}")
        if not 0.0 <= float(p) <= 1.0:
            raise ValueError(f"p must be in [0, 1], got {p}")
        if str(mix) not in {"random", "crossdomain"}:
            raise ValueError(f"mix must be 'random' or 'crossdomain', got {mix!r}")
        self.p = float(p)
        self.alpha = float(alpha)
        self.eps = float(eps)
        self.mix = str(mix)
        self._activated = True
        self._dist = torch.distributions.Beta(self.alpha, self.alpha)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(p={self.p}, alpha={self.alpha}, "
            f"eps={self.eps}, mix={self.mix!r}, activated={self._activated})"
        )

    def set_activation_status(self, status: bool = True) -> None:
        self._activated = bool(status)

    def update_mix_method(self, mix: str = "random") -> None:
        if str(mix) not in {"random", "crossdomain"}:
            raise ValueError(f"mix must be 'random' or 'crossdomain', got {mix!r}")
        self.mix = str(mix)

    def _random_perm(self, batch_size: int, device: torch.device) -> torch.Tensor:
        return torch.randperm(int(batch_size), device=device)

    def _crossdomain_perm(self, batch_size: int, device: torch.device) -> torch.Tensor:
        if int(batch_size) < 2:
            return torch.arange(int(batch_size), device=device)
        perm = torch.arange(int(batch_size) - 1, -1, -1, device=device)
        first, second = perm.chunk(2)
        first = first[torch.randperm(first.numel(), device=device)]
        second = second[torch.randperm(second.numel(), device=device)]
        return torch.cat([first, second], dim=0)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or not self._activated:
            return x
        if self.p <= 0.0 or random.random() > self.p:
            return x
        if x.ndim != 4:
            raise ValueError(f"MixStyle expects a 4D tensor [B,C,H,W], got shape {tuple(x.shape)}")

        batch_size = int(x.size(0))
        if batch_size <= 1:
            return x

        mu = x.mean(dim=(2, 3), keepdim=True)
        var = x.var(dim=(2, 3), keepdim=True, unbiased=False)
        sig = (var + self.eps).sqrt()
        mu_detached = mu.detach()
        sig_detached = sig.detach()
        x_normed = (x - mu_detached) / sig_detached

        lmda = self._dist.sample((batch_size, 1, 1, 1)).to(device=x.device, dtype=x.dtype)
        if self.mix == "random":
            perm = self._random_perm(batch_size, x.device)
        else:
            perm = self._crossdomain_perm(batch_size, x.device)

        mu2 = mu_detached[perm]
        sig2 = sig_detached[perm]
        mu_mix = mu_detached * lmda + mu2 * (1.0 - lmda)
        sig_mix = sig_detached * lmda + sig2 * (1.0 - lmda)
        return x_normed * sig_mix + mu_mix


def _iter_mixstyle_modules(model: nn.Module) -> Iterator[MixStyle]:
    for module in model.modules():
        if isinstance(module, MixStyle):
            yield module


def deactivate_mixstyle(model: nn.Module) -> None:
    for module in _iter_mixstyle_modules(model):
        module.set_activation_status(False)


def activate_mixstyle(model: nn.Module) -> None:
    for module in _iter_mixstyle_modules(model):
        module.set_activation_status(True)


def random_mixstyle(model: nn.Module) -> None:
    for module in _iter_mixstyle_modules(model):
        module.update_mix_method("random")


def crossdomain_mixstyle(model: nn.Module) -> None:
    for module in _iter_mixstyle_modules(model):
        module.update_mix_method("crossdomain")


@contextmanager
def run_without_mixstyle(model: nn.Module):
    deactivate_mixstyle(model)
    try:
        yield
    finally:
        activate_mixstyle(model)


@contextmanager
def run_with_mixstyle(model: nn.Module):
    activate_mixstyle(model)
    try:
        yield
    finally:
        deactivate_mixstyle(model)
