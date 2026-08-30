from __future__ import annotations

from copy import deepcopy

import torch
from torch import nn


class RobustBatchNorm2d(nn.Module):
    def __init__(self, batch_norm: nn.BatchNorm2d, alpha: float):
        super().__init__()
        if batch_norm.running_mean is None or batch_norm.running_var is None:
            raise ValueError("RoTTA RBN requires source BatchNorm running statistics")
        self.alpha = float(alpha)
        self.eps = float(batch_norm.eps)
        self.weight = nn.Parameter(deepcopy(batch_norm.weight.detach()))
        self.bias = nn.Parameter(deepcopy(batch_norm.bias.detach()))
        self.register_buffer("source_mean", batch_norm.running_mean.detach().clone())
        self.register_buffer("source_var", batch_norm.running_var.detach().clone())
        self.register_buffer("updates", torch.zeros((), dtype=torch.long))

    def forward(self, value: torch.Tensor) -> torch.Tensor:
        if self.training:
            variance, mean = torch.var_mean(value, dim=(0, 2, 3), unbiased=False)
            blended_mean = (1.0 - self.alpha) * self.source_mean + self.alpha * mean
            blended_var = (1.0 - self.alpha) * self.source_var + self.alpha * variance
            self.source_mean.copy_(blended_mean.detach())
            self.source_var.copy_(blended_var.detach())
            self.updates.add_(1)
        else:
            blended_mean, blended_var = self.source_mean, self.source_var
        normalized = (value - blended_mean.view(1, -1, 1, 1)) / torch.sqrt(blended_var.view(1, -1, 1, 1) + self.eps)
        return normalized * self.weight.view(1, -1, 1, 1) + self.bias.view(1, -1, 1, 1)


def replace_batch_norm_with_rbn(model: nn.Module, alpha: float) -> list[str]:
    names = [name for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)]
    for name in names:
        parent_name, _, child_name = name.rpartition(".")
        parent = model.get_submodule(parent_name) if parent_name else model
        original = getattr(parent, child_name)
        setattr(parent, child_name, RobustBatchNorm2d(original, alpha))
    return names
