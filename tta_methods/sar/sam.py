from __future__ import annotations

import torch


class SAM:
    """Minimal same-batch sharpness-aware optimizer wrapper."""

    def __init__(self, parameters: list[torch.nn.Parameter], lr: float, momentum: float, weight_decay: float, rho: float):
        self.parameters = list(parameters)
        self.rho = float(rho)
        self.base_optimizer = torch.optim.SGD(
            self.parameters, lr=lr, momentum=momentum, weight_decay=weight_decay
        )
        self.perturbations: dict[torch.nn.Parameter, torch.Tensor] = {}

    def zero_grad(self, set_to_none: bool = True) -> None:
        self.base_optimizer.zero_grad(set_to_none=set_to_none)

    @torch.no_grad()
    def first_step(self) -> None:
        norms = [parameter.grad.norm(p=2) for parameter in self.parameters if parameter.grad is not None]
        if not norms:
            return
        grad_norm = torch.norm(torch.stack(norms), p=2)
        scale = self.rho / (grad_norm + 1e-12)
        self.perturbations = {}
        for parameter in self.parameters:
            if parameter.grad is None:
                continue
            perturbation = parameter.grad * scale
            parameter.add_(perturbation)
            self.perturbations[parameter] = perturbation
        self.zero_grad()

    @torch.no_grad()
    def second_step(self) -> None:
        for parameter, perturbation in self.perturbations.items():
            parameter.sub_(perturbation)
        self.base_optimizer.step()
        self.perturbations = {}
        self.zero_grad()
