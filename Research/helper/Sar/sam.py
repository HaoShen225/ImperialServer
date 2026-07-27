"""Sharpness-Aware Minimization optimizer used by SAR.

The wrapper keeps the base optimizer state and the temporary SAM perturbations
in one optimizer state dictionary, so model recovery and episodic resets can
restore an adaptation run exactly.
"""

from __future__ import annotations

from typing import Any, Callable, Iterable, Optional, Type

import torch
from torch.optim import Optimizer


class SAM(Optimizer):
    """Wrap a PyTorch optimizer with the two-step SAM update."""

    def __init__(
        self,
        params: Iterable[torch.nn.Parameter],
        base_optimizer: Type[Optimizer],
        rho: float = 0.05,
        adaptive: bool = False,
        **kwargs: Any,
    ) -> None:
        if float(rho) < 0.0:
            raise ValueError("SAM rho must be non-negative.")

        defaults = dict(rho=float(rho), adaptive=bool(adaptive), **kwargs)
        super().__init__(params, defaults)
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
        self.state = self.base_optimizer.state
        for group in self.param_groups:
            group.setdefault("rho", float(rho))
            group.setdefault("adaptive", bool(adaptive))
        self.last_grad_norm: Optional[float] = None

    @torch.no_grad()
    def _grad_norm(self) -> torch.Tensor:
        shared_device: Optional[torch.device] = None
        terms = []
        for group in self.param_groups:
            adaptive = bool(group["adaptive"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                shared_device = parameter.device if shared_device is None else shared_device
                scale = parameter.abs() if adaptive else 1.0
                terms.append((scale * parameter.grad).norm(p=2).to(shared_device))
        if not terms:
            device = shared_device if shared_device is not None else torch.device("cpu")
            return torch.zeros((), device=device)
        return torch.norm(torch.stack(terms), p=2)

    @torch.no_grad()
    def first_step(self, zero_grad: bool = False) -> bool:
        """Move to the local worst-case point; return whether a move occurred."""
        grad_norm = self._grad_norm()
        self.last_grad_norm = float(grad_norm.detach().cpu())
        if not torch.isfinite(grad_norm) or float(grad_norm) <= 0.0:
            if zero_grad:
                self.zero_grad(set_to_none=True)
            return False

        for group in self.param_groups:
            scale = float(group["rho"]) / (grad_norm + 1e-12)
            adaptive = bool(group["adaptive"])
            for parameter in group["params"]:
                if parameter.grad is None:
                    continue
                multiplier = parameter.pow(2) if adaptive else 1.0
                perturbation = multiplier * parameter.grad * scale.to(parameter)
                parameter.add_(perturbation)
                self.state[parameter]["e_w"] = perturbation

        if zero_grad:
            self.zero_grad(set_to_none=True)
        return True

    @torch.no_grad()
    def cancel_step(self, zero_grad: bool = False) -> None:
        """Undo a first step without applying the base optimizer."""
        for group in self.param_groups:
            for parameter in group["params"]:
                perturbation = self.state[parameter].pop("e_w", None)
                if perturbation is not None:
                    parameter.sub_(perturbation)
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def second_step(self, zero_grad: bool = False) -> None:
        """Return to the original point and apply the base optimizer update."""
        self.cancel_step(zero_grad=False)
        self.base_optimizer.step()
        if zero_grad:
            self.zero_grad(set_to_none=True)

    @torch.no_grad()
    def step(self, closure: Optional[Callable[[], torch.Tensor]] = None) -> torch.Tensor:
        """Perform a complete SAM step using a gradient-producing closure."""
        if closure is None:
            raise RuntimeError("SAM.step requires a closure.")
        closure = torch.enable_grad()(closure)
        loss = closure()
        if not self.first_step(zero_grad=True):
            return loss
        closure()
        self.second_step(zero_grad=True)
        return loss

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        super().load_state_dict(state_dict)
        self.base_optimizer.param_groups = self.param_groups
        self.base_optimizer.state = self.state


__all__ = ["SAM"]
