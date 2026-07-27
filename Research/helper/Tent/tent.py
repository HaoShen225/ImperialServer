"""Tent: fully test-time adaptation by entropy minimization.

This module follows the online adaptation behavior of the reference Tent
implementation while supporting segmentation models whose forward pass returns
``(features, logits)`` (as the UNet in this project does).
"""

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ModelState = Dict[str, torch.Tensor]
OptimizerState = Dict[str, Any]


def model_logits(output: Any) -> torch.Tensor:
    """Extract segmentation logits from a supported model output."""
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
    """Return categorical entropy for every sample/spatial position.

    The class dimension is dimension 1, so segmentation logits shaped
    ``[B, C, H, W]`` produce entropy shaped ``[B, H, W]``.
    """
    if logits.ndim < 2:
        raise ValueError("Logits must have a class dimension at index 1.")
    return -(F.softmax(logits, dim=1) * F.log_softmax(logits, dim=1)).sum(dim=1)


def configure_model(model: nn.Module) -> nn.Module:
    """Configure a model for Tent adaptation.

    All parameters are frozen except affine scale and bias in BatchNorm2d.
    BatchNorm uses statistics from each incoming test batch instead of stored
    source-domain running statistics.
    """
    model.train()
    model.requires_grad_(False)

    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            module.requires_grad_(True)
            module.track_running_stats = False
            module.running_mean = None
            module.running_var = None

    return model


def collect_params(model: nn.Module) -> Tuple[List[nn.Parameter], List[str]]:
    """Collect the BatchNorm2d affine parameters adapted by Tent."""
    params: List[nn.Parameter] = []
    names: List[str] = []

    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name, None)
            if parameter is not None and parameter.requires_grad:
                params.append(parameter)
                full_name = f"{module_name}.{parameter_name}" if module_name else parameter_name
                names.append(full_name)

    if not params:
        raise RuntimeError(
            "Tent found no trainable BatchNorm2d affine parameters. "
            "Call configure_model(model) before collect_params(model)."
        )
    return params, names


def copy_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> Tuple[ModelState, OptimizerState]:
    """Copy model and optimizer state for exact episodic resets."""
    return deepcopy(model.state_dict()), deepcopy(optimizer.state_dict())


def load_model_and_optimizer(
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    model_state: ModelState,
    optimizer_state: OptimizerState,
) -> None:
    """Restore model and optimizer state in place."""
    model.load_state_dict(model_state, strict=True)
    optimizer.load_state_dict(optimizer_state)


def check_model(model: nn.Module) -> None:
    """Validate that a model is correctly configured for Tent."""
    if not model.training:
        raise AssertionError("Tent requires the model to be in train mode.")

    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError("Tent requires at least one trainable parameter.")

    batch_norm_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter in (module.weight, module.bias)
        if parameter is not None
    }
    invalid = [name for name, parameter in trainable if id(parameter) not in batch_norm_parameter_ids]
    if invalid:
        raise AssertionError(
            "Only BatchNorm2d affine parameters may be trainable for Tent; "
            f"found: {', '.join(invalid)}"
        )

    adaptive_batch_norms = [
        module
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        and any(parameter is not None and parameter.requires_grad for parameter in (module.weight, module.bias))
    ]
    if not adaptive_batch_norms:
        raise AssertionError("Tent requires at least one adaptive BatchNorm2d layer.")
    if any(module.track_running_stats for module in adaptive_batch_norms):
        raise AssertionError("Adaptive BatchNorm2d layers must use test-batch statistics.")


@torch.enable_grad()
def forward_and_adapt(
    images: torch.Tensor,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
) -> torch.Tensor:
    """Run one forward pass and one Tent entropy-minimization update.

    Returned logits are from the forward pass that produced the update. The
    updated parameters therefore affect subsequent batches, matching online
    Tent behavior.
    """
    logits = model_logits(model(images))
    loss = softmax_entropy(logits).mean()

    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()
    return logits


class Tent(nn.Module):
    """Wrap a configured model with online Tent adaptation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        steps: int = 1,
        episodic: bool = False,
    ) -> None:
        super().__init__()
        if int(steps) < 1:
            raise ValueError("Tent steps must be at least 1.")

        check_model(model)
        self.model = model
        self.optimizer = optimizer
        self.steps = int(steps)
        self.episodic = bool(episodic)
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()

        logits: Optional[torch.Tensor] = None
        for _ in range(self.steps):
            logits = forward_and_adapt(images, self.model, self.optimizer)

        if logits is None:  # guarded by steps validation; keeps typing explicit
            raise RuntimeError("Tent did not execute an adaptation step.")
        return logits

    def reset(self) -> None:
        """Restore the model and optimizer to their initial Tent state."""
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )


__all__: Sequence[str] = (
    "Tent",
    "check_model",
    "collect_params",
    "configure_model",
    "copy_model_and_optimizer",
    "forward_and_adapt",
    "load_model_and_optimizer",
    "model_logits",
    "softmax_entropy",
)
