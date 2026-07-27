"""GraTA tools for multiclass medical image segmentation.

This adapts the official GraTA optimization procedure to mutually exclusive
multiclass segmentation: entropy is computed with softmax and the consistency
objective is soft-target cross entropy. Only BatchNorm2d affine parameters are
adapted, matching the parameterization used by the paper's reference code.
"""

from copy import deepcopy
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F


ModelState = Dict[str, torch.Tensor]
OptimizerState = Dict[str, Any]
OptionalGradient = Optional[torch.Tensor]
WEAK_TRANSFORMS = ("identity", "hflip", "vflip", "rot90", "rot180", "rot270")


def model_logits(output: Any) -> torch.Tensor:
    """Extract logits from a tensor, ``(features, logits)``, or mapping."""
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
    """Return categorical entropy at every sample/spatial position."""
    if logits.ndim < 2:
        raise ValueError("Logits must have a class dimension at index 1.")
    return -(logits.softmax(dim=1) * logits.log_softmax(dim=1)).sum(dim=1)


def soft_target_cross_entropy(logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Cross entropy between logits and detached multiclass probabilities."""
    if logits.shape != targets.shape:
        raise ValueError(f"Logit/target shapes differ: {tuple(logits.shape)} vs {tuple(targets.shape)}")
    return -(targets.detach() * logits.log_softmax(dim=1)).sum(dim=1).mean()


def configure_model(model: nn.Module) -> nn.Module:
    """Freeze a model and enable test-batch adaptation of BN affine values."""
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
    """Collect trainable BatchNorm2d affine parameters and their names."""
    params: List[nn.Parameter] = []
    names: List[str] = []
    for module_name, module in model.named_modules():
        if not isinstance(module, nn.BatchNorm2d):
            continue
        for parameter_name in ("weight", "bias"):
            parameter = getattr(module, parameter_name, None)
            if parameter is not None and parameter.requires_grad:
                params.append(parameter)
                names.append(f"{module_name}.{parameter_name}" if module_name else parameter_name)
    if not params:
        raise RuntimeError("GraTA found no trainable BatchNorm2d affine parameters.")
    return params, names


def check_model(model: nn.Module) -> None:
    """Validate the GraTA model configuration."""
    if not model.training:
        raise AssertionError("GraTA requires train mode.")
    trainable = [(name, parameter) for name, parameter in model.named_parameters() if parameter.requires_grad]
    if not trainable:
        raise AssertionError("GraTA requires trainable BatchNorm2d affine parameters.")
    bn_parameter_ids = {
        id(parameter)
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter in (module.weight, module.bias)
        if parameter is not None
    }
    invalid = [name for name, parameter in trainable if id(parameter) not in bn_parameter_ids]
    if invalid:
        raise AssertionError(f"Only BatchNorm2d affine parameters may be trainable: {', '.join(invalid)}")
    adaptive_bn = [
        module
        for module in model.modules()
        if isinstance(module, nn.BatchNorm2d)
        and any(parameter is not None and parameter.requires_grad for parameter in (module.weight, module.bias))
    ]
    if not adaptive_bn:
        raise AssertionError("GraTA requires at least one adaptive BatchNorm2d layer.")
    if any(module.track_running_stats for module in adaptive_bn):
        raise AssertionError("Adaptive BatchNorm2d layers must use current-batch statistics.")


def transform_weak(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    """Apply one of the six deterministic GraTA weak spatial views."""
    if transform == "identity":
        return tensor
    if transform == "hflip":
        return tensor.flip(-1)
    if transform == "vflip":
        return tensor.flip(-2)
    if transform == "rot90":
        return torch.rot90(tensor, 1, dims=(-2, -1))
    if transform == "rot180":
        return torch.rot90(tensor, 2, dims=(-2, -1))
    if transform == "rot270":
        return torch.rot90(tensor, 3, dims=(-2, -1))
    raise ValueError(f"Unknown weak transform: {transform}")


def invert_weak_transform(tensor: torch.Tensor, transform: str) -> torch.Tensor:
    """Map a weak-view prediction back to the original coordinates."""
    if transform in ("identity", "hflip", "vflip", "rot180"):
        return transform_weak(tensor, transform)
    if transform == "rot90":
        return torch.rot90(tensor, 3, dims=(-2, -1))
    if transform == "rot270":
        return torch.rot90(tensor, 1, dims=(-2, -1))
    raise ValueError(f"Unknown weak transform: {transform}")


@torch.no_grad()
def weak_ensemble_target(model: nn.Module, images: torch.Tensor) -> torch.Tensor:
    """Average aligned probabilities from identity, flips, and rotations."""
    predictions: List[torch.Tensor] = []
    for transform in WEAK_TRANSFORMS:
        logits = model_logits(model(transform_weak(images, transform)))
        aligned_logits = invert_weak_transform(logits, transform)
        predictions.append(aligned_logits.softmax(dim=1))
    target = torch.stack(predictions, dim=0).mean(dim=0)
    return target.detach()


def _probability_mask(images: torch.Tensor, probability: float, *, per_channel: bool) -> torch.Tensor:
    channels = images.shape[1] if per_channel else 1
    shape = (images.shape[0], channels, 1, 1)
    mask = torch.rand(shape, device=images.device) < float(probability)
    if not per_channel:
        mask = mask.expand(-1, images.shape[1], -1, -1)
    return mask


def _random_uniform(images: torch.Tensor, low: float, high: float, *, per_channel: bool) -> torch.Tensor:
    channels = images.shape[1] if per_channel else 1
    values = torch.empty((images.shape[0], channels, 1, 1), device=images.device, dtype=images.dtype)
    values.uniform_(float(low), float(high))
    if not per_channel:
        values = values.expand(-1, images.shape[1], -1, -1)
    return values


def _gaussian_blur_channel(image: torch.Tensor, sigma: float) -> torch.Tensor:
    radius = max(1, int(round(3.0 * float(sigma))))
    coordinates = torch.arange(-radius, radius + 1, device=image.device, dtype=image.dtype)
    kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * float(sigma) ** 2))
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d).view(1, 1, 2 * radius + 1, 2 * radius + 1)
    padded = F.pad(image.view(1, 1, *image.shape), (radius, radius, radius, radius), mode="reflect")
    return F.conv2d(padded, kernel_2d).view_as(image)


def strong_style_augmentation(images: torch.Tensor) -> torch.Tensor:
    """Apply the reference GraTA strong style transform in pure PyTorch.

    The order and ranges match the reference implementation: brightness,
    contrast, inverted gamma, Gaussian noise, and Gaussian blur.
    """
    if images.ndim != 4 or not images.is_floating_point():
        raise ValueError("Strong augmentation expects floating [B, C, H, W] images.")
    augmented = images.clone()

    brightness_mask = _probability_mask(augmented, 0.75, per_channel=False)
    brightness = _random_uniform(augmented, 0.5, 1.5, per_channel=True)
    augmented = torch.where(brightness_mask, augmented * brightness, augmented)

    contrast_mask = _probability_mask(augmented, 0.75, per_channel=False)
    contrast = _random_uniform(augmented, 0.5, 1.5, per_channel=True)
    spatial_mean = augmented.mean(dim=(-2, -1), keepdim=True)
    contrasted = (augmented - spatial_mean) * contrast + spatial_mean
    augmented = torch.where(contrast_mask, contrasted, augmented)

    gamma_mask = _probability_mask(augmented, 0.75, per_channel=False)
    gamma = _random_uniform(augmented, 0.5, 2.0, per_channel=True)
    inverted_gamma = 1.0 - (1.0 - augmented.clamp(0.0, 1.0)).pow(gamma)
    augmented = torch.where(gamma_mask, inverted_gamma, augmented)

    noise_sample_mask = _probability_mask(augmented, 0.5, per_channel=False)
    noise_variance = _random_uniform(augmented, 0.0, 0.05, per_channel=False)
    noisy = augmented + torch.randn_like(augmented) * noise_variance.sqrt()
    augmented = torch.where(noise_sample_mask, noisy, augmented)

    blur_mask = _probability_mask(augmented, 0.5, per_channel=False)
    blurred = augmented.clone()
    for batch_index in range(augmented.shape[0]):
        if not bool(blur_mask[batch_index, 0, 0, 0]):
            continue
        for channel_index in range(augmented.shape[1]):
            if bool(torch.rand((), device=augmented.device) < 0.5):
                sigma = float(torch.empty((), device=augmented.device).uniform_(0.5, 1.5).item())
                blurred[batch_index, channel_index] = _gaussian_blur_channel(
                    augmented[batch_index, channel_index], sigma
                )
    return blurred.clamp(0.0, 1.0)


def gradient_cosine(
    auxiliary_gradients: Sequence[OptionalGradient],
    pseudo_gradients: Sequence[OptionalGradient],
    eps: float = 1e-12,
) -> torch.Tensor:
    """Calculate cosine similarity across two sequences of parameter gradients."""
    if len(auxiliary_gradients) != len(pseudo_gradients):
        raise ValueError("Gradient sequences must have equal length.")
    pairs = [
        (auxiliary, pseudo)
        for auxiliary, pseudo in zip(auxiliary_gradients, pseudo_gradients)
        if auxiliary is not None and pseudo is not None
    ]
    if not pairs:
        raise RuntimeError("No shared auxiliary/pseudo gradients were produced.")
    inner = sum((auxiliary * pseudo).sum() for auxiliary, pseudo in pairs)
    auxiliary_norm = torch.sqrt(sum(auxiliary.square().sum() for auxiliary, _ in pairs))
    pseudo_norm = torch.sqrt(sum(pseudo.square().sum() for _, pseudo in pairs))
    cosine = inner / (auxiliary_norm * pseudo_norm + float(eps))
    return cosine.clamp(-1.0, 1.0)


def dynamic_lr_activation(cosine: torch.Tensor) -> torch.Tensor:
    """Reference GraTA activation: one quarter of squared shifted cosine."""
    return 0.25 * (cosine.clamp(-1.0, 1.0) + 1.0).square()


@torch.no_grad()
def perturb_parameters(
    parameters: Sequence[nn.Parameter], gradients: Sequence[OptionalGradient]
) -> List[torch.Tensor]:
    """Save parameters and apply the reference raw-gradient perturbation."""
    if len(parameters) != len(gradients):
        raise ValueError("Parameter and gradient sequences must have equal length.")
    originals = [parameter.detach().clone() for parameter in parameters]
    for parameter, gradient in zip(parameters, gradients):
        if gradient is not None:
            parameter.sub_(gradient)
    return originals


@torch.no_grad()
def restore_parameters(parameters: Sequence[nn.Parameter], originals: Sequence[torch.Tensor]) -> None:
    """Restore parameters saved by :func:`perturb_parameters`."""
    if len(parameters) != len(originals):
        raise ValueError("Parameter and original-value sequences must have equal length.")
    for parameter, original in zip(parameters, originals):
        parameter.copy_(original)


def copy_model_and_optimizer(
    model: nn.Module, optimizer: torch.optim.Optimizer
) -> Tuple[ModelState, OptimizerState]:
    """Copy model and optimizer state for episodic/reset behavior."""
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


class GraTa(nn.Module):
    """Online gradient alignment-based test-time adaptation."""

    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        base_lr: Optional[float] = None,
        steps: int = 1,
        episodic: bool = False,
        perturb_eps: float = 1e-12,
    ) -> None:
        super().__init__()
        if int(steps) < 1:
            raise ValueError("GraTA steps must be at least 1.")
        if not optimizer.param_groups:
            raise ValueError("GraTA requires an optimizer with parameters.")
        check_model(model)
        self.model = model
        self.optimizer = optimizer
        self.parameters_to_adapt, self.parameter_names = collect_params(model)
        self.base_lr = float(optimizer.param_groups[0]["lr"] if base_lr is None else base_lr)
        self.steps = int(steps)
        self.episodic = bool(episodic)
        self.perturb_eps = float(perturb_eps)
        self.last_aux_loss = float("nan")
        self.last_pseudo_loss = float("nan")
        self.last_cosine = float("nan")
        self.last_learning_rate = self.base_lr
        self.model_state, self.optimizer_state = copy_model_and_optimizer(model, optimizer)

    @staticmethod
    def _clone_gradients(parameters: Sequence[nn.Parameter]) -> List[OptionalGradient]:
        return [None if parameter.grad is None else parameter.grad.detach().clone() for parameter in parameters]

    @torch.enable_grad()
    def _adapt_once(self, images: torch.Tensor) -> None:
        self.optimizer.zero_grad(set_to_none=True)
        auxiliary_logits = model_logits(self.model(images))
        auxiliary_loss = softmax_entropy(auxiliary_logits).mean()
        auxiliary_loss.backward()
        auxiliary_gradients = self._clone_gradients(self.parameters_to_adapt)

        originals = perturb_parameters(self.parameters_to_adapt, auxiliary_gradients)
        self.optimizer.zero_grad(set_to_none=True)
        try:
            targets = weak_ensemble_target(self.model, images)
            strong_images = strong_style_augmentation(images)
            strong_logits = model_logits(self.model(strong_images))
            pseudo_loss = soft_target_cross_entropy(strong_logits, targets)
            pseudo_loss.backward()
            pseudo_gradients = self._clone_gradients(self.parameters_to_adapt)
            cosine = gradient_cosine(auxiliary_gradients, pseudo_gradients, eps=self.perturb_eps)
        finally:
            restore_parameters(self.parameters_to_adapt, originals)

        learning_rate = self.base_lr * float(dynamic_lr_activation(cosine).item())
        for group in self.optimizer.param_groups:
            group["lr"] = learning_rate
        self.optimizer.step()
        self.optimizer.zero_grad(set_to_none=True)

        self.last_aux_loss = float(auxiliary_loss.detach().item())
        self.last_pseudo_loss = float(pseudo_loss.detach().item())
        self.last_cosine = float(cosine.detach().item())
        self.last_learning_rate = float(learning_rate)

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if self.episodic:
            self.reset()
        for _ in range(self.steps):
            self._adapt_once(images)
        with torch.no_grad():
            return model_logits(self.model(images))

    def reset(self) -> None:
        """Restore the initial model and optimizer adaptation state."""
        load_model_and_optimizer(
            self.model,
            self.optimizer,
            self.model_state,
            self.optimizer_state,
        )
        self.last_aux_loss = float("nan")
        self.last_pseudo_loss = float("nan")
        self.last_cosine = float("nan")
        self.last_learning_rate = self.base_lr


__all__ = (
    "GraTa",
    "WEAK_TRANSFORMS",
    "check_model",
    "collect_params",
    "configure_model",
    "copy_model_and_optimizer",
    "dynamic_lr_activation",
    "gradient_cosine",
    "invert_weak_transform",
    "load_model_and_optimizer",
    "model_logits",
    "perturb_parameters",
    "restore_parameters",
    "soft_target_cross_entropy",
    "softmax_entropy",
    "strong_style_augmentation",
    "transform_weak",
    "weak_ensemble_target",
)
