"""M&Ms multi-class adaptation of GraTA's official optimization mechanism."""

from __future__ import annotations

import math

import torch

from ..base import AdaptationResult, BaseTTA
from ..common import collect_bn_affine, configure_bn_for_batch_stats, pixel_entropy
from .alignment import (
    aligned_learning_rate,
    gradient_cosine,
    perturb_parameters,
    restore_parameters,
)
from .augment import OFFICIAL_WEAK_VIEWS, strong_style_augmentation, weak_probability_ensemble


class GraTA(BaseTTA):
    """Align entropy and consistency gradients for online segmentation TTA."""

    def setup(self) -> None:
        validate_weak_views(self.cfg["weak_views"])
        configure_bn_for_batch_stats(self.model)
        self.parameters, self.parameter_names = collect_bn_affine(self.model)
        if not self.parameters:
            raise ValueError("GraTA requires BatchNorm affine parameters")
        self.base_lr = float(self.cfg["lr"])
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters,
            lr=self.base_lr,
            betas=(float(self.cfg["beta1"]), float(self.cfg["beta2"])),
            weight_decay=float(self.cfg["weight_decay"]),
        )

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        original_logits = self.model(images)["logits"]
        entropy_loss = pixel_entropy(original_logits).mean()
        entropy_loss.backward()
        entropy_gradients = [
            None if parameter.grad is None else parameter.grad.detach().clone()
            for parameter in self.parameters
        ]

        originals = perturb_parameters(
            self.parameters,
            entropy_gradients,
            scale=float(self.cfg["perturbation_scale"]),
        )
        consistency_loss: torch.Tensor | None = None
        cosine: torch.Tensor | None = None
        entropy_norm: torch.Tensor | None = None
        consistency_norm: torch.Tensor | None = None
        augmentation_diagnostics: dict[str, float] = {}
        try:
            weak_target = weak_probability_ensemble(
                self.model, images, tuple(self.cfg["weak_views"])
            ).detach()
            strong_images, augmentation_diagnostics = strong_style_augmentation(
                images, self.cfg["strong_augmentation"], self.generator
            )
            self.optimizer.zero_grad(set_to_none=True)
            strong_logits = self.model(strong_images)["logits"]
            consistency_loss = -(
                weak_target * strong_logits.log_softmax(dim=1)
            ).sum(dim=1).mean()
            consistency_loss.backward()
            consistency_gradients = [
                None if parameter.grad is None else parameter.grad.detach().clone()
                for parameter in self.parameters
            ]
            cosine, entropy_norm, consistency_norm = gradient_cosine(
                entropy_gradients,
                consistency_gradients,
                epsilon=float(self.cfg["cosine_epsilon"]),
            )
        finally:
            restore_parameters(self.parameters, originals)

        if consistency_loss is None or cosine is None or entropy_norm is None or consistency_norm is None:
            raise RuntimeError("GraTA consistency update did not complete")
        scalar_values = (
            entropy_loss,
            consistency_loss,
            cosine,
            entropy_norm,
            consistency_norm,
        )
        if not all(torch.isfinite(value).all() for value in scalar_values):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("GraTA produced a non-finite loss or gradient diagnostic")

        effective_lr = aligned_learning_rate(self.base_lr, cosine)
        for group in self.optimizer.param_groups:
            group["lr"] = effective_lr
        has_update = effective_lr > 0.0 and float(consistency_norm.detach().cpu()) > 0.0
        self.optimizer.step()

        extras = {
            "entropy_loss": float(entropy_loss.detach().cpu()),
            "consistency_loss": float(consistency_loss.detach().cpu()),
            "gradient_cosine": float(cosine.detach().cpu()),
            "entropy_gradient_norm": float(entropy_norm.detach().cpu()),
            "consistency_gradient_norm": float(consistency_norm.detach().cpu()),
            "effective_lr": effective_lr,
            "weak_view_count": float(len(self.cfg["weak_views"])),
            **augmentation_diagnostics,
        }
        if not all(math.isfinite(value) for value in extras.values()):
            raise RuntimeError("GraTA produced non-finite adaptation metadata")
        return AdaptationResult(
            loss=float(consistency_loss.detach().cpu()),
            n_seen=int(images.shape[0]),
            n_selected=int(images.shape[0]),
            updated=has_update,
            extras=extras,
        )


def validate_weak_views(views: list[str]) -> None:
    """Keep the verified profile tied to the paper's exact six-view set."""
    if tuple(views) != OFFICIAL_WEAK_VIEWS:
        raise ValueError(
            f"GraTA weak views must be {list(OFFICIAL_WEAK_VIEWS)!r}, got {views!r}"
        )
