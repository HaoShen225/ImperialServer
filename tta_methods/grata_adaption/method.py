"""GraTA with single-pass hard teachers and CSL-weighted consistency."""

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
from .augment import (
    WEAK_VIEWS,
    apply_per_sample_weak_views,
    sample_weak_views,
    strong_style_augmentation,
)
from .csl import CSLSelection, select_csl_weights, weighted_cross_entropy


class GraTAAdaption(BaseTTA):
    """Align entropy and reliable consistency gradients for online TTA."""

    def setup(self) -> None:
        self._validate_profile()
        configure_bn_for_batch_stats(self.model)
        self.parameters, self.parameter_names = collect_bn_affine(self.model)
        if not self.parameters:
            raise ValueError("GraTA-Adaption requires BatchNorm affine parameters")
        self.base_lr = float(self.cfg["lr"])
        self.optimizer = self._build_optimizer()

    def _validate_profile(self) -> None:
        if tuple(self.cfg["weak_views"]) != WEAK_VIEWS:
            raise ValueError(
                f"GraTA-Adaption weak views must be {list(WEAK_VIEWS)!r}, "
                f"got {self.cfg['weak_views']!r}"
            )
        weak_view_samples = int(self.cfg["weak_view_samples"])
        if not 1 <= weak_view_samples <= len(WEAK_VIEWS):
            raise ValueError(
                f"weak_view_samples must be in [1, {len(WEAK_VIEWS)}], "
                f"got {weak_view_samples}"
            )
        if self.cfg["selector_scope"] not in {"global", "classwise"}:
            raise ValueError("selector_scope must be 'global' or 'classwise'")
        if self.cfg["weight_mode"] not in {"hard", "smooth"}:
            raise ValueError("weight_mode must be 'hard' or 'smooth'")
        if self.cfg["loss_normalization"] != "weight_sum":
            raise ValueError("GraTA-Adaption requires weight_sum loss normalization")

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters,
            lr=self.base_lr,
            betas=(float(self.cfg["beta1"]), float(self.cfg["beta2"])),
            weight_decay=float(self.cfg["weight_decay"]),
        )

    def _selection(self, probabilities: torch.Tensor) -> CSLSelection:
        return select_csl_weights(
            probabilities,
            scope=str(self.cfg["selector_scope"]),
            mode=str(self.cfg["weight_mode"]),
            alpha=float(self.cfg["csl_alpha"]),
            epsilon=float(self.cfg["csl_epsilon"]),
            classwise_min_pixels=int(self.cfg["classwise_min_pixels"]),
        )

    def _selection_extras(self, selection: CSLSelection) -> dict[str, float]:
        extras = {
            "reliable_pixel_coverage": float(selection.hard_mask.float().mean().cpu()),
            "mean_csl_weight": float(selection.weights.float().mean().cpu()),
            "global_fallback_count": float(selection.global_fallbacks),
            "classwise_group_count": float(selection.classwise_groups),
            "classwise_fallback_count": float(selection.classwise_fallbacks),
            "classwise_fallback_rate": (
                float(selection.classwise_fallbacks / selection.classwise_groups)
                if selection.classwise_groups
                else 0.0
            ),
        }
        for class_id in range(4):
            predicted = selection.labels == class_id
            count = int(predicted.sum())
            extras[f"reliable_coverage_class_{class_id}"] = (
                float(selection.hard_mask[predicted].float().mean().cpu()) if count else 0.0
            )
            extras[f"mean_weight_class_{class_id}"] = (
                float(selection.weights[predicted].float().mean().cpu()) if count else 0.0
            )
        return extras

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
        selection: CSLSelection | None = None
        augmentation_totals: dict[str, float] = {}
        try:
            with torch.no_grad():
                teacher_probabilities = self.model(images)["logits"].softmax(dim=1)
                selection = self._selection(teacher_probabilities)
                sampled = sample_weak_views(
                    batch_size=int(images.shape[0]),
                    samples_per_image=int(self.cfg["weak_view_samples"]),
                    views=tuple(self.cfg["weak_views"]),
                    generator=self.generator,
                )

            self.optimizer.zero_grad(set_to_none=True)
            view_losses: list[torch.Tensor] = []
            for view_index in range(int(self.cfg["weak_view_samples"])):
                views = [sampled[image_index][view_index] for image_index in range(len(sampled))]
                weak_images = apply_per_sample_weak_views(images, views)
                weak_labels = apply_per_sample_weak_views(selection.labels, views)
                weak_weights = apply_per_sample_weak_views(selection.weights, views)
                strong_images, diagnostics = strong_style_augmentation(
                    weak_images, self.cfg["strong_augmentation"], self.generator
                )
                for key, value in diagnostics.items():
                    augmentation_totals[key] = augmentation_totals.get(key, 0.0) + value
                student_logits = self.model(strong_images)["logits"]
                view_loss = weighted_cross_entropy(
                    student_logits,
                    weak_labels,
                    weak_weights,
                    epsilon=float(self.cfg["csl_epsilon"]),
                )
                if view_loss is not None:
                    view_losses.append(view_loss)

            if view_losses:
                consistency_loss = torch.stack(view_losses).mean()
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

        if selection is None:
            raise RuntimeError("GraTA-Adaption teacher selection did not complete")
        probe_payload = {
            "csl_reliable": {
                "selected": selection.hard_mask.detach(),
                "labels": selection.labels.detach(),
                "weights": selection.weights.detach(),
            }
        }
        selection_extras = self._selection_extras(selection)
        view_count = float(self.cfg["weak_view_samples"])
        augmentation_diagnostics = {
            key: value / view_count for key, value in augmentation_totals.items()
        }

        if consistency_loss is None:
            self.optimizer.zero_grad(set_to_none=True)
            return AdaptationResult(
                loss=0.0,
                n_seen=int(images.shape[0]),
                n_selected=int(selection.hard_mask.flatten(1).any(dim=1).sum()),
                updated=False,
                extras={
                    "entropy_loss": float(entropy_loss.detach().cpu()),
                    "consistency_loss": 0.0,
                    "gradient_cosine": 0.0,
                    "entropy_gradient_norm": 0.0,
                    "consistency_gradient_norm": 0.0,
                    "effective_lr": 0.0,
                    "teacher_forward_count": 1.0,
                    "student_forward_count": view_count,
                    "weak_view_samples": view_count,
                    **selection_extras,
                    **augmentation_diagnostics,
                },
                probe_payload=probe_payload,
            )

        if cosine is None or entropy_norm is None or consistency_norm is None:
            raise RuntimeError("GraTA-Adaption consistency gradient did not complete")
        scalar_values = (
            entropy_loss,
            consistency_loss,
            cosine,
            entropy_norm,
            consistency_norm,
        )
        if not all(torch.isfinite(value).all() for value in scalar_values):
            self.optimizer.zero_grad(set_to_none=True)
            raise RuntimeError("GraTA-Adaption produced a non-finite loss or gradient")

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
            "teacher_forward_count": 1.0,
            "student_forward_count": view_count,
            "weak_view_samples": view_count,
            **selection_extras,
            **augmentation_diagnostics,
        }
        if not all(math.isfinite(value) for value in extras.values()):
            raise RuntimeError("GraTA-Adaption produced non-finite adaptation metadata")
        return AdaptationResult(
            loss=float(consistency_loss.detach().cpu()),
            n_seen=int(images.shape[0]),
            n_selected=int(selection.hard_mask.flatten(1).any(dim=1).sum()),
            updated=has_update,
            extras=extras,
            probe_payload=probe_payload,
        )
