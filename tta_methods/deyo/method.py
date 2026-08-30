from __future__ import annotations

import math

import torch

from ..base import AdaptationResult, BaseTTA
from ..common import build_optimizer, collect_bn_affine, configure_bn_for_batch_stats, pixel_entropy
from .transform import patch_shuffle


class DeYO(BaseTTA):
    def setup(self) -> None:
        configure_bn_for_batch_stats(self.model)
        self.parameters, self.parameter_names = collect_bn_affine(self.model, ("encoder.layer4",))
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return build_optimizer(self.parameters, self.cfg)

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)["logits"]
        entropies = pixel_entropy(logits)
        predicted = logits.argmax(dim=1)
        entropy_margin = float(self.cfg["entropy_margin_factor"]) * math.log(logits.shape[1])
        selected_entropy = entropies < entropy_margin
        if bool(self.cfg["foreground_only"]):
            selected_entropy &= predicted > 0
        first_count = int(selected_entropy.sum())
        if first_count == 0:
            return AdaptationResult(n_seen=int(images.numel() // images.shape[1]), n_selected=0, updated=False, extras={"entropy_selected": 0.0})
        destroyed = patch_shuffle(images.detach(), int(self.cfg["patch_grid"]), self.generator)
        with torch.no_grad():
            destroyed_probabilities = self.model(destroyed)["logits"].softmax(dim=1)
        probabilities = logits.softmax(dim=1)
        original_class_probability = probabilities.gather(1, predicted.unsqueeze(1)).squeeze(1)
        destroyed_class_probability = destroyed_probabilities.gather(1, predicted.unsqueeze(1)).squeeze(1)
        plpd = original_class_probability - destroyed_class_probability
        selected = selected_entropy & (plpd > float(self.cfg["plpd_threshold"]))
        selected_count = int(selected.sum())
        if selected_count == 0:
            return AdaptationResult(n_seen=int(entropies.numel()), n_selected=0, updated=False, extras={"entropy_selected": float(first_count)})
        entropy_weight_margin = float(self.cfg["entropy_weight_margin_factor"]) * math.log(logits.shape[1])
        coefficients = torch.exp(entropy_weight_margin - entropies.detach()) + torch.exp(plpd.detach())
        loss = (entropies[selected] * coefficients[selected]).mean()
        loss.backward()
        self.optimizer.step()
        return AdaptationResult(
            loss=float(loss.detach()),
            n_seen=int(entropies.numel()),
            n_selected=selected_count,
            updated=True,
            extras={"entropy_selected": float(first_count), "mean_selected_plpd": float(plpd[selected].mean())},
        )
