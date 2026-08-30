from __future__ import annotations

import math
from typing import Any

import torch

from ..base import AdaptationResult, BaseTTA
from ..common import collect_bn_affine, configure_bn_for_batch_stats, restore_named_parameters, slice_entropy
from .sam import SAM


class SAR(BaseTTA):
    def setup(self) -> None:
        configure_bn_for_batch_stats(self.model)
        self.parameters, self.parameter_names = collect_bn_affine(self.model, ("encoder.layer4",))
        self.optimizer = self._build_optimizer()
        self.ema_loss: float | None = None
        self.recovery_count = 0

    def _build_optimizer(self) -> SAM:
        return SAM(
            self.parameters,
            lr=float(self.cfg["lr"]),
            momentum=float(self.cfg["momentum"]),
            weight_decay=float(self.cfg["weight_decay"]),
            rho=float(self.cfg["rho"]),
        )

    def capture_state(self) -> dict[str, Any]:
        state = super().capture_state()
        state.update({"ema_loss": None, "recovery_count": 0})
        return state

    def reset(self) -> None:
        super().reset()
        self.ema_loss = None
        self.recovery_count = 0

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert isinstance(self.optimizer, SAM)
        margin = float(self.cfg["entropy_margin_factor"]) * math.log(4)
        self.optimizer.zero_grad()
        first_logits = self.model(images)["logits"]
        first_entropy = slice_entropy(first_logits)
        first_selected = first_entropy < margin
        if not first_selected.any():
            return AdaptationResult(n_seen=int(images.shape[0]), n_selected=0, updated=False)
        first_loss = first_entropy[first_selected].mean()
        first_loss.backward()
        self.optimizer.first_step()
        second_logits = self.model(images)["logits"]
        second_entropy = slice_entropy(second_logits)
        second_selected = second_entropy < margin
        if not second_selected.any():
            for parameter, perturbation in self.optimizer.perturbations.items():
                parameter.data.sub_(perturbation)
            self.optimizer.perturbations = {}
            self.optimizer.zero_grad()
            return AdaptationResult(loss=float(first_loss.detach()), n_seen=int(images.shape[0]), n_selected=0, updated=False)
        second_loss = second_entropy[second_selected].mean()
        second_loss.backward()
        self.optimizer.second_step()
        loss_value = float(second_loss.detach())
        decay = float(self.cfg["recovery_ema"])
        self.ema_loss = loss_value if self.ema_loss is None else decay * self.ema_loss + (1.0 - decay) * loss_value
        recovered = self.ema_loss < float(self.cfg["recovery_threshold"])
        if recovered:
            restore_named_parameters(self.model, self.source_parameter_state)
            self.optimizer = self._build_optimizer()
            self.recovery_count += 1
        return AdaptationResult(
            loss=loss_value,
            n_seen=int(images.shape[0]),
            n_selected=int(second_selected.sum()),
            updated=True,
            extras={"ema_loss": float(self.ema_loss), "recovered": float(recovered), "recovery_count": float(self.recovery_count)},
        )
