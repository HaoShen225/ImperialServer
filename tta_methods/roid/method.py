from __future__ import annotations

import torch
from torch.nn import functional as F

from ..base import AdaptationResult, BaseTTA
from ..common import build_optimizer, collect_bn_affine, configure_bn_for_batch_stats, pixel_entropy


def _normalize(values: torch.Tensor) -> torch.Tensor:
    span = values.max() - values.min()
    if float(span.detach()) <= 1e-12:
        return torch.zeros_like(values)
    return (values - values.min()) / span


def _intensity_augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    batch = images.shape[0]
    brightness = (0.85 + 0.3 * torch.rand(batch, 1, 1, 1, generator=generator)).to(images.device)
    contrast = (0.85 + 0.3 * torch.rand(batch, 1, 1, 1, generator=generator)).to(images.device)
    mean = images.mean(dim=(-2, -1), keepdim=True)
    value = ((images - mean) * contrast + mean) * brightness
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype).to(images.device) * 0.01
    return (value + noise).clamp(0.0, 1.0)


class RoID(BaseTTA):
    def setup(self) -> None:
        if bool(self.cfg["prior_correction"]):
            raise ValueError("The locked segmentation RoID profile disables prior correction")
        configure_bn_for_batch_stats(self.model)
        self.parameters, self.parameter_names = collect_bn_affine(self.model)
        self.optimizer = self._build_optimizer()
        self.class_probs_ema = torch.full((4,), 0.25, device=self.device)

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return build_optimizer(self.parameters, self.cfg)

    def capture_state(self) -> dict[str, object]:
        state = super().capture_state()
        state["class_probs_ema"] = torch.full((4,), 0.25)
        return state

    def reset(self) -> None:
        super().reset()
        self.class_probs_ema = self._initial_state["class_probs_ema"].to(self.device)

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)["logits"]
        probabilities = logits.softmax(dim=1)
        descriptors = probabilities.mean(dim=(-2, -1))
        diversity = 1.0 - F.cosine_similarity(self.class_probs_ema.unsqueeze(0), descriptors, dim=1)
        diversity_normalized = _normalize(diversity)
        selected = diversity_normalized >= diversity_normalized.mean()
        certainty = -pixel_entropy(logits).mean(dim=(-2, -1))
        certainty_normalized = _normalize(certainty)
        weights = torch.exp(diversity_normalized * certainty_normalized / float(self.cfg["temperature"]))
        weights = weights * selected
        clipped = probabilities.clamp(0.0, 0.99)
        slr = -(clipped * torch.log(clipped / (1.0 - clipped) + 1e-5)).sum(dim=1).mean(dim=(-2, -1))
        per_slice = slr
        if bool(self.cfg["consistency"]) and selected.any():
            augmented_logits = self.model(_intensity_augment(images, self.generator))["logits"]
            forward_ce = -(probabilities.detach() * augmented_logits.log_softmax(dim=1)).sum(dim=1).mean(dim=(-2, -1))
            reverse_ce = -(augmented_logits.softmax(dim=1) * logits.detach().log_softmax(dim=1)).sum(dim=1).mean(dim=(-2, -1))
            per_slice = per_slice + 0.5 * (forward_ce + reverse_ce)
        loss = (per_slice * weights).sum() / images.shape[0]
        updated = bool(selected.any())
        if updated:
            loss.backward()
            self.optimizer.step()
            momentum = float(self.cfg["source_weight_momentum"])
            with torch.no_grad():
                for name, parameter in self.model.named_parameters():
                    if parameter.requires_grad:
                        parameter.mul_(momentum).add_(self.source_parameter_state[name].to(parameter.device), alpha=1.0 - momentum)
        probability_momentum = float(self.cfg["probability_momentum"])
        self.class_probs_ema = probability_momentum * self.class_probs_ema + (1.0 - probability_momentum) * descriptors.detach().mean(dim=0)
        return AdaptationResult(
            loss=float(loss.detach()),
            n_seen=int(images.shape[0]),
            n_selected=int(selected.sum()),
            updated=updated,
            extras={"mean_diversity": float(diversity.mean().detach()), "prior_correction": 0.0},
        )
