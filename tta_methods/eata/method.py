from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import torch
from torch.nn import functional as F

from ..base import AdaptationResult, BaseTTA
from ..common import build_optimizer, collect_bn_affine, configure_bn_for_batch_stats, slice_entropy


class EATA(BaseTTA):
    def setup(self) -> None:
        if self.cfg["descriptor"] != "spatial_mean_class_probability":
            raise ValueError("Unsupported EATA segmentation descriptor")
        configure_bn_for_batch_stats(self.model)
        self.parameters, self.parameter_names = collect_bn_affine(self.model)
        self.optimizer = self._build_optimizer()
        self.current_model_probs: torch.Tensor | None = None
        self.fisher: dict[str, torch.Tensor] = {}
        self.fisher_source: dict[str, torch.Tensor] = {}
        fisher_path = self.cfg.get("fisher_path")
        if fisher_path:
            artifact = torch.load(Path(fisher_path), map_location="cpu", weights_only=False)
            self.fisher = artifact["fisher"]
            self.fisher_source = artifact["source_parameters"]
        self.total_reliable = 0
        self.total_selected = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return build_optimizer(self.parameters, self.cfg)

    def capture_state(self) -> dict[str, Any]:
        state = super().capture_state()
        state.update({"current_model_probs": None, "total_reliable": 0, "total_selected": 0})
        return state

    def reset(self) -> None:
        super().reset()
        self.current_model_probs = None
        self.total_reliable = 0
        self.total_selected = 0

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        logits = self.model(images)["logits"]
        entropies = slice_entropy(logits)
        margin = float(self.cfg["entropy_margin_factor"]) * math.log(logits.shape[1])
        reliable = entropies < margin
        descriptors = logits.softmax(dim=1).mean(dim=(-2, -1))
        selected = reliable.clone()
        if self.current_model_probs is not None and reliable.any():
            similarities = F.cosine_similarity(
                self.current_model_probs.unsqueeze(0), descriptors[reliable], dim=1
            ).abs()
            reliable_indices = reliable.nonzero(as_tuple=False).flatten()
            selected[reliable_indices] = similarities < float(self.cfg["redundancy_margin"])
        selected_count = int(selected.sum())
        reliable_count = int(reliable.sum())
        if selected_count:
            chosen_descriptors = descriptors[selected].detach().mean(dim=0)
            if self.current_model_probs is None:
                self.current_model_probs = chosen_descriptors
            else:
                momentum = float(self.cfg["probability_momentum"])
                self.current_model_probs = momentum * self.current_model_probs + (1.0 - momentum) * chosen_descriptors
            chosen_entropy = entropies[selected]
            loss = (chosen_entropy * torch.exp(margin - chosen_entropy.detach())).mean()
            named = dict(self.model.named_parameters())
            for name, fisher in self.fisher.items():
                if name in named:
                    source = self.fisher_source[name].to(self.device)
                    loss = loss + float(self.cfg["fisher_alpha"]) * (fisher.to(self.device) * (named[name] - source).square()).sum()
            loss.backward()
            self.optimizer.step()
            loss_value: float | None = float(loss.detach())
        else:
            loss_value = None
        self.total_reliable += reliable_count
        self.total_selected += selected_count
        return AdaptationResult(
            loss=loss_value,
            n_seen=int(images.shape[0]),
            n_selected=selected_count,
            updated=selected_count > 0,
            extras={"n_reliable": float(reliable_count), "total_selected": float(self.total_selected)},
        )
