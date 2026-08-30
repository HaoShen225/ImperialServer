from __future__ import annotations

from copy import deepcopy
import math
from typing import Any

import torch

from ..base import AdaptationResult, BaseTTA
from ..common import cpu_state_dict, ema_update, slice_entropy
from .memory import CSTUMemory
from .rbn import RobustBatchNorm2d, replace_batch_norm_with_rbn


def _dominant_foreground_or_empty(labels: torch.Tensor) -> int:
    counts = torch.stack([(labels == class_id).sum() for class_id in (1, 2, 3)])
    if int(counts.sum()) == 0:
        return 0
    return int(counts.argmax()) + 1


def _intensity_augment(images: torch.Tensor, generator: torch.Generator) -> torch.Tensor:
    batch = images.shape[0]
    brightness = (0.8 + 0.4 * torch.rand(batch, 1, 1, 1, generator=generator)).to(images.device)
    contrast = (0.8 + 0.4 * torch.rand(batch, 1, 1, 1, generator=generator)).to(images.device)
    gamma = (0.8 + 0.4 * torch.rand(batch, 1, 1, 1, generator=generator)).to(images.device)
    mean = images.mean(dim=(-2, -1), keepdim=True)
    value = (images - mean) * contrast + mean
    value = (value * brightness).clamp(1e-6, 1.0).pow(gamma)
    noise = torch.randn(images.shape, generator=generator, dtype=images.dtype).to(images.device) * 0.02
    return (value + noise).clamp(0.0, 1.0)


class RoTTA(BaseTTA):
    prediction_source = "ema_teacher"

    def setup(self) -> None:
        if self.cfg["memory_category_key"] != "dominant_foreground_or_empty":
            raise ValueError("Unsupported RoTTA segmentation memory category")
        self.model.requires_grad_(False)
        self.rbn_names = replace_batch_norm_with_rbn(self.model, float(self.cfg["rbn_alpha"]))
        self.parameters = []
        for module in self.model.modules():
            if isinstance(module, RobustBatchNorm2d):
                module.weight.requires_grad_(True)
                module.bias.requires_grad_(True)
                self.parameters.extend([module.weight, module.bias])
        self.teacher = deepcopy(self.model).to(self.device)
        self.teacher.requires_grad_(False)
        self.memory = CSTUMemory(
            int(self.cfg["memory_capacity"]), 4,
            float(self.cfg["lambda_timeliness"]), float(self.cfg["lambda_uncertainty"]),
        )
        self.seen = 0
        self.update_count = 0
        self.optimizer = self._build_optimizer()

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters,
            lr=float(self.cfg["lr"]),
            betas=(float(self.cfg["beta"]), 0.999),
            weight_decay=float(self.cfg["weight_decay"]),
        )

    def capture_state(self) -> dict[str, Any]:
        state = super().capture_state()
        state.update({"teacher": cpu_state_dict(self.teacher), "memory": self.memory.state_dict(), "seen": 0, "update_count": 0})
        return state

    def reset(self) -> None:
        super().reset()
        self.teacher.load_state_dict(self._initial_state["teacher"], strict=True)
        self.memory.load_state_dict(self._initial_state["memory"])
        self.seen = 0
        self.update_count = 0

    def _update_model(self) -> float | None:
        images, ages = self.memory.get()
        if not images:
            return None
        replay = torch.stack(images).to(self.device)
        self.model.train()
        self.teacher.train()
        augmented = _intensity_augment(replay, self.generator)
        with torch.no_grad():
            teacher_logits = self.teacher(replay)["logits"]
        student_logits = self.model(augmented)["logits"]
        per_pixel = -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1)
        per_slice = per_pixel.mean(dim=(-2, -1))
        age_tensor = torch.tensor(ages, device=self.device, dtype=per_slice.dtype)
        weights = torch.sigmoid(-age_tensor)
        loss = (per_slice * weights).mean()
        assert self.optimizer is not None
        self.optimizer.zero_grad(set_to_none=True)
        loss.backward()
        self.optimizer.step()
        ema_update(self.teacher, self.model, 1.0 - float(self.cfg["teacher_nu"]))
        self.model.eval()
        self.teacher.eval()
        self.update_count += 1
        return float(loss.detach())

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        self.model.eval()
        self.teacher.eval()
        with torch.no_grad():
            teacher_logits = self.teacher(images)["logits"]
            labels = teacher_logits.argmax(dim=1)
            uncertainty = slice_entropy(teacher_logits) / math.log(teacher_logits.shape[1])
        accepted = 0
        losses = []
        for index in range(images.shape[0]):
            category = _dominant_foreground_or_empty(labels[index])
            accepted += int(self.memory.add(images[index], category, float(uncertainty[index])))
            self.seen += 1
            if self.seen % int(self.cfg["update_frequency"]) == 0:
                value = self._update_model()
                if value is not None:
                    losses.append(value)
        return AdaptationResult(
            loss=float(sum(losses) / len(losses)) if losses else None,
            n_seen=int(images.shape[0]),
            n_selected=accepted,
            updated=bool(losses),
            extras={"memory_size": float(len(self.memory)), "model_updates": float(self.update_count)},
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(images)["logits"]
