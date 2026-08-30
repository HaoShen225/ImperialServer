from __future__ import annotations

from copy import deepcopy
from typing import Any

import torch

from ..base import AdaptationResult, BaseTTA
from ..common import cpu_state_dict, ema_update, remove_bn_running_buffers
from .augment import teacher_augmentation_ensemble


class CoTTA(BaseTTA):
    prediction_source = "ema_teacher"

    def setup(self) -> None:
        self.model.train()
        self.model.requires_grad_(True)
        remove_bn_running_buffers(self.model)
        self.teacher = deepcopy(self.model).to(self.device)
        self.anchor = deepcopy(self.model).to(self.device)
        self.teacher.requires_grad_(False)
        self.anchor.requires_grad_(False)
        self.anchor.eval()
        self.parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = self._build_optimizer()
        self.restore_count = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        head, body = [], []
        for name, parameter in self.model.named_parameters():
            (head if name.startswith("segmentation_head.") else body).append(parameter)
        return torch.optim.SGD(
            [
                {"params": body, "lr": float(self.cfg["lr"])},
                {"params": head, "lr": float(self.cfg["lr"]) * float(self.cfg["head_lr_multiplier"])},
            ],
            momentum=float(self.cfg["momentum"]),
            weight_decay=float(self.cfg["weight_decay"]),
        )

    def capture_state(self) -> dict[str, Any]:
        state = super().capture_state()
        state.update({
            "teacher": cpu_state_dict(self.teacher),
            "anchor": cpu_state_dict(self.anchor),
            "restore_count": 0,
        })
        return state

    def reset(self) -> None:
        super().reset()
        self.teacher.load_state_dict(self._initial_state["teacher"], strict=True)
        self.anchor.load_state_dict(self._initial_state["anchor"], strict=True)
        self.restore_count = 0

    @torch.no_grad()
    def _teacher_target(self, images: torch.Tensor) -> torch.Tensor:
        anchor_logits = self.anchor(images)["logits"]
        anchor_confidence = anchor_logits.softmax(dim=1).amax(dim=1).mean()
        if float(anchor_confidence) < float(self.cfg["confidence_gate"]):
            return teacher_augmentation_ensemble(
                self.teacher,
                images,
                [float(value) for value in self.cfg["augmentation_scales"]],
                float(self.cfg["horizontal_flip_probability"]),
                self.generator,
            )
        return self.teacher(images)["logits"]

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert self.optimizer is not None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        teacher_logits = self._teacher_target(images)
        student_logits = self.model(images)["logits"]
        loss = -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean()
        loss.backward()
        self.optimizer.step()
        ema_update(self.teacher, self.model, float(self.cfg["teacher_momentum"]))
        probability = float(self.cfg["restore_probability"])
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                mask = (torch.rand(parameter.shape, generator=self.generator) < probability).to(parameter.device)
                restored = int(mask.sum())
                if restored:
                    source = self.source_parameter_state[name].to(parameter.device)
                    parameter.copy_(torch.where(mask, source, parameter))
                    self.restore_count += restored
        return AdaptationResult(
            loss=float(loss.detach()),
            n_seen=int(images.shape[0]),
            n_selected=int(images.shape[0]),
            updated=True,
            extras={"restored_parameters": float(self.restore_count)},
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(images)["logits"]
