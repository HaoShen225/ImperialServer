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
        self.teacher.eval()
        self.anchor.eval()
        self.anchor_parameters = dict(self.anchor.named_parameters())
        self.restore_generator = torch.Generator(device=self.device)
        self.restore_generator.manual_seed(int(self.cfg.get("method_seed", 0)))
        self.parameters = [parameter for parameter in self.model.parameters() if parameter.requires_grad]
        self.optimizer = self._build_optimizer()
        self.restore_count = 0

    def _build_optimizer(self) -> torch.optim.Optimizer:
        return torch.optim.Adam(
            self.parameters,
            lr=float(self.cfg["lr"]),
            betas=(float(self.cfg["beta1"]), float(self.cfg["beta2"])),
            weight_decay=float(self.cfg["weight_decay"]),
        )

    def capture_state(self) -> dict[str, Any]:
        state = super().capture_state()
        state.update({
            "teacher": cpu_state_dict(self.teacher),
            "anchor": cpu_state_dict(self.anchor),
            "restore_generator": self.restore_generator.get_state().clone(),
            "restore_count": 0,
        })
        return state

    def reset(self) -> None:
        super().reset()
        self.teacher.load_state_dict(self._initial_state["teacher"], strict=True)
        self.anchor.load_state_dict(self._initial_state["anchor"], strict=True)
        self.restore_generator.set_state(self._initial_state["restore_generator"])
        self.restore_count = 0

    @torch.no_grad()
    def _teacher_target(self, images: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        anchor_logits = self.anchor(images)["logits"]
        anchor_confidence = anchor_logits.softmax(dim=1).amax(dim=1).mean(dim=(-2, -1))
        low_confidence = anchor_confidence < float(self.cfg["confidence_gate"])
        teacher_logits = self.teacher(images)["logits"]
        if bool(low_confidence.any()):
            low_logits = teacher_augmentation_ensemble(
                self.teacher,
                images[low_confidence],
                [float(value) for value in self.cfg["augmentation_scales"]],
                [bool(value) for value in self.cfg["augmentation_flips"]],
                standard_logits=teacher_logits[low_confidence],
            )
            teacher_logits = teacher_logits.clone()
            teacher_logits[low_confidence] = low_logits
        return teacher_logits, low_confidence, anchor_confidence

    def adapt(self, images: torch.Tensor) -> AdaptationResult:
        assert self.optimizer is not None
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        teacher_logits, low_confidence, anchor_confidence = self._teacher_target(images)
        student_logits = self.model(images)["logits"]
        loss = -(teacher_logits.softmax(dim=1) * student_logits.log_softmax(dim=1)).sum(dim=1).mean()
        loss.backward()
        self.optimizer.step()
        ema_update(self.teacher, self.model, float(self.cfg["teacher_momentum"]))
        probability = float(self.cfg["restore_probability"])
        restored_this_step = 0
        with torch.no_grad():
            for name, parameter in self.model.named_parameters():
                mask = torch.rand(
                    parameter.shape,
                    generator=self.restore_generator,
                    device=parameter.device,
                ) < probability
                restored = int(mask.sum())
                if restored:
                    source = self.anchor_parameters[name]
                    parameter.copy_(torch.where(mask, source, parameter))
                    self.restore_count += restored
                    restored_this_step += restored
        n_augmented = int(low_confidence.sum())
        return AdaptationResult(
            loss=float(loss.detach()),
            n_seen=int(images.shape[0]),
            n_selected=int(images.shape[0]),
            updated=True,
            extras={
                "anchor_confidence_mean": float(anchor_confidence.mean()),
                "augmentation_triggered_slices": float(n_augmented),
                "augmentation_coverage": float(n_augmented / images.shape[0]),
                "teacher_views_when_triggered": float(
                    len(self.cfg["augmentation_scales"]) * len(self.cfg["augmentation_flips"])
                ),
                "restored_parameters_step": float(restored_this_step),
                "restored_parameters_cumulative": float(self.restore_count),
            },
        )

    @torch.no_grad()
    def predict(self, images: torch.Tensor) -> torch.Tensor:
        self.teacher.eval()
        return self.teacher(images)["logits"]
