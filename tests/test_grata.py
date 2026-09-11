from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from conftest import method_config
from tta_methods import build_method
from tta_methods.grata.alignment import (
    aligned_learning_rate,
    gradient_cosine,
    perturb_parameters,
    restore_parameters,
)
from tta_methods.grata.augment import (
    OFFICIAL_WEAK_VIEWS,
    apply_weak_view,
    invert_weak_view,
    strong_style_augmentation,
    weak_probability_ensemble,
)


@pytest.mark.parametrize("view", OFFICIAL_WEAK_VIEWS)
def test_weak_views_are_exactly_invertible(view):
    value = torch.arange(2 * 3 * 5 * 7).reshape(2, 3, 5, 7)
    assert torch.equal(invert_weak_view(apply_weak_view(value, view), view), value)


def test_weak_probability_ensemble_inverse_maps_predictions(images):
    class PointwiseModel(nn.Module):
        def forward(self, value):
            return {"logits": torch.cat([value, value * 2.0, -value, value * 0.5], dim=1)}

    model = PointwiseModel()
    expected = model(images[:2])["logits"].softmax(dim=1)
    observed = weak_probability_ensemble(model, images[:2], OFFICIAL_WEAK_VIEWS)
    assert torch.allclose(observed, expected)


def test_strong_style_augmentation_is_reproducible_and_bounded(config, images):
    augmentation_cfg = deepcopy(config["methods"]["grata"]["strong_augmentation"])
    for key in (
        "brightness_probability",
        "contrast_probability",
        "gamma_probability",
        "noise_probability",
        "blur_probability",
        "blur_channel_probability",
    ):
        augmentation_cfg[key] = 1.0
    first_generator = torch.Generator().manual_seed(71)
    second_generator = torch.Generator().manual_seed(71)
    first, first_diagnostics = strong_style_augmentation(
        images[:2], augmentation_cfg, first_generator
    )
    second, second_diagnostics = strong_style_augmentation(
        images[:2], augmentation_cfg, second_generator
    )
    assert torch.equal(first, second)
    assert first_diagnostics == second_diagnostics
    assert not torch.equal(first, images[:2])
    assert float(first.min()) >= 0.0
    assert float(first.max()) <= 1.0
    assert all(value == 1.0 for value in first_diagnostics.values())


def test_gradient_alignment_primitives_restore_parameters_and_map_lr():
    parameter = nn.Parameter(torch.tensor([1.0, -2.0]))
    gradient = torch.tensor([0.25, -0.5])
    originals = perturb_parameters([parameter], [gradient], scale=1.0)
    assert torch.equal(parameter, torch.tensor([0.75, -1.5]))
    restore_parameters([parameter], originals)
    assert torch.equal(parameter, torch.tensor([1.0, -2.0]))

    cosine, first_norm, second_norm = gradient_cosine(
        [torch.tensor([1.0, 0.0])], [torch.tensor([0.0, 2.0])], epsilon=1e-12
    )
    assert cosine == pytest.approx(0.0)
    assert first_norm == pytest.approx(1.0)
    assert second_norm == pytest.approx(2.0)
    assert aligned_learning_rate(1e-4, -1.0) == pytest.approx(0.0)
    assert aligned_learning_rate(1e-4, 0.0) == pytest.approx(2.5e-5)
    assert aligned_learning_rate(1e-4, 1.0) == pytest.approx(1e-4)


def test_grata_uses_locked_bn_affine_adam_profile(config, tiny_model):
    method = build_method(
        "grata",
        deepcopy(tiny_model),
        method_config(config, "grata"),
        config["tta"],
        torch.device("cpu"),
    )
    assert isinstance(method.optimizer, torch.optim.Adam)
    group = method.optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(1e-4)
    assert group["betas"] == pytest.approx((0.9, 0.999))
    assert group["weight_decay"] == pytest.approx(0.0)
    expected = {
        f"{module_name}.{parameter_name}"
        for module_name, module in method.model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter_name in ("weight", "bias")
    }
    assert set(method.trainable_parameter_names()) == expected
    for module in method.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.running_mean is None
            assert module.running_var is None


def test_grata_reports_alignment_and_reset_replays_exactly(config, tiny_model, images):
    method = build_method(
        "grata",
        deepcopy(tiny_model),
        method_config(config, "grata"),
        config["tta"],
        torch.device("cpu"),
    )
    first_logits, first_info = method.process_batch(images)
    method.reset()
    second_logits, second_info = method.process_batch(images)

    assert torch.equal(first_logits, second_logits)
    assert first_info.updated
    assert first_info.n_seen == first_info.n_selected == images.shape[0]
    replayed_keys = set(first_info.extras) - {
        "adaptation_seconds", "prediction_seconds"
    }
    assert replayed_keys == set(second_info.extras) - {
        "adaptation_seconds", "prediction_seconds"
    }
    assert all(
        first_info.extras[key] == second_info.extras[key]
        for key in replayed_keys
    )
    assert first_info.extras["weak_view_count"] == 6.0
    assert -1.0 <= first_info.extras["gradient_cosine"] <= 1.0
    assert 0.0 <= first_info.extras["effective_lr"] <= 1e-4
    assert first_info.extras["entropy_gradient_norm"] > 0.0
    assert first_info.extras["consistency_gradient_norm"] > 0.0
