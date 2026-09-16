from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from conftest import method_config
from run_tta import _csl_probe_counts, attach_csl_label_probe, run_volume
from tta_methods import build_method
from tta_methods.grata_adaption.augment import (
    WEAK_VIEWS,
    apply_per_sample_weak_views,
    apply_weak_view,
    sample_weak_views,
)
from tta_methods.grata_adaption.csl import (
    select_csl_weights,
    weighted_cross_entropy,
)


VARIANTS = {
    "grata_adaption_global_hard": ("global", "hard"),
    "grata_adaption_global_smooth": ("global", "smooth"),
    "grata_adaption_classwise_hard": ("classwise", "hard"),
    "grata_adaption_classwise_smooth": ("classwise", "smooth"),
}


def _selection(probabilities, scope, mode, minimum=16):
    return select_csl_weights(
        probabilities,
        scope=scope,
        mode=mode,
        alpha=8.0,
        epsilon=1e-8,
        classwise_min_pixels=minimum,
    )


def test_global_hard_and_smooth_share_reliable_region():
    generator = torch.Generator().manual_seed(19)
    probabilities = torch.randn(2, 4, 16, 16, generator=generator).softmax(dim=1)
    hard = _selection(probabilities, "global", "hard")
    smooth = _selection(probabilities, "global", "smooth")

    assert torch.equal(hard.labels, smooth.labels)
    assert torch.equal(hard.hard_mask, smooth.hard_mask)
    assert torch.equal(hard.weights, hard.hard_mask.float())
    assert float(smooth.weights.min()) >= 0.0
    assert float(smooth.weights.max()) <= 1.0
    assert torch.equal(
        smooth.weights[smooth.hard_mask],
        torch.ones_like(smooth.weights[smooth.hard_mask]),
    )
    assert torch.all(smooth.weights >= hard.weights)


def test_classwise_small_class_uses_same_image_global_fallback():
    generator = torch.Generator().manual_seed(29)
    logits = torch.randn(1, 4, 16, 16, generator=generator) * 0.1
    logits[:, 0] += 2.0
    logits[:, 1, :4, :4] += 4.0
    probabilities = logits.softmax(dim=1)
    global_selection = _selection(probabilities, "global", "smooth", minimum=128)
    classwise = _selection(probabilities, "classwise", "smooth", minimum=128)
    small_class = classwise.labels == 1

    assert int(small_class.sum()) < 128
    assert classwise.classwise_fallbacks >= 1
    assert torch.equal(
        classwise.weights[small_class], global_selection.weights[small_class]
    )
    assert torch.equal(
        classwise.hard_mask[small_class], global_selection.hard_mask[small_class]
    )


def test_weighted_cross_entropy_normalizes_by_realized_weight_sum():
    logits = torch.tensor([[[[2.0, -1.0]], [[-1.0, 2.0]]]])
    labels = torch.tensor([[[0, 1]]])
    weights = torch.tensor([[[1.0, 0.25]]])
    pixel_loss = torch.nn.functional.cross_entropy(logits, labels, reduction="none")
    observed = weighted_cross_entropy(logits, labels, weights, epsilon=1e-8)
    expected = (pixel_loss * weights).sum() / weights.sum()
    assert observed is not None
    assert observed == pytest.approx(float(expected))
    assert weighted_cross_entropy(logits, labels, torch.zeros_like(weights), 1e-8) is None


def test_per_image_weak_views_are_unique_replayable_and_geometry_aligned():
    first = sample_weak_views(4, 3, WEAK_VIEWS, torch.Generator().manual_seed(71))
    second = sample_weak_views(4, 3, WEAK_VIEWS, torch.Generator().manual_seed(71))
    assert first == second
    assert all(len(set(views)) == 3 for views in first)

    value = torch.arange(2 * 4 * 4).reshape(2, 4, 4)
    views = ["horizontal_flip", "rotate_90"]
    observed = apply_per_sample_weak_views(value, views)
    expected = torch.cat(
        [apply_weak_view(value[index : index + 1], view) for index, view in enumerate(views)]
    )
    assert torch.equal(observed, expected)


@pytest.mark.parametrize("name", sorted(VARIANTS))
def test_all_grata_adaption_aliases_run_locked_profiles(
    name, config, tiny_model, images
):
    method = build_method(
        name,
        deepcopy(tiny_model),
        method_config(config, name),
        config["tta"],
        torch.device("cpu"),
    )
    logits, info = method.process_batch(images)
    scope, mode = VARIANTS[name]

    assert method.cfg["selector_scope"] == scope
    assert method.cfg["weight_mode"] == mode
    assert isinstance(method.optimizer, torch.optim.Adam)
    for module in method.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.running_mean is None
            assert module.running_var is None
    assert logits.shape == (4, 4, 16, 16)
    assert info.updated
    assert info.probe_payload is not None
    assert info.probe_payload["csl_reliable"]["selected"].shape == images.shape[:1] + images.shape[-2:]
    assert info.extras["teacher_forward_count"] == 1.0
    assert info.extras["student_forward_count"] == 1.0
    assert 0.0 <= info.extras["reliable_pixel_coverage"] <= 1.0
    assert 0.0 <= info.extras["mean_csl_weight"] <= 1.0


def test_grata_adaption_reset_replays_model_and_method_rng(
    config, tiny_model, images
):
    name = "grata_adaption_classwise_smooth"
    method = build_method(
        name,
        deepcopy(tiny_model),
        method_config(config, name),
        config["tta"],
        torch.device("cpu"),
    )
    first_logits, first_info = method.process_batch(images)
    method.reset()
    second_logits, second_info = method.process_batch(images)
    assert torch.equal(first_logits, second_logits)
    assert first_info.n_selected == second_info.n_selected
    replayed = set(first_info.extras) - {"adaptation_seconds", "prediction_seconds"}
    assert replayed == set(second_info.extras) - {"adaptation_seconds", "prediction_seconds"}
    assert all(first_info.extras[key] == second_info.extras[key] for key in replayed)


def test_teacher_is_single_pass_and_n_students_are_evaluated(
    config, tiny_model, images
):
    cfg = method_config(config, "grata_adaption_global_smooth")
    cfg["weak_view_samples"] = 2
    method = build_method(
        "grata_adaption_global_smooth",
        deepcopy(tiny_model),
        cfg,
        config["tta"],
        torch.device("cpu"),
    )
    calls = []
    handle = method.model.register_forward_hook(lambda *_: calls.append(1))
    try:
        info = method.adapt(images[:2])
    finally:
        handle.remove()
    assert len(calls) == 4  # entropy + one teacher + two student views
    assert info.extras["teacher_forward_count"] == 1.0
    assert info.extras["student_forward_count"] == 2.0


def test_csl_probe_reports_hard_and_weighted_accuracy():
    target = torch.tensor([[[0, 1], [3, 3]]])
    payload = {
        "selected": torch.tensor([[[True, True], [False, True]]]),
        "labels": torch.tensor([[[0, 1], [2, 3]]]),
        "weights": torch.tensor([[[0.2, 1.0], [0.5, 1.0]]]),
    }
    probe = _csl_probe_counts(payload, target)
    assert probe["seen_pixels"] == 4
    assert probe["reliable_pixels"] == 3
    assert probe["reliable_coverage"] == pytest.approx(0.75)
    assert probe["reliable_accuracy"] == pytest.approx(1.0)
    assert probe["effective_weight_coverage"] == pytest.approx(0.675)
    assert probe["weighted_accuracy"] == pytest.approx(2.2 / 2.7)
    assert probe["reliable_foreground_coverage"] == pytest.approx(2.0 / 3.0)
    assert probe["weighted_foreground_accuracy"] == pytest.approx(2.0 / 2.5)


def test_csl_probe_cannot_change_method_state(config, tiny_model, images):
    method = build_method(
        "grata_adaption_global_smooth",
        deepcopy(tiny_model),
        method_config(config, "grata_adaption_global_smooth"),
        config["tta"],
        torch.device("cpu"),
    )
    prediction, records = run_volume(method, images, 4, torch.device("cpu"))
    state_before = {
        name: value.detach().clone() for name, value in method.model.state_dict().items()
    }
    probe = attach_csl_label_probe(records, torch.zeros_like(prediction))
    state_after = method.model.state_dict()
    assert probe is not None
    assert "csl_label_probe" in records[0]
    assert "_probe_payload" not in records[0]
    assert all(torch.equal(value, state_after[name]) for name, value in state_before.items())
