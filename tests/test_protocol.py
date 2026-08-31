from __future__ import annotations

import inspect
from copy import deepcopy

import torch

from run_tta import (
    _probe_stage_counts,
    _validate_evaluation_target,
    attach_entropy_label_probe,
    run_volume,
)
from tta_methods import build_method


def test_run_volume_signature_has_no_label_boundary():
    assert list(inspect.signature(run_volume).parameters) == ["method", "images", "batch_size", "device"]


def test_target_label_permutation_cannot_change_prediction(config, tiny_model, images):
    method_cfg = dict(config["methods"]["source"])
    method = build_method("source", tiny_model, method_cfg, config["tta"], torch.device("cpu"))
    first, _ = run_volume(method, images, 4, torch.device("cpu"))
    arbitrary_target = torch.randint(0, 4, (4, 16, 16))
    arbitrary_target = arbitrary_target.flatten()[torch.randperm(arbitrary_target.numel())].reshape_as(arbitrary_target)
    assert arbitrary_target.shape == first.shape
    method.reset()
    second, _ = run_volume(method, images, 4, torch.device("cpu"))
    assert torch.equal(first, second)


def test_timing_modes_produce_distinct_first_batch(config, tiny_model, images):
    method_cfg = dict(config["methods"]["tent"])
    before_cfg = dict(config["tta"])
    before_cfg["timing"] = "predict_then_adapt"
    after_cfg = dict(config["tta"])
    after_cfg["timing"] = "adapt_then_predict"
    first = build_method("tent", tiny_model, method_cfg, before_cfg, torch.device("cpu"))
    second_model = __import__("copy").deepcopy(tiny_model)
    second = build_method("tent", second_model, method_cfg, after_cfg, torch.device("cpu"))
    logits_before, _ = first.process_batch(images)
    logits_after, _ = second.process_batch(images)
    assert not torch.equal(logits_before, logits_after)


def test_empty_evaluation_target_is_rejected():
    target = torch.zeros((4, 16, 16), dtype=torch.int64)
    with __import__("pytest").raises(ValueError, match="C/PATIENT/ED"):
        _validate_evaluation_target(target, [1, 2, 3], "C", "PATIENT", "ED")


def test_entropy_label_probe_counts_all_and_foreground_pixels():
    target = torch.tensor(
        [
            [[0, 1], [2, 0]],
            [[0, 3], [3, 0]],
        ]
    )
    payload = {
        "selected": torch.tensor([True, False]),
        "labels": torch.tensor(
            [
                [[0, 1], [0, 0]],
                [[3, 3], [3, 3]],
            ]
        ),
    }
    probe = _probe_stage_counts(payload, target)
    assert probe == {
        "seen_slices": 2,
        "selected_slices": 1,
        "selection_coverage": 0.5,
        "selected_pixels": 4,
        "correct_pixels": 3,
        "pixel_accuracy": 0.75,
        "gt_foreground_pixels": 2,
        "correct_gt_foreground_pixels": 1,
        "foreground_pixel_accuracy": 0.5,
    }


def test_entropy_label_probe_empty_selection_uses_null_accuracies():
    target = torch.ones((2, 2, 2), dtype=torch.int64)
    probe = _probe_stage_counts(
        {"selected": torch.tensor([False, False]), "labels": torch.zeros_like(target)},
        target,
    )
    assert probe["selected_slices"] == 0
    assert probe["selected_pixels"] == 0
    assert probe["pixel_accuracy"] is None
    assert probe["foreground_pixel_accuracy"] is None


def test_probe_targets_do_not_change_sar_state(config, tiny_model, images):
    method_cfg = dict(config["methods"]["sar"])
    method_cfg["entropy_margin_factor"] = 2.0
    method_cfg["recovery_threshold"] = -1.0
    method = build_method(
        "sar", tiny_model, method_cfg, config["tta"], torch.device("cpu")
    )
    prediction, records = run_volume(method, images, 4, torch.device("cpu"))
    state_before_probe = {
        name: value.detach().clone() for name, value in method.model.state_dict().items()
    }
    zero_target_records = deepcopy(records)
    foreground_target_records = deepcopy(records)
    zero_probe = attach_entropy_label_probe(
        zero_target_records, torch.zeros_like(prediction)
    )
    foreground_probe = attach_entropy_label_probe(
        foreground_target_records, torch.ones_like(prediction)
    )
    state_after_probe = method.model.state_dict()
    assert all(torch.equal(value, state_after_probe[name]) for name, value in state_before_probe.items())
    assert zero_probe != foreground_probe
    assert "_probe_payload" not in zero_target_records[0]
    assert "entropy_label_probe" in zero_target_records[0]
