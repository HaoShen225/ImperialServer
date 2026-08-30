from __future__ import annotations

import inspect

import torch

from run_tta import run_volume
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
