from __future__ import annotations

from copy import deepcopy

import pytest
import torch

from run_tent_lr_sweep import configure_sweep, resolve_learning_rate
from tta_methods import build_method


@pytest.mark.parametrize(
    ("text", "expected", "tag"),
    [
        ("1e-4", 1e-4, "1e-4"),
        ("1e-3", 1e-3, "1e-3"),
        ("1e-2", 1e-2, "1e-2"),
        ("1e-1", 1e-1, "1e-1"),
        ("1", 1.0, "1"),
    ],
)
def test_resolve_tent_sweep_learning_rate(text, expected, tag):
    value, resolved_tag = resolve_learning_rate(text)
    assert value == pytest.approx(expected)
    assert resolved_tag == tag


def test_tent_sweep_rejects_unplanned_learning_rate():
    with pytest.raises(ValueError, match="must be one of"):
        resolve_learning_rate("0.02")


@pytest.mark.parametrize(
    ("stream_mode", "batch_size"),
    [("patient_volume", 4), ("slice_random", 8)],
)
def test_configure_tent_sweep_is_isolated(
    config, tmp_path, stream_mode, batch_size
):
    original = deepcopy(config)
    resolved = configure_sweep(config, 0.01, "1e-2", stream_mode, tmp_path)
    assert config == original
    assert resolved["methods"]["tent"]["lr"] == pytest.approx(0.01)
    assert resolved["methods"]["tent"]["profile_kind"] == "lr_sweep"
    assert resolved["tta"]["stream_mode"] == stream_mode
    assert resolved["tta"]["batch_size"] == batch_size
    assert resolved["tta"]["results_dir"] == str(tmp_path / "lr_1e-2")


@pytest.mark.parametrize("learning_rate", [1e-4, 1e-3, 1e-2, 1e-1, 1.0])
def test_every_tent_sweep_learning_rate_executes_one_finite_step(
    config, tiny_model, images, learning_rate
):
    method_cfg = deepcopy(config["methods"]["tent"])
    method_cfg["lr"] = learning_rate
    method_cfg["profile_kind"] = "lr_sweep"
    method = build_method(
        "tent", tiny_model, method_cfg, config["tta"], torch.device("cpu")
    )
    logits, adaptation = method.process_batch(images)
    assert torch.isfinite(logits).all()
    assert adaptation.updated
    assert adaptation.loss is not None
    assert learning_rate == pytest.approx(method.optimizer.param_groups[0]["lr"])
