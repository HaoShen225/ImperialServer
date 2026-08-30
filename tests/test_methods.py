from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from conftest import method_config
from tta_methods import METHODS, build_method
from tta_methods.rotta.rbn import RobustBatchNorm2d


@pytest.mark.parametrize("name", sorted(METHODS))
def test_every_method_constructs_and_processes(name, config, tiny_model, images):
    method = build_method(name, deepcopy(tiny_model), method_config(config, name), config["tta"], torch.device("cpu"))
    logits, info = method.process_batch(images)
    assert logits.shape == (4, 4, 16, 16)
    assert info.n_seen > 0
    assert "parameter_drift" in info.extras


@pytest.mark.parametrize("name", ["tent", "eata", "sar", "cotta", "roid", "deyo"])
def test_batch_stat_methods_have_no_running_buffers(name, config, tiny_model):
    method = build_method(name, deepcopy(tiny_model), method_config(config, name), config["tta"], torch.device("cpu"))
    for module in method.model.modules():
        if isinstance(module, nn.BatchNorm2d):
            assert module.running_mean is None
            assert module.running_var is None


@pytest.mark.parametrize("name", ["cotta", "rotta", "roid", "deyo"])
def test_stochastic_method_reset_replays_exactly(name, config, tiny_model, images):
    method = build_method(name, deepcopy(tiny_model), method_config(config, name), config["tta"], torch.device("cpu"))
    first_logits, first_info = method.process_batch(images)
    method.reset()
    second_logits, second_info = method.process_batch(images)
    assert torch.equal(first_logits, second_logits)
    assert first_info.n_selected == second_info.n_selected
    assert first_info.updated == second_info.updated


def test_rotta_rbn_buffers_evolve_and_reset(config, tiny_model, images):
    method = build_method("rotta", deepcopy(tiny_model), method_config(config, "rotta"), config["tta"], torch.device("cpu"))
    initial = {name: value.clone() for name, value in method.model.state_dict().items() if name.endswith("source_mean")}
    method.process_batch(images)
    evolved = {name: value.clone() for name, value in method.model.state_dict().items() if name.endswith("source_mean")}
    assert any(not torch.equal(initial[name], evolved[name]) for name in initial)
    method.reset()
    restored = {name: value for name, value in method.model.state_dict().items() if name.endswith("source_mean")}
    assert all(torch.equal(initial[name], restored[name]) for name in initial)
    assert any(isinstance(module, RobustBatchNorm2d) for module in method.model.modules())


def test_unknown_method_fails(config, tiny_model):
    with pytest.raises(ValueError, match="Unknown method"):
        build_method("unknown", tiny_model, {}, config["tta"], torch.device("cpu"))


@pytest.mark.parametrize("name", sorted(METHODS))
def test_only_declared_trainable_parameters_can_change(name, config, tiny_model, images):
    method = build_method(name, deepcopy(tiny_model), method_config(config, name), config["tta"], torch.device("cpu"))
    before = {key: value.detach().clone() for key, value in method.model.named_parameters()}
    _, info = method.process_batch(images)
    after = dict(method.model.named_parameters())
    changed = {key for key in before if not torch.equal(before[key], after[key].detach())}
    assert changed.issubset(set(method.trainable_parameter_names()))
    if info.updated and name not in {"source"}:
        assert changed
