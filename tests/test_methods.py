from __future__ import annotations

from copy import deepcopy

import pytest
import torch
from torch import nn

from conftest import method_config
from tta_methods import METHODS, build_method
from tta_methods.rotta.rbn import RobustBatchNorm2d
from tta_methods.sar.sam import SAM


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


def test_tent_uses_locked_sgd_profile(config, tiny_model):
    method = build_method(
        "tent", deepcopy(tiny_model), method_config(config, "tent"), config["tta"], torch.device("cpu")
    )
    assert config["tta"]["batch_size"] == 4
    assert isinstance(method.optimizer, torch.optim.SGD)
    assert len(method.optimizer.param_groups) == 1
    group = method.optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(6.25e-5)
    assert group["momentum"] == pytest.approx(0.9)
    assert group["weight_decay"] == pytest.approx(0.0)
    expected = {
        f"{module_name}.{parameter_name}"
        for module_name, module in method.model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter_name in ("weight", "bias")
    }
    assert set(method.trainable_parameter_names()) == expected


def test_sar_uses_locked_sam_sgd_profile(config, tiny_model):
    method = build_method(
        "sar", deepcopy(tiny_model), method_config(config, "sar"), config["tta"], torch.device("cpu")
    )
    assert config["tta"]["batch_size"] == 4
    assert isinstance(method.optimizer, SAM)
    assert isinstance(method.optimizer.base_optimizer, torch.optim.SGD)
    assert len(method.optimizer.base_optimizer.param_groups) == 1
    group = method.optimizer.base_optimizer.param_groups[0]
    assert group["lr"] == pytest.approx(6.25e-5)
    assert group["momentum"] == pytest.approx(0.9)
    assert group["weight_decay"] == pytest.approx(0.0)
    assert method.optimizer.rho == pytest.approx(0.05)
    expected = {
        f"{module_name}.{parameter_name}"
        for module_name, module in method.model.named_modules()
        if isinstance(module, nn.BatchNorm2d) and not module_name.startswith("encoder.layer4")
        for parameter_name in ("weight", "bias")
    }
    assert set(method.trainable_parameter_names()) == expected


def test_sam_first_step_only_perturbs_and_second_step_uses_sgd_momentum():
    parameter = nn.Parameter(torch.tensor([1.0, -1.0]))
    optimizer = SAM([parameter], lr=0.1, momentum=0.9, weight_decay=0.0, rho=0.05)
    original = parameter.detach().clone()

    first_gradient = torch.tensor([3.0, 4.0])
    parameter.grad = first_gradient.clone()
    optimizer.first_step()
    expected_perturbation = first_gradient * (0.05 / first_gradient.norm())
    assert torch.allclose(parameter, original + expected_perturbation)
    assert optimizer.base_optimizer.state == {}

    second_gradient = torch.tensor([2.0, -1.0])
    parameter.grad = second_gradient.clone()
    optimizer.second_step()
    assert torch.allclose(parameter, original - 0.1 * second_gradient)
    momentum_buffer = optimizer.base_optimizer.state[parameter]["momentum_buffer"]
    assert torch.equal(momentum_buffer, second_gradient)

    next_original = parameter.detach().clone()
    parameter.grad = torch.tensor([1.0, 0.0])
    optimizer.first_step()
    next_second_gradient = torch.tensor([-1.0, 3.0])
    parameter.grad = next_second_gradient.clone()
    optimizer.second_step()
    expected_buffer = 0.9 * second_gradient + next_second_gradient
    assert torch.allclose(optimizer.base_optimizer.state[parameter]["momentum_buffer"], expected_buffer)
    assert torch.allclose(parameter, next_original - 0.1 * expected_buffer)


def test_sar_second_filter_is_subset_of_first(config, tiny_model, images, monkeypatch):
    import importlib

    sar_module = importlib.import_module("tta_methods.sar.method")
    entropies = iter(
        [
            torch.tensor([0.1, 1.0], requires_grad=True),
            torch.tensor([0.1, 0.1], requires_grad=True),
        ]
    )
    monkeypatch.setattr(sar_module, "slice_entropy", lambda logits: next(entropies))
    method = build_method(
        "sar", deepcopy(tiny_model), method_config(config, "sar"), config["tta"], torch.device("cpu")
    )
    info = method.adapt(images[:2])
    first = info.probe_payload["first_filter"]["selected"]
    second = info.probe_payload["second_filter"]["selected"]
    assert torch.equal(first, torch.tensor([True, False]))
    assert torch.equal(second, torch.tensor([True, False]))
    assert torch.all(second <= first)


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
