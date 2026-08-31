from __future__ import annotations

import torch
from torch import nn

import model as model_module
from model import build_model
from train_source import build_source_optimizer, source_loss
from utils import set_seed, state_dict_sha256


def test_resunet34_shape_and_bn_decoder(config):
    model = build_model(config, pretrained_override=False)
    model.eval()
    with torch.no_grad():
        output = model(torch.rand(1, 1, 64, 64), return_features=True)
    assert output["logits"].shape == (1, 4, 64, 64)
    assert output["features"].shape[-2:] == (64, 64)
    assert any(isinstance(module, nn.BatchNorm2d) for name, module in model.named_modules() if name.startswith("decoder"))


def test_stochastic_initialization_is_seeded_and_uses_no_pretrained_weights(config, monkeypatch):
    original = model_module.resnet34
    weights_arguments = []

    def capture_weights(*args, **kwargs):
        weights_arguments.append(kwargs.get("weights"))
        return original(*args, **kwargs)

    monkeypatch.setattr(model_module, "resnet34", capture_weights)
    set_seed(2022)
    first = build_model(config)
    first_hash = state_dict_sha256(first.state_dict())
    del first
    set_seed(2022)
    second = build_model(config)
    second_hash = state_dict_sha256(second.state_dict())
    del second
    set_seed(2023)
    third = build_model(config)
    third_hash = state_dict_sha256(third.state_dict())

    assert config["experiment"]["initialization_profile"] == "stochastic"
    assert config["model"]["pretrained_encoder"] is False
    assert weights_arguments == [None, None, None]
    assert first_hash == second_hash
    assert first_hash != third_hash


def test_source_loss_and_optimizer_groups(tiny_model):
    logits = tiny_model(torch.rand(2, 1, 16, 16))["logits"]
    target = torch.randint(0, 4, (2, 16, 16))
    loss, parts = source_loss(logits, target)
    assert torch.isfinite(loss)
    assert set(parts) == {"cross_entropy", "dice_loss"}


def test_source_optimizer_exempts_bias_and_bn(config, tiny_model):
    optimizer = build_source_optimizer(tiny_model, config)
    assert len(optimizer.param_groups) == 4
    assert {group["weight_decay"] for group in optimizer.param_groups} == {0.0, config["source"]["weight_decay"]}
