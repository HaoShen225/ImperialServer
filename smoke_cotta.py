"""Run a real-checkpoint CoTTA smoke test for either locked target stream."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn

from data import MMSTargetSliceDataset, build_target_slice_loader, build_target_stream
from metrics import evaluate_slice
from model import build_model, load_source_checkpoint
from run_tta import run_random_slice_batch, run_volume
from tta_methods import build_method
from utils import get_device, load_config, set_seed, state_dict_sha256


LOCKED_COTTA = {
    "profile_kind": "official_segmentation_mms_adapted",
    "steps": 1,
    "update_scope": "all",
    "bn_policy": "batch_no_running",
    "optimizer": "adam",
    "lr": 7.5e-6,
    "beta1": 0.9,
    "beta2": 0.999,
    "weight_decay": 0.0,
    "teacher_momentum": 0.999,
    "confidence_gate": 0.9,
    "restore_probability": 0.01,
    "augmentation_scales": [0.5, 0.75, 1.0, 1.25, 1.5, 1.75, 2.0],
    "augmentation_flips": [False, True],
}


def _validate_locked_config(method_cfg: dict, learning_rate: float | None = None) -> None:
    expected_cfg = deepcopy(LOCKED_COTTA)
    if learning_rate is not None:
        expected_cfg["profile_kind"] = "lr_sweep"
        expected_cfg["lr"] = learning_rate
    for key, expected in expected_cfg.items():
        if method_cfg.get(key) != expected:
            raise RuntimeError(
                f"Unexpected locked CoTTA setting {key}: {method_cfg.get(key)!r} != {expected!r}"
            )


def _validate_method(method, method_cfg: dict) -> None:
    if not isinstance(method.optimizer, torch.optim.Adam):
        raise RuntimeError("CoTTA smoke is not using Adam")
    if len(method.optimizer.param_groups) != 1:
        raise RuntimeError("CoTTA must use one full-model optimizer group")
    group = method.optimizer.param_groups[0]
    if group["lr"] != method_cfg["lr"] or group["betas"] != (
        method_cfg["beta1"], method_cfg["beta2"]
    ) or group["weight_decay"] != method_cfg["weight_decay"]:
        raise RuntimeError("CoTTA optimizer differs from the smoke configuration")
    if set(method.trainable_parameter_names()) != {
        name for name, _ in method.model.named_parameters()
    }:
        raise RuntimeError("CoTTA does not expose the complete model to Adam")
    for module in method.model.modules():
        if isinstance(module, nn.BatchNorm2d) and (
            module.running_mean is not None or module.running_var is not None
        ):
            raise RuntimeError("CoTTA student BatchNorm retained running-statistics buffers")
    if method.teacher.training or method.anchor.training:
        raise RuntimeError("CoTTA teacher or anchor is not in evaluation mode")


def _validate_adaptation(adaptation: dict, batch_size: int, n_views: int) -> None:
    if not adaptation["updated"] or int(adaptation["n_seen"]) != batch_size:
        raise RuntimeError("CoTTA smoke did not adapt on the complete arrival batch")
    if int(adaptation["n_selected"]) != batch_size:
        raise RuntimeError("CoTTA smoke did not select the complete arrival batch")
    if adaptation["loss"] is None or not math.isfinite(float(adaptation["loss"])):
        raise RuntimeError("CoTTA smoke produced a non-finite loss")
    extras = adaptation["extras"]
    if int(extras["augmentation_triggered_slices"]) != batch_size:
        raise RuntimeError("Forced CoTTA smoke did not exercise augmentation for every slice")
    if float(extras["augmentation_coverage"]) != 1.0:
        raise RuntimeError("Forced CoTTA augmentation coverage is not one")
    if int(extras["teacher_views_when_triggered"]) != n_views:
        raise RuntimeError("CoTTA smoke used the wrong teacher-view count")
    if float(extras["parameter_drift"]) <= 0.0:
        raise RuntimeError("CoTTA smoke did not change student parameters")
    for key in ("adaptation_seconds", "prediction_seconds", "restored_parameters_step"):
        value = float(extras[key])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"Invalid CoTTA diagnostic {key}={value}")


def _validate_metrics(predictions: torch.Tensor, targets: torch.Tensor, cfg: dict) -> None:
    classes = [int(value) for value in cfg["evaluation"]["classes"]]
    names = {int(key): value for key, value in cfg["evaluation"]["class_names"].items()}
    for prediction, target in zip(predictions, targets):
        metrics, _ = evaluate_slice(
            prediction.numpy(), target.numpy(), classes=classes, class_names=names
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"CoTTA smoke produced non-finite metrics: {metrics}")


def smoke(
    cfg: dict,
    stream_mode: str,
    seed: int,
    vendor: str,
    device: torch.device,
    smoke_scales: list[float] | None,
    learning_rate: float | None,
) -> None:
    set_seed(int(cfg["experiment"]["harness_seed"]), deterministic=True)
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    model = build_model(cfg, pretrained_override=False)
    load_source_checkpoint(model, checkpoint, map_location="cpu")

    method_cfg = deepcopy(cfg["methods"]["cotta"])
    if learning_rate is not None:
        method_cfg["profile_kind"] = "lr_sweep"
        method_cfg["lr"] = learning_rate
    _validate_locked_config(method_cfg, learning_rate)
    if smoke_scales is not None:
        method_cfg["augmentation_scales"] = smoke_scales
    # The smoke must exercise the ensemble independently of checkpoint confidence.
    method_cfg["confidence_gate"] = 1.1
    method = build_method("cotta", model, method_cfg, cfg["tta"], device)
    _validate_method(method, method_cfg)
    anchor_hash = state_dict_sha256(method.anchor.state_dict())
    teacher_hash = state_dict_sha256(method.teacher.state_dict())

    batch_size = int(cfg["tta"]["batch_size"])
    if stream_mode == "slice_random":
        loader = build_target_slice_loader(vendor, cfg, order_seed=seed, batch_size=batch_size)
        dataset = loader.dataset
        if not isinstance(dataset, MMSTargetSliceDataset):
            raise TypeError("CoTTA random-slice smoke received an unexpected dataset")
        batch = next(iter(loader))
        if list(batch["slice_id"]) != dataset.slice_order[: len(batch["slice_id"])]:
            raise RuntimeError("CoTTA smoke batch violates the seeded random-slice order")
        predictions, adaptation, _ = run_random_slice_batch(method, batch["image"], device)
        targets = dataset.load_masks(list(batch["mask_path"]))
        order_hash = dataset.slice_order_sha256
    else:
        dataset = build_target_stream(vendor, cfg, order_seed=seed)
        volume = dataset[0]
        images = volume["image"][:batch_size]
        predictions, records = run_volume(method, images, batch_size, device)
        if len(records) != 1:
            raise RuntimeError("CoTTA patient smoke unexpectedly split one arrival batch")
        adaptation = records[0]
        targets = dataset.load_mask(volume)[:batch_size]
        order_hash = dataset.target_order_sha256

    if predictions.shape != targets.shape:
        raise RuntimeError(
            f"CoTTA smoke prediction shape {tuple(predictions.shape)} != {tuple(targets.shape)}"
        )
    _validate_adaptation(
        adaptation,
        int(predictions.shape[0]),
        len(method_cfg["augmentation_scales"]) * len(method_cfg["augmentation_flips"]),
    )
    _validate_metrics(predictions, targets, cfg)
    if state_dict_sha256(method.anchor.state_dict()) != anchor_hash:
        raise RuntimeError("CoTTA source anchor changed during adaptation")
    if state_dict_sha256(method.teacher.state_dict()) == teacher_hash:
        raise RuntimeError("CoTTA EMA teacher did not change during adaptation")
    print(
        f"[SMOKE] method=cotta stream={stream_mode} batch_size={batch_size} "
        f"device={device.type} vendor={vendor} seed={seed} views="
        f"{int(adaptation['extras']['teacher_views_when_triggered'])} "
        f"order={order_hash} passed"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stream-mode", choices=["patient_volume", "slice_random"], required=True
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--vendor", choices=["B", "C", "D"], default="C")
    parser.add_argument("--batch-size", type=int)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--smoke-scales",
        nargs="+",
        type=float,
        help="Use fewer scales for a fast CPU smoke; omit for the full seven-scale GPU smoke",
    )
    parser.add_argument(
        "--learning-rate",
        type=float,
        help="Override only CoTTA's learning rate for an LR-sweep smoke test",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["tta"]["stream_mode"] = args.stream_mode
    cfg["tta"]["batch_size"] = args.batch_size or (
        4 if args.stream_mode == "patient_volume" else 8
    )
    if cfg["tta"]["timing"] != "adapt_then_predict" or cfg["tta"]["reset"] != "vendor":
        raise RuntimeError("CoTTA smoke requires adapt_then_predict with vendor reset")
    smoke(
        cfg,
        args.stream_mode,
        args.seed,
        args.vendor,
        get_device(args.device),
        args.smoke_scales,
        args.learning_rate,
    )


if __name__ == "__main__":
    main()
