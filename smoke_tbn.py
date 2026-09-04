"""Run a real-checkpoint TBN smoke test for either locked target stream."""

from __future__ import annotations

import argparse
import math
from copy import deepcopy
from pathlib import Path

import torch
from torch import nn

from data import MMSTargetSliceDataset, build_target_slice_loader, build_target_stream
from metrics import evaluate_slice, evaluate_volume
from model import build_model, load_source_checkpoint
from run_tta import run_random_slice_batch, run_volume
from tta_methods import build_method
from utils import get_device, load_config, set_seed, state_dict_sha256


LOCKED_TBN = {
    "profile_kind": "batch_statistics_only",
    "update_scope": "none",
    "bn_policy": "batch_no_running",
}


def _validate_method(method, method_cfg: dict) -> None:
    for key, expected in LOCKED_TBN.items():
        if method_cfg.get(key) != expected:
            raise RuntimeError(
                f"Unexpected locked TBN setting {key}: "
                f"{method_cfg.get(key)!r} != {expected!r}"
            )
    if method.optimizer is not None:
        raise RuntimeError("TBN must not construct an optimizer")
    if method.trainable_parameter_names():
        raise RuntimeError("TBN unexpectedly exposes trainable parameters")
    if method.model.training:
        raise RuntimeError("TBN must keep the model in evaluation mode")
    batch_norms = [
        module for module in method.model.modules()
        if isinstance(module, nn.BatchNorm2d)
    ]
    if not batch_norms:
        raise RuntimeError("TBN smoke found no BatchNorm2d layers")
    for module in batch_norms:
        if module.training:
            raise RuntimeError("TBN BatchNorm module unexpectedly entered training mode")
        if module.track_running_stats:
            raise RuntimeError("TBN BatchNorm still tracks running statistics")
        if module.running_mean is not None or module.running_var is not None:
            raise RuntimeError("TBN BatchNorm retained running-statistics buffers")


def _validate_adaptation(adaptation: dict, batch_size: int) -> None:
    if adaptation["updated"]:
        raise RuntimeError("TBN smoke performed a parameter update")
    if int(adaptation["n_seen"]) != batch_size:
        raise RuntimeError("TBN smoke recorded the wrong arrival batch size")
    if int(adaptation["n_selected"]) != 0 or adaptation["loss"] is not None:
        raise RuntimeError("TBN smoke reported an optimizer selection or loss")
    extras = adaptation["extras"]
    if float(extras["parameter_drift"]) != 0.0:
        raise RuntimeError("TBN changed source parameters")
    for key in ("adaptation_seconds", "prediction_seconds"):
        value = float(extras[key])
        if not math.isfinite(value) or value < 0.0:
            raise RuntimeError(f"Invalid TBN diagnostic {key}={value}")


def _finite_metrics(predictions: torch.Tensor, targets: torch.Tensor, cfg: dict, stream_mode: str) -> None:
    classes = [int(value) for value in cfg["evaluation"]["classes"]]
    names = {
        int(key): value for key, value in cfg["evaluation"]["class_names"].items()
    }
    if stream_mode == "patient_volume":
        scores = evaluate_volume(
            predictions.numpy(), targets.numpy(), classes=classes, class_names=names
        )
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError(f"TBN smoke produced non-finite metrics: {scores}")
        return
    for prediction, target in zip(predictions, targets):
        scores, _ = evaluate_slice(
            prediction.numpy(), target.numpy(), classes=classes, class_names=names
        )
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError(f"TBN smoke produced non-finite metrics: {scores}")


def smoke(
    cfg: dict,
    stream_mode: str,
    seed: int,
    vendor: str,
    device: torch.device,
) -> None:
    set_seed(int(cfg["experiment"]["harness_seed"]), deterministic=True)
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    model = build_model(cfg, pretrained_override=False)
    load_source_checkpoint(model, checkpoint, map_location="cpu")
    method_cfg = deepcopy(cfg["methods"]["tbn"])
    method = build_method("tbn", model, method_cfg, cfg["tta"], device)
    _validate_method(method, method_cfg)
    initial_hash = state_dict_sha256(method.model.state_dict())
    batch_size = int(cfg["tta"]["batch_size"])

    if stream_mode == "slice_random":
        loader = build_target_slice_loader(
            vendor, cfg, order_seed=seed, batch_size=batch_size
        )
        dataset = loader.dataset
        if not isinstance(dataset, MMSTargetSliceDataset):
            raise TypeError("TBN random-slice smoke received an unexpected dataset")
        batch = next(iter(loader))
        if int(batch["image"].shape[0]) != batch_size:
            raise RuntimeError("TBN random-slice smoke did not receive a full batch")
        predictions, adaptation, _ = run_random_slice_batch(
            method, batch["image"], device
        )
        targets = dataset.load_masks(list(batch["mask_path"]))
        order_hash = dataset.slice_order_sha256
    else:
        dataset = build_target_stream(vendor, cfg, order_seed=seed)
        volume = next(
            dataset[index]
            for index in range(len(dataset))
            if int(dataset[index]["n_slices"]) >= batch_size
        )
        images = volume["image"][:batch_size]
        predictions, records = run_volume(method, images, batch_size, device)
        if len(records) != 1:
            raise RuntimeError("TBN patient smoke unexpectedly split one arrival batch")
        adaptation = records[0]
        targets = dataset.load_mask(volume)[:batch_size]
        order_hash = dataset.target_order_sha256

    if predictions.shape != targets.shape:
        raise RuntimeError(
            f"TBN smoke prediction shape {tuple(predictions.shape)} != "
            f"{tuple(targets.shape)}"
        )
    _validate_adaptation(adaptation, batch_size)
    _finite_metrics(predictions, targets, cfg, stream_mode)
    if state_dict_sha256(method.model.state_dict()) != initial_hash:
        raise RuntimeError("TBN model state changed during inference")
    print(
        f"[SMOKE] method=tbn stream={stream_mode} batch_size={batch_size} "
        f"device={device.type} vendor={vendor} seed={seed} order={order_hash} passed"
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
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    cfg["tta"]["stream_mode"] = args.stream_mode
    cfg["tta"]["batch_size"] = args.batch_size or (
        4 if args.stream_mode == "patient_volume" else 8
    )
    if cfg["tta"]["timing"] != "adapt_then_predict" or cfg["tta"]["reset"] != "vendor":
        raise RuntimeError("TBN smoke requires adapt_then_predict with vendor reset")
    smoke(cfg, args.stream_mode, args.seed, args.vendor, get_device(args.device))


if __name__ == "__main__":
    main()
