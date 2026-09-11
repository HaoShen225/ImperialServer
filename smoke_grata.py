"""Run a real-checkpoint GraTA smoke test for either locked target stream."""

from __future__ import annotations

import argparse
from copy import deepcopy
import math
from pathlib import Path

import torch
from torch import nn

from data import MMSTargetSliceDataset, build_target_slice_loader, build_target_stream
from metrics import evaluate_slice, evaluate_volume
from model import build_model, load_source_checkpoint
from run_tta import run_random_slice_batch, run_volume
from tta_methods import build_method
from utils import get_device, load_config, set_seed


def _validate_method(
    method, method_cfg: dict, learning_rate: float | None = None
) -> None:
    expected_profile = (
        "official_mechanism_mms_multiclass"
        if learning_rate is None
        else "lr_sweep"
    )
    if method_cfg["profile_kind"] != expected_profile:
        raise RuntimeError(f"GraTA smoke requires the {expected_profile} profile")
    if learning_rate is not None and method_cfg["lr"] != learning_rate:
        raise RuntimeError("GraTA smoke did not apply the requested learning-rate ceiling")
    if not isinstance(method.optimizer, torch.optim.Adam):
        raise RuntimeError("GraTA smoke is not using Adam")
    group = method.optimizer.param_groups[0]
    if group["lr"] != method_cfg["lr"] or group["betas"] != (
        method_cfg["beta1"], method_cfg["beta2"]
    ) or group["weight_decay"] != method_cfg["weight_decay"]:
        raise RuntimeError("GraTA Adam settings differ from the locked configuration")

    expected = {
        f"{module_name}.{parameter_name}"
        for module_name, module in method.model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        for parameter_name in ("weight", "bias")
    }
    if set(method.trainable_parameter_names()) != expected:
        raise RuntimeError("GraTA does not expose exactly all BN affine parameters")
    for module in method.model.modules():
        if isinstance(module, nn.BatchNorm2d) and (
            module.running_mean is not None or module.running_var is not None
        ):
            raise RuntimeError("GraTA BatchNorm retained running-statistics buffers")


def _validate_adaptation(adaptation: dict, batch_size: int, base_lr: float) -> None:
    if int(adaptation["n_seen"]) != batch_size or int(adaptation["n_selected"]) != batch_size:
        raise RuntimeError("GraTA smoke did not adapt on the complete arrival batch")
    if not adaptation["updated"]:
        raise RuntimeError("GraTA smoke did not update BN affine parameters")
    if adaptation["loss"] is None or not math.isfinite(float(adaptation["loss"])):
        raise RuntimeError("GraTA smoke produced a non-finite consistency loss")
    extras = adaptation["extras"]
    required = {
        "entropy_loss", "consistency_loss", "gradient_cosine",
        "entropy_gradient_norm", "consistency_gradient_norm", "effective_lr",
        "weak_view_count", "adaptation_seconds", "prediction_seconds", "parameter_drift",
    }
    if not required.issubset(extras):
        raise RuntimeError(f"Missing GraTA diagnostics: {sorted(required - set(extras))}")
    if not all(math.isfinite(float(extras[key])) for key in required):
        raise RuntimeError("GraTA smoke produced a non-finite diagnostic")
    if int(extras["weak_view_count"]) != 6:
        raise RuntimeError("GraTA smoke did not use the official six weak views")
    if not -1.0 <= float(extras["gradient_cosine"]) <= 1.0:
        raise RuntimeError("GraTA gradient cosine is outside [-1, 1]")
    if not 0.0 <= float(extras["effective_lr"]) <= base_lr:
        raise RuntimeError("GraTA effective learning rate is outside [0, beta]")
    if float(extras["parameter_drift"]) <= 0.0:
        raise RuntimeError("GraTA smoke did not change model parameters")


def _finite_metrics(
    predictions: torch.Tensor, targets: torch.Tensor, cfg: dict, stream_mode: str
) -> None:
    classes = [int(value) for value in cfg["evaluation"]["classes"]]
    names = {
        int(key): value for key, value in cfg["evaluation"]["class_names"].items()
    }
    if stream_mode == "patient_volume":
        scores = evaluate_volume(
            predictions.numpy(), targets.numpy(), classes=classes, class_names=names
        )
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError(f"GraTA smoke produced non-finite metrics: {scores}")
        return
    for prediction, target in zip(predictions, targets):
        scores, _ = evaluate_slice(
            prediction.numpy(), target.numpy(), classes=classes, class_names=names
        )
        if not all(math.isfinite(value) for value in scores.values()):
            raise RuntimeError(f"GraTA smoke produced non-finite metrics: {scores}")


def smoke(
    cfg: dict,
    stream_mode: str,
    seed: int,
    vendor: str,
    device: torch.device,
    learning_rate: float | None,
) -> None:
    set_seed(int(cfg["experiment"]["harness_seed"]), deterministic=True)
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    model = build_model(cfg, pretrained_override=False)
    load_source_checkpoint(model, checkpoint, map_location="cpu")
    method_cfg = deepcopy(cfg["methods"]["grata"])
    if learning_rate is not None:
        method_cfg["profile_kind"] = "lr_sweep"
        method_cfg["lr"] = learning_rate
    method = build_method("grata", model, method_cfg, cfg["tta"], device)
    _validate_method(method, method_cfg, learning_rate)
    batch_size = int(cfg["tta"]["batch_size"])

    if stream_mode == "slice_random":
        loader = build_target_slice_loader(
            vendor, cfg, order_seed=seed, batch_size=batch_size
        )
        dataset = loader.dataset
        if not isinstance(dataset, MMSTargetSliceDataset):
            raise TypeError("GraTA random-slice smoke received an unexpected dataset")
        batch = next(iter(loader))
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
        predictions, records = run_volume(
            method, volume["image"][:batch_size], batch_size, device
        )
        if len(records) != 1:
            raise RuntimeError("GraTA patient smoke unexpectedly split one arrival batch")
        adaptation = records[0]
        targets = dataset.load_mask(volume)[:batch_size]
        order_hash = dataset.target_order_sha256

    if predictions.shape != targets.shape:
        raise RuntimeError(
            f"GraTA prediction shape {tuple(predictions.shape)} != target shape {tuple(targets.shape)}"
        )
    _validate_adaptation(adaptation, batch_size, float(method_cfg["lr"]))
    _finite_metrics(predictions, targets, cfg, stream_mode)
    print(
        f"[SMOKE] method=grata stream={stream_mode} batch_size={batch_size} "
        f"device={device.type} vendor={vendor} seed={seed} "
        f"cosine={float(adaptation['extras']['gradient_cosine']):.6f} "
        f"effective_lr={float(adaptation['extras']['effective_lr']):.8f} "
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
        "--learning-rate",
        type=float,
        help="Override GraTA's dynamic learning-rate ceiling for sweep smoke tests",
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
        raise RuntimeError("GraTA smoke requires adapt_then_predict with vendor reset")
    smoke(
        cfg,
        args.stream_mode,
        args.seed,
        args.vendor,
        get_device(args.device),
        args.learning_rate,
    )


if __name__ == "__main__":
    main()
