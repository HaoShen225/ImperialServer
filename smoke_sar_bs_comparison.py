"""GPU smoke test for SAR patient BS=8 and random-slice BS=4 at LR=1e-3."""

from __future__ import annotations

import argparse
import gc
import math
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data import MMSTargetSliceDataset, build_target_slice_loader, build_target_stream
from metrics import evaluate_slice, evaluate_volume
from model import build_model, load_source_checkpoint
from run_sar_bs_comparison import (
    BATCH_SIZES,
    LEARNING_RATE,
    SOURCE_SEEDS,
    STREAM_MODES,
    configure_sar_comparison,
)
from run_tta import attach_entropy_label_probe, run_random_slice_batch, run_volume
from tta_methods import build_method
from tta_methods.sar.sam import SAM
from utils import get_device, load_config, set_seed


def _validate_method(method: Any) -> None:
    if not isinstance(method.optimizer, SAM):
        raise RuntimeError("SAR smoke is not using SAM")
    if not isinstance(method.optimizer.base_optimizer, torch.optim.SGD):
        raise RuntimeError("SAR smoke SAM base optimizer is not SGD")
    group = method.optimizer.base_optimizer.param_groups[0]
    if (
        float(group["lr"]) != LEARNING_RATE
        or float(group["momentum"]) != 0.9
        or float(group["weight_decay"]) != 0.0
        or float(method.cfg["rho"]) != 0.05
    ):
        raise RuntimeError("SAR smoke has the wrong optimizer profile")
    expected = {
        f"{module_name}.{parameter_name}"
        for module_name, module in method.model.named_modules()
        if isinstance(module, nn.BatchNorm2d)
        and not module_name.startswith("encoder.layer4")
        for parameter_name in ("weight", "bias")
    }
    if set(method.trainable_parameter_names()) != expected:
        raise RuntimeError("SAR smoke has the wrong BN-affine trainable scope")


def _validate_probe(record: dict[str, Any], batch_size: int) -> None:
    if int(record["arrival_batch_size"]) != batch_size or int(record["n_seen"]) != batch_size:
        raise RuntimeError("SAR smoke has inconsistent arrival metadata")
    probe = record.get("entropy_label_probe")
    if probe is None:
        raise RuntimeError("SAR smoke is missing its label-aware entropy probe")
    first = probe["first_filter"]
    second = probe["second_filter"]
    if int(second["selected_slices"]) > int(first["selected_slices"]):
        raise RuntimeError("SAR second filter is not a subset of the first")
    if int(record["n_selected"]) != int(second["selected_slices"]):
        raise RuntimeError("SAR selected count differs from the second filter")
    if bool(record["updated"]) != (int(second["selected_slices"]) > 0):
        raise RuntimeError("SAR update flag differs from its final selection")
    if record["loss"] is not None and not math.isfinite(float(record["loss"])):
        raise RuntimeError("SAR smoke produced a non-finite entropy loss")
    drift = float(record["extras"]["parameter_drift"])
    if not math.isfinite(drift) or drift < 0.0:
        raise RuntimeError("SAR smoke produced invalid parameter drift")
    for key in ("adaptation_seconds", "prediction_seconds"):
        if not math.isfinite(float(record["extras"][key])):
            raise RuntimeError(f"SAR smoke produced invalid {key}")


def _validate_metrics(
    predictions: torch.Tensor,
    targets: torch.Tensor,
    cfg: dict[str, Any],
    stream_mode: str,
) -> None:
    classes = [int(value) for value in cfg["evaluation"]["classes"]]
    names = {int(key): value for key, value in cfg["evaluation"]["class_names"].items()}
    if stream_mode == "patient_volume":
        values = evaluate_volume(
            predictions.numpy(), targets.numpy(), classes=classes, class_names=names
        ).values()
        if not all(math.isfinite(float(value)) for value in values):
            raise RuntimeError("Patient-volume SAR smoke produced non-finite metrics")
    else:
        for prediction, target in zip(predictions, targets):
            values = evaluate_slice(
                prediction.numpy(), target.numpy(), classes=classes, class_names=names
            )[0].values()
            if not all(math.isfinite(float(value)) for value in values):
                raise RuntimeError("Random-slice SAR smoke produced non-finite metrics")


def smoke_one(
    base_cfg: dict[str, Any],
    stream_mode: str,
    seed: int,
    vendor: str,
    device: torch.device,
) -> None:
    cfg = configure_sar_comparison(base_cfg, stream_mode)
    set_seed(int(cfg["experiment"]["harness_seed"]), deterministic=True)
    model = build_model(cfg, pretrained_override=False)
    load_source_checkpoint(
        model,
        Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt",
        map_location="cpu",
    )
    method = build_method("sar", model, cfg["methods"]["sar"], cfg["tta"], device)
    _validate_method(method)
    batch_size = BATCH_SIZES[stream_mode]

    if stream_mode == "patient_volume":
        dataset = build_target_stream(vendor, cfg, order_seed=seed)
        volume = next(
            dataset[index]
            for index in range(len(dataset))
            if int(dataset[index]["n_slices"]) >= batch_size
        )
        predictions, records = run_volume(
            method, volume["image"][:batch_size], batch_size, device
        )
        targets = dataset.load_mask(volume)[:batch_size]
        aggregate_probe = attach_entropy_label_probe(records, targets)
        if len(records) != 1 or aggregate_probe is None:
            raise RuntimeError("Patient SAR smoke did not produce one complete probed batch")
        record = records[0]
        order_hash = dataset.target_order_sha256
    else:
        loader = build_target_slice_loader(vendor, cfg, order_seed=seed, batch_size=batch_size)
        dataset = loader.dataset
        if not isinstance(dataset, MMSTargetSliceDataset):
            raise TypeError("Random-slice SAR smoke received an unexpected dataset")
        batch = next(iter(loader))
        predictions, record, payload = run_random_slice_batch(method, batch["image"], device)
        targets = dataset.load_masks(list(batch["mask_path"]))
        if payload is None:
            raise RuntimeError("Random-slice SAR smoke did not produce a probe payload")
        record["_probe_payload"] = payload
        if attach_entropy_label_probe([record], targets) is None:
            raise RuntimeError("Random-slice SAR smoke did not attach its probe")
        order_hash = dataset.slice_order_sha256

    if predictions.shape != targets.shape:
        raise RuntimeError("SAR smoke prediction/target shape mismatch")
    _validate_probe(record, batch_size)
    if not bool(record["updated"]):
        raise RuntimeError(f"SAR {stream_mode} smoke did not execute a second-pass SGD update")
    _validate_metrics(predictions, targets, cfg, stream_mode)
    print(
        f"[SMOKE] method=sar stream={stream_mode} batch_size={batch_size} "
        f"lr={LEARNING_RATE} device={device.type} vendor={vendor} seed={seed} "
        f"order={order_hash} passed"
    )
    del method, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--stream-modes", nargs="+", choices=STREAM_MODES, default=list(STREAM_MODES))
    parser.add_argument("--seed", type=int, choices=SOURCE_SEEDS, default=2022)
    parser.add_argument("--vendor", choices=["B", "C", "D"], default="C")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] {torch.cuda.get_device_name(0)} torch={torch.__version__} "
            f"runtime={torch.version.cuda}"
        )
    base_cfg = load_config(args.config)
    for stream_mode in args.stream_modes:
        smoke_one(base_cfg, stream_mode, args.seed, args.vendor, device)
    print("[SMOKE] all requested SAR comparison cells passed")


if __name__ == "__main__":
    main()
