"""Smoke-test Source, TENT, and SAR on the real random-slice stream."""

from __future__ import annotations

import argparse
import gc
import math
from copy import deepcopy
from pathlib import Path
from typing import Any

import torch
from torch import nn

from data import MMSTargetSliceDataset, build_target_slice_loader
from metrics import evaluate_slice
from model import build_model, load_source_checkpoint
from run_tta import attach_entropy_label_probe, run_random_slice_batch
from tta_methods import build_method
from tta_methods.sar.sam import SAM
from utils import get_device, load_config, set_seed


LOCKED_METHOD_CONFIGS = {
    "tent": {
        "optimizer": "sgd",
        "lr": 6.25e-5,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "steps": 1,
    },
    "sar": {
        "optimizer": "sgd_sam",
        "lr": 6.25e-5,
        "momentum": 0.9,
        "weight_decay": 0.0,
        "rho": 0.05,
        "steps": 1,
        "entropy_margin_factor": 0.4,
        "recovery_ema": 0.9,
        "recovery_threshold": 0.2,
    },
}


def _finite_metrics(predictions: torch.Tensor, targets: torch.Tensor, cfg: dict[str, Any]) -> None:
    classes = [int(value) for value in cfg["evaluation"]["classes"]]
    names = {int(key): value for key, value in cfg["evaluation"]["class_names"].items()}
    for prediction, target in zip(predictions, targets):
        metrics, _ = evaluate_slice(
            prediction.numpy(), target.numpy(), classes=classes, class_names=names
        )
        if not all(math.isfinite(value) for value in metrics.values()):
            raise RuntimeError(f"Smoke test produced non-finite metrics: {metrics}")


def _validate_locked_setup(method_name: str, method: Any, method_cfg: dict[str, Any]) -> None:
    for key, expected in LOCKED_METHOD_CONFIGS.get(method_name, {}).items():
        if method_cfg.get(key) != expected:
            raise RuntimeError(
                f"Unexpected {method_name} setting {key}: {method_cfg.get(key)!r} != {expected!r}"
            )
    if method_name == "source":
        if method.trainable_parameter_names():
            raise RuntimeError("Source-only exposes trainable parameters")
        return

    if method_name == "tent":
        if not isinstance(method.optimizer, torch.optim.SGD):
            raise RuntimeError("TENT is not using SGD")
        group = method.optimizer.param_groups[0]
        if (group["lr"], group["momentum"], group["weight_decay"]) != (6.25e-5, 0.9, 0.0):
            raise RuntimeError(f"Unexpected TENT optimizer group: {group}")
        expected_names = {
            f"{module_name}.{parameter_name}"
            for module_name, module in method.model.named_modules()
            if isinstance(module, nn.BatchNorm2d)
            for parameter_name in ("weight", "bias")
        }
    else:
        if not isinstance(method.optimizer, SAM) or not isinstance(
            method.optimizer.base_optimizer, torch.optim.SGD
        ):
            raise RuntimeError("SAR is not using SAM with an SGD base optimizer")
        group = method.optimizer.base_optimizer.param_groups[0]
        if (group["lr"], group["momentum"], group["weight_decay"]) != (6.25e-5, 0.9, 0.0):
            raise RuntimeError(f"Unexpected SAR optimizer group: {group}")
        expected_names = {
            f"{module_name}.{parameter_name}"
            for module_name, module in method.model.named_modules()
            if isinstance(module, nn.BatchNorm2d)
            and not module_name.startswith("encoder.layer4")
            for parameter_name in ("weight", "bias")
        }
    if set(method.trainable_parameter_names()) != expected_names:
        raise RuntimeError(f"{method_name} trainable scope differs from the locked BN affine scope")
    for module in method.model.modules():
        if isinstance(module, nn.BatchNorm2d) and (
            module.running_mean is not None or module.running_var is not None
        ):
            raise RuntimeError(f"{method_name} BatchNorm still has running-statistics buffers")


def _validate_probe(probe: dict[str, Any]) -> None:
    first = probe["first_filter"]
    second = probe["second_filter"]
    if int(second["selected_slices"]) > int(first["selected_slices"]):
        raise RuntimeError("SAR second entropy filter is not a subset of the first")
    for stage in (first, second):
        for key in ("selection_coverage", "pixel_accuracy", "foreground_pixel_accuracy"):
            value = stage[key]
            if value is not None and not (0.0 <= float(value) <= 1.0):
                raise RuntimeError(f"Invalid SAR probe metric {key}={value}")


def smoke_method(
    cfg: dict[str, Any],
    method_name: str,
    seed: int,
    vendor: str,
    max_batches: int,
    device: torch.device,
) -> None:
    set_seed(int(cfg["experiment"]["harness_seed"]), deterministic=True)
    model = build_model(cfg, pretrained_override=False)
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    load_source_checkpoint(model, checkpoint, map_location="cpu")
    method_cfg = deepcopy(cfg["methods"][method_name])
    method = build_method(method_name, model, method_cfg, cfg["tta"], device)
    _validate_locked_setup(method_name, method, method_cfg)

    loader = build_target_slice_loader(
        vendor, cfg, order_seed=seed, batch_size=int(cfg["tta"]["batch_size"])
    )
    dataset = loader.dataset
    if not isinstance(dataset, MMSTargetSliceDataset):
        raise TypeError("Random-slice smoke received an unexpected dataset type")
    expected_offset = 0
    saw_update = False
    processed_batches = 0
    for batch_index, batch in enumerate(loader):
        if batch_index >= max_batches:
            break
        expected_ids = dataset.slice_order[expected_offset : expected_offset + len(batch["slice_id"])]
        if list(batch["slice_id"]) != expected_ids:
            raise RuntimeError("Smoke batch does not follow the seeded slice order")

        # This is the label-free adaptation boundary. Masks are loaded only afterwards.
        predictions, adaptation, probe_payload = run_random_slice_batch(
            method, batch["image"], device
        )
        targets = dataset.load_masks(list(batch["mask_path"]))
        if probe_payload is not None:
            adaptation["_probe_payload"] = probe_payload
            probe = attach_entropy_label_probe([adaptation], targets)
        else:
            probe = None
        if predictions.shape != targets.shape:
            raise RuntimeError(
                f"Prediction shape {tuple(predictions.shape)} != target shape {tuple(targets.shape)}"
            )
        arrival_size = int(predictions.shape[0])
        if arrival_size > int(cfg["tta"]["batch_size"]) or int(adaptation["n_seen"]) != arrival_size:
            raise RuntimeError(f"Invalid smoke arrival metadata: {adaptation}")
        _finite_metrics(predictions, targets, cfg)

        if method_name == "source":
            if adaptation["updated"] or adaptation["n_selected"] != 0:
                raise RuntimeError("Source-only smoke performed an adaptation update")
            if float(adaptation["extras"]["parameter_drift"]) != 0.0:
                raise RuntimeError("Source-only smoke changed model parameters")
        elif method_name == "tent":
            if not adaptation["updated"] or int(adaptation["n_selected"]) != arrival_size:
                raise RuntimeError("TENT smoke did not update on the complete batch")
            if adaptation["loss"] is None or not math.isfinite(float(adaptation["loss"])):
                raise RuntimeError("TENT smoke produced a non-finite loss")
            if float(adaptation["extras"]["parameter_drift"]) <= 0.0:
                raise RuntimeError("TENT smoke did not change BN affine parameters")
            saw_update = True
        else:
            if probe is None:
                raise RuntimeError("SAR smoke did not emit the entropy-label probe")
            _validate_probe(probe)
            second_selected = int(probe["second_filter"]["selected_slices"])
            if int(adaptation["n_selected"]) != second_selected:
                raise RuntimeError("SAR selected count differs from the probe")
            if bool(adaptation["updated"]) != (second_selected > 0):
                raise RuntimeError("SAR update flag differs from the final entropy selection")
            if adaptation["loss"] is not None and not math.isfinite(float(adaptation["loss"])):
                raise RuntimeError("SAR smoke produced a non-finite loss")
            saw_update = saw_update or bool(adaptation["updated"])

        processed_batches += 1
        expected_offset += arrival_size
        if method_name != "sar" or saw_update:
            break

    if processed_batches == 0:
        raise RuntimeError("Smoke test did not process a batch")
    if method_name in {"tent", "sar"} and not saw_update:
        raise RuntimeError(
            f"{method_name} did not execute an update in the first {max_batches} batches"
        )
    print(
        f"[SMOKE] method={method_name} device={device.type} vendor={vendor} "
        f"seed={seed} batches={processed_batches} order={dataset.slice_order_sha256} passed"
    )
    del method, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--methods", nargs="+", default=["source", "tent", "sar"],
        choices=["source", "tent", "sar"],
    )
    parser.add_argument("--seed", type=int, default=2022)
    parser.add_argument("--vendor", choices=["B", "C", "D"], default="C")
    parser.add_argument("--max-batches", type=int, default=16)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.max_batches < 1:
        raise ValueError("max-batches must be positive")
    cfg = load_config(args.config)
    cfg["tta"]["stream_mode"] = "slice_random"
    if args.batch_size < 1:
        raise ValueError("batch-size must be positive")
    cfg["tta"]["batch_size"] = args.batch_size
    if cfg["tta"]["timing"] != "adapt_then_predict" or cfg["tta"]["reset"] != "vendor":
        raise RuntimeError("Random-slice smoke requires adapt_then_predict with vendor reset")
    device = get_device(args.device)
    if device.type == "cuda":
        print(
            f"[CUDA] {torch.cuda.get_device_name(0)} torch={torch.__version__} "
            f"runtime={torch.version.cuda}"
        )
    for method_name in args.methods:
        smoke_method(cfg, method_name, args.seed, args.vendor, args.max_batches, device)
    print("[SMOKE] completed")


if __name__ == "__main__":
    main()
