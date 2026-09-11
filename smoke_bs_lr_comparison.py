"""Smoke-test all cells of the locked BS(8/4), LR=1e-3 comparison."""

from __future__ import annotations

import argparse
from copy import deepcopy
import gc
import math
from pathlib import Path
from typing import Any

import torch

from data import MMSTargetSliceDataset, build_target_slice_loader, build_target_stream
from metrics import evaluate_slice, evaluate_volume
from model import build_model, load_source_checkpoint
from run_bs_lr_comparison import (
    BATCH_SIZES,
    LEARNING_RATE,
    METHODS,
    SOURCE_SEEDS,
    STREAM_MODES,
    configure_comparison,
)
from run_tta import run_random_slice_batch, run_volume
from tta_methods import build_method
from utils import get_device, load_config, set_seed, state_dict_sha256


def _validate_setup(method: Any, method_name: str) -> None:
    if method_name in {"source", "tbn"}:
        if method.optimizer is not None or method.trainable_parameter_names():
            raise RuntimeError(f"{method_name} smoke unexpectedly exposes an optimizer")
        return
    if method_name == "tent":
        if not isinstance(method.optimizer, torch.optim.SGD):
            raise RuntimeError("TENT smoke is not using SGD")
        group = method.optimizer.param_groups[0]
        if (
            group["lr"] != LEARNING_RATE
            or group["momentum"] != 0.9
            or group["weight_decay"] != 0.0
        ):
            raise RuntimeError("TENT smoke has the wrong optimizer profile")
        return
    if not isinstance(method.optimizer, torch.optim.Adam):
        raise RuntimeError(f"{method_name} smoke is not using Adam")
    if method.optimizer.param_groups[0]["lr"] != LEARNING_RATE:
        raise RuntimeError(f"{method_name} smoke is not using LR=1e-3")


def _validate_adaptation(record: dict[str, Any], method_name: str, batch_size: int) -> None:
    if int(record["arrival_batch_size"]) != batch_size or int(record["n_seen"]) != batch_size:
        raise RuntimeError(f"{method_name} smoke has the wrong arrival batch size")
    drift = float(record["extras"]["parameter_drift"])
    if not math.isfinite(drift):
        raise RuntimeError(f"{method_name} smoke produced non-finite parameter drift")
    if method_name in {"source", "tbn"}:
        if record["updated"] or record["n_selected"] != 0 or drift != 0.0:
            raise RuntimeError(f"{method_name} smoke changed model parameters")
        return
    if int(record["n_selected"]) != batch_size:
        raise RuntimeError(f"{method_name} smoke did not select the complete batch")
    if record["loss"] is None or not math.isfinite(float(record["loss"])):
        raise RuntimeError(f"{method_name} smoke produced a non-finite loss")
    if method_name in {"tent", "cotta"} and (not record["updated"] or drift <= 0.0):
        raise RuntimeError(f"{method_name} smoke did not update parameters")
    if method_name == "grata":
        effective_lr = float(record["extras"]["effective_lr"])
        cosine = float(record["extras"]["gradient_cosine"])
        expected_lr = LEARNING_RATE * 0.25 * (cosine + 1.0) ** 2
        if (
            not math.isclose(effective_lr, expected_lr, rel_tol=1e-9, abs_tol=1e-12)
            or not 0.0 <= effective_lr <= LEARNING_RATE
            or int(record["extras"]["weak_view_count"]) != 6
        ):
            raise RuntimeError("GraTA smoke has invalid gradient-alignment diagnostics")


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
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Patient-volume smoke produced non-finite metrics")
        return
    for prediction, target in zip(predictions, targets):
        values = evaluate_slice(
            prediction.numpy(), target.numpy(), classes=classes, class_names=names
        )[0].values()
        if not all(math.isfinite(value) for value in values):
            raise RuntimeError("Random-slice smoke produced non-finite metrics")


def smoke_one(
    base_cfg: dict[str, Any],
    method_name: str,
    stream_mode: str,
    seed: int,
    vendor: str,
    device: torch.device,
    fast_cotta: bool,
) -> None:
    cfg = configure_comparison(base_cfg, method_name, stream_mode)
    set_seed(int(cfg["experiment"]["harness_seed"]), deterministic=True)
    model = build_model(cfg, pretrained_override=False)
    checkpoint = Path(cfg["source"]["checkpoint_dir"]) / f"seed{seed}_best.pt"
    load_source_checkpoint(model, checkpoint, map_location="cpu")
    method_cfg = deepcopy(cfg["methods"][method_name])
    if method_name == "cotta" and fast_cotta:
        method_cfg["augmentation_scales"] = [1.0]
    method = build_method(method_name, model, method_cfg, cfg["tta"], device)
    _validate_setup(method, method_name)
    initial_hash = state_dict_sha256(method.model.state_dict())
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
        if len(records) != 1:
            raise RuntimeError("Patient-volume smoke split one full arrival batch")
        record = records[0]
        targets = dataset.load_mask(volume)[:batch_size]
        order_hash = dataset.target_order_sha256
    else:
        loader = build_target_slice_loader(
            vendor, cfg, order_seed=seed, batch_size=batch_size
        )
        dataset = loader.dataset
        if not isinstance(dataset, MMSTargetSliceDataset):
            raise TypeError("Random-slice smoke received an unexpected dataset")
        batch = next(iter(loader))
        if int(batch["image"].shape[0]) != batch_size:
            raise RuntimeError("Random-slice smoke did not receive a full arrival batch")
        predictions, record, _ = run_random_slice_batch(method, batch["image"], device)
        targets = dataset.load_masks(list(batch["mask_path"]))
        order_hash = dataset.slice_order_sha256

    if predictions.shape != targets.shape:
        raise RuntimeError(f"{method_name} smoke prediction/target shape mismatch")
    _validate_adaptation(record, method_name, batch_size)
    _validate_metrics(predictions, targets, cfg, stream_mode)
    if method_name in {"source", "tbn"} and state_dict_sha256(method.model.state_dict()) != initial_hash:
        raise RuntimeError(f"{method_name} state dict changed during smoke")
    print(
        f"[SMOKE] method={method_name} stream={stream_mode} batch_size={batch_size} "
        f"lr={'N/A' if method_name in {'source', 'tbn'} else LEARNING_RATE} "
        f"device={device.type} vendor={vendor} seed={seed} order={order_hash} passed"
    )
    del method, model
    gc.collect()
    if device.type == "cuda":
        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--methods", nargs="+", choices=METHODS, default=list(METHODS))
    parser.add_argument("--stream-modes", nargs="+", choices=STREAM_MODES, default=list(STREAM_MODES))
    parser.add_argument("--seed", type=int, choices=SOURCE_SEEDS, default=2022)
    parser.add_argument("--vendor", choices=["B", "C", "D"], default="C")
    parser.add_argument("--device", choices=["cpu", "cuda", "auto"], default="cpu")
    parser.add_argument(
        "--fast-cotta",
        action="store_true",
        help="Use one CoTTA scale for the CPU smoke; GPU preflight must omit this flag",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    device = get_device(args.device)
    if args.fast_cotta and device.type != "cpu":
        raise ValueError("--fast-cotta is reserved for the local CPU smoke")
    if device.type == "cuda":
        print(
            f"[CUDA] {torch.cuda.get_device_name(0)} torch={torch.__version__} "
            f"runtime={torch.version.cuda}"
        )
    base_cfg = load_config(args.config)
    for stream_mode in args.stream_modes:
        for method_name in args.methods:
            smoke_one(
                base_cfg,
                method_name,
                stream_mode,
                args.seed,
                args.vendor,
                device,
                args.fast_cotta,
            )
    print("[SMOKE] all requested comparison cells passed")


if __name__ == "__main__":
    main()
