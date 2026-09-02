"""Train the Vendor-A source model and optionally prepare EATA Fisher state."""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from data import build_source_loaders, build_source_validation_volumes, split_volume_into_batches
from metrics import evaluate_volume
from model import build_model, load_source_checkpoint
from utils import (
    file_sha256,
    get_device,
    load_config,
    run_metadata,
    save_json,
    set_seed,
    state_dict_sha256,
)


def foreground_soft_dice_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> torch.Tensor:
    probabilities = logits.softmax(dim=1)
    one_hot = F.one_hot(target, num_classes=logits.shape[1]).permute(0, 3, 1, 2).to(probabilities.dtype)
    intersection = (probabilities[:, 1:] * one_hot[:, 1:]).sum(dim=(0, 2, 3))
    denominator = probabilities[:, 1:].sum(dim=(0, 2, 3)) + one_hot[:, 1:].sum(dim=(0, 2, 3))
    dice = (2.0 * intersection + epsilon) / (denominator + epsilon)
    return 1.0 - dice.mean()


def source_loss(logits: torch.Tensor, target: torch.Tensor, epsilon: float = 1e-6) -> tuple[torch.Tensor, dict[str, float]]:
    cross_entropy = F.cross_entropy(logits, target)
    dice = foreground_soft_dice_loss(logits, target, epsilon)
    total = cross_entropy + dice
    return total, {"cross_entropy": float(cross_entropy.detach()), "dice_loss": float(dice.detach())}


def build_source_optimizer(model: nn.Module, cfg: dict[str, Any]) -> torch.optim.Optimizer:
    source_cfg = cfg["source"]
    groups: dict[tuple[str, bool], list[nn.Parameter]] = {
        ("encoder", True): [], ("encoder", False): [],
        ("decoder", True): [], ("decoder", False): [],
    }
    for name, parameter in model.named_parameters():
        family = "encoder" if name.startswith("encoder.") else "decoder"
        use_decay = parameter.ndim > 1 and not name.endswith(".bias")
        groups[(family, use_decay)].append(parameter)
    parameter_groups = []
    for (family, use_decay), parameters in groups.items():
        if not parameters:
            continue
        lr = float(source_cfg["encoder_lr"] if family == "encoder" else source_cfg["decoder_lr"])
        parameter_groups.append({
            "params": parameters,
            "lr": lr,
            "initial_lr": lr,
            "weight_decay": float(source_cfg["weight_decay"]) if use_decay else 0.0,
        })
    return torch.optim.AdamW(parameter_groups, betas=(0.9, 0.999), eps=1e-8)


def set_epoch_learning_rates(optimizer: torch.optim.Optimizer, epoch: int, epochs: int, cfg: dict[str, Any]) -> None:
    warmup = int(cfg["source"]["warmup_epochs"])
    minimum = float(cfg["source"]["min_lr"])
    for group in optimizer.param_groups:
        base = float(group["initial_lr"])
        if epoch < warmup:
            lr = base * (epoch + 1) / max(1, warmup)
        else:
            denominator = max(1, epochs - warmup - 1)
            progress = min(1.0, (epoch - warmup) / denominator)
            lr = minimum + 0.5 * (base - minimum) * (1.0 + math.cos(math.pi * progress))
        group["lr"] = lr


def train_one_epoch(
    model: nn.Module,
    loader: Iterable[dict[str, Any]],
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epsilon: float,
    max_batches: int | None = None,
) -> dict[str, float]:
    model.train()
    totals = {"loss": 0.0, "cross_entropy": 0.0, "dice_loss": 0.0}
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        images = batch["image"].to(device)
        masks = batch["mask"].to(device)
        optimizer.zero_grad(set_to_none=True)
        logits = model(images)["logits"]
        loss, parts = source_loss(logits, masks, epsilon)
        if not torch.isfinite(loss):
            raise FloatingPointError("Non-finite source training loss")
        loss.backward()
        optimizer.step()
        totals["loss"] += float(loss.detach())
        totals["cross_entropy"] += parts["cross_entropy"]
        totals["dice_loss"] += parts["dice_loss"]
        batches += 1
    if batches == 0:
        raise RuntimeError("Source training loader produced no batches")
    return {key: value / batches for key, value in totals.items()}


@torch.no_grad()
def validate_source(
    model: nn.Module,
    dataset: Any,
    cfg: dict[str, Any],
    device: torch.device,
    max_volumes: int | None = None,
    batch_size: int | None = None,
) -> dict[str, float]:
    model.eval()
    volume_scores = []
    for index in range(len(dataset)):
        if max_volumes is not None and index >= max_volumes:
            break
        volume = dataset[index]
        predictions = []
        for batch in split_volume_into_batches(volume["image"], batch_size or int(cfg["source"]["batch_size"])):
            logits = model(batch.to(device))["logits"]
            predictions.append(logits.argmax(dim=1).cpu())
        prediction = torch.cat(predictions).numpy()
        target = dataset.load_mask(volume).numpy()
        volume_scores.append(evaluate_volume(prediction, target))
    if not volume_scores:
        raise RuntimeError("Source validation produced no volumes")
    keys = volume_scores[0].keys()
    return {key: float(np.mean([score[key] for score in volume_scores])) for key in keys}


def train_source_seed(
    cfg: dict[str, Any],
    seed: int,
    device: torch.device,
    smoke_test: bool = False,
) -> Path:
    set_seed(seed, deterministic=True)
    smoke_cfg = cfg["source"]["smoke"]
    batch_size = int(smoke_cfg["batch_size"] if smoke_test else cfg["source"]["batch_size"])
    epochs = int(smoke_cfg["epochs"] if smoke_test else cfg["source"]["epochs"])
    train_loader, _ = build_source_loaders(cfg, seed=seed, batch_size=batch_size)
    val_volumes = build_source_validation_volumes(cfg)
    model = build_model(cfg).to(device)
    initialization_profile = str(cfg["experiment"]["initialization_profile"])
    initial_model_sha256 = state_dict_sha256(model.state_dict())
    initial_parameters = {name: parameter.detach().cpu().clone() for name, parameter in model.named_parameters()}
    optimizer = build_source_optimizer(model, cfg)
    best_dice = -math.inf
    patience = 0
    checkpoint_dir = Path(cfg["source"]["checkpoint_dir"])
    checkpoint_path = checkpoint_dir / ("smoke" if smoke_test else "") / (
        f"seed{seed}_smoke.pt" if smoke_test else f"seed{seed}_best.pt"
    )
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    history = []
    for epoch in range(epochs):
        set_epoch_learning_rates(optimizer, epoch, epochs, cfg)
        train_metrics = train_one_epoch(
            model,
            train_loader,
            optimizer,
            device,
            epsilon=float(cfg["source"]["dice_epsilon"]),
            max_batches=int(smoke_cfg["max_train_batches"]) if smoke_test else None,
        )
        val_metrics = validate_source(
            model,
            val_volumes,
            cfg,
            device,
            max_volumes=int(smoke_cfg["max_val_volumes"]) if smoke_test else None,
            batch_size=batch_size,
        )
        entry = {"epoch": epoch + 1, "lr": [group["lr"] for group in optimizer.param_groups], "train": train_metrics, "validation": val_metrics}
        history.append(entry)
        score = val_metrics["dice_macro"]
        if score > best_dice:
            best_dice = score
            patience = 0
            torch.save({
                "model": model.state_dict(),
                "optimizer": optimizer.state_dict(),
                "epoch": epoch + 1,
                "validation_dice_macro": score,
                "seed": seed,
                "smoke_test": smoke_test,
                "initialization_profile": initialization_profile,
                "initial_model_sha256": initial_model_sha256,
                "config": cfg,
                "protocol_sha256": file_sha256(cfg["data"]["protocol_file"]),
            }, checkpoint_path)
        else:
            patience += 1
            if not smoke_test and patience >= int(cfg["source"]["early_stopping_patience"]):
                break
    if not any(
        not torch.equal(initial_parameters[name], parameter.detach().cpu())
        for name, parameter in model.named_parameters()
    ):
        raise RuntimeError("Source training completed without changing any model parameter")
    metadata_path = checkpoint_path.with_suffix(".json")
    save_json({
        "checkpoint": str(checkpoint_path),
        "checkpoint_sha256": file_sha256(checkpoint_path),
        "initialization_profile": initialization_profile,
        "initial_model_sha256": initial_model_sha256,
        "slice_filter": cfg["data"]["slice_filter"],
        "training_slices": len(train_loader.dataset),
        "validation_slices": val_volumes.n_slices,
        "best_validation_dice_macro": best_dice,
        "history": history,
        "runtime": run_metadata(Path(__file__).resolve().parent),
    }, metadata_path)
    load_source_checkpoint(model, checkpoint_path, map_location=device)
    reloaded = build_model(cfg, pretrained_override=False).to(device)
    load_source_checkpoint(reloaded, checkpoint_path, map_location=device)
    reloaded.eval()
    model.eval()
    probe = next(iter(train_loader))["image"][:1].to(device)
    with torch.no_grad():
        expected = model(probe)["logits"]
        actual = reloaded(probe)["logits"]
    if not torch.equal(expected, actual):
        raise RuntimeError("Reloaded source checkpoint does not reproduce logits")
    return checkpoint_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    seed_group = parser.add_mutually_exclusive_group(required=False)
    seed_group.add_argument("--seed", type=int)
    seed_group.add_argument("--all-seeds", action="store_true")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--fisher", action="store_true")
    parser.add_argument("--smoke-test", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    configured = [int(seed) for seed in cfg["experiment"]["source_seeds"]]
    seeds = configured if args.all_seeds else [args.seed if args.seed is not None else configured[0]]
    device = get_device(args.device)
    for seed in seeds:
        if seed not in configured:
            raise ValueError(f"Seed {seed} is not in the locked source seed list")
        checkpoint = train_source_seed(cfg, seed, device, smoke_test=args.smoke_test)
        print(f"source checkpoint: {checkpoint} sha256={file_sha256(checkpoint)}")
        if args.fisher:
            from tta_methods.eata.fisher import estimate_fisher

            model = build_model(cfg, pretrained_override=False).to(device)
            load_source_checkpoint(model, checkpoint, map_location=device)
            loader, _ = build_source_loaders(cfg, seed=seed)
            output = (
                Path(cfg["source"]["checkpoint_dir"]) / "smoke" / f"fisher_seed{seed}_smoke.pt"
                if args.smoke_test
                else Path(cfg["source"]["checkpoint_dir"]) / f"fisher_seed{seed}.pt"
            )
            estimate_fisher(
                model,
                loader,
                cfg["methods"]["eata"],
                device,
                output,
                source_checkpoint_sha256=file_sha256(checkpoint),
                initialization_profile=str(cfg["experiment"]["initialization_profile"]),
            )
            print(f"fisher artifact: {output} sha256={file_sha256(output)}")


if __name__ == "__main__":
    main()
