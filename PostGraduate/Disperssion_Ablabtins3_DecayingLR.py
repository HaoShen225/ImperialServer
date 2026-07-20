from __future__ import annotations

import argparse
import csv
import json
import math
import random
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn

from helper.backbone import ModelConfig, build_model
from helper.backbone_losses import backbone_training_loss
from helper.dataloaders import (
    DATA_ROOT,
    DOMAIN_NAMES,
    FOREGROUND_CLASS_NAMES,
    SOURCE_DOMAIN_NAME,
    build_test_dataset,
    build_train_dataset,
    make_loader,
    run_test_flow,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "DecayingLR"
SOURCE_DOMAIN = SOURCE_DOMAIN_NAME
EXPERIMENT_NAME = "DecayingLR"


SCHEDULER_NAMES = ("Cosine-0.1x", "Cosine-0.01x", "Plateau", "Poly")
SCHEDULER_ALIASES = {name.lower(): name for name in SCHEDULER_NAMES}
SCHEDULER_ALIASES.update(
    {
        "cosine0.1x": "Cosine-0.1x",
        "cosine_0.1x": "Cosine-0.1x",
        "cosine-0p1x": "Cosine-0.1x",
        "cosine0.01x": "Cosine-0.01x",
        "cosine_0.01x": "Cosine-0.01x",
        "cosine-0p01x": "Cosine-0.01x",
        "poly": "Poly",
        "plateau": "Plateau",
    }
)


@dataclass(frozen=True)
class GroupConfig:
    name: str
    physical_batch: int
    bn_batch: int
    effective_grad_batch: int
    grad_accum_steps: int = 1
    ghost_bn_virtual_batch: int = 0
    fixed_steps_to_bs4: bool = False
    description: str = ""


GROUP_CONFIGS: Dict[str, GroupConfig] = {
    "A_BS4": GroupConfig(
        name="A_BS4",
        physical_batch=4,
        bn_batch=4,
        effective_grad_batch=4,
        description="physical BS=4, BN BS=4, effective grad BS=4",
    ),
    "B_BS8": GroupConfig(
        name="B_BS8",
        physical_batch=8,
        bn_batch=8,
        effective_grad_batch=8,
        description="physical BS=8, BN BS=8, effective grad BS=8",
    ),
    "C_BS8_FixedSteps": GroupConfig(
        name="C_BS8_FixedSteps",
        physical_batch=8,
        bn_batch=8,
        effective_grad_batch=8,
        fixed_steps_to_bs4=True,
        description="BS=8 with optimizer steps matched to A_BS4",
    ),
    "D_BS4_Accum2": GroupConfig(
        name="D_BS4_Accum2",
        physical_batch=4,
        bn_batch=4,
        effective_grad_batch=8,
        grad_accum_steps=2,
        description="physical BS=4, BN BS=4, gradient accumulation=2",
    ),
    "E_BS8_GhostBN4": GroupConfig(
        name="E_BS8_GhostBN4",
        physical_batch=8,
        bn_batch=4,
        effective_grad_batch=8,
        ghost_bn_virtual_batch=4,
        description="physical BS=8 with GhostBN virtual BS=4",
    ),
}
GROUP_ALIASES = {name.lower(): name for name in GROUP_CONFIGS}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def snapshot_rng_state() -> Dict[str, Any]:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
    }


def restore_rng_state(state: Dict[str, Any]) -> None:
    random.setstate(state["python"])
    np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if torch.cuda.is_available() and state.get("cuda"):
        torch.cuda.set_rng_state_all(state["cuda"])


def resolve_device(name: str) -> torch.device:
    if str(name).strip().lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    device = torch.device(str(name))
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA requested but torch.cuda.is_available() is False.")
    return device


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    return p if p.is_absolute() else base / p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def normalize_group(name: str) -> str:
    key = str(name).strip()
    if key in GROUP_CONFIGS:
        return key
    low = key.lower()
    if low in GROUP_ALIASES:
        return GROUP_ALIASES[low]
    raise ValueError(f"Unknown group {name!r}; expected one of {', '.join(GROUP_CONFIGS)}")


def parse_group_list(text: str) -> List[str]:
    groups = [normalize_group(name) for name in parse_str_list(text)]
    if not groups:
        raise ValueError("At least one group must be provided.")
    out: List[str] = []
    seen: set[str] = set()
    for group in groups:
        if group not in seen:
            seen.add(group)
            out.append(group)
    return out


def scheduler_tag(name: str) -> str:
    return (
        normalize_scheduler(name)
        .replace("-", "")
        .replace(".", "p")
        .replace("x", "x")
    )


def normalize_scheduler(name: str) -> str:
    key = str(name).strip()
    if key in SCHEDULER_NAMES:
        return key
    low = key.lower()
    if low in SCHEDULER_ALIASES:
        return SCHEDULER_ALIASES[low]
    raise ValueError(f"Unknown scheduler {name!r}; expected one of {', '.join(SCHEDULER_NAMES)}")


def parse_scheduler_list(text: str) -> List[str]:
    schedulers = [normalize_scheduler(name) for name in parse_str_list(text)]
    if not schedulers:
        raise ValueError("At least one scheduler must be provided.")
    out: List[str] = []
    seen: set[str] = set()
    for scheduler in schedulers:
        if scheduler not in seen:
            seen.add(scheduler)
            out.append(scheduler)
    return out


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            f = float(value)
            if np.isfinite(f):
                out.append(f)
        except Exception:
            pass
    return out


def finite_mean(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(np.mean(xs)) if xs else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0 if len(xs) == 1 else float("nan")


def safe_float(value: Any) -> float:
    try:
        f = float(value)
        return f if np.isfinite(f) else float("nan")
    except Exception:
        return float("nan")


def write_json(path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    ensure_dir(path.parent)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    keys: List[str] = []
    for row in rows:
        for key in row:
            if key not in keys:
                keys.append(key)
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=keys)
        writer.writeheader()
        writer.writerows(rows)


def read_csv(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open("r", newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def model_config(args: argparse.Namespace) -> ModelConfig:
    return ModelConfig(
        in_ch=1,
        num_classes=3,
        base_ch=int(args.base_ch),
        latent_ch=int(args.latent_ch),
        model_type="d_l2_disp_bn",
        use_l2_norm=True,
        use_batch_norm=True,
    )


def run_dir_for(args: argparse.Namespace, scheduler: str, group: str, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.result_root)
        / "runs"
        / scheduler_tag(scheduler)
        / normalize_group(group)
        / f"shot{int(shot)}"
        / f"Seed{int(seed)}"
    )


class GhostBatchNorm2d(nn.BatchNorm2d):
    def __init__(self, *args: Any, virtual_batch_size: int = 4, **kwargs: Any) -> None:
        super().__init__(*args, **kwargs)
        self.virtual_batch_size = int(virtual_batch_size)

    @classmethod
    def from_batch_norm(cls, module: nn.BatchNorm2d, virtual_batch_size: int) -> "GhostBatchNorm2d":
        ghost = cls(
            module.num_features,
            eps=module.eps,
            momentum=module.momentum,
            affine=module.affine,
            track_running_stats=module.track_running_stats,
            virtual_batch_size=int(virtual_batch_size),
        )
        if module.affine:
            ghost.weight.data.copy_(module.weight.data)
            ghost.bias.data.copy_(module.bias.data)
        if module.track_running_stats:
            ghost.running_mean.data.copy_(module.running_mean.data)
            ghost.running_var.data.copy_(module.running_var.data)
            ghost.num_batches_tracked.data.copy_(module.num_batches_tracked.data)
        return ghost

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        if (not self.training) or input.shape[0] <= int(self.virtual_batch_size):
            return super().forward(input)
        chunks = torch.split(input, int(self.virtual_batch_size), dim=0)
        return torch.cat([super(GhostBatchNorm2d, self).forward(chunk) for chunk in chunks], dim=0)


def replace_batch_norm_with_ghost(module: nn.Module, virtual_batch_size: int) -> None:
    for name, child in list(module.named_children()):
        if isinstance(child, nn.BatchNorm2d):
            setattr(module, name, GhostBatchNorm2d.from_batch_norm(child, int(virtual_batch_size)))
        else:
            replace_batch_norm_with_ghost(child, int(virtual_batch_size))


def build_group_model(args: argparse.Namespace, group_cfg: GroupConfig, device: torch.device) -> nn.Module:
    model = build_model(model_config(args))
    if int(group_cfg.ghost_bn_virtual_batch) > 0:
        replace_batch_norm_with_ghost(model, int(group_cfg.ghost_bn_virtual_batch))
    return model.to(device)


def iter_bn_modules(model: nn.Module) -> List[tuple[str, nn.BatchNorm2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)]


def snapshot_bn_state(model: nn.Module) -> Dict[str, Dict[str, torch.Tensor]]:
    out: Dict[str, Dict[str, torch.Tensor]] = {}
    for name, module in iter_bn_modules(model):
        state: Dict[str, torch.Tensor] = {}
        if module.running_mean is not None:
            state["running_mean"] = module.running_mean.detach().clone()
        if module.running_var is not None:
            state["running_var"] = module.running_var.detach().clone()
        if module.num_batches_tracked is not None:
            state["num_batches_tracked"] = module.num_batches_tracked.detach().clone()
        out[name] = state
    return out


def restore_bn_state(model: nn.Module, snapshot: Dict[str, Dict[str, torch.Tensor]]) -> None:
    modules = dict(iter_bn_modules(model))
    with torch.no_grad():
        for name, state in snapshot.items():
            module = modules.get(name)
            if module is None:
                continue
            if "running_mean" in state and module.running_mean is not None:
                module.running_mean.copy_(state["running_mean"])
            if "running_var" in state and module.running_var is not None:
                module.running_var.copy_(state["running_var"])
            if "num_batches_tracked" in state and module.num_batches_tracked is not None:
                module.num_batches_tracked.copy_(state["num_batches_tracked"])


class BNProbeRecorder:
    def __init__(self, model: nn.Module) -> None:
        self.rows: List[Dict[str, Any]] = []
        self.context: Dict[str, Any] = {}
        self.enabled = True
        self.handles = []
        for name, module in iter_bn_modules(model):
            self.handles.append(module.register_forward_pre_hook(self._make_hook(name)))

    def close(self) -> None:
        for handle in self.handles:
            handle.remove()
        self.handles.clear()

    def set_context(self, **kwargs: Any) -> None:
        self.context = dict(kwargs)

    def _make_hook(self, name: str):
        def hook(module: nn.BatchNorm2d, inputs: tuple[torch.Tensor, ...]) -> None:
            if not self.enabled:
                return
            if not inputs:
                return
            x = inputs[0]
            if not torch.is_tensor(x) or x.dim() != 4:
                return
            with torch.no_grad():
                xd = x.detach()
                batch_mean = xd.mean(dim=(0, 2, 3))
                batch_var = xd.var(dim=(0, 2, 3), unbiased=False)
                running_mean = module.running_mean.detach() if module.running_mean is not None else torch.zeros_like(batch_mean)
                running_var = module.running_var.detach() if module.running_var is not None else torch.ones_like(batch_var)
                row = dict(self.context)
                row.update(
                    {
                        "bn_layer": name,
                        "bn_module_type": module.__class__.__name__,
                        "bn_num_features": int(module.num_features),
                        "bn_input_batch": int(xd.shape[0]),
                        "batch_mean_running_l2": float(torch.norm(batch_mean - running_mean).cpu()),
                        "batch_var_running_l2": float(torch.norm(batch_var - running_var).cpu()),
                        "batch_mean_std_across_channels": float(batch_mean.std(unbiased=False).cpu()),
                        "virtual_batch_size": int(getattr(module, "virtual_batch_size", 0) or 0),
                        "virtual_chunk_count": 0,
                        "virtual_batch_mean_running_l2_mean": float("nan"),
                        "virtual_batch_var_running_l2_mean": float("nan"),
                        "virtual_batch_mean_std_across_channels_mean": float("nan"),
                    }
                )
                vbs = int(getattr(module, "virtual_batch_size", 0) or 0)
                if module.training and vbs > 0 and xd.shape[0] > vbs:
                    mean_l2: List[float] = []
                    var_l2: List[float] = []
                    mean_std: List[float] = []
                    chunks = torch.split(xd, vbs, dim=0)
                    for chunk in chunks:
                        cm = chunk.mean(dim=(0, 2, 3))
                        cv = chunk.var(dim=(0, 2, 3), unbiased=False)
                        mean_l2.append(float(torch.norm(cm - running_mean).cpu()))
                        var_l2.append(float(torch.norm(cv - running_var).cpu()))
                        mean_std.append(float(cm.std(unbiased=False).cpu()))
                    row["virtual_chunk_count"] = int(len(chunks))
                    row["virtual_batch_mean_running_l2_mean"] = finite_mean(mean_l2)
                    row["virtual_batch_var_running_l2_mean"] = finite_mean(var_l2)
                    row["virtual_batch_mean_std_across_channels_mean"] = finite_mean(mean_std)
                self.rows.append(row)

        return hook


def build_train_data(args: argparse.Namespace, shot: int, seed: int):
    return build_train_dataset(
        SOURCE_DOMAIN,
        shot=int(shot),
        seed=int(seed),
        data_root=Path(args.data_root),
        min_fg_ratio=float(args.min_fg_ratio),
        resize_hw=int(args.resize_hw),
        split_csv=None,
        use_split=False,
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )


def build_loss_eval_batches(args: argparse.Namespace, train_case_ids: Sequence[str], device: torch.device) -> Dict[str, Dict[str, Any]]:
    caches: Dict[str, Dict[str, Any]] = {}
    for domain in DOMAIN_NAMES:
        dataset = build_test_dataset(
            domain,
            exclude_case_ids=train_case_ids,
            data_root=Path(args.data_root),
            min_fg_ratio=float(args.min_fg_ratio),
            resize_hw=int(args.resize_hw),
            max_cases=int(args.max_test_cases) if int(args.max_test_cases) > 0 else None,
            split_csv=None,
            use_split=False,
            slice_policy=str(args.slice_policy),
            num_middle_slices=int(args.num_middle_slices),
            filter_min_fg=bool(args.filter_min_fg),
        )
        batches: List[tuple[torch.Tensor, torch.Tensor]] = []
        loader = make_loader(dataset, batch_size=int(args.eval_batch_size), shuffle=False, device=device)
        for img, mask, _meta in loader:
            batches.append((img.detach().cpu(), mask.detach().cpu()))
        caches[domain] = {
            "n_cases": int(len(dataset.grouped_case_ids())),
            "n_slices": int(len(dataset)),
            "batches": batches,
        }
    return caches


def compute_training_losses(
    model: nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
) -> Dict[str, torch.Tensor]:
    out = model(img, return_features=True)
    return backbone_training_loss(
        out["logits"],
        mask,
        out["features"]["dec1"],
        num_classes=3,
        dice_weight=float(args.dice_weight),
        lambda_disp=float(args.lambda_disp),
        disp_margin=float(args.disp_margin),
    )


def scheduler_min_lr(scheduler: str, args: argparse.Namespace) -> float:
    scheduler_name = normalize_scheduler(scheduler)
    base_lr = float(args.lr)
    if scheduler_name == "Cosine-0.1x":
        return 0.1 * base_lr
    if scheduler_name == "Cosine-0.01x":
        return 0.01 * base_lr
    if scheduler_name == "Plateau":
        return float(args.plateau_min_lr)
    return 0.0


def optimizer_name(args: argparse.Namespace) -> str:
    suffix = "_nesterov" if bool(args.nesterov) else ""
    return f"SGD_m{float(args.momentum):g}{suffix}_wd{float(args.weight_decay):g}"


def scheduler_common_fields(scheduler: str, args: argparse.Namespace) -> Dict[str, Any]:
    scheduler_name = normalize_scheduler(scheduler)
    return {
        "scheduler": scheduler_name,
        "scheduler_tag": scheduler_tag(scheduler_name),
        "base_lr": float(args.lr),
        "min_lr": float(scheduler_min_lr(scheduler_name, args)),
        "warmup_ratio": float(args.warmup_ratio),
        "poly_power": float(args.poly_power),
        "optimizer_name": optimizer_name(args),
    }


def set_optimizer_lr(optimizer: torch.optim.Optimizer, lr: float) -> None:
    for group in optimizer.param_groups:
        group["lr"] = float(lr)


def current_optimizer_lr(optimizer: torch.optim.Optimizer) -> float:
    if not optimizer.param_groups:
        return float("nan")
    return float(optimizer.param_groups[0]["lr"])


def scheduled_lr_for_step(
    *,
    scheduler: str,
    args: argparse.Namespace,
    planned_step: int,
    target_steps: int,
    current_lr: float,
) -> float:
    scheduler_name = normalize_scheduler(scheduler)
    if scheduler_name == "Plateau":
        return float(current_lr)

    base_lr = float(args.lr)
    min_lr = float(scheduler_min_lr(scheduler_name, args))
    target_steps = max(1, int(target_steps))
    planned_step = min(max(1, int(planned_step)), target_steps)
    warmup_steps = int(round(float(args.warmup_ratio) * float(target_steps)))
    warmup_steps = min(max(0, warmup_steps), target_steps)

    if warmup_steps > 0 and planned_step <= warmup_steps:
        return base_lr * float(planned_step) / float(warmup_steps)

    if scheduler_name.startswith("Cosine"):
        decay_steps = max(1, target_steps - warmup_steps)
        progress = float(planned_step - warmup_steps) / float(decay_steps)
        progress = min(max(progress, 0.0), 1.0)
        cosine = 0.5 * (1.0 + math.cos(math.pi * progress))
        return min_lr + (base_lr - min_lr) * cosine

    if scheduler_name == "Poly":
        completed_steps = max(0, int(planned_step) - 1)
        progress = float(completed_steps) / float(target_steps)
        progress = min(max(progress, 0.0), 1.0)
        return base_lr * math.pow(max(0.0, 1.0 - progress), float(args.poly_power))

    return base_lr


def build_optimizer(args: argparse.Namespace, model: nn.Module) -> torch.optim.Optimizer:
    return torch.optim.SGD(
        model.parameters(),
        lr=float(args.lr),
        momentum=float(args.momentum),
        nesterov=bool(args.nesterov),
        weight_decay=float(args.weight_decay),
    )


def build_plateau_scheduler(args: argparse.Namespace, optimizer: torch.optim.Optimizer):
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=float(args.plateau_factor),
        patience=int(args.plateau_patience),
        cooldown=int(args.plateau_cooldown),
        min_lr=float(args.plateau_min_lr),
    )


@torch.no_grad()
def evaluate_loss(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    group_cfg: GroupConfig,
    scheduler: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    loss_eval_batches: Dict[str, Dict[str, Any]],
    output_dir: Path,
    epoch: int,
    optimizer_steps_completed: int,
    target_optimizer_steps: int,
    lr: float,
) -> tuple[List[Dict[str, Any]], Dict[str, Any]]:
    model.eval()
    rows: List[Dict[str, Any]] = []
    common = {
        "experiment": EXPERIMENT_NAME,
        **scheduler_common_fields(scheduler, args),
        "lr": float(lr),
        "group": group_cfg.name,
        "physical_batch": int(group_cfg.physical_batch),
        "bn_batch": int(group_cfg.bn_batch),
        "effective_grad_batch": int(group_cfg.effective_grad_batch),
        "grad_accum_steps": int(group_cfg.grad_accum_steps),
        "ghost_bn_virtual_batch": int(group_cfg.ghost_bn_virtual_batch),
        "fixed_steps_to_bs4": bool(group_cfg.fixed_steps_to_bs4),
        "rho": float(args.rho),
        "lambda_disp": float(args.lambda_disp),
        "dice_weight": float(args.dice_weight),
        "shot": int(shot),
        "seed": int(seed),
        "epoch": int(epoch),
        "optimizer_steps_completed": int(optimizer_steps_completed),
        "target_optimizer_steps": int(target_optimizer_steps),
        "output_dir": str(output_dir),
    }
    for domain in DOMAIN_NAMES:
        cache = loss_eval_batches[domain]
        loss_sum = 0.0
        seg_sum = 0.0
        disp_sum = 0.0
        weight_sum = 0
        for img, mask in cache["batches"]:
            img = img.to(device)
            mask = mask.to(device)
            losses = compute_training_losses(model, img, mask, args)
            batch_n = int(img.shape[0])
            loss_sum += float(losses["loss"].detach().cpu()) * batch_n
            seg_sum += float(losses["seg_loss"].detach().cpu()) * batch_n
            disp_sum += float(losses["disp_loss"].detach().cpu()) * batch_n
            weight_sum += batch_n
        denom = max(1, int(weight_sum))
        row = {
            **common,
            "domain": domain,
            "n_cases": int(cache["n_cases"]),
            "n_slices": int(cache["n_slices"]),
            "eval_loss": float(loss_sum / denom),
            "eval_seg_loss": float(seg_sum / denom),
            "eval_disp_loss": float(disp_sum / denom),
        }
        rows.append(row)

    source_rows = [row for row in rows if row.get("domain") == SOURCE_DOMAIN]
    target_rows = [row for row in rows if row.get("domain") != SOURCE_DOMAIN]
    summary = {
        **common,
        "source_loss": finite_mean(row.get("eval_loss") for row in source_rows),
        "source_seg_loss": finite_mean(row.get("eval_seg_loss") for row in source_rows),
        "source_disp_loss": finite_mean(row.get("eval_disp_loss") for row in source_rows),
        "mean_5target_loss": finite_mean(row.get("eval_loss") for row in target_rows),
        "mean_5target_seg_loss": finite_mean(row.get("eval_seg_loss") for row in target_rows),
        "mean_5target_disp_loss": finite_mean(row.get("eval_disp_loss") for row in target_rows),
        "mean_6domain_loss": finite_mean(row.get("eval_loss") for row in rows),
        "mean_6domain_seg_loss": finite_mean(row.get("eval_seg_loss") for row in rows),
        "mean_6domain_disp_loss": finite_mean(row.get("eval_disp_loss") for row in rows),
    }
    return rows, summary


def detach_grads(grads: Sequence[torch.Tensor | None]) -> List[torch.Tensor | None]:
    return [None if grad is None else grad.detach().clone() for grad in grads]


def flatten_grads(params: Sequence[nn.Parameter], grads: Sequence[torch.Tensor | None]) -> torch.Tensor:
    chunks: List[torch.Tensor] = []
    for param, grad in zip(params, grads):
        if grad is None:
            chunks.append(torch.zeros_like(param, memory_format=torch.preserve_format).reshape(-1))
        else:
            chunks.append(grad.detach().reshape(-1))
    if not chunks:
        device = params[0].device if params else torch.device("cpu")
        return torch.empty(0, device=device)
    return torch.cat(chunks)


def tensor_norm(value: torch.Tensor) -> float:
    return float(torch.norm(value.detach()).cpu()) if value.numel() else 0.0


def tensor_cosine(a: torch.Tensor, b: torch.Tensor) -> float:
    if a.numel() == 0 or b.numel() == 0:
        return float("nan")
    denom = torch.norm(a) * torch.norm(b)
    if float(denom.detach().cpu()) <= 0.0:
        return float("nan")
    return float((torch.dot(a, b) / denom).detach().cpu())


def add_perturbation_from_grads(
    params: Sequence[nn.Parameter],
    grads: Sequence[torch.Tensor | None],
    *,
    rho: float,
    eps: float,
) -> List[tuple[nn.Parameter, torch.Tensor]]:
    grad_vec = flatten_grads(params, grads)
    grad_norm = torch.norm(grad_vec)
    scale = float(rho) / (grad_norm + float(eps))
    perturbations: List[tuple[nn.Parameter, torch.Tensor]] = []
    with torch.no_grad():
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            e_w = grad.detach() * scale
            param.add_(e_w)
            perturbations.append((param, e_w))
    return perturbations


def restore_perturbation(perturbations: Sequence[tuple[nn.Parameter, torch.Tensor]]) -> None:
    with torch.no_grad():
        for param, e_w in perturbations:
            param.sub_(e_w)


def add_grads_to_parameters(
    params: Sequence[nn.Parameter],
    grads_list: Sequence[Sequence[torch.Tensor | None]],
    *,
    scale: float,
) -> None:
    for grads in grads_list:
        for param, grad in zip(params, grads):
            if grad is None:
                continue
            contribution = grad.detach() * float(scale)
            if param.grad is None:
                param.grad = contribution.clone()
            else:
                param.grad.add_(contribution)


def current_param_grad_vector(params: Sequence[nn.Parameter]) -> torch.Tensor:
    grads = [None if p.grad is None else p.grad.detach() for p in params]
    return flatten_grads(params, grads)


def grads_for_loss(loss: torch.Tensor, params: Sequence[nn.Parameter]) -> List[torch.Tensor | None]:
    grads = torch.autograd.grad(loss, params, allow_unused=True)
    return list(grads)


def sadg_micro_step(
    *,
    model: nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    params: Sequence[nn.Parameter],
    bn_probe: BNProbeRecorder,
    context: Dict[str, Any],
) -> Dict[str, Any]:
    bn_probe.enabled = bool(getattr(args, "enable_bn_probe", False))
    bn_probe.set_context(**context, forward_phase="epsilon")
    epsilon_losses = compute_training_losses(model, img, mask, args)
    epsilon_loss = epsilon_losses["seg_loss"]
    g1 = grads_for_loss(epsilon_loss, params)
    g1_vec = flatten_grads(params, g1)
    g1_norm = tensor_norm(g1_vec)

    perturbations = add_perturbation_from_grads(
        params,
        g1,
        rho=float(args.rho),
        eps=float(args.sadg_eps),
    )

    update_losses: Dict[str, torch.Tensor] | None = None
    g2: List[torch.Tensor | None] | None = None
    try:
        bn_probe.set_context(**context, forward_phase="update")
        update_losses = compute_training_losses(model, img, mask, args)
        g2 = grads_for_loss(update_losses["loss"], params)
    finally:
        restore_perturbation(perturbations)

    if update_losses is None or g2 is None:
        raise RuntimeError("SADG update gradients were not computed.")
    g2_vec = flatten_grads(params, g2)
    return {
        "train_loss": float(update_losses["loss"].detach().cpu()),
        "train_seg_loss": float(update_losses["seg_loss"].detach().cpu()),
        "train_disp_loss": float(update_losses["disp_loss"].detach().cpu()),
        "epsilon_loss": float(epsilon_loss.detach().cpu()),
        "g1_norm": g1_norm,
        "g2_norm": tensor_norm(g2_vec),
        "g1_g2_cosine": tensor_cosine(g1_vec, g2_vec),
        "perturb_norm": float(args.rho) if g1_norm > 0.0 else 0.0,
        "g2_grads": detach_grads(g2),
    }


def meta_values(meta: Dict[str, Any], key: str) -> List[Any]:
    value = meta.get(key, [])
    if torch.is_tensor(value):
        return value.detach().cpu().tolist()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (list, tuple)):
        return list(value)
    return [value]


def combine_meta(metas: Sequence[Dict[str, Any]]) -> Dict[str, List[Any]]:
    keys: List[str] = []
    for meta in metas:
        for key in meta:
            if key not in keys:
                keys.append(key)
    out: Dict[str, List[Any]] = {key: [] for key in keys}
    for meta in metas:
        for key in keys:
            out[key].extend(meta_values(meta, key))
    return out


def batch_composition_probe(img: torch.Tensor, mask: torch.Tensor, meta: Dict[str, Any]) -> Dict[str, float]:
    case_ids = [str(x) for x in meta_values(meta, "case_id")]
    slice_indices = [safe_float(x) for x in meta_values(meta, "sagittal_x_index")]
    fg_ratios = [safe_float(x) for x in meta_values(meta, "foreground_ratio")]
    mask_f = mask.detach()
    class1_ratios = (mask_f == 1).float().mean(dim=(1, 2)).detach().cpu().numpy().astype(float).tolist()
    class2_ratios = (mask_f == 2).float().mean(dim=(1, 2)).detach().cpu().numpy().astype(float).tolist()
    img_cpu = img.detach().float().cpu()
    sample_means = img_cpu.mean(dim=(1, 2, 3)).numpy().astype(float).tolist()
    sample_stds = img_cpu.std(dim=(1, 2, 3), unbiased=False).numpy().astype(float).tolist()
    return {
        "batch_size_observed": int(img.shape[0]),
        "batch_unique_cases": int(len(set(case_ids))),
        "batch_slice_idx_std": finite_std(slice_indices),
        "batch_fg_ratio_mean": finite_mean(fg_ratios),
        "batch_fg_ratio_std": finite_std(fg_ratios),
        "batch_class1_ratio_std": finite_std(class1_ratios),
        "batch_class2_ratio_std": finite_std(class2_ratios),
        "input_sample_mean_var": float(np.var(sample_means)) if sample_means else float("nan"),
        "input_sample_std_var": float(np.var(sample_stds)) if sample_stds else float("nan"),
    }


def half_gradient_probe(
    *,
    model: nn.Module,
    params: Sequence[nn.Parameter],
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    bn_probe: BNProbeRecorder,
) -> Dict[str, float]:
    n = int(img.shape[0])
    if n < 2:
        return {
            "grad_cosine_half1_half2": float("nan"),
            "epsilon_cosine_half1_half2": float("nan"),
        }
    split = n // 2
    if split <= 0 or split >= n:
        return {
            "grad_cosine_half1_half2": float("nan"),
            "epsilon_cosine_half1_half2": float("nan"),
        }
    was_enabled = bn_probe.enabled
    bn_probe.enabled = False
    snapshot = snapshot_bn_state(model)
    try:
        halves = ((img[:split], mask[:split]), (img[split:], mask[split:]))
        eps_vecs: List[torch.Tensor] = []
        full_vecs: List[torch.Tensor] = []
        for half_img, half_mask in halves:
            restore_bn_state(model, snapshot)
            losses = compute_training_losses(model, half_img, half_mask, args)
            eps_vecs.append(flatten_grads(params, grads_for_loss(losses["seg_loss"], params)))
            restore_bn_state(model, snapshot)
            losses = compute_training_losses(model, half_img, half_mask, args)
            full_vecs.append(flatten_grads(params, grads_for_loss(losses["loss"], params)))
        restore_bn_state(model, snapshot)
    finally:
        bn_probe.enabled = was_enabled
    return {
        "grad_cosine_half1_half2": tensor_cosine(full_vecs[0], full_vecs[1]),
        "epsilon_cosine_half1_half2": tensor_cosine(eps_vecs[0], eps_vecs[1]),
    }


def optimizer_steps_per_epoch(n_slices: int, physical_batch: int, accum_steps: int, max_train_steps: int) -> int:
    micro_batches = int(math.ceil(float(n_slices) / float(physical_batch)))
    if int(max_train_steps) > 0:
        micro_batches = min(micro_batches, int(max_train_steps))
    return int(math.ceil(float(micro_batches) / float(accum_steps)))


def target_optimizer_steps(args: argparse.Namespace, group_cfg: GroupConfig, n_slices: int) -> int:
    base_steps = optimizer_steps_per_epoch(n_slices, 4, 1, int(args.max_train_steps))
    own_steps = optimizer_steps_per_epoch(
        n_slices,
        int(group_cfg.physical_batch),
        int(group_cfg.grad_accum_steps),
        int(args.max_train_steps),
    )
    steps_per_epoch = base_steps if bool(group_cfg.fixed_steps_to_bs4) else own_steps
    return max(1, int(args.epochs) * int(steps_per_epoch))


def train_or_load_group(
    *,
    args: argparse.Namespace,
    device: torch.device,
    scheduler: str,
    group: str,
    shot: int,
    seed: int,
) -> tuple[
    nn.Module,
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    List[Dict[str, Any]],
    Dict[str, Any],
    Path,
]:
    set_seed(int(seed))
    scheduler_name = normalize_scheduler(scheduler)
    group_name = normalize_group(group)
    group_cfg = GROUP_CONFIGS[group_name]
    run_dir = run_dir_for(args, scheduler_name, group_name, shot, seed)

    train_dataset = build_train_data(args, shot, seed)
    rng_state = snapshot_rng_state()
    try:
        loss_eval_batches = build_loss_eval_batches(args, [str(x) for x in train_dataset.selected_case_ids], device)
    finally:
        restore_rng_state(rng_state)
    slice_indices_by_case = train_dataset.slice_indices_by_case()
    target_steps = target_optimizer_steps(args, group_cfg, len(train_dataset))
    dataset_meta = {
        "experiment": EXPERIMENT_NAME,
        **scheduler_common_fields(scheduler_name, args),
        "group": group_name,
        "physical_batch": int(group_cfg.physical_batch),
        "bn_batch": int(group_cfg.bn_batch),
        "effective_grad_batch": int(group_cfg.effective_grad_batch),
        "grad_accum_steps": int(group_cfg.grad_accum_steps),
        "ghost_bn_virtual_batch": int(group_cfg.ghost_bn_virtual_batch),
        "fixed_steps_to_bs4": bool(group_cfg.fixed_steps_to_bs4),
        "lambda_disp": float(args.lambda_disp),
        "dice_weight": float(args.dice_weight),
        "rho": float(args.rho),
        "lr": float(args.lr),
        "momentum": float(args.momentum),
        "nesterov": bool(args.nesterov),
        "checkpoint_source": "new_training",
        "checkpoint_path": "",
        "save_model_params": False,
        "source_domain": SOURCE_DOMAIN,
        "shot": int(shot),
        "seed": int(seed),
        "target_optimizer_steps": int(target_steps),
        "train_case_ids": train_dataset.selected_case_ids,
        "n_train_slices": int(len(train_dataset)),
        "slice_policy": train_dataset.slice_policy,
        "num_middle_slices": int(train_dataset.num_middle_slices),
        "filter_min_fg": bool(train_dataset.filter_min_fg),
        "min_fg_ratio": float(args.min_fg_ratio),
        "resize_hw": int(args.resize_hw),
        "slice_indices_by_case": slice_indices_by_case,
        "loss_eval_batch_cache": True,
    }
    ensure_dir(run_dir)
    write_json(run_dir / "dataset_metadata.json", dataset_meta)
    write_json(run_dir / "run_config.json", {**vars(args), **dataset_meta, "output_dir": str(run_dir)})

    model = build_group_model(args, group_cfg, device)
    optimizer = build_optimizer(args, model)
    plateau_scheduler = build_plateau_scheduler(args, optimizer) if scheduler_name == "Plateau" else None
    params = [p for p in model.parameters() if p.requires_grad]
    bn_probe = BNProbeRecorder(model)
    bn_probe.enabled = False

    train_rows: List[Dict[str, Any]] = []
    step_probe_rows: List[Dict[str, Any]] = []
    eval_loss_rows: List[Dict[str, Any]] = []
    eval_loss_summary_rows: List[Dict[str, Any]] = []
    lr_rows: List[Dict[str, Any]] = []
    global_step = 0
    epoch = 0

    print(
        f"[TRAIN] scheduler={scheduler_name} group={group_name} shot={shot} seed={seed} rho={float(args.rho):.6g} "
        f"target_steps={target_steps} physical_bs={group_cfg.physical_batch} accum={group_cfg.grad_accum_steps} "
        f"cases={train_dataset.selected_case_ids} slices={len(train_dataset)}"
    )

    try:
        while global_step < target_steps:
            epoch += 1
            model.train()
            loader = make_loader(train_dataset, batch_size=int(group_cfg.physical_batch), shuffle=True, device=device)
            epoch_step_rows: List[Dict[str, Any]] = []
            micro_imgs: List[torch.Tensor] = []
            micro_masks: List[torch.Tensor] = []
            micro_metas: List[Dict[str, Any]] = []
            micro_metrics: List[Dict[str, Any]] = []
            micro_grads: List[List[torch.Tensor | None]] = []
            optimizer.zero_grad(set_to_none=True)

            def flush_accumulation() -> None:
                nonlocal global_step
                if not micro_metrics or global_step >= target_steps:
                    return
                planned_step = global_step + 1
                step_lr = scheduled_lr_for_step(
                    scheduler=scheduler_name,
                    args=args,
                    planned_step=planned_step,
                    target_steps=target_steps,
                    current_lr=current_optimizer_lr(optimizer),
                )
                set_optimizer_lr(optimizer, step_lr)
                optimizer.zero_grad(set_to_none=True)
                add_grads_to_parameters(params, micro_grads, scale=1.0 / float(len(micro_grads)))
                combined_img = torch.cat(micro_imgs, dim=0)
                combined_mask = torch.cat(micro_masks, dim=0)
                combined_meta = combine_meta(micro_metas)
                row: Dict[str, Any] = {
                    "experiment": EXPERIMENT_NAME,
                    **scheduler_common_fields(scheduler_name, args),
                    "lr": float(step_lr),
                    "group": group_name,
                    "shot": int(shot),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "optimizer_step": int(planned_step),
                    "physical_batch": int(group_cfg.physical_batch),
                    "bn_batch": int(group_cfg.bn_batch),
                    "effective_grad_batch": int(group_cfg.effective_grad_batch),
                    "grad_accum_steps": int(group_cfg.grad_accum_steps),
                    "micro_batches_accumulated": int(len(micro_metrics)),
                    "rho": float(args.rho),
                    "lambda_disp": float(args.lambda_disp),
                    "dice_weight": float(args.dice_weight),
                    "train_loss": finite_mean(m["train_loss"] for m in micro_metrics),
                    "train_seg_loss": finite_mean(m["train_seg_loss"] for m in micro_metrics),
                    "train_disp_loss": finite_mean(m["train_disp_loss"] for m in micro_metrics),
                    "epsilon_loss": finite_mean(m["epsilon_loss"] for m in micro_metrics),
                    "g1_norm": finite_mean(m["g1_norm"] for m in micro_metrics),
                    "g2_norm": finite_mean(m["g2_norm"] for m in micro_metrics),
                    "g1_g2_cosine": finite_mean(m["g1_g2_cosine"] for m in micro_metrics),
                    "perturb_norm": finite_mean(m["perturb_norm"] for m in micro_metrics),
                }
                row.update(batch_composition_probe(combined_img, combined_mask, combined_meta))
                row.update(
                    half_gradient_probe(
                        model=model,
                        params=params,
                        img=combined_img,
                        mask=combined_mask,
                        args=args,
                        bn_probe=bn_probe,
                    )
                    if int(args.probe_every) > 0 and planned_step % int(args.probe_every) == 0
                    else {
                        "grad_cosine_half1_half2": float("nan"),
                        "epsilon_cosine_half1_half2": float("nan"),
                    }
                )
                row["grad_norm"] = tensor_norm(current_param_grad_vector(params))
                optimizer.step()
                global_step += 1
                lr_rows.append(
                    {
                        "experiment": EXPERIMENT_NAME,
                        **scheduler_common_fields(scheduler_name, args),
                        "lr": float(step_lr),
                        "group": group_name,
                        "shot": int(shot),
                        "seed": int(seed),
                        "epoch": int(epoch),
                        "optimizer_step": int(global_step),
                        "target_optimizer_steps": int(target_steps),
                        "plateau_monitor_value": float("nan"),
                        "output_dir": str(run_dir),
                    }
                )
                step_probe_rows.append(row)
                epoch_step_rows.append(row)
                micro_imgs.clear()
                micro_masks.clear()
                micro_metas.clear()
                micro_metrics.clear()
                micro_grads.clear()

            for micro_step, (img, mask, meta) in enumerate(loader, start=1):
                if int(args.max_train_steps) > 0 and micro_step > int(args.max_train_steps):
                    break
                if global_step >= target_steps:
                    break
                img = img.to(device)
                mask = mask.to(device)
                planned_step = global_step + 1
                context = {
                    "experiment": EXPERIMENT_NAME,
                    **scheduler_common_fields(scheduler_name, args),
                    "lr": float(current_optimizer_lr(optimizer)),
                    "group": group_name,
                    "shot": int(shot),
                    "seed": int(seed),
                    "epoch": int(epoch),
                    "optimizer_step": int(planned_step),
                    "micro_step": int(micro_step),
                    "micro_index_in_accum": int(len(micro_metrics) + 1),
                    "physical_batch": int(group_cfg.physical_batch),
                    "bn_batch": int(group_cfg.bn_batch),
                    "effective_grad_batch": int(group_cfg.effective_grad_batch),
                    "grad_accum_steps": int(group_cfg.grad_accum_steps),
                    "rho": float(args.rho),
                    "lambda_disp": float(args.lambda_disp),
                    "dice_weight": float(args.dice_weight),
                }
                metrics = sadg_micro_step(
                    model=model,
                    img=img,
                    mask=mask,
                    args=args,
                    params=params,
                    bn_probe=bn_probe,
                    context=context,
                )
                micro_imgs.append(img.detach())
                micro_masks.append(mask.detach())
                micro_metas.append(meta)
                micro_grads.append(metrics.pop("g2_grads"))
                micro_metrics.append(metrics)
                if len(micro_metrics) >= int(group_cfg.grad_accum_steps):
                    flush_accumulation()
            flush_accumulation()

            epoch_lr = finite_mean(r.get("lr") for r in epoch_step_rows)
            row = {
                "experiment": EXPERIMENT_NAME,
                **scheduler_common_fields(scheduler_name, args),
                "lr": float(epoch_lr),
                "group": group_name,
                "shot": int(shot),
                "seed": int(seed),
                "epoch": int(epoch),
                "optimizer_steps_completed": int(global_step),
                "target_optimizer_steps": int(target_steps),
                "physical_batch": int(group_cfg.physical_batch),
                "bn_batch": int(group_cfg.bn_batch),
                "effective_grad_batch": int(group_cfg.effective_grad_batch),
                "grad_accum_steps": int(group_cfg.grad_accum_steps),
                "rho": float(args.rho),
                "lambda_disp": float(args.lambda_disp),
                "dice_weight": float(args.dice_weight),
                "train_loss": finite_mean(r.get("train_loss") for r in epoch_step_rows),
                "train_seg_loss": finite_mean(r.get("train_seg_loss") for r in epoch_step_rows),
                "train_disp_loss": finite_mean(r.get("train_disp_loss") for r in epoch_step_rows),
                "g1_norm": finite_mean(r.get("g1_norm") for r in epoch_step_rows),
                "g2_norm": finite_mean(r.get("g2_norm") for r in epoch_step_rows),
                "grad_norm": finite_mean(r.get("grad_norm") for r in epoch_step_rows),
                "g1_g2_cosine": finite_mean(r.get("g1_g2_cosine") for r in epoch_step_rows),
                "grad_cosine_half1_half2": finite_mean(r.get("grad_cosine_half1_half2") for r in epoch_step_rows),
                "epsilon_cosine_half1_half2": finite_mean(r.get("epsilon_cosine_half1_half2") for r in epoch_step_rows),
                "train_case_ids": "|".join(train_dataset.selected_case_ids),
                "n_train_slices": int(len(train_dataset)),
                "output_dir": str(run_dir),
            }
            train_rows.append(row)
            last_eval_summary: Dict[str, Any] = {}
            if int(args.loss_eval_every) > 0 and (epoch % int(args.loss_eval_every) == 0 or global_step >= target_steps):
                was_enabled = bn_probe.enabled
                bn_probe.enabled = False
                try:
                    new_eval_loss_rows, last_eval_summary = evaluate_loss(
                        args=args,
                        model=model,
                        device=device,
                        group_cfg=group_cfg,
                        scheduler=scheduler_name,
                        shot=int(shot),
                        seed=int(seed),
                        train_case_ids=[str(x) for x in train_dataset.selected_case_ids],
                        loss_eval_batches=loss_eval_batches,
                        output_dir=run_dir,
                        epoch=int(epoch),
                        optimizer_steps_completed=int(global_step),
                        target_optimizer_steps=int(target_steps),
                        lr=current_optimizer_lr(optimizer),
                    )
                finally:
                    bn_probe.enabled = was_enabled
                monitor_value = safe_float(last_eval_summary.get(str(args.plateau_monitor), float("nan")))
                if plateau_scheduler is not None and np.isfinite(monitor_value):
                    plateau_scheduler.step(monitor_value)
                    last_eval_summary["lr_after_plateau_step"] = current_optimizer_lr(optimizer)
                    if lr_rows:
                        lr_rows[-1]["plateau_monitor_value"] = monitor_value
                        lr_rows[-1]["lr_after_plateau_step"] = current_optimizer_lr(optimizer)
                else:
                    last_eval_summary["lr_after_plateau_step"] = current_optimizer_lr(optimizer)
                eval_loss_rows.extend(new_eval_loss_rows)
                eval_loss_summary_rows.append(last_eval_summary)
            write_csv(run_dir / "training_log.csv", train_rows)
            write_csv(run_dir / "train_step_probes.csv", step_probe_rows)
            write_csv(run_dir / "bn_layer_step_probes.csv", bn_probe.rows)
            write_csv(run_dir / "eval_loss_curves.csv", eval_loss_rows)
            write_csv(run_dir / "eval_loss_summary_curves.csv", eval_loss_summary_rows)
            write_csv(run_dir / "lr_curves.csv", lr_rows)
            eval_text = ""
            if last_eval_summary:
                eval_text = f" eval6={safe_float(last_eval_summary.get('mean_6domain_loss')):.6f} "
            print(
                f"  epoch_cycle {epoch:03d} steps={global_step:04d}/{target_steps:04d} "
                f"lr={safe_float(row['lr']):.6g} "
                f"loss={safe_float(row['train_loss']):.6f} "
                f"{eval_text}"
                f"grad_cos={safe_float(row['grad_cosine_half1_half2']):.6f} "
                f"eps_cos={safe_float(row['epsilon_cosine_half1_half2']):.6f}"
            )
    finally:
        bn_probe.close()

    dataset_meta["final_lr"] = current_optimizer_lr(optimizer)
    return model, train_rows, step_probe_rows, bn_probe.rows, eval_loss_rows, eval_loss_summary_rows, lr_rows, dataset_meta, run_dir


def eval_rows_to_class_rows(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        for cls, class_name in FOREGROUND_CLASS_NAMES.items():
            rows.append(
                {
                    "experiment": row.get("experiment", EXPERIMENT_NAME),
                    "scheduler": row.get("scheduler", ""),
                    "scheduler_tag": row.get("scheduler_tag", ""),
                    "lr": row.get("lr", float("nan")),
                    "base_lr": row.get("base_lr", float("nan")),
                    "min_lr": row.get("min_lr", float("nan")),
                    "warmup_ratio": row.get("warmup_ratio", float("nan")),
                    "poly_power": row.get("poly_power", float("nan")),
                    "optimizer_name": row.get("optimizer_name", ""),
                    "group": row.get("group", ""),
                    "shot": int(row.get("shot", 0)),
                    "seed": int(row.get("seed", 0)),
                    "domain": row.get("domain", ""),
                    "class_id": int(cls),
                    "class_name": class_name,
                    "n_cases": int(row.get("n_cases", 0)),
                    "n_slices": int(row.get("n_slices", 0)),
                    "case_dice": row.get(f"case_dice_{class_name}", float("nan")),
                    "case_hd95": row.get(f"case_hd95_{class_name}", float("nan")),
                    "slice_dice": row.get(f"slice_dice_{class_name}", float("nan")),
                    "slice_hd95": row.get(f"slice_hd95_{class_name}", float("nan")),
                    "output_dir": row.get("output_dir", ""),
                }
            )
    return rows


def evaluate_model(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    group_cfg: GroupConfig,
    scheduler: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    output_dir: Path,
    checkpoint_source: str,
    final_lr: float,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for domain in DOMAIN_NAMES:
        metrics = run_test_flow(
            model,
            domain,
            exclude_case_ids=train_case_ids,
            data_root=Path(args.data_root),
            min_fg_ratio=float(args.min_fg_ratio),
            resize_hw=int(args.resize_hw),
            batch_size=int(args.eval_batch_size),
            device=device,
            max_cases=int(args.max_test_cases) if int(args.max_test_cases) > 0 else None,
            use_split=False,
            eval_set="all_domains",
            slice_policy=str(args.slice_policy),
            num_middle_slices=int(args.num_middle_slices),
            filter_min_fg=bool(args.filter_min_fg),
        )
        row = {
            "experiment": EXPERIMENT_NAME,
            **scheduler_common_fields(scheduler, args),
            "lr": float(final_lr),
            "group": group_cfg.name,
            "physical_batch": int(group_cfg.physical_batch),
            "bn_batch": int(group_cfg.bn_batch),
            "effective_grad_batch": int(group_cfg.effective_grad_batch),
            "grad_accum_steps": int(group_cfg.grad_accum_steps),
            "ghost_bn_virtual_batch": int(group_cfg.ghost_bn_virtual_batch),
            "fixed_steps_to_bs4": bool(group_cfg.fixed_steps_to_bs4),
            "rho": float(args.rho),
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
            "checkpoint_source": checkpoint_source,
            "shot": int(shot),
            "seed": int(seed),
            "domain": metrics["domain"],
            "n_cases": int(metrics["n_cases"]),
            "n_slices": int(metrics["n_slices"]),
            "slice_policy": metrics["slice_policy"],
            "num_middle_slices": int(metrics["num_middle_slices"]),
            "excluded_case_ids": "|".join(metrics["excluded_case_ids"]),
            "output_dir": str(output_dir),
        }
        row.update(metrics["summary"])
        eval_rows.append(row)
        for case_row in metrics["case_rows"]:
            case_rows.append(
                {
                    "experiment": EXPERIMENT_NAME,
                    **scheduler_common_fields(scheduler, args),
                    "lr": float(final_lr),
                    "group": group_cfg.name,
                    "shot": int(shot),
                    "seed": int(seed),
                    "domain": metrics["domain"],
                    **case_row,
                }
            )
        print(
            f"[EVAL] group={group_cfg.name} shot={shot} seed={seed} domain={metrics['domain']} "
            f"case_dice={float(row['case_dice']):.6f} case_hd95={float(row['case_hd95']):.6f} "
            f"slice_dice={float(row['slice_dice']):.6f} slice_hd95={float(row['slice_hd95']):.6f}"
        )
    return eval_rows, case_rows


def summarize_experiment(
    *,
    group_cfg: GroupConfig,
    shot: int,
    seed: int,
    dataset_meta: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    step_rows: Sequence[Dict[str, Any]],
    eval_loss_summary_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    last_train = train_rows[-1] if train_rows else {}
    last_eval_loss = eval_loss_summary_rows[-1] if eval_loss_summary_rows else {}
    source_rows = [row for row in eval_rows if row.get("domain", "") == SOURCE_DOMAIN]
    target_rows = [row for row in eval_rows if row.get("domain", "") != SOURCE_DOMAIN]
    source = source_rows[0] if source_rows else {}
    return {
        "experiment": EXPERIMENT_NAME,
        "scheduler": dataset_meta.get("scheduler", ""),
        "scheduler_tag": dataset_meta.get("scheduler_tag", ""),
        "lr": float(dataset_meta.get("final_lr", float("nan"))),
        "base_lr": float(dataset_meta.get("base_lr", float("nan"))),
        "min_lr": float(dataset_meta.get("min_lr", float("nan"))),
        "warmup_ratio": float(dataset_meta.get("warmup_ratio", float("nan"))),
        "poly_power": float(dataset_meta.get("poly_power", float("nan"))),
        "optimizer_name": dataset_meta.get("optimizer_name", ""),
        "group": group_cfg.name,
        "physical_batch": int(group_cfg.physical_batch),
        "bn_batch": int(group_cfg.bn_batch),
        "effective_grad_batch": int(group_cfg.effective_grad_batch),
        "grad_accum_steps": int(group_cfg.grad_accum_steps),
        "ghost_bn_virtual_batch": int(group_cfg.ghost_bn_virtual_batch),
        "fixed_steps_to_bs4": bool(group_cfg.fixed_steps_to_bs4),
        "rho": float(dataset_meta.get("rho", float("nan"))),
        "lambda_disp": float(dataset_meta.get("lambda_disp", float("nan"))),
        "dice_weight": float(dataset_meta.get("dice_weight", float("nan"))),
        "checkpoint_source": dataset_meta.get("checkpoint_source", ""),
        "checkpoint_path": dataset_meta.get("checkpoint_path", ""),
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": "|".join(str(x) for x in dataset_meta.get("train_case_ids", [])),
        "n_train_slices": int(dataset_meta.get("n_train_slices", 0)),
        "target_optimizer_steps": int(dataset_meta.get("target_optimizer_steps", 0)),
        "actual_optimizer_steps": int(len(step_rows)),
        "final_train_loss": last_train.get("train_loss", float("nan")),
        "final_train_seg_loss": last_train.get("train_seg_loss", float("nan")),
        "final_train_disp_loss": last_train.get("train_disp_loss", float("nan")),
        "final_mean_6domain_loss": last_eval_loss.get("mean_6domain_loss", float("nan")),
        "best_mean_6domain_loss": min(finite_values(row.get("mean_6domain_loss") for row in eval_loss_summary_rows), default=float("nan")),
        "final_mean_5target_loss": last_eval_loss.get("mean_5target_loss", float("nan")),
        "best_mean_5target_loss": min(finite_values(row.get("mean_5target_loss") for row in eval_loss_summary_rows), default=float("nan")),
        "final_source_loss": last_eval_loss.get("source_loss", float("nan")),
        "best_source_loss": min(finite_values(row.get("source_loss") for row in eval_loss_summary_rows), default=float("nan")),
        "source_case_dice": source.get("case_dice", float("nan")),
        "source_case_hd95": source.get("case_hd95", float("nan")),
        "source_slice_dice": source.get("slice_dice", float("nan")),
        "source_slice_hd95": source.get("slice_hd95", float("nan")),
        "mean_6domain_case_dice": finite_mean(row.get("case_dice") for row in eval_rows),
        "mean_6domain_case_hd95": finite_mean(row.get("case_hd95") for row in eval_rows),
        "mean_6domain_slice_dice": finite_mean(row.get("slice_dice") for row in eval_rows),
        "mean_6domain_slice_hd95": finite_mean(row.get("slice_hd95") for row in eval_rows),
        "mean_5target_case_dice": finite_mean(row.get("case_dice") for row in target_rows),
        "mean_5target_case_hd95": finite_mean(row.get("case_hd95") for row in target_rows),
        "mean_5target_slice_dice": finite_mean(row.get("slice_dice") for row in target_rows),
        "mean_5target_slice_hd95": finite_mean(row.get("slice_hd95") for row in target_rows),
        "output_dir": str(output_dir),
    }


def summarize_rows(rows: Sequence[Dict[str, Any]], group_fields: Sequence[str], metrics: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in rows:
        groups[tuple(row.get(field, "") for field in group_fields)].append(row)
    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        group_rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        first = group_rows[0] if group_rows else {}
        for field in ("scheduler_tag", "optimizer_name"):
            if field not in item and field in first:
                item[field] = first.get(field, "")
        for field in ("lr", "base_lr", "min_lr", "warmup_ratio", "poly_power"):
            if field not in item:
                item[field] = finite_mean(row.get(field) for row in group_rows)
        item["n_rows"] = int(len(group_rows))
        for metric in metrics:
            item[f"{metric}_mean"] = finite_mean(row.get(metric) for row in group_rows)
            item[f"{metric}_std"] = finite_std(row.get(metric) for row in group_rows)
        out.append(item)
    return out


def series_oscillation_metrics(values: Sequence[Any]) -> Dict[str, float]:
    xs = finite_values(values)
    if not xs:
        return {
            "mean": float("nan"),
            "cv": float("nan"),
            "positive_jump_ratio": float("nan"),
            "large_positive_jump_ratio_5pct": float("nan"),
        }
    n = len(xs)
    late = xs[int(n * 0.75) :] or xs
    mean_value = float(np.mean(late))
    std_value = float(np.std(late, ddof=1)) if len(late) > 1 else 0.0
    diffs = np.diff(np.asarray(late, dtype=float))
    if diffs.size:
        previous = np.asarray(late[:-1], dtype=float)
        positive_jump_ratio = float(np.mean(diffs > 0.0))
        large_jump_ratio = float(np.mean(diffs > 0.05 * np.maximum(previous, 1e-12)))
    else:
        positive_jump_ratio = float("nan")
        large_jump_ratio = float("nan")
    return {
        "mean": mean_value,
        "cv": float(std_value / mean_value) if mean_value else float("nan"),
        "positive_jump_ratio": positive_jump_ratio,
        "large_positive_jump_ratio_5pct": large_jump_ratio,
    }


def loss_oscillation_rows(
    training_rows: Sequence[Dict[str, Any]],
    eval_loss_summary_rows: Sequence[Dict[str, Any]],
) -> List[Dict[str, Any]]:
    keys = ("scheduler", "group", "shot", "seed")
    grouped_train: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    grouped_eval: Dict[tuple[Any, ...], List[Dict[str, Any]]] = defaultdict(list)
    for row in training_rows:
        grouped_train[tuple(row.get(key, "") for key in keys)].append(row)
    for row in eval_loss_summary_rows:
        grouped_eval[tuple(row.get(key, "") for key in keys)].append(row)
    out: List[Dict[str, Any]] = []
    for key in sorted(set(grouped_train) | set(grouped_eval), key=lambda vals: tuple(str(v) for v in vals)):
        train_group = sorted(grouped_train.get(key, []), key=lambda row: safe_float(row.get("optimizer_steps_completed", 0)))
        eval_group = sorted(grouped_eval.get(key, []), key=lambda row: safe_float(row.get("optimizer_steps_completed", 0)))
        first = (train_group or eval_group or [{}])[0]
        train_stats = series_oscillation_metrics([row.get("train_loss") for row in train_group])
        eval6_stats = series_oscillation_metrics([row.get("mean_6domain_loss") for row in eval_group])
        eval5_stats = series_oscillation_metrics([row.get("mean_5target_loss") for row in eval_group])
        row = {field: value for field, value in zip(keys, key)}
        row.update(
            {
                "scheduler_tag": first.get("scheduler_tag", ""),
                "lr": finite_mean(item.get("lr") for item in (train_group or eval_group)),
                "base_lr": finite_mean(item.get("base_lr") for item in (train_group or eval_group)),
                "min_lr": finite_mean(item.get("min_lr") for item in (train_group or eval_group)),
                "warmup_ratio": finite_mean(item.get("warmup_ratio") for item in (train_group or eval_group)),
                "poly_power": finite_mean(item.get("poly_power") for item in (train_group or eval_group)),
                "optimizer_name": first.get("optimizer_name", ""),
                "n_train_points": int(len(train_group)),
                "n_eval_loss_points": int(len(eval_group)),
                "late25_train_loss_mean": train_stats["mean"],
                "late25_train_loss_cv": train_stats["cv"],
                "late25_train_loss_positive_jump_ratio": train_stats["positive_jump_ratio"],
                "late25_train_loss_large_up_5pct": train_stats["large_positive_jump_ratio_5pct"],
                "late25_mean_6domain_loss_mean": eval6_stats["mean"],
                "late25_mean_6domain_loss_cv": eval6_stats["cv"],
                "late25_mean_6domain_loss_positive_jump_ratio": eval6_stats["positive_jump_ratio"],
                "late25_mean_6domain_loss_large_up_5pct": eval6_stats["large_positive_jump_ratio_5pct"],
                "late25_mean_5target_loss_mean": eval5_stats["mean"],
                "late25_mean_5target_loss_cv": eval5_stats["cv"],
                "late25_mean_5target_loss_positive_jump_ratio": eval5_stats["positive_jump_ratio"],
                "late25_mean_5target_loss_large_up_5pct": eval5_stats["large_positive_jump_ratio_5pct"],
            }
        )
        out.append(row)
    return out


def write_analysis(
    result_root: Path,
    experiment_rows: Sequence[Dict[str, Any]],
    training_rows: Sequence[Dict[str, Any]],
    eval_loss_summary_rows: Sequence[Dict[str, Any]],
    step_rows: Sequence[Dict[str, Any]],
    bn_rows: Sequence[Dict[str, Any]],
) -> None:
    perf_metrics = (
        "source_case_dice",
        "source_case_hd95",
        "mean_5target_case_dice",
        "mean_5target_case_hd95",
        "mean_5target_slice_dice",
        "mean_5target_slice_hd95",
        "final_train_loss",
        "final_mean_6domain_loss",
        "best_mean_6domain_loss",
        "final_mean_5target_loss",
        "best_mean_5target_loss",
    )
    batch_metrics = (
        "batch_unique_cases",
        "batch_slice_idx_std",
        "batch_fg_ratio_mean",
        "batch_fg_ratio_std",
        "batch_class1_ratio_std",
        "batch_class2_ratio_std",
        "input_sample_mean_var",
        "input_sample_std_var",
    )
    grad_metrics = (
        "grad_norm",
        "g1_norm",
        "g2_norm",
        "g1_g2_cosine",
        "grad_cosine_half1_half2",
        "epsilon_cosine_half1_half2",
    )
    bn_metrics = (
        "batch_mean_running_l2",
        "batch_var_running_l2",
        "batch_mean_std_across_channels",
        "virtual_batch_mean_running_l2_mean",
        "virtual_batch_var_running_l2_mean",
        "virtual_batch_mean_std_across_channels_mean",
    )
    scheduler_summary = summarize_rows(experiment_rows, ("scheduler", "group", "shot"), perf_metrics)
    group_summary = summarize_rows(experiment_rows, ("scheduler", "group"), perf_metrics)
    by_scheduler = summarize_rows(experiment_rows, ("scheduler",), perf_metrics)
    batch_summary = summarize_rows(step_rows, ("scheduler", "group"), batch_metrics)
    grad_summary = summarize_rows(step_rows, ("scheduler", "group"), grad_metrics)
    bn_summary = summarize_rows(bn_rows, ("scheduler", "group", "forward_phase"), bn_metrics)
    oscillation = loss_oscillation_rows(training_rows, eval_loss_summary_rows)

    write_csv(result_root / "scheduler_summary.csv", scheduler_summary)
    write_csv(result_root / "analysis_group_summary.csv", group_summary)
    write_csv(result_root / "analysis_by_scheduler.csv", by_scheduler)
    write_csv(result_root / "analysis_batch_composition.csv", batch_summary)
    write_csv(result_root / "analysis_gradient_alignment.csv", grad_summary)
    write_csv(result_root / "analysis_bn_stats.csv", bn_summary)
    write_csv(result_root / "loss_oscillation_summary.csv", oscillation)

    lines = [
        "# DecayingLR Analysis",
        "",
        f"- Source domain: `{SOURCE_DOMAIN}`",
        f"- Experiment rows: {len(experiment_rows)}",
        f"- Training curve rows: {len(training_rows)}",
        f"- Eval loss summary rows: {len(eval_loss_summary_rows)}",
        f"- Train step probe rows: {len(step_rows)}",
        f"- BN probe rows: {len(bn_rows)}",
        "",
        "## Scheduler Performance",
        "| scheduler | target Dice | target HD95 | final 6-domain loss | best 6-domain loss |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in sorted(by_scheduler, key=lambda r: str(r.get("scheduler", ""))):
        lines.append(
            f"| {row.get('scheduler', '')} | "
            f"{safe_float(row.get('mean_5target_case_dice_mean')):.4f} | "
            f"{safe_float(row.get('mean_5target_case_hd95_mean')):.4f} | "
            f"{safe_float(row.get('final_mean_6domain_loss_mean')):.4f} | "
            f"{safe_float(row.get('best_mean_6domain_loss_mean')):.4f} |"
        )
    lines.extend(
        [
            "",
            "## Output Files",
            "- `scheduler_summary.csv`",
            "- `loss_oscillation_summary.csv`",
            "- `eval_loss_curves.csv`",
            "- `eval_loss_summary_curves.csv`",
            "- `lr_curves.csv`",
        ]
    )
    (result_root / "decaying_lr_analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_all_outputs(
    result_root: Path,
    *,
    training_rows: Sequence[Dict[str, Any]],
    step_probe_rows: Sequence[Dict[str, Any]],
    bn_probe_rows: Sequence[Dict[str, Any]],
    eval_loss_rows: Sequence[Dict[str, Any]],
    eval_loss_summary_rows: Sequence[Dict[str, Any]],
    lr_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    case_rows: Sequence[Dict[str, Any]],
    experiment_rows: Sequence[Dict[str, Any]],
) -> None:
    class_rows = eval_rows_to_class_rows(eval_rows)
    write_csv(result_root / "training_curves.csv", training_rows)
    write_csv(result_root / "train_step_probes.csv", step_probe_rows)
    write_csv(result_root / "bn_layer_step_probes.csv", bn_probe_rows)
    write_csv(result_root / "eval_loss_curves.csv", eval_loss_rows)
    write_csv(result_root / "eval_loss_summary_curves.csv", eval_loss_summary_rows)
    write_csv(result_root / "lr_curves.csv", lr_rows)
    write_csv(result_root / "eval_metrics.csv", eval_rows)
    write_csv(result_root / "eval_case_metrics.csv", case_rows)
    write_csv(result_root / "eval_domain_class_metrics.csv", class_rows)
    write_csv(result_root / "experiment_summary.csv", experiment_rows)
    write_csv(
        result_root / "group_summary.csv",
        summarize_rows(
            experiment_rows,
            ("scheduler", "group"),
            ("mean_5target_case_dice", "mean_5target_case_hd95", "source_case_dice", "source_case_hd95", "final_mean_6domain_loss"),
        ),
    )
    write_analysis(result_root, experiment_rows, training_rows, eval_loss_summary_rows, step_probe_rows, bn_probe_rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    result_root = ensure_dir(resolve_path(args.result_root))
    groups = parse_group_list(args.groups)
    schedulers = parse_scheduler_list(args.schedulers)
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)

    all_training_rows: List[Dict[str, Any]] = []
    all_step_probe_rows: List[Dict[str, Any]] = []
    all_bn_probe_rows: List[Dict[str, Any]] = []
    all_eval_loss_rows: List[Dict[str, Any]] = []
    all_eval_loss_summary_rows: List[Dict[str, Any]] = []
    all_lr_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []
    all_experiment_rows: List[Dict[str, Any]] = []

    print(f"[DEVICE] {device}")
    print(f"[RESULT_ROOT] {result_root}")
    print("[SAVE_MODEL_PARAMS] False")
    print(f"[MATRIX] schedulers={schedulers} groups={groups} shots={shots} seeds={seeds}")
    print(f"[OPTIMIZER] {optimizer_name(args)} lr={float(args.lr):.6g}")
    print(f"[OBJECTIVE] epsilon=CE+{float(args.dice_weight)}Dice update=CE+{float(args.dice_weight)}Dice+{float(args.lambda_disp)}Disp rho={float(args.rho)}")
    print(f"[SLICE] policy={args.slice_policy} filter_min_fg={args.filter_min_fg} min_fg_ratio={args.min_fg_ratio}")

    for scheduler_name in schedulers:
        for group in groups:
            group_cfg = GROUP_CONFIGS[normalize_group(group)]
            for shot in shots:
                for seed in seeds:
                    (
                        model,
                        train_rows,
                        step_rows,
                        bn_rows,
                        eval_loss_rows,
                        eval_loss_summary_rows,
                        lr_rows,
                        dataset_meta,
                        output_dir,
                    ) = train_or_load_group(
                        args=args,
                        device=device,
                        scheduler=scheduler_name,
                        group=group_cfg.name,
                        shot=int(shot),
                        seed=int(seed),
                    )
                    eval_rows, case_rows = evaluate_model(
                        args=args,
                        model=model,
                        device=device,
                        group_cfg=group_cfg,
                        scheduler=scheduler_name,
                        shot=int(shot),
                        seed=int(seed),
                        train_case_ids=[str(x) for x in dataset_meta["train_case_ids"]],
                        output_dir=output_dir,
                        checkpoint_source=str(dataset_meta.get("checkpoint_source", "")),
                        final_lr=float(dataset_meta.get("final_lr", float("nan"))),
                    )
                    all_training_rows.extend(train_rows)
                    all_step_probe_rows.extend(step_rows)
                    all_bn_probe_rows.extend(bn_rows)
                    all_eval_loss_rows.extend(eval_loss_rows)
                    all_eval_loss_summary_rows.extend(eval_loss_summary_rows)
                    all_lr_rows.extend(lr_rows)
                    all_eval_rows.extend(eval_rows)
                    all_case_rows.extend(case_rows)
                    all_experiment_rows.append(
                        summarize_experiment(
                            group_cfg=group_cfg,
                            shot=int(shot),
                            seed=int(seed),
                            dataset_meta=dataset_meta,
                            train_rows=train_rows,
                            step_rows=step_rows,
                            eval_loss_summary_rows=eval_loss_summary_rows,
                            eval_rows=eval_rows,
                            output_dir=output_dir,
                        )
                    )
                    write_all_outputs(
                        result_root,
                        training_rows=all_training_rows,
                        step_probe_rows=all_step_probe_rows,
                        bn_probe_rows=all_bn_probe_rows,
                        eval_loss_rows=all_eval_loss_rows,
                        eval_loss_summary_rows=all_eval_loss_summary_rows,
                        lr_rows=all_lr_rows,
                        eval_rows=all_eval_rows,
                        case_rows=all_case_rows,
                        experiment_rows=all_experiment_rows,
                    )
                    del model
                    if device.type == "cuda":
                        torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run DecayingLR ablations for Seg-SADG source training.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=1000)
    parser.add_argument("--groups", default="A_BS4,B_BS8,C_BS8_FixedSteps,E_BS8_GhostBN4")
    parser.add_argument("--schedulers", default="Cosine-0.1x,Cosine-0.01x,Plateau,Poly")
    parser.add_argument("--shots", default="3,4,5")
    parser.add_argument("--seeds", default="3")
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-2)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--weight_decay", type=float, default=3e-5)
    parser.add_argument("--nesterov", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--warmup_ratio", type=float, default=0.05)
    parser.add_argument("--poly_power", type=float, default=0.9)
    parser.add_argument("--loss_eval_every", type=int, default=1)
    parser.add_argument("--plateau_monitor", default="mean_6domain_loss")
    parser.add_argument("--plateau_factor", type=float, default=0.5)
    parser.add_argument("--plateau_patience", type=int, default=20)
    parser.add_argument("--plateau_cooldown", type=int, default=5)
    parser.add_argument("--plateau_min_lr", type=float, default=1e-4)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--lambda_disp", type=float, default=0.05)
    parser.add_argument("--disp_margin", type=float, default=0.0)
    parser.add_argument("--rho", type=float, default=0.05)
    parser.add_argument("--sadg_eps", type=float, default=1e-12)
    parser.add_argument("--probe_every", type=int, default=0)
    parser.add_argument("--enable_bn_probe", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--resize_hw", type=int, default=224)
    parser.add_argument("--min_fg_ratio", type=float, default=0.05)
    parser.add_argument("--filter_min_fg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slice_policy", default="all_filtered", choices=("center9", "all", "all_filtered"))
    parser.add_argument("--num_middle_slices", type=int, default=9)
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--latent_ch", type=int, default=64)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_test_cases", type=int, default=0)
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
