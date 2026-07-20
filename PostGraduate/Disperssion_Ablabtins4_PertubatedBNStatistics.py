from __future__ import annotations

import argparse
from contextlib import contextmanager
import csv
import json
import random
import shutil
import types
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from helper.backbone import ModelConfig, build_model
from helper.backbone_losses import (
    apply_sadg_perturbation_from_loss,
    backbone_training_loss,
    restore_sadg_perturbation,
)
from helper.dataloaders import (
    DATA_ROOT,
    DOMAIN_NAMES,
    FOREGROUND_CLASS_NAMES,
    SOURCE_DOMAIN_NAME,
    build_train_dataset,
    make_loader,
    run_test_flow,
)
from helper.load_model_params import build_backbone_from_checkpoint
# 用扰动参数后的模型计算梯度并反传

PROJECT_ROOT = Path(__file__).resolve().parent
EXPERIMENT_NAME = "PertubatedBNStatistics"
BASE_METHOD = "Disperssion+Seg-SADG"
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / EXPERIMENT_NAME
DEFAULT_NEW_BACKBONE_ROOT = PROJECT_ROOT / "backbones" / EXPERIMENT_NAME
SOURCE_DOMAIN = SOURCE_DOMAIN_NAME

SEG_SADG_ABLATION = "Seg-SADG"
FIXED_RHO = 0.05

BASELINE_METHOD = "Baseline-SegSADG"
BN_CONSISTENCY_METHOD = "BN-Consistency"
BN_ADVERSARIAL_METHOD = "BN-Adversarial"
BN_SENSITIVITY_METHOD = "BN-Sensitivity"
BN_AFFINE_METHOD = "BN-Affine"
BN_COMPACT_METHOD = "BN-Compact"
BN_METHODS = (
    BASELINE_METHOD,
    BN_CONSISTENCY_METHOD,
    BN_ADVERSARIAL_METHOD,
    BN_SENSITIVITY_METHOD,
    BN_AFFINE_METHOD,
    BN_COMPACT_METHOD,
)
BN_METHOD_ALIASES = {
    "baseline": BASELINE_METHOD,
    "baseline-segsadg": BASELINE_METHOD,
    "baseline_seg_sadg": BASELINE_METHOD,
    "baseline-seg-sadg": BASELINE_METHOD,
    "seg-sadg": BASELINE_METHOD,
    "bn-cons": BN_CONSISTENCY_METHOD,
    "bn_consistency": BN_CONSISTENCY_METHOD,
    "consistency": BN_CONSISTENCY_METHOD,
    "bn-adv": BN_ADVERSARIAL_METHOD,
    "bn_adversarial": BN_ADVERSARIAL_METHOD,
    "adversarial": BN_ADVERSARIAL_METHOD,
    "bn-sens": BN_SENSITIVITY_METHOD,
    "bn_sensitivity": BN_SENSITIVITY_METHOD,
    "sensitivity": BN_SENSITIVITY_METHOD,
    "bn-affine": BN_AFFINE_METHOD,
    "bn_affine": BN_AFFINE_METHOD,
    "affine": BN_AFFINE_METHOD,
    "bn-compact": BN_COMPACT_METHOD,
    "bn_compact": BN_COMPACT_METHOD,
    "compact": BN_COMPACT_METHOD,
}


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


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


def rho_tag(value: float) -> str:
    text = f"{float(value):.12g}".replace("-", "m").replace(".", "p")
    return f"rho{text}"


def bn_method_tag(name: str) -> str:
    return normalize_bn_method(name).lower().replace("-", "_").replace("+", "_")


def normalize_bn_method(name: str) -> str:
    key = str(name).strip().lower()
    if key in BN_METHOD_ALIASES:
        return BN_METHOD_ALIASES[key]
    for method in BN_METHODS:
        if key == method.lower():
            return method
    raise ValueError(f"Unknown BN method {name!r}; expected one of {', '.join(BN_METHODS)}")


def parse_method_list(text: str) -> List[str]:
    methods = [normalize_bn_method(name) for name in parse_str_list(text)]
    if not methods:
        raise ValueError("At least one method must be provided.")
    seen: set[str] = set()
    out: List[str] = []
    for method in methods:
        if method not in seen:
            seen.add(method)
            out.append(method)
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


def method_run_dir(args: argparse.Namespace, method: str, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.new_backbone_root)
        / normalize_bn_method(method)
        / f"shot{int(shot)}"
        / f"Seed{int(seed)}"
    )


def save_model_files(run_dir: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(run_dir)
    final_path = run_dir / "checkpoint_final.pt"
    baseline_path = run_dir / "baseline_model_with_metadata.pt"
    torch.save(payload, final_path)
    shutil.copyfile(final_path, baseline_path)


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


def epsilon_source_name() -> str:
    return "CE + Dice"


def bn_config_fields(args: argparse.Namespace, method: str) -> Dict[str, Any]:
    method = normalize_bn_method(method)
    return {
        "experiment": EXPERIMENT_NAME,
        "base_method": BASE_METHOD,
        "loss_mode": method,
        "ablation": method,
        "sadg_method": SEG_SADG_ABLATION,
        "bn_method": method,
        "bn_method_tag": bn_method_tag(method),
        "bn_layers": str(args.bn_layers),
        "bn_mu_std": float(args.bn_mu_std),
        "bn_logvar_std": float(args.bn_logvar_std),
        "bn_adv_radius": float(args.bn_adv_radius),
        "lambda_bn_cons": float(args.lambda_bn_cons),
        "lambda_bn_adv": float(args.lambda_bn_adv),
        "lambda_bn_sens": float(args.lambda_bn_sens),
        "lambda_bn_affine": float(args.lambda_bn_affine),
        "lambda_bn_compact": float(args.lambda_bn_compact),
        "lambda_disp": float(args.lambda_disp),
        "dice_weight": float(args.dice_weight),
        "rho": float(args.rho),
        "rho_tag": rho_tag(float(args.rho)),
    }


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    method: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    n_train_slices: int,
    slice_indices_by_case: Dict[str, List[int]],
) -> Dict[str, Any]:
    cfg = model_config(args)
    method = normalize_bn_method(method)
    method_fields = bn_config_fields(args, method)
    return {
        "model_state_dict": model.state_dict(),
        "metadata": {
            **method_fields,
            "sadg_eps": float(args.sadg_eps),
            "source_domain": SOURCE_DOMAIN,
            "shot": int(shot),
            "seed": int(seed),
            "train_case_ids": list(train_case_ids),
            "n_train_slices": int(n_train_slices),
            "use_split": False,
            "slice_policy": str(args.slice_policy),
            "num_middle_slices": int(args.num_middle_slices),
            "filter_min_fg": bool(args.filter_min_fg),
            "min_fg_ratio": float(args.min_fg_ratio),
            "resize_hw": int(args.resize_hw),
            "model_type": cfg.model_type,
            "batch_norm_momentum": 0.1,
            "slice_indices_by_case": slice_indices_by_case,
            "model_config": {
                "in_ch": int(cfg.in_ch),
                "num_classes": int(cfg.num_classes),
                "base_ch": int(cfg.base_ch),
                "latent_ch": int(cfg.latent_ch),
                "model_type": str(cfg.model_type),
                "use_l2_norm": bool(cfg.use_l2_norm),
                "use_batch_norm": bool(cfg.use_batch_norm),
            },
            "objective": {
                "segmentation": f"CE + {float(args.dice_weight)} Dice",
                "dispersion": "L2_disperssion_loss",
                "lambda_disp": float(args.lambda_disp),
                "dice_weight": float(args.dice_weight),
                "disp_margin": float(args.disp_margin),
                "epsilon_source": epsilon_source_name(),
                "update_source": "CE + Dice + Dispersion + selected BN regularizer at theta + epsilon",
            },
            "args": vars(args),
        },
    }


def normalize_training_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    method: str,
    shot: int,
    seed: int,
    output_dir: Path,
    checkpoint_source: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    method = normalize_bn_method(method)
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row.update(bn_config_fields(args, method))
        row["shot"] = int(shot)
        row["seed"] = int(seed)
        row["checkpoint_source"] = checkpoint_source
        row["output_dir"] = str(output_dir)
        out.append(row)
    return out


def compute_training_losses(
    model: torch.nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    *,
    include_outputs: bool = False,
) -> Dict[str, torch.Tensor]:
    out = model(img, return_features=True)
    losses = backbone_training_loss(
        out["logits"],
        mask,
        out["features"]["dec1"],
        num_classes=3,
        dice_weight=float(args.dice_weight),
        lambda_disp=float(args.lambda_disp),
        disp_margin=float(args.disp_margin),
    )
    if include_outputs:
        losses["_logits"] = out["logits"]
        losses["_dec1"] = out["features"]["dec1"]
    return losses


def select_epsilon_loss(losses: Dict[str, torch.Tensor]) -> torch.Tensor:
    return losses["seg_loss"]


def iter_bn_modules(model: nn.Module) -> List[tuple[str, nn.BatchNorm2d]]:
    return [(name, module) for name, module in model.named_modules() if isinstance(module, nn.BatchNorm2d)]


def selected_bn_modules(model: nn.Module, scope: str) -> List[tuple[str, nn.BatchNorm2d]]:
    scope = str(scope).strip().lower()
    modules = iter_bn_modules(model)
    if scope == "all":
        return modules
    if scope == "shallow":
        prefixes = ("base.enc1.", "base.enc2.", "enc1.", "enc2.")
        return [(name, module) for name, module in modules if name.startswith(prefixes)]
    if scope == "encoder":
        tokens = (".enc1.", ".enc2.", ".enc3.", ".bottleneck.", "base.enc1.", "base.enc2.", "base.enc3.", "base.bottleneck.")
        return [(name, module) for name, module in modules if any(token in f".{name}." for token in tokens)]
    if scope == "all_except_dec1":
        return [(name, module) for name, module in modules if ".dec1." not in f".{name}."]
    raise ValueError(f"Unknown --bn_layers {scope!r}; expected shallow/all/encoder/all_except_dec1.")


@contextmanager
def temporarily_disable_bn_tracking(model: nn.Module):
    states: List[tuple[nn.BatchNorm2d, bool]] = []
    for module in model.modules():
        if isinstance(module, nn.BatchNorm2d):
            states.append((module, bool(module.track_running_stats)))
            module.track_running_stats = False
    try:
        yield
    finally:
        for module, track_running_stats in states:
            module.track_running_stats = track_running_stats


class BNStatPerturbation:
    def __init__(
        self,
        modules: Sequence[tuple[str, nn.BatchNorm2d]],
        *,
        args: argparse.Namespace,
        mode: str,
        fixed_deltas: Dict[str, tuple[torch.Tensor, torch.Tensor]] | None = None,
    ):
        self.modules = list(modules)
        self.args = args
        self.mode = str(mode)
        self.fixed_deltas = fixed_deltas or {}
        self.original_forwards: List[tuple[nn.BatchNorm2d, Any]] = []
        self.delta_vars: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}

    def __enter__(self) -> "BNStatPerturbation":
        for name, module in self.modules:
            self.original_forwards.append((module, module.forward))
            module.forward = types.MethodType(self._make_forward(name), module)
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        for module, original_forward in self.original_forwards:
            module.forward = original_forward
        self.original_forwards.clear()

    def _make_forward(self, name: str):
        def forward(module: nn.BatchNorm2d, x: torch.Tensor) -> torch.Tensor:
            dims = (0, 2, 3)
            mu = x.mean(dim=dims, keepdim=True)
            var = x.var(dim=dims, unbiased=False, keepdim=True)
            scale = torch.sqrt(var.detach() + float(module.eps))
            if self.mode == "random":
                delta_mu = torch.randn_like(mu)
                delta_logvar = torch.randn_like(var)
            elif self.mode == "zero_vars":
                delta_mu = torch.zeros_like(mu, requires_grad=True)
                delta_logvar = torch.zeros_like(var, requires_grad=True)
                self.delta_vars[name] = (delta_mu, delta_logvar)
            elif self.mode == "fixed":
                delta_mu, delta_logvar = self.fixed_deltas[name]
                delta_mu = delta_mu.to(device=x.device, dtype=x.dtype)
                delta_logvar = delta_logvar.to(device=x.device, dtype=x.dtype)
            else:
                raise ValueError(f"Unknown BNStatPerturbation mode: {self.mode}")

            mu_hat = mu + float(self.args.bn_mu_std) * scale * delta_mu
            var_hat = var * torch.exp(float(self.args.bn_logvar_std) * delta_logvar)
            y = (x - mu_hat) / torch.sqrt(var_hat + float(module.eps))
            if module.affine:
                weight = module.weight.view(1, -1, 1, 1)
                bias = module.bias.view(1, -1, 1, 1)
                y = y * weight + bias
            return y

        return forward


def zero_tensor(device: torch.device) -> torch.Tensor:
    return torch.tensor(0.0, device=device)


def foreground_kl_loss(clean_logits: torch.Tensor, shifted_logits: torch.Tensor) -> torch.Tensor:
    clean_prob = torch.softmax(clean_logits.detach(), dim=1)
    shifted_log_prob = torch.log_softmax(shifted_logits, dim=1)
    clean_log_prob = torch.log_softmax(clean_logits.detach(), dim=1)
    return (clean_prob * (clean_log_prob - shifted_log_prob)).sum(dim=1).mean()


def bn_affine_regularizer(modules: Sequence[tuple[str, nn.BatchNorm2d]], device: torch.device) -> torch.Tensor:
    terms: List[torch.Tensor] = []
    for _name, module in modules:
        if module.affine:
            terms.append((module.weight - 1.0).pow(2).mean() + module.bias.pow(2).mean())
    return torch.stack(terms).mean() if terms else zero_tensor(device)


def bn_compact_regularizer(model: nn.Module, img: torch.Tensor, modules: Sequence[tuple[str, nn.BatchNorm2d]]) -> torch.Tensor:
    captured: List[torch.Tensor] = []
    handles = []

    def hook(_module: nn.BatchNorm2d, inputs: tuple[torch.Tensor, ...]) -> None:
        captured.append(inputs[0])

    try:
        for _name, module in modules:
            handles.append(module.register_forward_pre_hook(hook))
        with temporarily_disable_bn_tracking(model):
            model(img, return_features=True)
    finally:
        for handle in handles:
            handle.remove()

    terms: List[torch.Tensor] = []
    for feat in captured:
        sample_mean = feat.mean(dim=(2, 3))
        sample_logstd = torch.log(feat.var(dim=(2, 3), unbiased=False).add(1e-6).sqrt())
        if sample_mean.shape[0] > 1:
            terms.append(sample_mean.var(dim=0, unbiased=False).mean())
            terms.append(sample_logstd.var(dim=0, unbiased=False).mean())
    return torch.stack(terms).mean() if terms else zero_tensor(img.device)


def bn_sensitivity_regularizer(
    model: nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    modules: Sequence[tuple[str, nn.BatchNorm2d]],
) -> torch.Tensor:
    if not modules:
        return zero_tensor(img.device)
    with temporarily_disable_bn_tracking(model):
        with BNStatPerturbation(modules, args=args, mode="zero_vars") as ctx:
            losses = compute_training_losses(model, img, mask, args)
            delta_vars = [v for pair in ctx.delta_vars.values() for v in pair]
            grads = torch.autograd.grad(
                losses["seg_loss"],
                delta_vars,
                retain_graph=True,
                create_graph=True,
                allow_unused=True,
            )
    terms = [grad.pow(2).mean() for grad in grads if grad is not None]
    return torch.stack(terms).mean() if terms else zero_tensor(img.device)


def normalized_adv_deltas(
    delta_vars: Dict[str, tuple[torch.Tensor, torch.Tensor]],
    grads: Sequence[torch.Tensor | None],
    *,
    radius: float,
) -> Dict[str, tuple[torch.Tensor, torch.Tensor]]:
    grad_iter = iter(grads)
    raw: Dict[str, tuple[torch.Tensor, torch.Tensor]] = {}
    sq_sum: torch.Tensor | None = None
    for name, (delta_mu, delta_logvar) in delta_vars.items():
        grad_mu = next(grad_iter)
        grad_logvar = next(grad_iter)
        grad_mu = torch.zeros_like(delta_mu) if grad_mu is None else grad_mu.detach()
        grad_logvar = torch.zeros_like(delta_logvar) if grad_logvar is None else grad_logvar.detach()
        raw[name] = (grad_mu, grad_logvar)
        value = grad_mu.pow(2).sum() + grad_logvar.pow(2).sum()
        sq_sum = value if sq_sum is None else sq_sum + value
    norm = torch.sqrt(sq_sum) if sq_sum is not None else None
    if norm is None or float(norm.detach().cpu()) == 0.0:
        return {name: (torch.zeros_like(mu), torch.zeros_like(logvar)) for name, (mu, logvar) in raw.items()}
    scale = float(radius) / (norm + 1e-12)
    return {name: (grad_mu * scale, grad_logvar * scale) for name, (grad_mu, grad_logvar) in raw.items()}


def bn_adversarial_regularizer(
    model: nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    modules: Sequence[tuple[str, nn.BatchNorm2d]],
) -> torch.Tensor:
    if not modules:
        return zero_tensor(img.device)
    with temporarily_disable_bn_tracking(model):
        with BNStatPerturbation(modules, args=args, mode="zero_vars") as ctx:
            losses = compute_training_losses(model, img, mask, args)
            delta_vars = [v for pair in ctx.delta_vars.values() for v in pair]
            grads = torch.autograd.grad(
                losses["seg_loss"],
                delta_vars,
                retain_graph=False,
                create_graph=False,
                allow_unused=True,
            )
            fixed = normalized_adv_deltas(ctx.delta_vars, grads, radius=float(args.bn_adv_radius))
    with temporarily_disable_bn_tracking(model):
        with BNStatPerturbation(modules, args=args, mode="fixed", fixed_deltas=fixed):
            adv_losses = compute_training_losses(model, img, mask, args)
    return adv_losses["loss"]


def compute_bn_regularizers(
    *,
    method: str,
    model: nn.Module,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    clean_logits: torch.Tensor,
) -> Dict[str, torch.Tensor]:
    method = normalize_bn_method(method)
    modules = selected_bn_modules(model, str(args.bn_layers))
    zeros = {
        "bn_cons_loss": zero_tensor(img.device),
        "bn_adv_loss": zero_tensor(img.device),
        "bn_sens_loss": zero_tensor(img.device),
        "bn_affine_loss": zero_tensor(img.device),
        "bn_compact_loss": zero_tensor(img.device),
    }
    if method == BN_CONSISTENCY_METHOD:
        with temporarily_disable_bn_tracking(model):
            with BNStatPerturbation(modules, args=args, mode="random"):
                shifted = model(img, return_features=True)
        zeros["bn_cons_loss"] = foreground_kl_loss(clean_logits, shifted["logits"])
    elif method == BN_ADVERSARIAL_METHOD:
        zeros["bn_adv_loss"] = bn_adversarial_regularizer(model, img, mask, args, modules)
    elif method == BN_SENSITIVITY_METHOD:
        zeros["bn_sens_loss"] = bn_sensitivity_regularizer(model, img, mask, args, modules)
    elif method == BN_AFFINE_METHOD:
        zeros["bn_affine_loss"] = bn_affine_regularizer(modules, img.device)
    elif method == BN_COMPACT_METHOD:
        zeros["bn_compact_loss"] = bn_compact_regularizer(model, img, modules)
    elif method != BASELINE_METHOD:
        raise ValueError(f"Unsupported BN method: {method}")

    total = (
        float(args.lambda_bn_cons) * zeros["bn_cons_loss"]
        + float(args.lambda_bn_adv) * zeros["bn_adv_loss"]
        + float(args.lambda_bn_sens) * zeros["bn_sens_loss"]
        + float(args.lambda_bn_affine) * zeros["bn_affine_loss"]
        + float(args.lambda_bn_compact) * zeros["bn_compact_loss"]
    )
    zeros["bn_total_reg_loss"] = total
    return zeros


def train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    method: str,
    params: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> Dict[str, float]:
    method = normalize_bn_method(method)

    optimizer.zero_grad(set_to_none=True)
    epsilon_losses = compute_training_losses(model, img, mask, args)
    epsilon_loss = select_epsilon_loss(epsilon_losses)
    sadg_state = apply_sadg_perturbation_from_loss(
        params,
        epsilon_loss,
        rho=float(args.rho),
        eps=float(args.sadg_eps),
        device=device,
    )
    perturbations = sadg_state["perturbations"]
    grad_norm = sadg_state["grad_norm"]
    perturb_norm = sadg_state["perturb_norm"]

    update_losses: Dict[str, torch.Tensor] | None = None
    bn_losses: Dict[str, torch.Tensor] | None = None
    optimizer.zero_grad(set_to_none=True)
    try:
        update_losses = compute_training_losses(model, img, mask, args, include_outputs=True)
        bn_losses = compute_bn_regularizers(
            method=method,
            model=model,
            img=img,
            mask=mask,
            args=args,
            clean_logits=update_losses["_logits"],
        )
        total_loss = update_losses["loss"] + bn_losses["bn_total_reg_loss"]
        total_loss.backward()
    finally:
        restore_sadg_perturbation(perturbations)

    optimizer.step()
    if update_losses is None or bn_losses is None:
        raise RuntimeError("SADG update losses were not computed.")
    return {
        "train_loss": float((update_losses["loss"] + bn_losses["bn_total_reg_loss"]).detach().cpu()),
        "train_base_loss": float(update_losses["loss"].detach().cpu()),
        "train_seg_loss": float(update_losses["seg_loss"].detach().cpu()),
        "train_disp_loss": float(update_losses["disp_loss"].detach().cpu()),
        "epsilon_loss": float(epsilon_loss.detach().cpu()),
        "sadg_grad_norm": float(grad_norm.detach().cpu()),
        "perturb_norm": float(perturb_norm),
        "bn_cons_loss": float(bn_losses["bn_cons_loss"].detach().cpu()),
        "bn_adv_loss": float(bn_losses["bn_adv_loss"].detach().cpu()),
        "bn_sens_loss": float(bn_losses["bn_sens_loss"].detach().cpu()),
        "bn_affine_loss": float(bn_losses["bn_affine_loss"].detach().cpu()),
        "bn_compact_loss": float(bn_losses["bn_compact_loss"].detach().cpu()),
        "bn_total_reg_loss": float(bn_losses["bn_total_reg_loss"].detach().cpu()),
    }


def train_or_load_method(
    *,
    args: argparse.Namespace,
    device: torch.device,
    method: str,
    shot: int,
    seed: int,
) -> tuple[torch.nn.Module, List[Dict[str, Any]], Dict[str, Any], Path]:
    set_seed(int(seed))
    method = normalize_bn_method(method)
    run_dir = method_run_dir(args, method, shot, seed)
    train_dataset = build_train_data(args, shot, seed)
    slice_indices_by_case = train_dataset.slice_indices_by_case()
    dataset_meta = {
        **bn_config_fields(args, method),
        "checkpoint_source": "new_training",
        "source_domain": SOURCE_DOMAIN,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": train_dataset.selected_case_ids,
        "n_train_slices": int(len(train_dataset)),
        "slice_policy": train_dataset.slice_policy,
        "num_middle_slices": int(train_dataset.num_middle_slices),
        "filter_min_fg": bool(train_dataset.filter_min_fg),
        "min_fg_ratio": float(args.min_fg_ratio),
        "resize_hw": int(args.resize_hw),
        "slice_indices_by_case": slice_indices_by_case,
    }
    run_config = {
        **vars(args),
        **dataset_meta,
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "dataset_metadata.json", dataset_meta)
    write_json(run_dir / "run_config.json", run_config)

    checkpoint = run_dir / "baseline_model_with_metadata.pt"
    dataset_meta["checkpoint_path"] = str(checkpoint)
    if checkpoint.exists() and bool(args.resume) and not bool(args.overwrite):
        model, _metadata = build_backbone_from_checkpoint(checkpoint, device=device, strict=True, eval_mode=True)
        rows = normalize_training_rows(
            read_csv(run_dir / "training_log.csv"),
            method=method,
            shot=int(shot),
            seed=int(seed),
            output_dir=run_dir,
            checkpoint_source="new_training",
            args=args,
        )
        print(f"[RESUME] {method} shot={shot} seed={seed}: {checkpoint}")
        return model, rows, dataset_meta, run_dir

    model = build_model(model_config(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    params = [p for p in model.parameters() if p.requires_grad]
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=True, device=device)
    log_rows: List[Dict[str, Any]] = []

    print(
        f"[TRAIN] {method} shot={shot} seed={seed} rho={float(args.rho):.6g} "
        f"lambda_disp={float(args.lambda_disp):.6g} cases={train_dataset.selected_case_ids} "
        f"slices={len(train_dataset)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss: List[float] = []
        epoch_base_loss: List[float] = []
        epoch_seg: List[float] = []
        epoch_disp: List[float] = []
        epoch_epsilon: List[float] = []
        epoch_grad_norm: List[float] = []
        epoch_perturb_norm: List[float] = []
        epoch_bn_cons: List[float] = []
        epoch_bn_adv: List[float] = []
        epoch_bn_sens: List[float] = []
        epoch_bn_affine: List[float] = []
        epoch_bn_compact: List[float] = []
        epoch_bn_total: List[float] = []
        steps = 0
        for step, (img, mask, _meta) in enumerate(loader, start=1):
            if int(args.max_train_steps) > 0 and step > int(args.max_train_steps):
                break
            img = img.to(device)
            mask = mask.to(device)
            step_row = train_step(
                model=model,
                optimizer=optimizer,
                img=img,
                mask=mask,
                args=args,
                method=method,
                params=params,
                device=device,
            )

            steps += 1
            epoch_loss.append(float(step_row["train_loss"]))
            epoch_base_loss.append(float(step_row["train_base_loss"]))
            epoch_seg.append(float(step_row["train_seg_loss"]))
            epoch_disp.append(float(step_row["train_disp_loss"]))
            epoch_epsilon.append(float(step_row["epsilon_loss"]))
            epoch_grad_norm.append(float(step_row["sadg_grad_norm"]))
            epoch_perturb_norm.append(float(step_row["perturb_norm"]))
            epoch_bn_cons.append(float(step_row["bn_cons_loss"]))
            epoch_bn_adv.append(float(step_row["bn_adv_loss"]))
            epoch_bn_sens.append(float(step_row["bn_sens_loss"]))
            epoch_bn_affine.append(float(step_row["bn_affine_loss"]))
            epoch_bn_compact.append(float(step_row["bn_compact_loss"]))
            epoch_bn_total.append(float(step_row["bn_total_reg_loss"]))

        row = {
            **bn_config_fields(args, method),
            "checkpoint_source": "new_training",
            "shot": int(shot),
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": finite_mean(epoch_loss),
            "train_base_loss": finite_mean(epoch_base_loss),
            "train_seg_loss": finite_mean(epoch_seg),
            "train_disp_loss": finite_mean(epoch_disp),
            "epsilon_loss": finite_mean(epoch_epsilon),
            "sadg_grad_norm": finite_mean(epoch_grad_norm),
            "perturb_norm": finite_mean(epoch_perturb_norm),
            "bn_cons_loss": finite_mean(epoch_bn_cons),
            "bn_adv_loss": finite_mean(epoch_bn_adv),
            "bn_sens_loss": finite_mean(epoch_bn_sens),
            "bn_affine_loss": finite_mean(epoch_bn_affine),
            "bn_compact_loss": finite_mean(epoch_bn_compact),
            "bn_total_reg_loss": finite_mean(epoch_bn_total),
            "steps": int(steps),
            "train_case_ids": "|".join(train_dataset.selected_case_ids),
            "n_train_slices": int(len(train_dataset)),
            "slice_policy": str(args.slice_policy),
            "num_middle_slices": int(args.num_middle_slices),
            "filter_min_fg": bool(args.filter_min_fg),
            "output_dir": str(run_dir),
        }
        log_rows.append(row)
        write_csv(run_dir / "training_log.csv", log_rows)
        print(
            f"  epoch {epoch:03d}/{int(args.epochs):03d} "
            f"loss={float(row['train_loss']):.6f} "
            f"base={float(row['train_base_loss']):.6f} "
            f"seg={float(row['train_seg_loss']):.6f} "
            f"disp={float(row['train_disp_loss']):.6f} "
            f"bn_reg={float(row['bn_total_reg_loss']):.6f} "
            f"grad_norm={float(row['sadg_grad_norm']):.6f}"
        )

    payload = checkpoint_payload(
        model,
        args=args,
        method=method,
        shot=int(shot),
        seed=int(seed),
        train_case_ids=train_dataset.selected_case_ids,
        n_train_slices=len(train_dataset),
        slice_indices_by_case=slice_indices_by_case,
    )
    save_model_files(run_dir, payload)
    return model, log_rows, dataset_meta, run_dir


def eval_rows_to_class_rows(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        for cls, class_name in FOREGROUND_CLASS_NAMES.items():
            rows.append(
                {
                    "loss_mode": row.get("loss_mode", ""),
                    "ablation": row.get("ablation", row.get("loss_mode", "")),
                    "sadg_method": row.get("sadg_method", row.get("ablation", "")),
                    "bn_method": row.get("bn_method", row.get("ablation", "")),
                    "bn_method_tag": row.get("bn_method_tag", ""),
                    "bn_layers": row.get("bn_layers", ""),
                    "bn_mu_std": row.get("bn_mu_std", float("nan")),
                    "bn_logvar_std": row.get("bn_logvar_std", float("nan")),
                    "bn_adv_radius": row.get("bn_adv_radius", float("nan")),
                    "lambda_bn_cons": row.get("lambda_bn_cons", float("nan")),
                    "lambda_bn_adv": row.get("lambda_bn_adv", float("nan")),
                    "lambda_bn_sens": row.get("lambda_bn_sens", float("nan")),
                    "lambda_bn_affine": row.get("lambda_bn_affine", float("nan")),
                    "lambda_bn_compact": row.get("lambda_bn_compact", float("nan")),
                    "lambda_disp": row.get("lambda_disp", float("nan")),
                    "dice_weight": row.get("dice_weight", float("nan")),
                    "rho": row.get("rho", float("nan")),
                    "rho_tag": row.get("rho_tag", ""),
                    "checkpoint_source": row.get("checkpoint_source", ""),
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


def class_metric_names() -> List[str]:
    names: List[str] = []
    for prefix in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
        for name in FOREGROUND_CLASS_NAMES.values():
            names.append(f"{prefix}_{name}")
    return names


def summarize_eval_groups(eval_rows: Sequence[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in eval_rows:
        groups.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "lambda_disp": finite_mean(row.get("lambda_disp") for row in rows),
                "dice_weight": finite_mean(row.get("dice_weight") for row in rows),
                "rho": finite_mean(row.get("rho") for row in rows),
                "bn_mu_std": finite_mean(row.get("bn_mu_std") for row in rows),
                "bn_logvar_std": finite_mean(row.get("bn_logvar_std") for row in rows),
                "bn_adv_radius": finite_mean(row.get("bn_adv_radius") for row in rows),
                "lambda_bn_cons": finite_mean(row.get("lambda_bn_cons") for row in rows),
                "lambda_bn_adv": finite_mean(row.get("lambda_bn_adv") for row in rows),
                "lambda_bn_sens": finite_mean(row.get("lambda_bn_sens") for row in rows),
                "lambda_bn_affine": finite_mean(row.get("lambda_bn_affine") for row in rows),
                "lambda_bn_compact": finite_mean(row.get("lambda_bn_compact") for row in rows),
                "n_rows": int(len(rows)),
                "n_cases_total": int(sum(int(float(row.get("n_cases", 0) or 0)) for row in rows)),
                "n_slices_total": int(sum(int(float(row.get("n_slices", 0) or 0)) for row in rows)),
                "case_dice": finite_mean(row.get("case_dice") for row in rows),
                "case_hd95": finite_mean(row.get("case_hd95") for row in rows),
                "slice_dice": finite_mean(row.get("slice_dice") for row in rows),
                "slice_hd95": finite_mean(row.get("slice_hd95") for row in rows),
            }
        )
        for metric in class_metric_names():
            item[metric] = finite_mean(row.get(metric) for row in rows)
        out.append(item)
    return out


def summarize_class_seed_average(class_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    group_fields = ("bn_method", "bn_method_tag", "bn_layers", "rho", "rho_tag", "shot", "domain", "class_id", "class_name")
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in class_rows:
        groups.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        seeds = sorted({int(row.get("seed", 0)) for row in rows})
        item.update(
            {
                "lambda_disp": finite_mean(row.get("lambda_disp") for row in rows),
                "dice_weight": finite_mean(row.get("dice_weight") for row in rows),
                "rho": finite_mean(row.get("rho") for row in rows),
                "bn_mu_std": finite_mean(row.get("bn_mu_std") for row in rows),
                "bn_logvar_std": finite_mean(row.get("bn_logvar_std") for row in rows),
                "bn_adv_radius": finite_mean(row.get("bn_adv_radius") for row in rows),
                "lambda_bn_cons": finite_mean(row.get("lambda_bn_cons") for row in rows),
                "lambda_bn_adv": finite_mean(row.get("lambda_bn_adv") for row in rows),
                "lambda_bn_sens": finite_mean(row.get("lambda_bn_sens") for row in rows),
                "lambda_bn_affine": finite_mean(row.get("lambda_bn_affine") for row in rows),
                "lambda_bn_compact": finite_mean(row.get("lambda_bn_compact") for row in rows),
                "n_seeds": int(len(seeds)),
                "seeds": "|".join(str(seed) for seed in seeds),
                "n_cases_mean": finite_mean(row.get("n_cases") for row in rows),
                "n_slices_mean": finite_mean(row.get("n_slices") for row in rows),
            }
        )
        for metric in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
            item[f"{metric}_mean"] = finite_mean(row.get(metric) for row in rows)
            item[f"{metric}_std"] = finite_std(row.get(metric) for row in rows)
        out.append(item)
    return out


def evaluate_model(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    method: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    output_dir: Path,
    checkpoint_source: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    method = normalize_bn_method(method)
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
            **bn_config_fields(args, method),
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
                    **bn_config_fields(args, method),
                    "checkpoint_source": checkpoint_source,
                    "shot": int(shot),
                    "seed": int(seed),
                    "domain": metrics["domain"],
                    **case_row,
                }
            )
        print(
            f"[EVAL] {method} shot={shot} seed={seed} domain={metrics['domain']} "
            f"case_dice={float(row['case_dice']):.6f} case_hd95={float(row['case_hd95']):.6f} "
            f"slice_dice={float(row['slice_dice']):.6f} slice_hd95={float(row['slice_hd95']):.6f}"
        )
        for class_name in FOREGROUND_CLASS_NAMES.values():
            print(
                f"  class={class_name} "
                f"case_dice={float(row[f'case_dice_{class_name}']):.6f} "
                f"case_hd95={float(row[f'case_hd95_{class_name}']):.6f} "
                f"slice_dice={float(row[f'slice_dice_{class_name}']):.6f} "
                f"slice_hd95={float(row[f'slice_hd95_{class_name}']):.6f}"
            )
    return eval_rows, case_rows


def summarize_experiment(
    *,
    method: str,
    shot: int,
    seed: int,
    dataset_meta: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    method = normalize_bn_method(method)
    last_train = train_rows[-1] if train_rows else {}
    target_rows = [row for row in eval_rows if row.get("domain", "") != SOURCE_DOMAIN]
    return {
        "loss_mode": method,
        "ablation": method,
        "sadg_method": SEG_SADG_ABLATION,
        "bn_method": method,
        "bn_method_tag": dataset_meta.get("bn_method_tag", bn_method_tag(method)),
        "bn_layers": dataset_meta.get("bn_layers", ""),
        "bn_mu_std": float(dataset_meta.get("bn_mu_std", float("nan"))),
        "bn_logvar_std": float(dataset_meta.get("bn_logvar_std", float("nan"))),
        "bn_adv_radius": float(dataset_meta.get("bn_adv_radius", float("nan"))),
        "lambda_bn_cons": float(dataset_meta.get("lambda_bn_cons", float("nan"))),
        "lambda_bn_adv": float(dataset_meta.get("lambda_bn_adv", float("nan"))),
        "lambda_bn_sens": float(dataset_meta.get("lambda_bn_sens", float("nan"))),
        "lambda_bn_affine": float(dataset_meta.get("lambda_bn_affine", float("nan"))),
        "lambda_bn_compact": float(dataset_meta.get("lambda_bn_compact", float("nan"))),
        "lambda_disp": float(dataset_meta.get("lambda_disp", float("nan"))),
        "dice_weight": float(dataset_meta.get("dice_weight", float("nan"))),
        "rho": float(dataset_meta.get("rho", float("nan"))),
        "rho_tag": dataset_meta.get("rho_tag", ""),
        "checkpoint_source": dataset_meta.get("checkpoint_source", ""),
        "checkpoint_path": dataset_meta.get("checkpoint_path", ""),
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": "|".join(str(x) for x in dataset_meta.get("train_case_ids", [])),
        "n_train_slices": int(dataset_meta.get("n_train_slices", 0)),
        "slice_policy": dataset_meta.get("slice_policy", ""),
        "filter_min_fg": bool(dataset_meta.get("filter_min_fg", False)),
        "final_train_loss": last_train.get("train_loss", float("nan")),
        "final_train_base_loss": last_train.get("train_base_loss", float("nan")),
        "final_train_seg_loss": last_train.get("train_seg_loss", float("nan")),
        "final_train_disp_loss": last_train.get("train_disp_loss", float("nan")),
        "final_bn_total_reg_loss": last_train.get("bn_total_reg_loss", float("nan")),
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


def write_all_summaries(
    result_root: Path,
    *,
    training_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    case_rows: Sequence[Dict[str, Any]],
    experiment_rows: Sequence[Dict[str, Any]],
) -> None:
    class_rows = eval_rows_to_class_rows(eval_rows)
    bn_curve_fields = {
        "experiment",
        "base_method",
        "loss_mode",
        "ablation",
        "sadg_method",
        "bn_method",
        "bn_method_tag",
        "bn_layers",
        "rho",
        "rho_tag",
        "lambda_disp",
        "dice_weight",
        "bn_mu_std",
        "bn_logvar_std",
        "bn_adv_radius",
        "lambda_bn_cons",
        "lambda_bn_adv",
        "lambda_bn_sens",
        "lambda_bn_affine",
        "lambda_bn_compact",
        "shot",
        "seed",
        "epoch",
        "bn_cons_loss",
        "bn_adv_loss",
        "bn_sens_loss",
        "bn_affine_loss",
        "bn_compact_loss",
        "bn_total_reg_loss",
        "output_dir",
    }
    bn_regularizer_rows = [{k: row.get(k, "") for k in row if k in bn_curve_fields} for row in training_rows]
    write_csv(result_root / "training_curves.csv", training_rows)
    write_csv(result_root / "bn_regularizer_curves.csv", bn_regularizer_rows)
    write_csv(result_root / "eval_metrics.csv", eval_rows)
    write_csv(result_root / "eval_case_metrics.csv", case_rows)
    write_csv(result_root / "eval_domain_class_metrics.csv", class_rows)
    write_csv(result_root / "seed_avg_domain_class_metrics.csv", summarize_class_seed_average(class_rows))
    write_csv(result_root / "method_summary.csv", summarize_eval_groups(eval_rows, ("bn_method", "bn_method_tag", "bn_layers", "rho", "rho_tag")))
    write_csv(result_root / "shot_summary.csv", summarize_eval_groups(eval_rows, ("bn_method", "bn_method_tag", "bn_layers", "rho", "rho_tag", "shot")))
    write_csv(result_root / "domain_summary.csv", summarize_eval_groups(eval_rows, ("bn_method", "bn_method_tag", "bn_layers", "rho", "rho_tag", "domain")))
    write_csv(result_root / "experiment_summary.csv", experiment_rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    result_root = ensure_dir(resolve_path(args.result_root))
    ensure_dir(resolve_path(args.new_backbone_root))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)
    methods = parse_method_list(args.methods)
    rho = float(FIXED_RHO)

    all_training_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []
    all_experiment_rows: List[Dict[str, Any]] = []

    print(f"[DEVICE] {device}")
    print(f"[RESULT_ROOT] {result_root}")
    print(f"[NEW_BACKBONE_ROOT] {resolve_path(args.new_backbone_root)}")
    print(f"[MATRIX] methods={methods} rho={rho} shots={shots} seeds={seeds} BS={args.batch_size}")
    print(
        f"[OBJECTIVE] CE + {float(args.dice_weight)} Dice + {float(args.lambda_disp)} Dispersion "
        f"+ selected BN regularizer, Seg-SADG rho={rho}"
    )
    print(f"[BN] layers={args.bn_layers} mu_std={args.bn_mu_std} logvar_std={args.bn_logvar_std}")
    print(f"[SLICE] policy={args.slice_policy} filter_min_fg={args.filter_min_fg} min_fg_ratio={args.min_fg_ratio}")

    for method in methods:
        run_args = argparse.Namespace(**{**vars(args), "rho": float(rho), "rho_tag": rho_tag(float(rho))})
        for shot in shots:
            for seed in seeds:
                model, train_rows, dataset_meta, output_dir = train_or_load_method(
                    args=run_args,
                    device=device,
                    method=method,
                    shot=int(shot),
                    seed=int(seed),
                )

                eval_rows, case_rows = evaluate_model(
                    args=run_args,
                    model=model,
                    device=device,
                    method=method,
                    shot=int(shot),
                    seed=int(seed),
                    train_case_ids=[str(x) for x in dataset_meta["train_case_ids"]],
                    output_dir=output_dir,
                    checkpoint_source=str(dataset_meta.get("checkpoint_source", "")),
                )
                all_training_rows.extend(train_rows)
                all_eval_rows.extend(eval_rows)
                all_case_rows.extend(case_rows)
                all_experiment_rows.append(
                    summarize_experiment(
                        method=method,
                        shot=int(shot),
                        seed=int(seed),
                        dataset_meta=dataset_meta,
                        train_rows=train_rows,
                        eval_rows=eval_rows,
                        output_dir=output_dir,
                    )
                )

                write_all_summaries(
                    result_root,
                    training_rows=all_training_rows,
                    eval_rows=all_eval_rows,
                    case_rows=all_case_rows,
                    experiment_rows=all_experiment_rows,
                )
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run PertubatedBNStatistics ablations on top of Disperssion + Seg-SADG.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--methods", default=",".join(BN_METHODS))
    parser.add_argument("--shots", default="3,4,5")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--lambda_disp", type=float, default=0.05)
    parser.add_argument("--disp_margin", type=float, default=0.0)
    parser.add_argument("--sadg_eps", type=float, default=1e-12)
    parser.add_argument("--bn_layers", default="shallow", choices=("shallow", "all", "encoder", "all_except_dec1"))
    parser.add_argument("--bn_mu_std", type=float, default=0.10)
    parser.add_argument("--bn_logvar_std", type=float, default=0.10)
    parser.add_argument("--bn_adv_radius", type=float, default=1.0)
    parser.add_argument("--lambda_bn_cons", type=float, default=0.05)
    parser.add_argument("--lambda_bn_adv", type=float, default=0.10)
    parser.add_argument("--lambda_bn_sens", type=float, default=1e-3)
    parser.add_argument("--lambda_bn_affine", type=float, default=1e-4)
    parser.add_argument("--lambda_bn_compact", type=float, default=1e-3)
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
    parser.add_argument("--new_backbone_root", default=str(DEFAULT_NEW_BACKBONE_ROOT))
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
