"""Small, method-agnostic infrastructure helpers."""

from __future__ import annotations

import hashlib
import json
import os
import random
import subprocess
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import yaml


_TOP_LEVEL_KEYS = {
    "experiment", "data", "model", "source", "evaluation", "tta", "methods"
}


def _reject_unknown_keys(value: Mapping[str, Any], allowed: set[str], where: str) -> None:
    unknown = set(value) - allowed
    if unknown:
        raise ValueError(f"Unknown configuration key(s) in {where}: {sorted(unknown)}")


def validate_config(cfg: Mapping[str, Any]) -> None:
    _reject_unknown_keys(cfg, _TOP_LEVEL_KEYS, "root")
    required = _TOP_LEVEL_KEYS
    missing = required - set(cfg)
    if missing:
        raise ValueError(f"Missing configuration section(s): {sorted(missing)}")

    _reject_unknown_keys(
        cfg["experiment"],
        {"source_seeds", "harness_seed", "target_vendors", "initialization_profile"},
        "experiment",
    )
    initialization_profile = cfg["experiment"].get("initialization_profile")
    if initialization_profile not in {"imagenet", "stochastic"}:
        raise ValueError("initialization_profile must be 'imagenet' or 'stochastic'")
    expected_pretrained = initialization_profile == "imagenet"
    if bool(cfg["model"]["pretrained_encoder"]) != expected_pretrained:
        raise ValueError(
            "model.pretrained_encoder is inconsistent with experiment.initialization_profile"
        )

    method_keys = {
        "source": {"profile_verified", "profile_kind", "method_seed"},
        "tent": {"profile_verified", "profile_kind", "method_seed", "steps", "update_scope", "bn_policy", "optimizer", "lr", "momentum", "weight_decay"},
        "eata": {"profile_verified", "profile_kind", "method_seed", "steps", "update_scope", "bn_policy", "optimizer", "lr", "momentum", "weight_decay", "entropy_margin_factor", "redundancy_margin", "probability_momentum", "descriptor", "fisher_path", "fisher_samples", "fisher_alpha"},
        "sar": {"profile_verified", "profile_kind", "method_seed", "steps", "update_scope", "bn_policy", "optimizer", "lr", "momentum", "weight_decay", "entropy_margin_factor", "rho", "recovery_ema", "recovery_threshold"},
        "cotta": {"profile_verified", "profile_kind", "method_seed", "steps", "update_scope", "bn_policy", "optimizer", "lr", "head_lr_multiplier", "momentum", "weight_decay", "teacher_momentum", "confidence_gate", "restore_probability", "augmentation_scales", "horizontal_flip_probability"},
        "rotta": {"profile_verified", "profile_kind", "method_seed", "steps", "update_scope", "optimizer", "lr", "beta", "weight_decay", "teacher_nu", "rbn_alpha", "memory_capacity", "update_frequency", "memory_category_key", "lambda_timeliness", "lambda_uncertainty"},
        "roid": {"profile_verified", "profile_kind", "method_seed", "steps", "update_scope", "bn_policy", "optimizer", "lr", "momentum", "weight_decay", "probability_momentum", "temperature", "source_weight_momentum", "consistency", "prior_correction"},
        "deyo": {"profile_verified", "profile_kind", "method_seed", "steps", "update_scope", "bn_policy", "optimizer", "lr", "momentum", "weight_decay", "entropy_margin_factor", "entropy_weight_margin_factor", "plpd_threshold", "patch_grid", "foreground_only"},
    }
    _reject_unknown_keys(cfg["methods"], set(method_keys), "methods")
    for name, allowed in method_keys.items():
        if name not in cfg["methods"]:
            raise ValueError(f"Missing method configuration: {name}")
        _reject_unknown_keys(cfg["methods"][name], allowed, f"methods.{name}")

    if cfg["evaluation"]["grid"] != "processed_256":
        raise ValueError("This locked profile supports only processed_256 evaluation")
    if cfg["evaluation"]["metrics"] != ["dice", "hd95_px"]:
        raise ValueError("Locked metrics must be exactly [dice, hd95_px]")
    if cfg["tta"]["timing"] not in {"adapt_then_predict", "predict_then_adapt"}:
        raise ValueError("Unknown TTA timing")
    if cfg["tta"]["reset"] not in {"patient", "vendor", "never"}:
        raise ValueError("Unknown TTA reset policy")


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as handle:
        cfg = yaml.safe_load(handle)
    if not isinstance(cfg, dict):
        raise ValueError("Configuration root must be a mapping")
    validate_config(cfg)
    return cfg


def set_seed(seed: int, deterministic: bool = True) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    if deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def state_dict_sha256(state: Mapping[str, torch.Tensor]) -> str:
    """Hash tensor names, metadata, and bytes without relying on torch.save."""
    digest = hashlib.sha256()
    for name, value in sorted(state.items()):
        tensor = value.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(tensor.dtype).encode("ascii"))
        digest.update(str(tuple(tensor.shape)).encode("ascii"))
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def save_json(value: Any, path: str | Path) -> None:
    destination = Path(path)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True, allow_nan=False)
        handle.write("\n")


def get_device(name: str) -> torch.device:
    if name == "auto":
        name = "cuda" if torch.cuda.is_available() else "cpu"
    if name == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is unavailable")
    if name not in {"cpu", "cuda"}:
        raise ValueError(f"Unsupported device: {name}")
    return torch.device(name)


def _git_commit(root: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=root, text=True, stderr=subprocess.DEVNULL
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def run_metadata(root: str | Path = ".") -> dict[str, Any]:
    root_path = Path(root).resolve()
    return {
        "git_commit": _git_commit(root_path),
        "python": os.sys.version.split()[0],
        "torch": torch.__version__,
        "cuda_runtime": torch.version.cuda,
        "numpy": np.__version__,
        "device_count": torch.cuda.device_count(),
    }
