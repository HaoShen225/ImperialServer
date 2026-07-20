from __future__ import annotations

from dataclasses import fields
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn

from helper.backbone import ModelConfig, build_model


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_BACKBONE_ROOT = PROJECT_ROOT / "backbones"
DEFAULT_CHECKPOINT_NAME = "baseline_model_with_metadata.pt"
CHECKPOINT_CANDIDATES = (
    "baseline_model_with_metadata.pt",
    "checkpoint_final.pt",
    "baseline_model.pt",
    "best_model.pt",
    "checkpoint_best.pt",
)


def _safe_torch_load(path: str | Path, map_location: str | torch.device = "cpu") -> Any:
    try:
        return torch.load(path, map_location=map_location, weights_only=False)
    except TypeError:
        return torch.load(path, map_location=map_location)


def _strip_module_prefix(state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    keys = list(state_dict.keys())
    if keys and all(str(key).startswith("module.") for key in keys):
        return {str(key)[7:]: value for key, value in state_dict.items()}
    return {str(key): value for key, value in state_dict.items()}


def extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, Mapping):
        for key in ("model_state_dict", "state_dict", "model"):
            value = payload.get(key)
            if isinstance(value, Mapping):
                return _strip_module_prefix(value)
        if payload and all(torch.is_tensor(value) for value in payload.values()):
            return _strip_module_prefix(payload)
    raise ValueError("Checkpoint does not contain a model state_dict.")


def extract_metadata(payload: Any) -> dict[str, Any]:
    if isinstance(payload, Mapping) and isinstance(payload.get("metadata"), Mapping):
        return dict(payload["metadata"])
    return {}


def model_config_from_metadata(metadata: Mapping[str, Any] | None = None) -> ModelConfig:
    metadata = metadata or {}
    raw = metadata.get("model_config", {})
    if not isinstance(raw, Mapping):
        raw = {}
    valid_fields = {field.name for field in fields(ModelConfig)}
    clean = {key: value for key, value in raw.items() if key in valid_fields}
    clean.setdefault("in_ch", 1)
    clean.setdefault("num_classes", 3)
    clean.setdefault("base_ch", 16)
    clean.setdefault("latent_ch", 64)
    clean.setdefault("model_type", metadata.get("model_type", "d_l2_disp_bn"))
    clean.setdefault("use_l2_norm", True)
    clean.setdefault("use_batch_norm", True)
    return ModelConfig(**clean)


def resolve_checkpoint_path(
    *,
    backbone_root: str | Path = DEFAULT_BACKBONE_ROOT,
    loss_mode: str,
    shot: int,
    seed: int,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
) -> Path:
    root = Path(backbone_root)
    run_dir = root / str(loss_mode) / f"shot{int(shot)}" / f"Seed{int(seed)}"
    requested = run_dir / checkpoint_name
    if requested.exists():
        return requested
    for name in CHECKPOINT_CANDIDATES:
        candidate = run_dir / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No checkpoint found in {run_dir}")


def load_model_params(
    model: nn.Module,
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
) -> dict[str, Any]:
    payload = _safe_torch_load(checkpoint_path, map_location=device)
    state_dict = extract_state_dict(payload)
    load_result = model.load_state_dict(state_dict, strict=bool(strict))
    metadata = extract_metadata(payload)
    metadata["checkpoint_path"] = str(Path(checkpoint_path))
    metadata["missing_keys"] = list(getattr(load_result, "missing_keys", []))
    metadata["unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []))
    return metadata


def build_backbone_from_checkpoint(
    checkpoint_path: str | Path,
    *,
    device: str | torch.device = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
    config: ModelConfig | None = None,
) -> tuple[nn.Module, dict[str, Any]]:
    payload = _safe_torch_load(checkpoint_path, map_location=device)
    metadata = extract_metadata(payload)
    model = build_model(config or model_config_from_metadata(metadata)).to(torch.device(device))
    state_dict = extract_state_dict(payload)
    load_result = model.load_state_dict(state_dict, strict=bool(strict))
    if eval_mode:
        model.eval()
    metadata["checkpoint_path"] = str(Path(checkpoint_path))
    metadata["missing_keys"] = list(getattr(load_result, "missing_keys", []))
    metadata["unexpected_keys"] = list(getattr(load_result, "unexpected_keys", []))
    return model, metadata


def load_backbone_by_spec(
    *,
    loss_mode: str,
    shot: int,
    seed: int,
    backbone_root: str | Path = DEFAULT_BACKBONE_ROOT,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
    device: str | torch.device = "cpu",
    strict: bool = True,
    eval_mode: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    checkpoint_path = resolve_checkpoint_path(
        backbone_root=backbone_root,
        loss_mode=loss_mode,
        shot=int(shot),
        seed=int(seed),
        checkpoint_name=checkpoint_name,
    )
    return build_backbone_from_checkpoint(
        checkpoint_path,
        device=device,
        strict=strict,
        eval_mode=eval_mode,
    )


__all__ = [
    "DEFAULT_BACKBONE_ROOT",
    "DEFAULT_CHECKPOINT_NAME",
    "ModelConfig",
    "build_backbone_from_checkpoint",
    "extract_metadata",
    "extract_state_dict",
    "load_backbone_by_spec",
    "load_model_params",
    "model_config_from_metadata",
    "resolve_checkpoint_path",
]
