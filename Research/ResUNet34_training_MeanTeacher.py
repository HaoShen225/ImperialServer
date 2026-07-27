"""Train ResUNet-34 ACDC source models with the established Mean Teacher protocol.

This entry point deliberately reuses ``backbone_training_MeanTeacher`` as the
training engine so that the U-Net and ResUNet-34 experiments share the same
data split, losses, EMA schedule, evaluation, and checkpoint schema.
"""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

import backbone_training_MeanTeacher as protocol
from helper.backbones.ResUNet34 import ResUNet34


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "ResUNet34_params"
DEFAULT_DATASET_ROOT = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/ACDC_normalized"
)
DEFAULT_TARGET_DATASET_ROOT = Path(
    "/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Cardiac/MMS_normalized"
)
ARCHITECTURE = "ResUNet34"
ENCODER = "ResNet34"
ENCODER_INITIALIZATION = "torchvision_resnet34_imagenet1k_v1"
ENCODER_INPUT_ADAPTATION = "rgb_channel_mean"
ENCODER_WEIGHTS_HASH_PREFIX = "b627a593"
DEFAULT_ENCODER_WEIGHTS = (
    PROJECT_ROOT / "Logs" / "pretrained" / "resnet34-b627a593.pth"
)
NUM_EXPECTED_RUNS = 15

_USE_L2_PROJECTION = False
_ENCODER_WEIGHTS_PATH = DEFAULT_ENCODER_WEIGHTS
_ENCODER_WEIGHTS_SHA256 = ""
_ENCODER_STATE_CACHE: Dict[str, torch.Tensor] | None = None
_ORIGINAL_CHECKPOINT_PAYLOAD = protocol.checkpoint_payload


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def imagenet_encoder_state(
    path: Path,
    expected_state: Mapping[str, torch.Tensor],
) -> Dict[str, torch.Tensor]:
    """Load and adapt the official torchvision ResNet-34 encoder state."""
    global _ENCODER_STATE_CACHE, _ENCODER_WEIGHTS_SHA256

    if _ENCODER_STATE_CACHE is not None:
        return _ENCODER_STATE_CACHE

    digest = file_sha256(path)
    if not digest.startswith(ENCODER_WEIGHTS_HASH_PREFIX):
        raise RuntimeError(
            f"Unexpected ResNet-34 weights SHA256 for {path}: {digest}; "
            f"expected prefix {ENCODER_WEIGHTS_HASH_PREFIX}."
        )

    payload = torch.load(path, map_location="cpu", weights_only=True)
    if not isinstance(payload, Mapping):
        raise TypeError(f"Expected a state-dict mapping in {path}, got {type(payload)!r}.")

    adapted: Dict[str, torch.Tensor] = {}
    for key, expected_value in expected_state.items():
        if key not in payload:
            if key.endswith(".num_batches_tracked"):
                adapted[key] = expected_value.clone()
                continue
            raise KeyError(f"Official ResNet-34 weights are missing encoder key {key!r}.")
        value = payload[key]
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Official weight {key!r} is not a tensor.")
        if key == "conv1.weight":
            if tuple(value.shape) != (64, 3, 7, 7):
                raise ValueError(
                    "Expected official conv1.weight shape (64, 3, 7, 7), "
                    f"got {tuple(value.shape)}."
                )
            value = value.mean(dim=1, keepdim=True)
        if value.shape != expected_value.shape:
            raise ValueError(
                f"Shape mismatch for encoder key {key!r}: "
                f"expected {tuple(expected_value.shape)}, got {tuple(value.shape)}."
            )
        adapted[key] = value

    _ENCODER_WEIGHTS_SHA256 = digest
    _ENCODER_STATE_CACHE = adapted
    return adapted


def build_model() -> ResUNet34:
    """Build the single-channel, four-class GraTA backbone."""
    model = ResUNet34(
        n_channels=1,
        n_classes=protocol.NUM_CLASSES,
        only_feature=False,
        use_l2_projection=_USE_L2_PROJECTION,
    )
    model.encoder.load_state_dict(
        imagenet_encoder_state(
            _ENCODER_WEIGHTS_PATH,
            model.encoder.state_dict(),
        ),
        strict=True,
    )
    return model


def checkpoint_payload(**kwargs: Any) -> Dict[str, Any]:
    """Add architecture provenance to the shared checkpoint format."""
    payload = _ORIGINAL_CHECKPOINT_PAYLOAD(**kwargs)
    metadata = payload.setdefault("metadata", {})
    metadata.update(
        {
            "architecture": ARCHITECTURE,
            "encoder": ENCODER,
            "encoder_initialization": ENCODER_INITIALIZATION,
            "encoder_weights": str(_ENCODER_WEIGHTS_PATH),
            "encoder_weights_sha256": _ENCODER_WEIGHTS_SHA256,
            "encoder_input_adaptation": ENCODER_INPUT_ADAPTATION,
            "use_l2_projection": bool(_USE_L2_PROJECTION),
        }
    )
    return payload


def backbone_id_for_run(labeled_cases_per_class: int, seed: int) -> str:
    projection = "L2" if _USE_L2_PROJECTION else "NoL2"
    return (
        f"MeanTeacher_{ARCHITECTURE}_ImageNet1K_V1_{projection}_"
        f"Patient{int(labeled_cases_per_class)}_Seed{int(seed)}_teacher"
    )


def build_arg_parser() -> argparse.ArgumentParser:
    parser = protocol.build_arg_parser()
    parser.description = "Train ACDC ResUNet-34 Mean Teacher source-domain backbones."
    parser.set_defaults(
        dataset_root=str(DEFAULT_DATASET_ROOT),
        target_dataset_root=str(DEFAULT_TARGET_DATASET_ROOT),
        output_root=str(DEFAULT_OUTPUT_ROOT),
    )
    parser.add_argument(
        "--use-l2-projection",
        action="store_true",
        help="Enable per-pixel unit-sphere projection before the segmentation head.",
    )
    parser.add_argument(
        "--encoder-weights",
        default=str(DEFAULT_ENCODER_WEIGHTS),
        help=(
            "Local torchvision ResNet-34 ImageNet-1K V1 state dict. "
            "The file must have the official b627a593 SHA256 prefix."
        ),
    )
    return parser


def validate_args(args: argparse.Namespace) -> tuple[list[int], list[int]]:
    patients = protocol.parse_int_list(args.patients)
    seeds = protocol.parse_int_list(args.seeds)
    if not patients:
        raise ValueError("--patients produced an empty list.")
    if not seeds:
        raise ValueError("--seeds produced an empty list.")
    if any(patient not in (1, 2, 3) for patient in patients):
        raise ValueError("ResUNet-34 training supports Patient settings 1, 2, and 3.")
    if any(seed not in range(5) for seed in seeds):
        raise ValueError("ResUNet-34 training supports seeds 0 through 4.")
    if int(args.labeled_batch_size) <= 0 or int(args.labeled_batch_size) >= int(args.batch_size):
        raise ValueError("--labeled-batch-size must satisfy 0 < labeled < batch-size.")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval-batch-size must be positive.")
    encoder_weights = Path(args.encoder_weights).expanduser()
    if not encoder_weights.is_file():
        raise FileNotFoundError(
            f"Official ResNet-34 encoder weights not found: {encoder_weights}"
        )
    return patients, seeds


def augment_run_config(
    output_root: Path,
    labeled_cases_per_class: int,
    seed: int,
) -> None:
    path = protocol.run_dir(output_root, labeled_cases_per_class, seed) / "run_config.json"
    if not path.is_file():
        return
    config = json.loads(path.read_text(encoding="utf-8"))
    config.update(
        {
            "architecture": ARCHITECTURE,
            "encoder": ENCODER,
            "encoder_initialization": ENCODER_INITIALIZATION,
            "encoder_weights": str(_ENCODER_WEIGHTS_PATH),
            "encoder_weights_sha256": _ENCODER_WEIGHTS_SHA256,
            "encoder_input_adaptation": ENCODER_INPUT_ADAPTATION,
            "use_l2_projection": bool(_USE_L2_PROJECTION),
        }
    )
    protocol.write_json(path, config)


def write_run_summary(
    output_root: Path,
    summary: Mapping[str, Any],
    labeled_cases_per_class: int,
    seed: int,
) -> None:
    payload = dict(summary)
    payload.update(
        {
            "labeled_cases_per_class": int(labeled_cases_per_class),
            "seed": int(seed),
            "architecture": ARCHITECTURE,
            "encoder": ENCODER,
            "encoder_initialization": ENCODER_INITIALIZATION,
            "encoder_weights": str(_ENCODER_WEIGHTS_PATH),
            "encoder_weights_sha256": _ENCODER_WEIGHTS_SHA256,
            "encoder_input_adaptation": ENCODER_INPUT_ADAPTATION,
            "use_l2_projection": bool(_USE_L2_PROJECTION),
        }
    )
    path = protocol.run_dir(output_root, labeled_cases_per_class, seed) / "run_summary.json"
    protocol.write_json(path, payload)


def rebuild_global_summary(output_root: Path) -> int:
    """Safely rebuild the global CSV while PBS array tasks finish concurrently."""
    protocol.ensure_dir(output_root)
    lock_path = output_root / ".mean_teacher_runs.lock"
    with lock_path.open("a+", encoding="utf-8") as lock_handle:
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_EX)
        rows: list[Dict[str, Any]] = []
        for patient in (1, 2, 3):
            for seed in range(5):
                summary_path = protocol.run_dir(output_root, patient, seed) / "run_summary.json"
                if not summary_path.is_file():
                    continue
                payload = json.loads(summary_path.read_text(encoding="utf-8"))
                if isinstance(payload, dict):
                    rows.append(payload)
        rows.sort(
            key=lambda row: (
                int(row.get("labeled_cases_per_class", -1)),
                int(row.get("seed", -1)),
            )
        )
        protocol.write_csv(output_root / "mean_teacher_runs.csv", rows)
        fcntl.flock(lock_handle.fileno(), fcntl.LOCK_UN)
    return len(rows)


def install_resunet_protocol_hooks() -> None:
    protocol.build_model = build_model
    protocol.checkpoint_payload = checkpoint_payload
    protocol.backbone_id_for_run = backbone_id_for_run


def main(argv: Sequence[str] | None = None) -> None:
    global _ENCODER_WEIGHTS_PATH, _USE_L2_PROJECTION

    parser = build_arg_parser()
    args = parser.parse_args(argv)
    patients, seeds = validate_args(args)
    _USE_L2_PROJECTION = bool(args.use_l2_projection)
    _ENCODER_WEIGHTS_PATH = Path(args.encoder_weights).expanduser().resolve()
    install_resunet_protocol_hooks()

    device = protocol.resolve_device(args.device)
    output_root = Path(args.output_root)
    protocol.ensure_dir(output_root)
    print(
        f"[DEVICE] {device} architecture={ARCHITECTURE} "
        f"encoder_init={ENCODER_INITIALIZATION} "
        f"encoder_weights={_ENCODER_WEIGHTS_PATH} "
        f"input_adaptation={ENCODER_INPUT_ADAPTATION} "
        f"use_l2_projection={_USE_L2_PROJECTION}"
    )

    for labeled_cases_per_class in patients:
        for seed in seeds:
            summary = protocol.train_one_run(
                args=args,
                device=device,
                labeled_cases_per_class=int(labeled_cases_per_class),
                seed=int(seed),
            )
            augment_run_config(output_root, labeled_cases_per_class, seed)
            write_run_summary(
                output_root,
                summary,
                labeled_cases_per_class,
                seed,
            )

    completed = rebuild_global_summary(output_root)
    print(
        f"[DONE] summary contains {completed}/{NUM_EXPECTED_RUNS} runs: "
        f"{output_root / 'mean_teacher_runs.csv'}"
    )


if __name__ == "__main__":
    main(sys.argv[1:])
