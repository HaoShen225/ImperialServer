from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch

from helper.backbone import ModelConfig, build_model
from helper.backbone_losses import backbone_training_loss
from helper.dataloaders import (
    DATA_ROOT,
    DEFAULT_SPLIT_CSV,
    DOMAIN_NAMES,
    FOREGROUND_CLASS_NAMES,
    SOURCE_DOMAIN_NAME,
    TARGET_DOMAIN_NAMES,
    build_train_dataset,
    make_loader,
    run_test_flow,
)


PROJECT_ROOT = Path(__file__).resolve().parent
SOURCE_DOMAIN = SOURCE_DOMAIN_NAME
BACKBONE_ROOT = PROJECT_ROOT / "beckbones"
SUMMARY_ROOT = PROJECT_ROOT / "TrainingData" / "DisperssionVsKnaked"
DEFAULT_OUTPUT_SUBDIR = "BatchSize4"

LOSS_CONFIGS = {
    "Disperssion": {"lambda_disp": 0.05},
    "PureCE": {"lambda_disp": 0.0},
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


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_loss_modes(text: str) -> List[str]:
    aliases = {
        "disperssion": "Disperssion",
        "dispersion": "Disperssion",
        "purece": "PureCE",
        "pure_ce": "PureCE",
        "ce": "PureCE",
    }
    out: List[str] = []
    for raw in str(text).split(","):
        key = raw.strip()
        if not key:
            continue
        canonical = aliases.get(key.lower(), key)
        if canonical not in LOSS_CONFIGS:
            raise ValueError(f"Unknown loss mode {key!r}; expected one of {sorted(LOSS_CONFIGS)}")
        out.append(canonical)
    return out


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


def finite_values(values: Iterable[Any]) -> List[float]:
    xs = []
    for value in values:
        try:
            v = float(value)
            if np.isfinite(v):
                xs.append(v)
        except Exception:
            pass
    return xs


def finite_mean(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(np.mean(xs)) if xs else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    xs = finite_values(values)
    return float(np.std(xs, ddof=1)) if len(xs) > 1 else 0.0 if len(xs) == 1 else float("nan")


def normalize_output_subdir(value: Any) -> str:
    text = str(value or "").strip().strip("/\\")
    if not text:
        return ""
    path = Path(text)
    if path.is_absolute() or any(part == ".." for part in path.parts):
        raise ValueError(f"output_subdir must be a relative child directory, got {value!r}")
    return str(path)


def backbone_root_for(output_subdir: Any = "") -> Path:
    subdir = normalize_output_subdir(output_subdir)
    return BACKBONE_ROOT / subdir if subdir else BACKBONE_ROOT


def summary_root_for(output_subdir: Any = "") -> Path:
    subdir = normalize_output_subdir(output_subdir)
    return SUMMARY_ROOT / subdir if subdir else SUMMARY_ROOT


def model_dir_for(loss_mode: str, shot: int, seed: int, output_subdir: Any = "") -> Path:
    return backbone_root_for(output_subdir) / loss_mode / f"shot{int(shot)}" / f"Seed{int(seed)}"


def class_metric_names() -> List[str]:
    names: List[str] = []
    for prefix in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
        for name in FOREGROUND_CLASS_NAMES.values():
            names.append(f"{prefix}_{name}")
    return names


def summarize_eval_groups(eval_rows: Sequence[Dict[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in eval_rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
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


def eval_rows_to_class_rows(eval_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    out: List[Dict[str, Any]] = []
    for row in eval_rows:
        for cls, class_name in FOREGROUND_CLASS_NAMES.items():
            out.append(
                {
                    "loss_mode": row.get("loss_mode", ""),
                    "shot": int(row.get("shot", 0)),
                    "seed": int(row.get("seed", 0)),
                    "eval_set": row.get("eval_set", ""),
                    "split_role": row.get("split_role", ""),
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
    return out


def summarize_class_seed_average(class_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    group_fields = ("loss_mode", "shot", "eval_set", "domain", "class_id", "class_name")
    groups: Dict[tuple[Any, ...], List[Dict[str, Any]]] = {}
    for row in class_rows:
        key = tuple(row.get(field, "") for field in group_fields)
        groups.setdefault(key, []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        seeds = sorted({int(row.get("seed", 0)) for row in rows})
        item.update(
            {
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


def write_eval_summary_tables(root: Path, eval_rows: Sequence[Dict[str, Any]]) -> None:
    target_rows = [row for row in eval_rows if row.get("eval_set") == "target"]
    source_rows = [row for row in eval_rows if row.get("eval_set") == "source_heldout"]
    main_rows = target_rows if target_rows else [row for row in eval_rows if row.get("eval_set") != "source_heldout"]
    class_rows = eval_rows_to_class_rows(eval_rows)
    target_class_rows = [row for row in class_rows if row.get("eval_set") == "target"]
    main_class_rows = target_class_rows if target_class_rows else [row for row in class_rows if row.get("eval_set") != "source_heldout"]
    source_class_rows = [row for row in class_rows if row.get("eval_set") == "source_heldout"]

    domain_group_fields = ("loss_mode", "shot", "domain") if target_rows else ("loss_mode", "domain")
    write_csv(root / "method_summary.csv", summarize_eval_groups(main_rows, ("loss_mode",)))
    write_csv(root / "shot_summary.csv", summarize_eval_groups(main_rows, ("loss_mode", "shot")))
    write_csv(root / "domain_summary.csv", summarize_eval_groups(main_rows, domain_group_fields))
    write_csv(root / "source_heldout_summary.csv", summarize_eval_groups(source_rows, ("loss_mode", "shot", "domain")))
    write_csv(root / "eval_domain_class_metrics.csv", class_rows)
    write_csv(root / "seed_avg_target_domain_class_metrics.csv", summarize_class_seed_average(main_class_rows))
    write_csv(root / "seed_avg_source_heldout_class_metrics.csv", summarize_class_seed_average(source_class_rows))


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    loss_mode: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    train_slices: int,
    slice_indices_by_case: Dict[str, List[int]],
) -> Dict[str, Any]:
    experiment_name = "PostGraduate_train_backbone_BatchSize4"
    if bool(args.use_split):
        experiment_name = "PostGraduate_train_backbone_Center9SeedAvg"
    return {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "experiment": experiment_name,
            "source_domain": SOURCE_DOMAIN,
            "loss_mode": loss_mode,
            "shot": int(shot),
            "seed": int(seed),
            "train_case_ids": list(train_case_ids),
            "n_train_slices": int(train_slices),
            "use_split": bool(args.use_split),
            "split_csv": str(args.split_csv),
            "slice_policy": str(args.slice_policy),
            "num_middle_slices": int(args.num_middle_slices),
            "filter_min_fg": bool(args.filter_min_fg),
            "slice_indices_by_case": slice_indices_by_case,
            "model_type": "d_l2_disp_bn",
            "batch_norm_momentum": 0.1,
            "model_config": {
                "in_ch": 1,
                "num_classes": 3,
                "base_ch": int(args.base_ch),
                "latent_ch": int(args.latent_ch),
                "use_l2_norm": True,
                "use_batch_norm": True,
            },
            "objective": {
                "segmentation": f"CE + {float(args.dice_weight)} Dice",
                "lambda_disp": float(LOSS_CONFIGS[loss_mode]["lambda_disp"]),
                "dispersion": "L2_disperssion_loss" if loss_mode == "Disperssion" else "disabled",
                "disp_margin": float(args.disp_margin),
            },
            "args": vars(args),
        },
    }


def save_model_files(root: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(root)
    final_path = root / "checkpoint_final.pt"
    baseline_path = root / "baseline_model_with_metadata.pt"
    torch.save(payload, final_path)
    shutil.copyfile(final_path, baseline_path)


def load_model_if_available(model: torch.nn.Module, root: Path, device: torch.device) -> bool:
    checkpoint = root / "baseline_model_with_metadata.pt"
    if not checkpoint.exists():
        return False
    payload = torch.load(checkpoint, map_location=device)
    model.load_state_dict(payload["model_state_dict"], strict=True)
    return True


def build_current_model(args: argparse.Namespace, device: torch.device) -> torch.nn.Module:
    model = build_model(
        ModelConfig(
            in_ch=1,
            num_classes=3,
            base_ch=int(args.base_ch),
            latent_ch=int(args.latent_ch),
            model_type="d_l2_disp_bn",
            use_l2_norm=True,
            use_batch_norm=True,
        )
    )
    return model.to(device)


def train_one(
    *,
    args: argparse.Namespace,
    device: torch.device,
    loss_mode: str,
    shot: int,
    seed: int,
) -> tuple[torch.nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(int(seed))
    out_dir = model_dir_for(loss_mode, shot, seed, args.output_subdir)
    train_dataset = build_train_dataset(
        SOURCE_DOMAIN,
        shot=int(shot),
        seed=int(seed),
        data_root=Path(args.data_root),
        min_fg_ratio=float(args.min_fg_ratio),
        resize_hw=int(args.resize_hw),
        split_csv=Path(args.split_csv),
        use_split=bool(args.use_split),
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )
    slice_indices_by_case = train_dataset.slice_indices_by_case()
    dataset_meta = {
        "source_domain": SOURCE_DOMAIN,
        "shot": int(shot),
        "seed": int(seed),
        "loss_mode": loss_mode,
        "use_split": bool(args.use_split),
        "split_csv": str(args.split_csv),
        "split_role": "source_train" if bool(args.use_split) else "random_seed_sample",
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
        "source_domain": SOURCE_DOMAIN,
        "loss_mode": loss_mode,
        "shot": int(shot),
        "seed": int(seed),
        "lambda_disp_effective": float(LOSS_CONFIGS[loss_mode]["lambda_disp"]),
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "run_config.json", run_config)
    write_json(out_dir / "dataset_metadata.json", dataset_meta)

    model = build_current_model(args, device)
    loaded = bool(args.resume) and load_model_if_available(model, out_dir, device)
    local_log_path = out_dir / "training_log.csv"
    if loaded:
        print(f"[RESUME] Loaded existing model: {out_dir / 'baseline_model_with_metadata.pt'}")
        return model, read_csv(local_log_path), dataset_meta

    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=True, device=device)
    lambda_disp = float(LOSS_CONFIGS[loss_mode]["lambda_disp"])
    log_rows: List[Dict[str, Any]] = []

    print(f"[TRAIN] {loss_mode} shot={shot} seed={seed} cases={train_dataset.selected_case_ids} slices={len(train_dataset)}")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss: List[float] = []
        epoch_seg: List[float] = []
        epoch_disp: List[float] = []
        steps = 0
        for step, (img, mask, _meta) in enumerate(loader, start=1):
            if int(args.max_train_steps) > 0 and step > int(args.max_train_steps):
                break
            img = img.to(device)
            mask = mask.to(device)
            out = model(img, return_features=True)
            losses = backbone_training_loss(
                out["logits"],
                mask,
                out["features"]["dec1"],
                num_classes=3,
                dice_weight=float(args.dice_weight),
                lambda_disp=lambda_disp,
                disp_margin=float(args.disp_margin),
            )
            optimizer.zero_grad(set_to_none=True)
            losses["loss"].backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            steps += 1
            epoch_loss.append(float(losses["loss"].detach().cpu()))
            epoch_seg.append(float(losses["seg_loss"].detach().cpu()))
            epoch_disp.append(float(losses["disp_loss"].detach().cpu()))

        row = {
            "loss_mode": loss_mode,
            "shot": int(shot),
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": finite_mean(epoch_loss),
            "train_seg_loss": finite_mean(epoch_seg),
            "train_disp_loss": finite_mean(epoch_disp),
            "lambda_disp": lambda_disp,
            "steps": int(steps),
            "train_case_ids": "|".join(train_dataset.selected_case_ids),
            "n_train_slices": int(len(train_dataset)),
            "slice_policy": str(args.slice_policy),
            "num_middle_slices": int(args.num_middle_slices),
            "filter_min_fg": bool(args.filter_min_fg),
            "use_split": bool(args.use_split),
            "output_dir": str(out_dir),
        }
        log_rows.append(row)
        write_csv(local_log_path, log_rows)
        print(
            f"  epoch {epoch:03d}/{int(args.epochs):03d} "
            f"loss={float(row['train_loss']):.6f} "
            f"seg={float(row['train_seg_loss']):.6f} "
            f"disp={float(row['train_disp_loss']):.6f}"
        )

    payload = checkpoint_payload(
        model,
        args=args,
        loss_mode=loss_mode,
        shot=shot,
        seed=seed,
        train_case_ids=train_dataset.selected_case_ids,
        train_slices=len(train_dataset),
        slice_indices_by_case=slice_indices_by_case,
    )
    save_model_files(out_dir, payload)
    return model, log_rows, dataset_meta


def _eval_domain(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    loss_mode: str,
    shot: int,
    seed: int,
    domain: str,
    split_role: str | None,
    eval_set: str,
    train_case_ids: Sequence[str],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    out_dir = model_dir_for(loss_mode, shot, seed, args.output_subdir)
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
        shot=int(shot),
        seed=int(seed),
        split_csv=Path(args.split_csv),
        split_role=split_role,
        use_split=bool(args.use_split),
        eval_set=eval_set,
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )
    row = {
        "loss_mode": loss_mode,
        "shot": int(shot),
        "seed": int(seed),
        "eval_set": eval_set,
        "split_role": str(split_role or ""),
        "domain": metrics["domain"],
        "n_cases": int(metrics["n_cases"]),
        "n_slices": int(metrics["n_slices"]),
        "slice_policy": metrics["slice_policy"],
        "num_middle_slices": int(metrics["num_middle_slices"]),
        "excluded_case_ids": "|".join(metrics["excluded_case_ids"]),
        "output_dir": str(out_dir),
    }
    row.update(metrics["summary"])

    case_rows = []
    for case_row in metrics["case_rows"]:
        case_rows.append(
            {
                "loss_mode": loss_mode,
                "shot": int(shot),
                "seed": int(seed),
                "eval_set": eval_set,
                "split_role": str(split_role or ""),
                "domain": metrics["domain"],
                **case_row,
            }
        )
    print(
        f"[EVAL] {loss_mode} shot={shot} seed={seed} eval_set={eval_set} domain={metrics['domain']} "
        f"case_dice={float(row['case_dice']):.6f} case_hd95={float(row['case_hd95']):.6f}"
    )
    return row, case_rows


def evaluate_one(
    *,
    args: argparse.Namespace,
    model: torch.nn.Module,
    device: torch.device,
    loss_mode: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    out_dir = model_dir_for(loss_mode, shot, seed, args.output_subdir)

    if bool(args.use_split):
        eval_domains = list(TARGET_DOMAIN_NAMES)
    else:
        eval_domains = list(DOMAIN_NAMES)

    for domain in eval_domains:
        row, rows = _eval_domain(
            args=args,
            model=model,
            device=device,
            loss_mode=loss_mode,
            shot=shot,
            seed=seed,
            domain=domain,
            split_role="target_test" if bool(args.use_split) else None,
            eval_set="target" if bool(args.use_split) else "all_domains",
            train_case_ids=train_case_ids,
        )
        eval_rows.append(row)
        case_rows.extend(rows)

    if bool(args.use_split) and not bool(args.skip_source_heldout):
        row, rows = _eval_domain(
            args=args,
            model=model,
            device=device,
            loss_mode=loss_mode,
            shot=shot,
            seed=seed,
            domain=SOURCE_DOMAIN,
            split_role="source_domain_unused",
            eval_set="source_heldout",
            train_case_ids=train_case_ids,
        )
        eval_rows.append(row)
        case_rows.extend(rows)

    write_csv(out_dir / "eval_metrics.csv", eval_rows)
    write_csv(out_dir / "eval_case_metrics.csv", case_rows)
    write_csv(out_dir / "eval_domain_class_metrics.csv", eval_rows_to_class_rows(eval_rows))
    return eval_rows, case_rows


def summarize_experiment(
    loss_mode: str,
    shot: int,
    seed: int,
    dataset_meta: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    output_subdir: Any = "",
) -> Dict[str, Any]:
    last_train = train_rows[-1] if train_rows else {}
    target_rows = [row for row in eval_rows if row.get("eval_set") == "target"]
    source_rows = [row for row in eval_rows if row.get("eval_set") == "source_heldout"]
    main_rows = target_rows if target_rows else [row for row in eval_rows if row.get("eval_set") != "source_heldout"]
    return {
        "loss_mode": loss_mode,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": "|".join(dataset_meta.get("train_case_ids", [])),
        "n_train_slices": int(dataset_meta.get("n_train_slices", 0)),
        "use_split": bool(dataset_meta.get("use_split", False)),
        "slice_policy": dataset_meta.get("slice_policy", ""),
        "num_middle_slices": int(dataset_meta.get("num_middle_slices", 0)),
        "filter_min_fg": bool(dataset_meta.get("filter_min_fg", False)),
        "final_train_loss": last_train.get("train_loss", float("nan")),
        "final_train_seg_loss": last_train.get("train_seg_loss", float("nan")),
        "final_train_disp_loss": last_train.get("train_disp_loss", float("nan")),
        "mean_6domain_case_dice": finite_mean(row.get("case_dice") for row in main_rows),
        "mean_6domain_case_hd95": finite_mean(row.get("case_hd95") for row in main_rows),
        "mean_6domain_slice_dice": finite_mean(row.get("slice_dice") for row in main_rows),
        "mean_6domain_slice_hd95": finite_mean(row.get("slice_hd95") for row in main_rows),
        "mean_target_case_dice": finite_mean(row.get("case_dice") for row in target_rows),
        "mean_target_case_hd95": finite_mean(row.get("case_hd95") for row in target_rows),
        "mean_target_slice_dice": finite_mean(row.get("slice_dice") for row in target_rows),
        "mean_target_slice_hd95": finite_mean(row.get("slice_hd95") for row in target_rows),
        "source_heldout_case_dice": finite_mean(row.get("case_dice") for row in source_rows),
        "source_heldout_case_hd95": finite_mean(row.get("case_hd95") for row in source_rows),
        "source_heldout_slice_dice": finite_mean(row.get("slice_dice") for row in source_rows),
        "source_heldout_slice_hd95": finite_mean(row.get("slice_hd95") for row in source_rows),
        "output_dir": str(model_dir_for(loss_mode, shot, seed, output_subdir)),
    }


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    backbone_root = ensure_dir(backbone_root_for(args.output_subdir))
    summary_root = ensure_dir(summary_root_for(args.output_subdir))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)
    loss_modes = parse_loss_modes(args.loss_modes)

    all_train_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []

    print(f"[DEVICE] {device}")
    print(f"[MATRIX] loss_modes={loss_modes} shots={shots} seeds={seeds}")
    print(f"[SPLIT] use_split={args.use_split} split_csv={args.split_csv}")
    print(f"[SLICE] policy={args.slice_policy} num_middle_slices={args.num_middle_slices} filter_min_fg={args.filter_min_fg}")
    print(f"[BACKBONE_ROOT] {backbone_root}")
    print(f"[SUMMARY_ROOT] {summary_root}")
    for loss_mode in loss_modes:
        for shot in shots:
            for seed in seeds:
                model, train_rows, dataset_meta = train_one(
                    args=args,
                    device=device,
                    loss_mode=loss_mode,
                    shot=int(shot),
                    seed=int(seed),
                )
                eval_rows, case_rows = evaluate_one(
                    args=args,
                    model=model,
                    device=device,
                    loss_mode=loss_mode,
                    shot=int(shot),
                    seed=int(seed),
                    train_case_ids=dataset_meta["train_case_ids"],
                )
                all_train_rows.extend(train_rows)
                all_eval_rows.extend(eval_rows)
                all_case_rows.extend(case_rows)
                summary_rows.append(
                    summarize_experiment(
                        loss_mode,
                        int(shot),
                        int(seed),
                        dataset_meta,
                        train_rows,
                        eval_rows,
                        output_subdir=args.output_subdir,
                    )
                )

                write_csv(summary_root / "training_curves.csv", all_train_rows)
                write_csv(summary_root / "eval_metrics.csv", all_eval_rows)
                write_csv(summary_root / "eval_case_metrics.csv", all_case_rows)
                write_csv(summary_root / "experiment_summary.csv", summary_rows)
                write_eval_summary_tables(summary_root, all_eval_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train PostGraduate SPIDER backbone matrix.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--split_csv", default=str(DEFAULT_SPLIT_CSV))
    parser.add_argument("--use_split", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shots", default="3,4,5")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--loss_modes", default="Disperssion,PureCE")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dice_weight", type=float, default=0.5)
    parser.add_argument("--disp_margin", type=float, default=0.0)
    parser.add_argument("--grad_clip", type=float, default=1.0)
    parser.add_argument("--resize_hw", type=int, default=224)
    parser.add_argument("--min_fg_ratio", type=float, default=0.05)
    parser.add_argument("--filter_min_fg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slice_policy", default="all_filtered", choices=("center9", "all", "all_filtered"))
    parser.add_argument("--num_middle_slices", type=int, default=9)
    parser.add_argument("--base_ch", type=int, default=16)
    parser.add_argument("--latent_ch", type=int, default=64)
    parser.add_argument("--max_train_steps", type=int, default=0)
    parser.add_argument("--max_test_cases", type=int, default=0)
    parser.add_argument("--output_subdir", default=DEFAULT_OUTPUT_SUBDIR)
    parser.add_argument("--skip_source_heldout", action="store_true")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
