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
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "DisperssionWithSADG"
DEFAULT_NEW_BACKBONE_ROOT = PROJECT_ROOT / "backbones" / "DisperssionWithSADG"
SOURCE_DOMAIN = SOURCE_DOMAIN_NAME

SEG_SADG_ABLATION = "Seg-SADG"
FIXED_RHO = 0.05
ABLATIONS = (SEG_SADG_ABLATION,)
ABLATION_ALIASES = {
    "seg": SEG_SADG_ABLATION,
    "seg-sadg": SEG_SADG_ABLATION,
    "seg_sadg": SEG_SADG_ABLATION,
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


def normalize_ablation(name: str) -> str:
    key = str(name).strip().lower()
    if key in ABLATION_ALIASES:
        return ABLATION_ALIASES[key]
    for ablation in ABLATIONS:
        if key == ablation.lower():
            return ablation
    raise ValueError(f"Unknown ablation {name!r}; expected one of {', '.join(ABLATIONS)}")


def parse_ablation_list(text: str) -> List[str]:
    ablations = [normalize_ablation(name) for name in parse_str_list(text)]
    if not ablations:
        raise ValueError("At least one ablation must be provided.")
    seen: set[str] = set()
    out: List[str] = []
    for ablation in ablations:
        if ablation not in seen:
            seen.add(ablation)
            out.append(ablation)
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


def ablation_run_dir(args: argparse.Namespace, ablation: str, shot: int, seed: int) -> Path:
    return (
        resolve_path(args.new_backbone_root)
        / normalize_ablation(ablation)
        / rho_tag(float(args.rho))
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


def epsilon_source_name(ablation: str) -> str:
    ablation = normalize_ablation(ablation)
    if ablation == SEG_SADG_ABLATION:
        return "CE + Dice"
    raise ValueError(f"Unsupported ablation: {ablation}")


def checkpoint_payload(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    ablation: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    n_train_slices: int,
    slice_indices_by_case: Dict[str, List[int]],
) -> Dict[str, Any]:
    cfg = model_config(args)
    ablation = normalize_ablation(ablation)
    return {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "experiment": "DisperssionWithSADG",
            "loss_mode": ablation,
            "ablation": ablation,
            "sadg_method": ablation,
            "rho": float(args.rho),
            "rho_tag": rho_tag(float(args.rho)),
            "sadg_eps": float(args.sadg_eps),
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
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
                "epsilon_source": epsilon_source_name(ablation),
                "update_source": "CE + Dice + Dispersion at theta + epsilon",
            },
            "args": vars(args),
        },
    }


def normalize_training_rows(
    rows: Sequence[Dict[str, Any]],
    *,
    ablation: str,
    shot: int,
    seed: int,
    output_dir: Path,
    checkpoint_source: str,
    args: argparse.Namespace,
) -> List[Dict[str, Any]]:
    ablation = normalize_ablation(ablation)
    out: List[Dict[str, Any]] = []
    for raw in rows:
        row = dict(raw)
        row["loss_mode"] = ablation
        row["ablation"] = ablation
        row["sadg_method"] = ablation
        row["lambda_disp"] = float(args.lambda_disp)
        row["dice_weight"] = float(args.dice_weight)
        row["rho"] = float(args.rho)
        row["rho_tag"] = rho_tag(float(args.rho))
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


def select_epsilon_loss(losses: Dict[str, torch.Tensor], ablation: str) -> torch.Tensor:
    ablation = normalize_ablation(ablation)
    if ablation == SEG_SADG_ABLATION:
        return losses["seg_loss"]
    raise ValueError(f"{ablation} does not use SADG epsilon loss.")


def train_step(
    *,
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    ablation: str,
    params: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> Dict[str, float]:
    ablation = normalize_ablation(ablation)

    optimizer.zero_grad(set_to_none=True)
    epsilon_losses = compute_training_losses(model, img, mask, args)
    epsilon_loss = select_epsilon_loss(epsilon_losses, ablation)
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
    optimizer.zero_grad(set_to_none=True)
    try:
        update_losses = compute_training_losses(model, img, mask, args)
        update_losses["loss"].backward()
    finally:
        restore_sadg_perturbation(perturbations)

    optimizer.step()
    if update_losses is None:
        raise RuntimeError("SADG update losses were not computed.")
    return {
        "train_loss": float(update_losses["loss"].detach().cpu()),
        "train_seg_loss": float(update_losses["seg_loss"].detach().cpu()),
        "train_disp_loss": float(update_losses["disp_loss"].detach().cpu()),
        "epsilon_loss": float(epsilon_loss.detach().cpu()),
        "sadg_grad_norm": float(grad_norm.detach().cpu()),
        "perturb_norm": float(perturb_norm),
    }


def train_or_load_ablation(
    *,
    args: argparse.Namespace,
    device: torch.device,
    ablation: str,
    shot: int,
    seed: int,
) -> tuple[torch.nn.Module, List[Dict[str, Any]], Dict[str, Any], Path]:
    set_seed(int(seed))
    ablation = normalize_ablation(ablation)
    run_dir = ablation_run_dir(args, ablation, shot, seed)
    train_dataset = build_train_data(args, shot, seed)
    slice_indices_by_case = train_dataset.slice_indices_by_case()
    dataset_meta = {
        "loss_mode": ablation,
        "ablation": ablation,
        "sadg_method": ablation,
        "lambda_disp": float(args.lambda_disp),
        "dice_weight": float(args.dice_weight),
        "rho": float(args.rho),
        "rho_tag": rho_tag(float(args.rho)),
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
            ablation=ablation,
            shot=int(shot),
            seed=int(seed),
            output_dir=run_dir,
            checkpoint_source="new_training",
            args=args,
        )
        print(f"[RESUME] {ablation} shot={shot} seed={seed}: {checkpoint}")
        return model, rows, dataset_meta, run_dir

    model = build_model(model_config(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    params = [p for p in model.parameters() if p.requires_grad]
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=True, device=device)
    log_rows: List[Dict[str, Any]] = []

    print(
        f"[TRAIN] {ablation} shot={shot} seed={seed} rho={float(args.rho):.6g} "
        f"lambda_disp={float(args.lambda_disp):.6g} cases={train_dataset.selected_case_ids} "
        f"slices={len(train_dataset)}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss: List[float] = []
        epoch_seg: List[float] = []
        epoch_disp: List[float] = []
        epoch_epsilon: List[float] = []
        epoch_grad_norm: List[float] = []
        epoch_perturb_norm: List[float] = []
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
                ablation=ablation,
                params=params,
                device=device,
            )

            steps += 1
            epoch_loss.append(float(step_row["train_loss"]))
            epoch_seg.append(float(step_row["train_seg_loss"]))
            epoch_disp.append(float(step_row["train_disp_loss"]))
            epoch_epsilon.append(float(step_row["epsilon_loss"]))
            epoch_grad_norm.append(float(step_row["sadg_grad_norm"]))
            epoch_perturb_norm.append(float(step_row["perturb_norm"]))

        row = {
            "loss_mode": ablation,
            "ablation": ablation,
            "sadg_method": ablation,
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
            "rho": float(args.rho),
            "rho_tag": rho_tag(float(args.rho)),
            "checkpoint_source": "new_training",
            "shot": int(shot),
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": finite_mean(epoch_loss),
            "train_seg_loss": finite_mean(epoch_seg),
            "train_disp_loss": finite_mean(epoch_disp),
            "epsilon_loss": finite_mean(epoch_epsilon),
            "sadg_grad_norm": finite_mean(epoch_grad_norm),
            "perturb_norm": finite_mean(epoch_perturb_norm),
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
            f"seg={float(row['train_seg_loss']):.6f} "
            f"disp={float(row['train_disp_loss']):.6f} "
            f"grad_norm={float(row['sadg_grad_norm']):.6f}"
        )

    payload = checkpoint_payload(
        model,
        args=args,
        ablation=ablation,
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
    group_fields = ("ablation", "rho", "rho_tag", "shot", "domain", "class_id", "class_name")
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
    ablation: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    output_dir: Path,
    checkpoint_source: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    ablation = normalize_ablation(ablation)
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
            "loss_mode": ablation,
            "ablation": ablation,
            "sadg_method": ablation,
            "lambda_disp": float(args.lambda_disp),
            "dice_weight": float(args.dice_weight),
            "rho": float(args.rho),
            "rho_tag": rho_tag(float(args.rho)),
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
                    "loss_mode": ablation,
                    "ablation": ablation,
                    "sadg_method": ablation,
                    "lambda_disp": float(args.lambda_disp),
                    "dice_weight": float(args.dice_weight),
                    "rho": float(args.rho),
                    "rho_tag": rho_tag(float(args.rho)),
                    "checkpoint_source": checkpoint_source,
                    "shot": int(shot),
                    "seed": int(seed),
                    "domain": metrics["domain"],
                    **case_row,
                }
            )
        print(
            f"[EVAL] {ablation} shot={shot} seed={seed} domain={metrics['domain']} "
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
    ablation: str,
    shot: int,
    seed: int,
    dataset_meta: Dict[str, Any],
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    ablation = normalize_ablation(ablation)
    last_train = train_rows[-1] if train_rows else {}
    target_rows = [row for row in eval_rows if row.get("domain", "") != SOURCE_DOMAIN]
    return {
        "loss_mode": ablation,
        "ablation": ablation,
        "sadg_method": ablation,
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
        "final_train_seg_loss": last_train.get("train_seg_loss", float("nan")),
        "final_train_disp_loss": last_train.get("train_disp_loss", float("nan")),
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
    write_csv(result_root / "training_curves.csv", training_rows)
    write_csv(result_root / "eval_metrics.csv", eval_rows)
    write_csv(result_root / "eval_case_metrics.csv", case_rows)
    write_csv(result_root / "eval_domain_class_metrics.csv", class_rows)
    write_csv(result_root / "seed_avg_domain_class_metrics.csv", summarize_class_seed_average(class_rows))
    write_csv(result_root / "ablation_summary.csv", summarize_eval_groups(eval_rows, ("ablation", "rho", "rho_tag")))
    write_csv(result_root / "shot_summary.csv", summarize_eval_groups(eval_rows, ("ablation", "rho", "rho_tag", "shot")))
    write_csv(result_root / "domain_summary.csv", summarize_eval_groups(eval_rows, ("ablation", "rho", "rho_tag", "domain")))
    write_csv(result_root / "experiment_summary.csv", experiment_rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    result_root = ensure_dir(resolve_path(args.result_root))
    ensure_dir(resolve_path(args.new_backbone_root))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)
    rhos = [float(FIXED_RHO)]
    ablation = SEG_SADG_ABLATION

    all_training_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []
    all_experiment_rows: List[Dict[str, Any]] = []

    print(f"[DEVICE] {device}")
    print(f"[RESULT_ROOT] {result_root}")
    print(f"[NEW_BACKBONE_ROOT] {resolve_path(args.new_backbone_root)}")
    print(f"[MATRIX] ablation={ablation} rhos={rhos} shots={shots} seeds={seeds} BS={args.batch_size}")
    print(
        f"[OBJECTIVE] CE + {float(args.dice_weight)} Dice + {float(args.lambda_disp)} Dispersion "
        f"rhos={rhos}"
    )
    print(f"[SLICE] policy={args.slice_policy} filter_min_fg={args.filter_min_fg} min_fg_ratio={args.min_fg_ratio}")

    for rho in rhos:
        run_args = argparse.Namespace(**{**vars(args), "rho": float(rho), "rho_tag": rho_tag(float(rho))})
        for shot in shots:
            for seed in seeds:
                model, train_rows, dataset_meta, output_dir = train_or_load_ablation(
                    args=run_args,
                    device=device,
                    ablation=ablation,
                    shot=int(shot),
                    seed=int(seed),
                )

                eval_rows, case_rows = evaluate_model(
                    args=run_args,
                    model=model,
                    device=device,
                    ablation=ablation,
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
                        ablation=ablation,
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
    parser = argparse.ArgumentParser(description="Run Seg-SADG rho=0.05 for DisperssionWithSADG source training.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
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
