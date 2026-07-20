from __future__ import annotations

import argparse
import csv
import json
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

from helper.FDA import (
    TARGET_STYLE_DOMAINS,
    FDAStyleBank,
    build_target_style_bank,
    fourier_domain_adapt,
    normalize_fda_style_mode,
)
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


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "FDA"
METHOD = "FDA"
DEFAULT_FDA_VARIANTS = ("FDA-GlobalMean", "FDA-DomainStylePool")
FIXED_SADG_RHO = 0.05


def set_seed(seed: int) -> None:
    random.seed(int(seed))
    np.random.seed(int(seed))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))


def resolve_device(name: str) -> torch.device:
    if str(name).lower() == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return torch.device(name)


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    if p.parts and str(p.parts[0]).lower() == "sota":
        return Path.cwd() / p
    return base / p


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def value_tag(name: str, value: float) -> str:
    text = f"{float(value):g}".replace("-", "m").replace(".", "p")
    return f"{name}{text}"


def finite_values(values: Iterable[Any]) -> List[float]:
    out: List[float] = []
    for value in values:
        try:
            number = float(value)
        except (TypeError, ValueError):
            continue
        if np.isfinite(number):
            out.append(number)
    return out


def finite_mean(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.mean(vals)) if vals else float("nan")


def finite_std(values: Iterable[Any]) -> float:
    vals = finite_values(values)
    return float(np.std(vals, ddof=1)) if len(vals) > 1 else 0.0


def write_json(path: Path, payload: Mapping[str, Any]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=True, indent=2, default=str)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    ensure_dir(path.parent)
    fieldnames: List[str] = []
    seen = set()
    for row in rows:
        for key in row.keys():
            if key not in seen:
                fieldnames.append(str(key))
                seen.add(key)
    with path.open("w", newline="", encoding="utf-8-sig") as f:
        if not fieldnames:
            f.write("")
            return
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: row.get(key, "") for key in fieldnames})


def reset_or_prepare_result_root(result_root: Path, overwrite: bool) -> Path:
    result_root = result_root.resolve()
    project_root = PROJECT_ROOT.resolve()
    if result_root.exists() and overwrite:
        if result_root == project_root or project_root not in result_root.parents:
            raise ValueError(f"Refusing to overwrite result_root outside {project_root}: {result_root}")
        shutil.rmtree(result_root)
    elif result_root.exists() and not overwrite:
        expected = (
            "training_curves.csv",
            "eval_metrics.csv",
            "eval_case_metrics.csv",
            "experiment_summary.csv",
            "domain_5seed_summary.csv",
            "overall_15run_domain_summary.csv",
        )
        if any((result_root / name).exists() for name in expected):
            raise FileExistsError(f"Result root already contains FDA outputs. Use --overwrite: {result_root}")
    return ensure_dir(result_root)


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


def build_train_data(args: argparse.Namespace, shot: int, seed: int):
    return build_train_dataset(
        SOURCE_DOMAIN_NAME,
        int(shot),
        int(seed),
        data_root=Path(args.data_root),
        min_fg_ratio=float(args.min_fg_ratio),
        resize_hw=int(args.resize_hw),
        use_split=True,
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )


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


def decorate_batch_with_fda(
    img: torch.Tensor,
    *,
    style_bank: FDAStyleBank,
    fda_variant: str,
    args: argparse.Namespace,
) -> torch.Tensor:
    style_lowfreq = style_bank.styles_for_batch(
        fda_variant,
        batch_size=int(img.shape[0]),
        device=img.device,
    )
    return fourier_domain_adapt(
        img,
        style_lowfreq,
        beta=float(args.fda_beta),
        eps=float(args.fda_eps),
        fft_norm=str(args.fda_fft_norm),
        clamp=True,
    )


def train_step(
    *,
    model: nn.Module,
    optimizer: torch.optim.Optimizer,
    img: torch.Tensor,
    mask: torch.Tensor,
    args: argparse.Namespace,
    params: Sequence[torch.nn.Parameter],
    device: torch.device,
) -> Dict[str, float]:
    optimizer.zero_grad(set_to_none=True)
    epsilon_losses = compute_training_losses(model, img, mask, args)
    epsilon_loss = epsilon_losses["seg_loss"]
    sadg_state = apply_sadg_perturbation_from_loss(
        params,
        epsilon_loss,
        rho=float(args.sadg_rho),
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


def run_dir_for(result_root: Path, fda_variant: str, shot: int, seed: int) -> Path:
    return result_root / "runs" / sanitize_variant(fda_variant) / f"shot{int(shot)}" / f"Seed{int(seed)}"


def sanitize_variant(fda_variant: str) -> str:
    return str(fda_variant).strip().replace("/", "_").replace("\\", "_")


def fda_metadata(
    args: argparse.Namespace,
    style_bank: FDAStyleBank,
    fda_variant: str,
) -> Dict[str, Any]:
    style_mode = normalize_fda_style_mode(fda_variant)
    return {
        "method": METHOD,
        "sota_method": METHOD,
        "fda_variant": str(fda_variant),
        "fda_style_mode": style_mode,
        "fda_beta": float(args.fda_beta),
        "fda_beta_tag": value_tag("beta", float(args.fda_beta)),
        "fda_eps": float(args.fda_eps),
        "fda_fft_norm": str(args.fda_fft_norm),
        "fda_radius": int(style_bank.radius),
        "fda_window_size": int(style_bank.window_size),
        "fda_style_domains": "|".join(style_bank.domain_names),
        "fda_style_slices_total": int(style_bank.style_slices_total),
        "fda_style_slices_by_domain": json.dumps(style_bank.domain_slice_counts, ensure_ascii=True),
        "sadg_rho": float(args.sadg_rho),
        "sadg_rho_tag": value_tag("sadg", float(args.sadg_rho)),
        "dice_weight": float(args.dice_weight),
        "lambda_disp": float(args.lambda_disp),
        "disp_margin": float(args.disp_margin),
    }


def train_one_run(
    *,
    args: argparse.Namespace,
    device: torch.device,
    shot: int,
    seed: int,
    result_root: Path,
    style_bank: FDAStyleBank,
    fda_variant: str,
) -> tuple[nn.Module, List[Dict[str, Any]], Dict[str, Any], Path]:
    set_seed(int(seed))
    train_dataset = build_train_data(args, int(shot), int(seed))
    run_dir = ensure_dir(run_dir_for(result_root, fda_variant, int(shot), int(seed)))
    metadata = {
        **fda_metadata(args, style_bank, fda_variant),
        "source_domain": SOURCE_DOMAIN_NAME,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": [str(x) for x in train_dataset.selected_case_ids],
        "n_train_slices": int(len(train_dataset)),
        "slice_policy": train_dataset.slice_policy,
        "num_middle_slices": int(train_dataset.num_middle_slices),
        "filter_min_fg": bool(train_dataset.filter_min_fg),
        "min_fg_ratio": float(args.min_fg_ratio),
        "resize_hw": int(args.resize_hw),
        "slice_indices_by_case": train_dataset.slice_indices_by_case(),
        "checkpoint_source": "not_saved",
        "model_parameters_saved": False,
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "dataset_metadata.json", metadata)
    write_json(run_dir / "run_config.json", {**vars(args), **metadata})

    model = build_model(model_config(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    params = [p for p in model.parameters() if p.requires_grad]
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=True, device=device)
    log_rows: List[Dict[str, Any]] = []

    print(
        f"[TRAIN] {METHOD}/{fda_variant} shot={shot} seed={seed} "
        f"beta={float(args.fda_beta):.6g} radius={style_bank.radius} "
        f"sadg_rho={float(args.sadg_rho):.6g} "
        f"cases={train_dataset.selected_case_ids} slices={len(train_dataset)}",
        flush=True,
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
            with torch.no_grad():
                img = decorate_batch_with_fda(
                    img,
                    style_bank=style_bank,
                    fda_variant=fda_variant,
                    args=args,
                )
            step_row = train_step(
                model=model,
                optimizer=optimizer,
                img=img,
                mask=mask,
                args=args,
                params=params,
                device=device,
            )
            steps += 1
            epoch_loss.append(step_row["train_loss"])
            epoch_seg.append(step_row["train_seg_loss"])
            epoch_disp.append(step_row["train_disp_loss"])
            epoch_epsilon.append(step_row["epsilon_loss"])
            epoch_grad_norm.append(step_row["sadg_grad_norm"])
            epoch_perturb_norm.append(step_row["perturb_norm"])

        row = {
            **fda_metadata(args, style_bank, fda_variant),
            "loss_mode": "SADG_CE_0p5Dice_0p05Dispersion",
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
            "train_case_ids": "|".join(str(x) for x in train_dataset.selected_case_ids),
            "n_train_slices": int(len(train_dataset)),
            "slice_policy": str(args.slice_policy),
            "num_middle_slices": int(args.num_middle_slices),
            "filter_min_fg": bool(args.filter_min_fg),
            "checkpoint_source": "not_saved",
            "model_parameters_saved": False,
            "output_dir": str(run_dir),
        }
        log_rows.append(row)
        write_csv(run_dir / "training_log.csv", log_rows)
        print(
            f"  epoch {epoch:03d}/{int(args.epochs):03d} "
            f"loss={float(row['train_loss']):.6f} "
            f"seg={float(row['train_seg_loss']):.6f} "
            f"disp={float(row['train_disp_loss']):.6f} "
            f"grad_norm={float(row['sadg_grad_norm']):.6f}",
            flush=True,
        )

    return model, log_rows, metadata, run_dir


def evaluate_model(
    *,
    args: argparse.Namespace,
    model: nn.Module,
    device: torch.device,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    output_dir: Path,
    style_bank: FDAStyleBank,
    fda_variant: str,
) -> tuple[List[Dict[str, Any]], List[Dict[str, Any]]]:
    eval_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    base_meta = {
        **fda_metadata(args, style_bank, fda_variant),
        "loss_mode": "SADG_CE_0p5Dice_0p05Dispersion",
        "shot": int(shot),
        "seed": int(seed),
        "checkpoint_source": "not_saved",
        "model_parameters_saved": False,
        "output_dir": str(output_dir),
    }
    model.eval()
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
            **base_meta,
            "domain": metrics["domain"],
            "n_cases": int(metrics["n_cases"]),
            "n_slices": int(metrics["n_slices"]),
            "slice_policy": metrics["slice_policy"],
            "num_middle_slices": int(metrics["num_middle_slices"]),
            "excluded_case_ids": "|".join(metrics["excluded_case_ids"]),
        }
        row.update(metrics["summary"])
        eval_rows.append(row)
        for case_row in metrics["case_rows"]:
            case_rows.append({**base_meta, "domain": metrics["domain"], **case_row})
        print(
            f"[EVAL] {METHOD}/{fda_variant} shot={shot} seed={seed} domain={metrics['domain']} "
            f"case_dice={float(row['case_dice']):.6f} case_hd95={float(row['case_hd95']):.6f} "
            f"slice_dice={float(row['slice_dice']):.6f} slice_hd95={float(row['slice_hd95']):.6f}",
            flush=True,
        )
    return eval_rows, case_rows


def class_metric_names() -> List[str]:
    names: List[str] = []
    for prefix in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
        for name in FOREGROUND_CLASS_NAMES.values():
            names.append(f"{prefix}_{name}")
    return names


def summarize_eval_groups(eval_rows: Sequence[Mapping[str, Any]], group_fields: Sequence[str]) -> List[Dict[str, Any]]:
    groups: Dict[tuple[Any, ...], List[Mapping[str, Any]]] = {}
    for row in eval_rows:
        groups.setdefault(tuple(row.get(field, "") for field in group_fields), []).append(row)

    out: List[Dict[str, Any]] = []
    for key in sorted(groups, key=lambda vals: tuple(str(v) for v in vals)):
        rows = groups[key]
        item = {field: value for field, value in zip(group_fields, key)}
        item.update(
            {
                "method": METHOD,
                "sota_method": METHOD,
                "fda_beta": finite_mean(row.get("fda_beta") for row in rows),
                "fda_radius": finite_mean(row.get("fda_radius") for row in rows),
                "fda_window_size": finite_mean(row.get("fda_window_size") for row in rows),
                "sadg_rho": finite_mean(row.get("sadg_rho") for row in rows),
                "lambda_disp": finite_mean(row.get("lambda_disp") for row in rows),
                "dice_weight": finite_mean(row.get("dice_weight") for row in rows),
                "n_rows": int(len(rows)),
                "n_cases_total": int(sum(int(float(row.get("n_cases", 0) or 0)) for row in rows)),
                "n_slices_total": int(sum(int(float(row.get("n_slices", 0) or 0)) for row in rows)),
                "case_dice_mean": finite_mean(row.get("case_dice") for row in rows),
                "case_dice_std": finite_std(row.get("case_dice") for row in rows),
                "case_hd95_mean": finite_mean(row.get("case_hd95") for row in rows),
                "case_hd95_std": finite_std(row.get("case_hd95") for row in rows),
                "slice_dice_mean": finite_mean(row.get("slice_dice") for row in rows),
                "slice_dice_std": finite_std(row.get("slice_dice") for row in rows),
                "slice_hd95_mean": finite_mean(row.get("slice_hd95") for row in rows),
                "slice_hd95_std": finite_std(row.get("slice_hd95") for row in rows),
            }
        )
        for metric in class_metric_names():
            item[f"{metric}_mean"] = finite_mean(row.get(metric) for row in rows)
            item[f"{metric}_std"] = finite_std(row.get(metric) for row in rows)
        out.append(item)
    return out


def experiment_summary_rows(
    eval_rows: Sequence[Mapping[str, Any]],
    training_rows: Sequence[Mapping[str, Any]],
) -> List[Dict[str, Any]]:
    train_by_run = {
        (str(row["fda_variant"]), int(row["shot"]), int(row["seed"])): row
        for row in training_rows
        if "fda_variant" in row and "shot" in row and "seed" in row
    }
    groups: Dict[tuple[str, int, int], List[Mapping[str, Any]]] = {}
    for row in eval_rows:
        groups.setdefault((str(row["fda_variant"]), int(row["shot"]), int(row["seed"])), []).append(row)

    out: List[Dict[str, Any]] = []
    for (variant, shot, seed), rows in sorted(groups.items()):
        last_train = train_by_run.get((variant, shot, seed), {})
        target_rows = [row for row in rows if row.get("domain", "") != SOURCE_DOMAIN_NAME]
        out.append(
            {
                "method": METHOD,
                "sota_method": METHOD,
                "fda_variant": variant,
                "shot": int(shot),
                "seed": int(seed),
                "fda_beta": finite_mean(row.get("fda_beta") for row in rows),
                "fda_radius": finite_mean(row.get("fda_radius") for row in rows),
                "fda_window_size": finite_mean(row.get("fda_window_size") for row in rows),
                "sadg_rho": finite_mean(row.get("sadg_rho") for row in rows),
                "lambda_disp": finite_mean(row.get("lambda_disp") for row in rows),
                "dice_weight": finite_mean(row.get("dice_weight") for row in rows),
                "final_train_loss": last_train.get("train_loss", float("nan")),
                "final_train_seg_loss": last_train.get("train_seg_loss", float("nan")),
                "final_train_disp_loss": last_train.get("train_disp_loss", float("nan")),
                "mean_6domain_case_dice": finite_mean(row.get("case_dice") for row in rows),
                "mean_6domain_case_hd95": finite_mean(row.get("case_hd95") for row in rows),
                "mean_6domain_slice_dice": finite_mean(row.get("slice_dice") for row in rows),
                "mean_6domain_slice_hd95": finite_mean(row.get("slice_hd95") for row in rows),
                "mean_5target_case_dice": finite_mean(row.get("case_dice") for row in target_rows),
                "mean_5target_case_hd95": finite_mean(row.get("case_hd95") for row in target_rows),
                "mean_5target_slice_dice": finite_mean(row.get("slice_dice") for row in target_rows),
                "mean_5target_slice_hd95": finite_mean(row.get("slice_hd95") for row in target_rows),
                "model_parameters_saved": False,
                "checkpoint_source": "not_saved",
            }
        )
    return out


def write_analysis_report(result_root: Path, overall_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# FDA SoTA Results",
        "",
        "Training objective: SADG epsilon loss = CE + 0.5 Dice; update loss = CE + 0.5 Dice + 0.05 Dispersion.",
        "FDA is used only as a train-set decorator. Evaluation runs on the original target images without FDA, TTA, or post-processing.",
        "No backbone parameters or model checkpoints are saved by this script.",
        "",
        "## Overall 15-run Domain Summary",
        "",
        "| Variant | Domain | Dice | HD95 | n_rows | n_cases |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            "| "
            + " | ".join(
                [
                    str(row.get("fda_variant", "")),
                    str(row.get("domain", "")),
                    f"{float(row.get('case_dice_mean', float('nan'))):.4f}",
                    f"{float(row.get('case_hd95_mean', float('nan'))):.2f}",
                    str(int(row.get("n_rows", 0))),
                    str(int(row.get("n_cases_total", 0))),
                ]
            )
            + " |"
        )
    lines.extend(
        [
            "",
            "## Files",
            "",
            "- `training_curves.csv`: per-epoch source training curves.",
            "- `eval_metrics.csv`: domain-level Dice/HD95.",
            "- `eval_case_metrics.csv`: case-level Dice/HD95.",
            "- `experiment_summary.csv`: one row per variant/shot/seed run.",
            "- `domain_5seed_summary.csv`: per variant/shot/domain 5-seed summary.",
            "- `overall_15run_domain_summary.csv`: per variant/domain summary over 3 shots x 5 seeds.",
        ]
    )
    (result_root / "analysis_report.md").write_text("\n".join(lines), encoding="utf-8")


def write_all_outputs(
    result_root: Path,
    *,
    training_rows: Sequence[Mapping[str, Any]],
    eval_rows: Sequence[Mapping[str, Any]],
    case_rows: Sequence[Mapping[str, Any]],
) -> None:
    write_csv(result_root / "training_curves.csv", training_rows)
    write_csv(result_root / "eval_metrics.csv", eval_rows)
    write_csv(result_root / "eval_case_metrics.csv", case_rows)
    experiments = experiment_summary_rows(eval_rows, training_rows)
    domain_5seed = summarize_eval_groups(eval_rows, ("fda_variant", "shot", "domain"))
    overall = summarize_eval_groups(eval_rows, ("fda_variant", "domain"))
    write_csv(result_root / "experiment_summary.csv", experiments)
    write_csv(result_root / "domain_5seed_summary.csv", domain_5seed)
    write_csv(result_root / "overall_15run_domain_summary.csv", overall)
    write_analysis_report(result_root, overall)


def assert_no_model_params_saved(result_root: Path) -> None:
    offenders = [p for p in result_root.rglob("*") if p.is_file() and p.suffix.lower() in {".pt", ".pth", ".ckpt"}]
    if offenders:
        joined = "\n".join(str(p) for p in offenders[:10])
        raise RuntimeError(f"FDA script must not save model parameter files, but found:\n{joined}")


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    result_root = reset_or_prepare_result_root(resolve_path(args.result_root), bool(args.overwrite))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)
    variants = parse_str_list(args.fda_variants)
    style_domains = parse_str_list(args.fda_style_domains)

    print(f"[DEVICE] {device}", flush=True)
    print(f"[RESULT_ROOT] {result_root}", flush=True)
    print(
        f"[FDA_STYLE_BANK] beta={float(args.fda_beta):.6g} domains={len(style_domains)} "
        f"resize={int(args.resize_hw)}",
        flush=True,
    )
    style_bank = build_target_style_bank(
        domains=style_domains,
        data_root=Path(args.data_root),
        resize_hw=int(args.resize_hw),
        beta=float(args.fda_beta),
        fft_norm=str(args.fda_fft_norm),
        clip_min=float(args.clip_min),
        clip_max=float(args.clip_max),
        style_batch_size=int(args.fda_style_batch_size),
    )
    print(
        f"[FDA_STYLE_BANK] radius={style_bank.radius} window={style_bank.window_size} "
        f"slices={style_bank.style_slices_total}",
        flush=True,
    )

    write_json(
        result_root / "run_config.json",
        {
            **vars(args),
            "method": METHOD,
            "objective": "SADG(CE+0.5Dice)+0.05Dispersion",
            "model_parameters_saved": False,
            "style_bank": style_bank.summary(),
        },
    )

    all_training_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []

    print(
        f"[METHOD] {METHOD} variants={variants} sadg_rho={float(args.sadg_rho):.6g} "
        f"shots={shots} seeds={seeds}",
        flush=True,
    )

    for variant in variants:
        normalize_fda_style_mode(variant)
        for shot in shots:
            for seed in seeds:
                model, train_rows, metadata, run_dir = train_one_run(
                    args=args,
                    device=device,
                    shot=int(shot),
                    seed=int(seed),
                    result_root=result_root,
                    style_bank=style_bank,
                    fda_variant=variant,
                )
                eval_rows, case_rows = evaluate_model(
                    args=args,
                    model=model,
                    device=device,
                    shot=int(shot),
                    seed=int(seed),
                    train_case_ids=[str(x) for x in metadata["train_case_ids"]],
                    output_dir=run_dir,
                    style_bank=style_bank,
                    fda_variant=variant,
                )
                all_training_rows.extend(train_rows)
                all_eval_rows.extend(eval_rows)
                all_case_rows.extend(case_rows)
                write_all_outputs(
                    result_root,
                    training_rows=all_training_rows,
                    eval_rows=all_eval_rows,
                    case_rows=all_case_rows,
                )
                assert_no_model_params_saved(result_root)
                del model
                if device.type == "cuda":
                    torch.cuda.empty_cache()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train FDA with SADG(CE+0.5Dice)+0.05Dispersion.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--device", default="cuda")
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
    parser.add_argument("--sadg_rho", type=float, default=FIXED_SADG_RHO)
    parser.add_argument("--sadg_eps", type=float, default=1e-12)
    parser.add_argument("--fda_variants", default=",".join(DEFAULT_FDA_VARIANTS))
    parser.add_argument("--fda_beta", type=float, default=0.01)
    parser.add_argument("--fda_eps", type=float, default=1e-6)
    parser.add_argument("--fda_fft_norm", default="backward", choices=("backward", "ortho", "forward"))
    parser.add_argument("--fda_style_batch_size", type=int, default=32)
    parser.add_argument("--fda_style_domains", default=",".join(TARGET_STYLE_DOMAINS))
    parser.add_argument("--clip_min", type=float, default=-3.0)
    parser.add_argument("--clip_max", type=float, default=3.0)
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
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
