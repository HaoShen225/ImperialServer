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
import torch.nn.functional as F

from helper.backbone import ModelConfig, build_model
from helper.dataloaders import (
    DATA_ROOT,
    DOMAIN_NAMES,
    FOREGROUND_CLASS_NAMES,
    SOURCE_DOMAIN_NAME,
    build_train_dataset,
    make_loader,
    run_test_flow,
)
from helper.load_model_params import build_backbone_from_checkpoint, resolve_checkpoint_path


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "CEwithVsWithoutDice"
DEFAULT_PURE_CE_ROOT = PROJECT_ROOT / "backbones" / "PureCE"
DEFAULT_CE_WITH_DICE_ROOT = PROJECT_ROOT / "backbones" / "CEwithDice"
SOURCE_DOMAIN = SOURCE_DOMAIN_NAME


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


def pure_ce_run_dir(args: argparse.Namespace, shot: int, seed: int) -> Path:
    return resolve_path(args.pure_ce_root) / f"shot{int(shot)}" / f"Seed{int(seed)}"


def method_run_dir(method_root: str | Path, shot: int, seed: int) -> Path:
    return resolve_path(method_root) / f"shot{int(shot)}" / f"Seed{int(seed)}"


def checkpoint_path_for(method_root: str | Path, shot: int, seed: int) -> Path:
    root = resolve_path(method_root)
    return resolve_checkpoint_path(backbone_root=root.parent, loss_mode=root.name, shot=int(shot), seed=int(seed))


def save_model_files(run_dir: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(run_dir)
    final_path = run_dir / "checkpoint_final.pt"
    baseline_path = run_dir / "baseline_model_with_metadata.pt"
    torch.save(payload, final_path)
    shutil.copyfile(final_path, baseline_path)


def load_checkpoint_model(method_root: str | Path, shot: int, seed: int, device: torch.device) -> tuple[torch.nn.Module, Dict[str, Any]]:
    ckpt = checkpoint_path_for(method_root, shot, seed)
    return build_backbone_from_checkpoint(ckpt, device=device, strict=True, eval_mode=True)


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


def pure_ce_checkpoint_payload(
    model: torch.nn.Module,
    *,
    args: argparse.Namespace,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    n_train_slices: int,
    slice_indices_by_case: Dict[str, List[int]],
) -> Dict[str, Any]:
    cfg = model_config(args)
    return {
        "model_state_dict": model.state_dict(),
        "metadata": {
            "experiment": "CEwithVsWithoutDice",
            "loss_mode": "PureCE",
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
                "segmentation": "CE",
                "dice_weight": 0.0,
                "dispersion": "disabled",
            },
            "args": vars(args),
        },
    }


def train_or_load_pure_ce(
    *,
    args: argparse.Namespace,
    device: torch.device,
    shot: int,
    seed: int,
) -> tuple[torch.nn.Module, List[Dict[str, Any]], Dict[str, Any]]:
    set_seed(int(seed))
    run_dir = pure_ce_run_dir(args, shot, seed)
    train_dataset = build_train_data(args, shot, seed)
    slice_indices_by_case = train_dataset.slice_indices_by_case()
    dataset_meta = {
        "loss_mode": "PureCE",
        "source_domain": SOURCE_DOMAIN,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": train_dataset.selected_case_ids,
        "n_train_slices": int(len(train_dataset)),
        "slice_policy": train_dataset.slice_policy,
        "num_middle_slices": int(train_dataset.num_middle_slices),
        "filter_min_fg": bool(train_dataset.filter_min_fg),
        "min_fg_ratio": float(args.min_fg_ratio),
        "slice_indices_by_case": slice_indices_by_case,
    }
    write_json(run_dir / "dataset_metadata.json", dataset_meta)
    write_json(run_dir / "run_config.json", {**vars(args), **dataset_meta, "output_dir": str(run_dir)})

    checkpoint = run_dir / "baseline_model_with_metadata.pt"
    if checkpoint.exists() and bool(args.resume) and not bool(args.overwrite_pure_ce):
        model, _metadata = build_backbone_from_checkpoint(checkpoint, device=device, strict=True, eval_mode=True)
        print(f"[RESUME] PureCE shot={shot} seed={seed}: {checkpoint}")
        return model, read_csv(run_dir / "training_log.csv"), dataset_meta

    model = build_model(model_config(args)).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    loader = make_loader(train_dataset, batch_size=int(args.batch_size), shuffle=True, device=device)
    log_rows: List[Dict[str, Any]] = []

    print(f"[TRAIN] PureCE shot={shot} seed={seed} cases={train_dataset.selected_case_ids} slices={len(train_dataset)}")
    for epoch in range(1, int(args.epochs) + 1):
        model.train()
        epoch_loss: List[float] = []
        steps = 0
        for step, (img, mask, _meta) in enumerate(loader, start=1):
            if int(args.max_train_steps) > 0 and step > int(args.max_train_steps):
                break
            img = img.to(device)
            mask = mask.to(device)
            out = model(img, return_features=True)
            logits = out["logits"] if isinstance(out, dict) else out
            loss = F.cross_entropy(logits, mask)

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(args.grad_clip))
            optimizer.step()

            steps += 1
            epoch_loss.append(float(loss.detach().cpu()))

        row = {
            "loss_mode": "PureCE",
            "shot": int(shot),
            "seed": int(seed),
            "epoch": int(epoch),
            "train_loss": finite_mean(epoch_loss),
            "train_ce_loss": finite_mean(epoch_loss),
            "train_dice_loss": 0.0,
            "steps": int(steps),
            "train_case_ids": "|".join(train_dataset.selected_case_ids),
            "n_train_slices": int(len(train_dataset)),
            "slice_policy": str(args.slice_policy),
            "filter_min_fg": bool(args.filter_min_fg),
            "output_dir": str(run_dir),
        }
        log_rows.append(row)
        write_csv(run_dir / "training_log.csv", log_rows)
        print(f"  epoch {epoch:03d}/{int(args.epochs):03d} ce={float(row['train_loss']):.6f}")

    payload = pure_ce_checkpoint_payload(
        model,
        args=args,
        shot=shot,
        seed=seed,
        train_case_ids=train_dataset.selected_case_ids,
        n_train_slices=len(train_dataset),
        slice_indices_by_case=slice_indices_by_case,
    )
    save_model_files(run_dir, payload)
    model.eval()
    return model, log_rows, dataset_meta


def class_metric_names() -> List[str]:
    names: List[str] = []
    for prefix in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
        for class_name in FOREGROUND_CLASS_NAMES.values():
            names.append(f"{prefix}_{class_name}")
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
    rows: List[Dict[str, Any]] = []
    for row in eval_rows:
        for cls, class_name in FOREGROUND_CLASS_NAMES.items():
            rows.append(
                {
                    "loss_mode": row.get("loss_mode", ""),
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


def summarize_class_seed_average(class_rows: Sequence[Dict[str, Any]]) -> List[Dict[str, Any]]:
    group_fields = ("loss_mode", "shot", "domain", "class_id", "class_name")
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
    loss_mode: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    output_dir: Path,
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
            "loss_mode": loss_mode,
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
                    "loss_mode": loss_mode,
                    "shot": int(shot),
                    "seed": int(seed),
                    "domain": metrics["domain"],
                    **case_row,
                }
            )
        print(
            f"[EVAL] {loss_mode} shot={shot} seed={seed} domain={metrics['domain']} "
            f"case_dice={float(row['case_dice']):.6f} case_hd95={float(row['case_hd95']):.6f}"
        )
    return eval_rows, case_rows


def summarize_experiment(
    *,
    loss_mode: str,
    shot: int,
    seed: int,
    train_case_ids: Sequence[str],
    n_train_slices: int,
    train_rows: Sequence[Dict[str, Any]],
    eval_rows: Sequence[Dict[str, Any]],
    output_dir: Path,
) -> Dict[str, Any]:
    last_train = train_rows[-1] if train_rows else {}
    return {
        "loss_mode": loss_mode,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": "|".join(str(x) for x in train_case_ids),
        "n_train_slices": int(n_train_slices),
        "final_train_loss": last_train.get("train_loss", float("nan")),
        "mean_6domain_case_dice": finite_mean(row.get("case_dice") for row in eval_rows),
        "mean_6domain_case_hd95": finite_mean(row.get("case_hd95") for row in eval_rows),
        "mean_6domain_slice_dice": finite_mean(row.get("slice_dice") for row in eval_rows),
        "mean_6domain_slice_hd95": finite_mean(row.get("slice_hd95") for row in eval_rows),
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
    write_csv(result_root / "method_summary.csv", summarize_eval_groups(eval_rows, ("loss_mode",)))
    write_csv(result_root / "shot_summary.csv", summarize_eval_groups(eval_rows, ("loss_mode", "shot")))
    write_csv(result_root / "domain_summary.csv", summarize_eval_groups(eval_rows, ("loss_mode", "domain")))
    write_csv(result_root / "experiment_summary.csv", experiment_rows)


def run(args: argparse.Namespace) -> None:
    device = resolve_device(args.device)
    result_root = ensure_dir(resolve_path(args.result_root))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)

    all_training_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []
    all_experiment_rows: List[Dict[str, Any]] = []

    print(f"[DEVICE] {device}")
    print(f"[RESULT_ROOT] {result_root}")
    print(f"[PURE_CE_ROOT] {resolve_path(args.pure_ce_root)}")
    print(f"[CE_WITH_DICE_ROOT] {resolve_path(args.ce_with_dice_root)}")
    print(f"[MATRIX] shots={shots} seeds={seeds} BS={args.batch_size}")
    print(f"[SLICE] policy={args.slice_policy} filter_min_fg={args.filter_min_fg} min_fg_ratio={args.min_fg_ratio}")

    for shot in shots:
        for seed in seeds:
            pure_model, pure_train_rows, pure_meta = train_or_load_pure_ce(
                args=args,
                device=device,
                shot=int(shot),
                seed=int(seed),
            )
            pure_eval_rows, pure_case_rows = evaluate_model(
                args=args,
                model=pure_model,
                device=device,
                loss_mode="PureCE",
                shot=int(shot),
                seed=int(seed),
                train_case_ids=pure_meta["train_case_ids"],
                output_dir=pure_ce_run_dir(args, shot, seed),
            )
            all_training_rows.extend(pure_train_rows)
            all_eval_rows.extend(pure_eval_rows)
            all_case_rows.extend(pure_case_rows)
            all_experiment_rows.append(
                summarize_experiment(
                    loss_mode="PureCE",
                    shot=int(shot),
                    seed=int(seed),
                    train_case_ids=pure_meta["train_case_ids"],
                    n_train_slices=int(pure_meta["n_train_slices"]),
                    train_rows=pure_train_rows,
                    eval_rows=pure_eval_rows,
                    output_dir=pure_ce_run_dir(args, shot, seed),
                )
            )
            del pure_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            ce_model, ce_metadata = load_checkpoint_model(args.ce_with_dice_root, int(shot), int(seed), device)
            ce_train_case_ids = ce_metadata.get("train_case_ids") or pure_meta["train_case_ids"]
            ce_train_rows = read_csv(method_run_dir(args.ce_with_dice_root, shot, seed) / "training_log.csv")
            ce_eval_rows, ce_case_rows = evaluate_model(
                args=args,
                model=ce_model,
                device=device,
                loss_mode="CEwithDice",
                shot=int(shot),
                seed=int(seed),
                train_case_ids=[str(x) for x in ce_train_case_ids],
                output_dir=method_run_dir(args.ce_with_dice_root, shot, seed),
            )
            all_training_rows.extend({**row, "loss_mode": "CEwithDice"} for row in ce_train_rows)
            all_eval_rows.extend(ce_eval_rows)
            all_case_rows.extend(ce_case_rows)
            all_experiment_rows.append(
                summarize_experiment(
                    loss_mode="CEwithDice",
                    shot=int(shot),
                    seed=int(seed),
                    train_case_ids=[str(x) for x in ce_train_case_ids],
                    n_train_slices=int(ce_metadata.get("n_train_slices", pure_meta["n_train_slices"])),
                    train_rows=ce_train_rows,
                    eval_rows=ce_eval_rows,
                    output_dir=method_run_dir(args.ce_with_dice_root, shot, seed),
                )
            )
            del ce_model
            if device.type == "cuda":
                torch.cuda.empty_cache()

            write_all_summaries(
                result_root,
                training_rows=all_training_rows,
                eval_rows=all_eval_rows,
                case_rows=all_case_rows,
                experiment_rows=all_experiment_rows,
            )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compare CE-only and CE+0.5 Dice backbones.")
    parser.add_argument("--data_root", default=str(DATA_ROOT))
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--pure_ce_root", default=str(DEFAULT_PURE_CE_ROOT))
    parser.add_argument("--ce_with_dice_root", default=str(DEFAULT_CE_WITH_DICE_ROOT))
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--shots", default="3,4,5")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--batch_size", type=int, default=4)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
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
    parser.add_argument("--resume", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--overwrite_pure_ce", action="store_true")
    return parser.parse_args()


def main() -> None:
    run(parse_args())


if __name__ == "__main__":
    main()
