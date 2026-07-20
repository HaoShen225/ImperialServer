from __future__ import annotations

import argparse
import csv
import json
import math
import random
import shutil
from pathlib import Path
from typing import Any, Dict, Iterable, List, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from helper.Evaluator import PatientStreamEvaluator, SliceWiceEvaluation
from helper.backbones.UNet import UNet
from helper.dataloader import ACDC_ROOT, MMS_ROOT, TestLoader, TrainLoader


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "backbone_params"
NUM_CLASSES = 4
DEFAULT_EVAL_VENDORS = ("A", "B", "C", "D")


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


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


def parse_str_list(text: str) -> List[str]:
    return [x.strip() for x in str(text).split(",") if x.strip()]


def finite_mean(values: Iterable[Any]) -> float:
    vals: List[float] = []
    for value in values:
        try:
            f = float(value)
        except Exception:
            continue
        if np.isfinite(f):
            vals.append(f)
    return float(np.mean(vals)) if vals else float("nan")


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def write_json(path: str | Path, payload: MappingLike) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    p.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path: str | Path, rows: Sequence[Dict[str, Any]]) -> None:
    p = Path(path)
    ensure_dir(p.parent)
    if not rows:
        p.write_text("", encoding="utf-8")
        return
    fields: List[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with p.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def run_dir(output_root: str | Path, labeled_cases_per_class: int, seed: int) -> Path:
    return Path(output_root) / f"Patient{int(labeled_cases_per_class)}" / f"Seed{int(seed)}"


def build_model() -> UNet:
    return UNet(n_channels=1, n_classes=NUM_CLASSES, only_feature=False)


def model_logits(model: torch.nn.Module, images: torch.Tensor) -> torch.Tensor:
    out = model(images)
    if isinstance(out, tuple):
        return out[-1]
    if isinstance(out, dict):
        return out["logits"]
    return out


def one_hot_mask(mask: torch.Tensor, num_classes: int = NUM_CLASSES) -> torch.Tensor:
    return F.one_hot(mask.long(), num_classes=int(num_classes)).permute(0, 3, 1, 2).float()


def soft_dice_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    num_classes: int = NUM_CLASSES,
    include_bg: bool = False,
    eps: float = 1e-8,
) -> torch.Tensor:
    prob = torch.softmax(logits, dim=1)
    target = one_hot_mask(mask, num_classes=num_classes)
    class_ids = range(int(num_classes)) if include_bg else range(1, int(num_classes))
    losses: List[torch.Tensor] = []
    for cls in class_ids:
        pc = prob[:, cls]
        tc = target[:, cls]
        inter = (pc * tc).sum(dim=(1, 2))
        denom = pc.sum(dim=(1, 2)) + tc.sum(dim=(1, 2))
        losses.append(1.0 - (2.0 * inter + eps) / (denom + eps))
    return torch.cat([loss.reshape(-1) for loss in losses]).mean() if losses else logits.sum() * 0.0


def supervised_segmentation_loss(
    logits: torch.Tensor,
    mask: torch.Tensor,
    *,
    dice_weight: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    ce = F.cross_entropy(logits, mask.long())
    dice = soft_dice_loss(logits, mask, num_classes=NUM_CLASSES, include_bg=False)
    total = ce + float(dice_weight) * dice
    return total, ce, dice


def sigmoid_rampup(current: int, rampup_length: int) -> float:
    if int(rampup_length) <= 0:
        return 1.0
    current = float(np.clip(current, 0, int(rampup_length)))
    phase = 1.0 - current / float(rampup_length)
    return float(math.exp(-5.0 * phase * phase))


def consistency_weight(epoch: int, *, max_lambda_u: float, rampup_epochs: int) -> float:
    return float(max_lambda_u) * sigmoid_rampup(int(epoch), int(rampup_epochs))


def ema_alpha(epoch: int, *, rampup_epochs: int, ema_alpha_rampup: float, ema_alpha: float) -> float:
    return float(ema_alpha_rampup) if int(epoch) <= int(rampup_epochs) else float(ema_alpha)


@torch.no_grad()
def update_ema_teacher(student: torch.nn.Module, teacher: torch.nn.Module, alpha: float) -> None:
    alpha = float(alpha)
    for teacher_param, student_param in zip(teacher.parameters(), student.parameters()):
        teacher_param.data.mul_(alpha).add_(student_param.data, alpha=1.0 - alpha)

    for teacher_buffer, student_buffer in zip(teacher.buffers(), student.buffers()):
        if torch.is_floating_point(teacher_buffer):
            teacher_buffer.data.mul_(alpha).add_(student_buffer.data, alpha=1.0 - alpha)
        else:
            teacher_buffer.copy_(student_buffer)


def add_student_noise(images: torch.Tensor, std: float) -> torch.Tensor:
    if float(std) <= 0.0:
        return images
    noisy = images + torch.randn_like(images) * float(std)
    return torch.clamp(noisy, 0.0, 1.0)


def checkpoint_payload(
    *,
    student: torch.nn.Module,
    teacher: torch.nn.Module,
    args: argparse.Namespace,
    labeled_cases_per_class: int,
    seed: int,
    loader: TrainLoader,
    log_rows: Sequence[Dict[str, Any]],
) -> Dict[str, Any]:
    final_row = dict(log_rows[-1]) if log_rows else {}
    metadata: Dict[str, Any] = {
        "method": "MeanTeacher",
        "primary_model": "teacher",
        "labeled_cases_per_class": int(labeled_cases_per_class),
        "seed": int(seed),
        "dataset_root": str(Path(args.dataset_root)),
        "num_classes": NUM_CLASSES,
        "label_map": {"0": "background", "1": "rv", "2": "myo", "3": "lv"},
        "labeled_patients_by_group": loader.labeled_patients_by_group,
        "labeled_patients": list(loader.labeled_patients),
        "unlabeled_patients": list(loader.unlabeled_patients),
        "labeled_slice_count": int(loader.labeled_slice_count),
        "unlabeled_slice_count": int(loader.unlabeled_slice_count),
        "epochs": int(args.epochs),
        "batch_size": int(args.batch_size),
        "labeled_batch_size": int(args.labeled_batch_size),
        "unlabeled_batch_size": int(args.batch_size) - int(args.labeled_batch_size),
        "optimizer": "AdamW",
        "lr": float(args.lr),
        "weight_decay": float(args.weight_decay),
        "dice_weight": float(args.dice_weight),
        "max_lambda_u": float(args.max_lambda_u),
        "rampup_epochs": int(args.rampup_epochs),
        "ema_alpha_rampup": float(args.ema_alpha_rampup),
        "ema_alpha": float(args.ema_alpha),
        "student_noise_std": float(args.student_noise_std),
        "grad_clip": float(args.grad_clip),
        "final_epoch": int(final_row.get("epoch", 0) or 0),
        "final_train_loss": final_row.get("train_loss", float("nan")),
        "final_supervised_loss": final_row.get("supervised_loss", float("nan")),
        "final_consistency_loss": final_row.get("consistency_loss", float("nan")),
        "args": vars(args),
    }
    return {
        "model_state_dict": teacher.state_dict(),
        "teacher_state_dict": teacher.state_dict(),
        "student_state_dict": student.state_dict(),
        "metadata": metadata,
    }


def save_checkpoint_files(run_path: Path, payload: Dict[str, Any]) -> None:
    ensure_dir(run_path)
    final_path = run_path / "checkpoint_final.pt"
    baseline_path = run_path / "baseline_model_with_metadata.pt"
    torch.save(payload, final_path)
    shutil.copyfile(final_path, baseline_path)


def backbone_id_for_run(labeled_cases_per_class: int, seed: int) -> str:
    return f"MeanTeacher_Patient{int(labeled_cases_per_class)}_Seed{int(seed)}_teacher"


def load_record_batch(records: Sequence[Any]) -> tuple[torch.Tensor, torch.Tensor, List[Dict[str, Any]]]:
    images: List[torch.Tensor] = []
    masks: List[torch.Tensor] = []
    metas: List[Dict[str, Any]] = []
    for record in records:
        image_arr = np.load(record.image_path).astype(np.float32, copy=False)
        if image_arr.ndim != 2:
            raise ValueError(f"Expected 2D image at {record.image_path}, got shape {image_arr.shape}")
        if record.mask_path is None:
            raise ValueError(f"Record {record.slice_id} has no mask_path")
        mask_arr = np.load(record.mask_path).astype(np.int64, copy=False)
        if mask_arr.ndim != 2:
            raise ValueError(f"Expected 2D mask at {record.mask_path}, got shape {mask_arr.shape}")
        images.append(torch.from_numpy(image_arr)[None, ...].float())
        masks.append(torch.from_numpy(mask_arr).long())
        metas.append(record.meta(include_mask_path=True))
    return torch.stack(images, dim=0), torch.stack(masks, dim=0), metas


def group_records_by_patient(records: Sequence[Any]) -> List[List[Any]]:
    groups: List[List[Any]] = []
    current_patient = ""
    current_records: List[Any] = []
    for record in records:
        patient_id = str(record.patient_id)
        if current_records and patient_id != current_patient:
            groups.append(current_records)
            current_records = []
        current_patient = patient_id
        current_records.append(record)
    if current_records:
        groups.append(current_records)
    return groups


@torch.no_grad()
def predict_labels(
    model: torch.nn.Module,
    images: torch.Tensor,
    *,
    device: torch.device,
    batch_size: int,
) -> torch.Tensor:
    model.eval()
    preds: List[torch.Tensor] = []
    step = max(1, int(batch_size))
    for start in range(0, int(images.shape[0]), step):
        logits = model_logits(model, images[start : start + step].to(device, non_blocking=True))
        preds.append(torch.argmax(logits, dim=1).detach().cpu())
    return torch.cat(preds, dim=0)


def eval_summary_row(
    *,
    summary: Dict[str, Any],
    mode: str,
    loader: TestLoader,
    labeled_cases_per_class: int,
    seed: int,
    checkpoint_path: Path,
    output_dir: Path,
) -> Dict[str, Any]:
    row = dict(summary)
    row.update(
        {
            "method": "MeanTeacher",
            "evaluation": "source_model",
            "mode": str(mode),
            "labeled_cases_per_class": int(labeled_cases_per_class),
            "seed": int(seed),
            "vendor": loader.vendor,
            "vendor_name": loader.vendor_name,
            "domain": loader.domain,
            "vendor_slice_count": int(loader.slice_count),
            "vendor_patient_count": int(len(loader.patient_ids)),
            "checkpoint": str(checkpoint_path),
            "output_dir": str(output_dir),
        }
    )
    return row


def evaluate_slice_stream(
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    run_path: Path,
    labeled_cases_per_class: int,
    seed: int,
    vendor: str,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    loader = TestLoader(
        vendor=vendor,
        batch_size=int(args.eval_batch_size),
        shuffle_all_slices=True,
        seed=int(seed),
        dataset_root=Path(args.target_dataset_root),
    )
    backbone_id = backbone_id_for_run(labeled_cases_per_class, seed)
    evaluator = SliceWiceEvaluation(domain=loader.domain, seed=int(seed), backbone_id=backbone_id)
    for step, batch in enumerate(loader, start=1):
        if int(args.max_eval_batches) > 0 and step > int(args.max_eval_batches):
            break
        preds = predict_labels(
            model,
            batch["images"],
            device=device,
            batch_size=int(args.eval_batch_size),
        )
        evaluator.update(preds, batch["masks"], batch["meta"], step=step, backbone_id=backbone_id)

    output_dir = run_path / "eval_source_model" / "slice" / loader.domain
    evaluator.save_csv(output_dir)
    return eval_summary_row(
        summary=evaluator.seed_summary(),
        mode="slice",
        loader=loader,
        labeled_cases_per_class=int(labeled_cases_per_class),
        seed=int(seed),
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )


def evaluate_patient_stream(
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    run_path: Path,
    labeled_cases_per_class: int,
    seed: int,
    vendor: str,
    checkpoint_path: Path,
) -> Dict[str, Any]:
    loader = TestLoader(
        vendor=vendor,
        batch_size=int(args.eval_batch_size),
        shuffle_all_slices=False,
        seed=int(seed),
        dataset_root=Path(args.target_dataset_root),
    )
    patient_groups = group_records_by_patient(loader.records)
    if int(args.max_eval_patients) > 0:
        patient_groups = patient_groups[: int(args.max_eval_patients)]

    backbone_id = backbone_id_for_run(labeled_cases_per_class, seed)
    evaluator = PatientStreamEvaluator(domain=loader.domain, seed=int(seed), backbone_id=backbone_id)
    for step, records in enumerate(patient_groups, start=1):
        images, masks, meta = load_record_batch(records)
        preds = predict_labels(
            model,
            images,
            device=device,
            batch_size=int(args.eval_batch_size),
        )
        evaluator.update(preds, masks, meta, step=step, backbone_id=backbone_id)

    output_dir = run_path / "eval_source_model" / "patient" / loader.domain
    evaluator.save_csv(output_dir)
    return eval_summary_row(
        summary=evaluator.seed_summary(),
        mode="patient",
        loader=loader,
        labeled_cases_per_class=int(labeled_cases_per_class),
        seed=int(seed),
        checkpoint_path=checkpoint_path,
        output_dir=output_dir,
    )


def evaluate_source_model(
    *,
    model: torch.nn.Module,
    args: argparse.Namespace,
    device: torch.device,
    run_path: Path,
    labeled_cases_per_class: int,
    seed: int,
    checkpoint_path: Path,
) -> List[Dict[str, Any]]:
    vendors = parse_str_list(args.eval_vendors)
    if not vendors:
        raise ValueError("--eval-vendors produced an empty list.")

    if args.eval_mode == "both":
        modes = ["patient", "slice"]
    else:
        modes = [str(args.eval_mode)]

    rows: List[Dict[str, Any]] = []
    print(f"[EVAL] Patient{labeled_cases_per_class} Seed{seed} source model on vendors={','.join(vendors)}")
    for mode in modes:
        for vendor in vendors:
            if mode == "patient":
                row = evaluate_patient_stream(
                    model=model,
                    args=args,
                    device=device,
                    run_path=run_path,
                    labeled_cases_per_class=int(labeled_cases_per_class),
                    seed=int(seed),
                    vendor=vendor,
                    checkpoint_path=checkpoint_path,
                )
            elif mode == "slice":
                row = evaluate_slice_stream(
                    model=model,
                    args=args,
                    device=device,
                    run_path=run_path,
                    labeled_cases_per_class=int(labeled_cases_per_class),
                    seed=int(seed),
                    vendor=vendor,
                    checkpoint_path=checkpoint_path,
                )
            else:
                raise ValueError(f"Unsupported eval mode: {mode}")
            rows.append(row)
            print(
                f"  {mode:7s} {row['domain']} "
                f"dice_mean={float(row.get('dice_mean', float('nan'))):.6f} "
                f"hd95_mean={float(row.get('hd95_mean', float('nan'))):.6f} "
                f"n_items={int(row.get('n_items', 0) or 0)}"
            )

    write_csv(run_path / "eval_source_model" / "eval_summary.csv", rows)
    return rows


def train_one_run(
    *,
    args: argparse.Namespace,
    device: torch.device,
    labeled_cases_per_class: int,
    seed: int,
) -> Dict[str, Any]:
    set_seed(int(seed))
    run_path = run_dir(args.output_root, labeled_cases_per_class, seed)
    baseline_path = run_path / "baseline_model_with_metadata.pt"

    if baseline_path.exists() and bool(args.resume) and not bool(args.overwrite):
        print(f"[SKIP] Patient{labeled_cases_per_class} Seed{seed}: {baseline_path}")
        return {"status": "skipped", "run_dir": str(run_path), "checkpoint": str(baseline_path)}

    ensure_dir(run_path)
    loader = TrainLoader(
        labeled_cases_per_class=int(labeled_cases_per_class),
        seed=int(seed),
        batch_size=int(args.batch_size),
        labeled_batch_size=int(args.labeled_batch_size),
        dataset_root=Path(args.dataset_root),
        shuffle_labeled=True,
        shuffle_unlabeled=True,
    )

    student = build_model().to(device)
    teacher = build_model().to(device)
    teacher.load_state_dict(student.state_dict())
    for param in teacher.parameters():
        param.requires_grad_(False)

    optimizer = torch.optim.AdamW(student.parameters(), lr=float(args.lr), weight_decay=float(args.weight_decay))
    log_rows: List[Dict[str, Any]] = []

    run_config = {
        "method": "MeanTeacher",
        "primary_model": "teacher",
        "run_dir": str(run_path),
        "labeled_cases_per_class": int(labeled_cases_per_class),
        "seed": int(seed),
        "device": str(device),
        "labeled_patients_by_group": loader.labeled_patients_by_group,
        "labeled_patients": list(loader.labeled_patients),
        "unlabeled_patients": list(loader.unlabeled_patients),
        "labeled_slice_count": int(loader.labeled_slice_count),
        "unlabeled_slice_count": int(loader.unlabeled_slice_count),
        "args": vars(args),
    }
    write_json(run_path / "run_config.json", run_config)

    print(
        f"[TRAIN] Patient{labeled_cases_per_class} Seed{seed} "
        f"labeled_slices={loader.labeled_slice_count} unlabeled_slices={loader.unlabeled_slice_count}"
    )
    for epoch in range(1, int(args.epochs) + 1):
        student.train()
        teacher.eval()
        lambda_u = consistency_weight(epoch, max_lambda_u=float(args.max_lambda_u), rampup_epochs=int(args.rampup_epochs))
        alpha = ema_alpha(
            epoch,
            rampup_epochs=int(args.rampup_epochs),
            ema_alpha_rampup=float(args.ema_alpha_rampup),
            ema_alpha=float(args.ema_alpha),
        )

        epoch_total: List[float] = []
        epoch_sup: List[float] = []
        epoch_ce: List[float] = []
        epoch_dice: List[float] = []
        epoch_cons: List[float] = []
        steps = 0

        for step, batch in enumerate(loader, start=1):
            if int(args.max_steps_per_epoch) > 0 and step > int(args.max_steps_per_epoch):
                break

            labeled_images = batch["labeled_images"].to(device, non_blocking=True)
            labeled_masks = batch["labeled_masks"].to(device, non_blocking=True)
            unlabeled_images = batch["unlabeled_images"].to(device, non_blocking=True)

            labeled_logits = model_logits(student, labeled_images)
            sup_loss, ce_loss, dice_loss = supervised_segmentation_loss(
                labeled_logits,
                labeled_masks,
                dice_weight=float(args.dice_weight),
            )

            student_unlabeled = add_student_noise(unlabeled_images, float(args.student_noise_std))
            student_logits_u = model_logits(student, student_unlabeled)
            student_prob_u = torch.softmax(student_logits_u, dim=1)

            with torch.no_grad():
                teacher_logits_u = model_logits(teacher, unlabeled_images)
                teacher_prob_u = torch.softmax(teacher_logits_u, dim=1)

            cons_loss = F.mse_loss(student_prob_u, teacher_prob_u)
            loss = sup_loss + float(lambda_u) * cons_loss

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            if float(args.grad_clip) > 0.0:
                torch.nn.utils.clip_grad_norm_(student.parameters(), float(args.grad_clip))
            optimizer.step()
            update_ema_teacher(student, teacher, alpha)

            steps += 1
            epoch_total.append(float(loss.detach().cpu()))
            epoch_sup.append(float(sup_loss.detach().cpu()))
            epoch_ce.append(float(ce_loss.detach().cpu()))
            epoch_dice.append(float(dice_loss.detach().cpu()))
            epoch_cons.append(float(cons_loss.detach().cpu()))

        row = {
            "method": "MeanTeacher",
            "labeled_cases_per_class": int(labeled_cases_per_class),
            "seed": int(seed),
            "epoch": int(epoch),
            "steps": int(steps),
            "lambda_u": float(lambda_u),
            "ema_alpha": float(alpha),
            "train_loss": finite_mean(epoch_total),
            "supervised_loss": finite_mean(epoch_sup),
            "ce_loss": finite_mean(epoch_ce),
            "dice_loss": finite_mean(epoch_dice),
            "consistency_loss": finite_mean(epoch_cons),
            "labeled_slice_count": int(loader.labeled_slice_count),
            "unlabeled_slice_count": int(loader.unlabeled_slice_count),
            "run_dir": str(run_path),
        }
        log_rows.append(row)
        write_csv(run_path / "training_log.csv", log_rows)
        print(
            f"  epoch {epoch:03d}/{int(args.epochs):03d} "
            f"loss={row['train_loss']:.6f} sup={row['supervised_loss']:.6f} "
            f"cons={row['consistency_loss']:.6f} lambda_u={lambda_u:.4f} alpha={alpha:.4f}"
        )

    payload = checkpoint_payload(
        student=student,
        teacher=teacher,
        args=args,
        labeled_cases_per_class=int(labeled_cases_per_class),
        seed=int(seed),
        loader=loader,
        log_rows=log_rows,
    )
    save_checkpoint_files(run_path, payload)
    checkpoint_path = run_path / "baseline_model_with_metadata.pt"
    eval_rows: List[Dict[str, Any]] = []
    if not bool(args.skip_eval):
        eval_rows = evaluate_source_model(
            model=teacher,
            args=args,
            device=device,
            run_path=run_path,
            labeled_cases_per_class=int(labeled_cases_per_class),
            seed=int(seed),
            checkpoint_path=checkpoint_path,
        )
    return {
        "status": "trained",
        "run_dir": str(run_path),
        "checkpoint": str(checkpoint_path),
        "eval_summary": str(run_path / "eval_source_model" / "eval_summary.csv") if eval_rows else "",
    }


def build_arg_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Train ACDC Mean Teacher source-domain backbones.")
    parser.add_argument("--patients", default="1,2,3", help="Comma-separated labeled cases per disease class.")
    parser.add_argument("--seeds", default="0,1,2,3,4", help="Comma-separated random seeds.")
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--labeled-batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--dice-weight", type=float, default=0.5)
    parser.add_argument("--max-lambda-u", type=float, default=1.0)
    parser.add_argument("--rampup-epochs", type=int, default=40)
    parser.add_argument("--ema-alpha-rampup", type=float, default=0.99)
    parser.add_argument("--ema-alpha", type=float, default=0.999)
    parser.add_argument("--student-noise-std", type=float, default=0.05)
    parser.add_argument("--grad-clip", type=float, default=5.0)
    parser.add_argument("--max-steps-per-epoch", type=int, default=0, help="0 means full epoch.")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--dataset-root", default=str(ACDC_ROOT))
    parser.add_argument("--target-dataset-root", default=str(MMS_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--eval-vendors", default=",".join(DEFAULT_EVAL_VENDORS))
    parser.add_argument("--eval-mode", default="patient", choices=("patient", "slice", "both"))
    parser.add_argument("--eval-batch-size", type=int, default=8)
    parser.add_argument("--max-eval-patients", type=int, default=0, help="0 means all patients.")
    parser.add_argument("--max-eval-batches", type=int, default=0, help="0 means all slice batches.")
    parser.add_argument("--skip-eval", action="store_true")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    parser = build_arg_parser()
    args = parser.parse_args(argv)
    device = resolve_device(args.device)
    patients = parse_int_list(args.patients)
    seeds = parse_int_list(args.seeds)

    if not patients:
        raise ValueError("--patients produced an empty list.")
    if not seeds:
        raise ValueError("--seeds produced an empty list.")
    if int(args.labeled_batch_size) <= 0 or int(args.labeled_batch_size) >= int(args.batch_size):
        raise ValueError("--labeled-batch-size must satisfy 0 < labeled < batch-size.")
    if int(args.eval_batch_size) <= 0:
        raise ValueError("--eval-batch-size must be positive.")

    ensure_dir(args.output_root)
    summaries: List[Dict[str, Any]] = []
    print(f"[DEVICE] {device}")
    for labeled_cases_per_class in patients:
        for seed in seeds:
            summary = train_one_run(
                args=args,
                device=device,
                labeled_cases_per_class=int(labeled_cases_per_class),
                seed=int(seed),
            )
            summary["labeled_cases_per_class"] = int(labeled_cases_per_class)
            summary["seed"] = int(seed)
            summaries.append(summary)

    write_csv(Path(args.output_root) / "mean_teacher_runs.csv", summaries)
    print(f"[DONE] wrote summary to {Path(args.output_root) / 'mean_teacher_runs.csv'}")


MappingLike = Dict[str, Any] | List[Any] | str | int | float | bool | None


if __name__ == "__main__":
    main()
