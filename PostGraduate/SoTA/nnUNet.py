from __future__ import annotations

import argparse
import csv
import importlib.util
import json
import os
import re
import shutil
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

import numpy as np
import SimpleITK as sitk

from helper.dataloaders import (
    DATA_ROOT,
    DOMAIN_NAMES,
    FOREGROUND_CLASS_NAMES,
    SOURCE_DOMAIN_NAME,
    build_test_dataset,
    build_train_dataset,
    case_dice_by_class,
    case_hd95_by_class,
    slice_dice_by_class,
    slice_hd95_by_class,
)


PROJECT_ROOT = Path(__file__).resolve().parent
DEFAULT_RESULT_ROOT = PROJECT_ROOT / "TrainingData" / "nnU-Net"
DEFAULT_PARAMETER_ROOT = PROJECT_ROOT / "parameters" / "nnU-Net"
METHOD = "nnU-Net"
SPIDER_DATASET_DIRNAME = "SPIDER_domain_strict_3Foreground"
SPIDER_SPLIT_RELATIVE_PATH = (
    Path("split")
    / "PostGraduateProject"
    / "spider_symphonytim_t1_fewshot_source_to_5domains_seed0-4.csv"
)
SERVER_SPINE_ROOT = Path("/rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Spine")


def resolve_path(path: str | Path, *, base: Path = PROJECT_ROOT) -> Path:
    p = Path(os.path.expandvars(os.path.expanduser(str(path))))
    if p.is_absolute():
        return p
    if p.parts and str(p.parts[0]).lower() == "sota":
        return Path.cwd() / p
    return base / p


def spider_root_candidate(path: str | Path) -> Path:
    return Path(os.path.expandvars(os.path.expanduser(str(path))))


def resolve_spider_data_root(path: str | Path) -> Path:
    """Accept either the Spine folder or the SPIDER_domain_strict_3Foreground folder."""
    p = spider_root_candidate(path)
    nested = p / SPIDER_DATASET_DIRNAME
    if nested.exists():
        return nested.resolve()
    if p.name == SPIDER_DATASET_DIRNAME:
        return p.resolve() if p.exists() else p
    if p.name.lower() == "spine":
        return nested
    return p.resolve() if p.exists() else p


def default_data_root() -> str:
    for env_name in ("SPIDER_DATA_ROOT", "SPINE_DATA_ROOT"):
        value = os.environ.get(env_name)
        if value:
            return str(resolve_spider_data_root(value))

    for candidate in (SERVER_SPINE_ROOT, DATA_ROOT):
        resolved = resolve_spider_data_root(candidate)
        if resolved.exists():
            return str(resolved)
    return str(DATA_ROOT)


def split_csv_for_data_root(data_root: str | Path) -> Path:
    return resolve_spider_data_root(data_root) / SPIDER_SPLIT_RELATIVE_PATH


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def parse_int_list(text: str) -> List[int]:
    return [int(x.strip()) for x in str(text).split(",") if x.strip()]


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


def slugify(text: str) -> str:
    value = re.sub(r"[^A-Za-z0-9]+", "_", str(text)).strip("_")
    return value or "item"


def dataset_id_for(dataset_id_base: int, shot: int, seed: int) -> int:
    if int(shot) not in {3, 4, 5}:
        raise ValueError(f"shot must be one of 3, 4, 5; got {shot}")
    if int(seed) not in {0, 1, 2, 3, 4}:
        raise ValueError(f"seed must be one of 0, 1, 2, 3, 4; got {seed}")
    return int(dataset_id_base) + (int(shot) - 3) * 5 + int(seed)


def dataset_name_for(dataset_id: int, shot: int, seed: int) -> str:
    return f"Dataset{int(dataset_id):03d}_SPIDER2D_shot{int(shot)}_seed{int(seed)}"


def run_dir_for(result_root: Path, shot: int, seed: int) -> Path:
    return result_root / "runs" / METHOD / f"shot{int(shot)}" / f"Seed{int(seed)}"


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
            raise FileExistsError(f"Result root already contains nnU-Net outputs. Use --overwrite: {result_root}")
    return ensure_dir(result_root)


def prepare_parameter_root(parameter_root: Path, overwrite: bool) -> Path:
    parameter_root = parameter_root.resolve()
    project_root = PROJECT_ROOT.resolve()
    if parameter_root.exists() and overwrite:
        if parameter_root == project_root or project_root not in parameter_root.parents:
            raise ValueError(f"Refusing to overwrite parameter_root outside {project_root}: {parameter_root}")
        shutil.rmtree(parameter_root)
    ensure_dir(parameter_root)
    ensure_dir(parameter_root / "nnUNet_raw")
    ensure_dir(parameter_root / "nnUNet_preprocessed")
    ensure_dir(parameter_root / "nnUNet_results")
    ensure_dir(parameter_root / "prediction_inputs")
    ensure_dir(parameter_root / "predictions")
    return parameter_root


def nnunet_env(parameter_root: Path, device: str) -> Dict[str, str]:
    env = os.environ.copy()
    env["nnUNet_raw"] = str(parameter_root / "nnUNet_raw")
    env["nnUNet_preprocessed"] = str(parameter_root / "nnUNet_preprocessed")
    env["nnUNet_results"] = str(parameter_root / "nnUNet_results")
    dev = str(device).lower().strip()
    if dev.startswith("cuda:"):
        env["CUDA_VISIBLE_DEVICES"] = dev.split(":", 1)[1]
    elif dev == "cpu":
        env["CUDA_VISIBLE_DEVICES"] = "-1"
    return env


def apply_nnunet_process_env(env: Dict[str, str], args: argparse.Namespace) -> Dict[str, str]:
    env["nnUNet_n_proc_DA"] = str(int(args.nnunet_n_proc_DA))
    env["nnUNet_def_n_proc"] = str(int(args.nnunet_def_n_proc))
    return env


def nnunet_stage_env(
    env: Mapping[str, str],
    *,
    n_proc_da: int | None = None,
    def_n_proc: int | None = None,
) -> Dict[str, str]:
    stage_env = dict(env)
    if n_proc_da is not None:
        stage_env["nnUNet_n_proc_DA"] = str(int(n_proc_da))
    if def_n_proc is not None:
        stage_env["nnUNet_def_n_proc"] = str(int(def_n_proc))
    return stage_env


def dependency_status() -> Dict[str, Any]:
    commands = ["nnUNetv2_plan_and_preprocess", "nnUNetv2_train", "nnUNetv2_predict"]
    return {
        "nnunetv2_importable": importlib.util.find_spec("nnunetv2") is not None,
        "commands": {cmd: shutil.which(cmd) for cmd in commands},
    }


def require_nnunetv2() -> None:
    status = dependency_status()
    missing = [cmd for cmd, path in status["commands"].items() if not path]
    if not status["nnunetv2_importable"] or missing:
        details = json.dumps(status, ensure_ascii=False, indent=2)
        raise RuntimeError(
            "nnU-Net v2 is not available in the current Python environment.\n"
            "Install it in Project1 first, for example:\n"
            "  conda run -n Project1 python -m pip install nnunetv2\n"
            f"Dependency status:\n{details}"
        )


def write_2d_mha(path: Path, array: np.ndarray, *, spacing_zy: Sequence[float], is_label: bool) -> None:
    ensure_dir(path.parent)
    arr = np.asarray(array)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D array for {path}, got shape={arr.shape}")
    if is_label:
        itk = sitk.GetImageFromArray(arr.astype(np.uint8))
    else:
        itk = sitk.GetImageFromArray(arr.astype(np.float32))
    if spacing_zy and len(tuple(spacing_zy)) >= 2:
        # SimpleITK 2D spacing order is x/y, while arrays are z/y.
        itk.SetSpacing((float(spacing_zy[1]), float(spacing_zy[0])))
    sitk.WriteImage(itk, str(path))


def read_prediction_mha(path: Path) -> np.ndarray:
    if not path.exists():
        raise FileNotFoundError(f"Missing nnU-Net prediction: {path}")
    arr = sitk.GetArrayFromImage(sitk.ReadImage(str(path)))
    arr = np.asarray(arr)
    if arr.ndim == 3 and 1 in arr.shape:
        arr = np.squeeze(arr)
    if arr.ndim != 2:
        raise ValueError(f"Expected 2D prediction in {path}, got shape={arr.shape}")
    return np.clip(arr, 0, 2).astype(np.int64)


def nnunet_metadata(args: argparse.Namespace, dataset_id: int, dataset_name: str) -> Dict[str, Any]:
    return {
        "method": METHOD,
        "sota_method": METHOD,
        "nnunet_dataset_id": int(dataset_id),
        "nnunet_dataset_name": str(dataset_name),
        "nnunet_config": str(args.nnunet_config),
        "nnunet_fold": str(args.nnunet_fold),
        "nnunet_trainer": str(args.nnunet_trainer),
        "parameter_root": str(resolve_path(args.parameter_root)),
        "model_parameters_saved": True,
        "checkpoint_source": "nnunetv2_results",
        "slice_policy": str(args.slice_policy),
        "num_middle_slices": int(args.num_middle_slices),
        "filter_min_fg": bool(args.filter_min_fg),
        "min_fg_ratio": float(args.min_fg_ratio),
        "resize_hw": int(args.resize_hw),
    }


def build_train_data(args: argparse.Namespace, shot: int, seed: int):
    data_root = resolve_spider_data_root(args.data_root)
    return build_train_dataset(
        SOURCE_DOMAIN_NAME,
        int(shot),
        int(seed),
        data_root=data_root,
        split_csv=split_csv_for_data_root(data_root),
        min_fg_ratio=float(args.min_fg_ratio),
        resize_hw=int(args.resize_hw),
        use_split=True,
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )


def build_eval_data(args: argparse.Namespace, domain: str, train_case_ids: Sequence[str]):
    data_root = resolve_spider_data_root(args.data_root)
    return build_test_dataset(
        domain,
        exclude_case_ids=train_case_ids,
        data_root=data_root,
        min_fg_ratio=float(args.min_fg_ratio),
        resize_hw=int(args.resize_hw),
        max_cases=int(args.max_test_cases) if int(args.max_test_cases) > 0 else None,
        use_split=False,
        slice_policy=str(args.slice_policy),
        num_middle_slices=int(args.num_middle_slices),
        filter_min_fg=bool(args.filter_min_fg),
    )


def prepare_nnunet_train_dataset(
    *,
    args: argparse.Namespace,
    parameter_root: Path,
    shot: int,
    seed: int,
    run_dir: Path,
) -> Dict[str, Any]:
    dataset_id = dataset_id_for(int(args.dataset_id_base), int(shot), int(seed))
    dataset_name = dataset_name_for(dataset_id, int(shot), int(seed))
    raw_dataset_dir = parameter_root / "nnUNet_raw" / dataset_name
    images_tr = raw_dataset_dir / "imagesTr"
    labels_tr = raw_dataset_dir / "labelsTr"

    if raw_dataset_dir.exists() and bool(args.overwrite):
        shutil.rmtree(raw_dataset_dir)
    ensure_dir(images_tr)
    ensure_dir(labels_tr)

    train_dataset = build_train_data(args, int(shot), int(seed))
    manifest_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(train_dataset.items):
        case_id = str(item["case_id"])
        x_idx = int(item["sagittal_x_index"])
        slice_id = f"s{int(shot)}_seed{int(seed)}_case{slugify(case_id)}_x{x_idx:04d}_{idx:04d}"
        image_path = images_tr / f"{slice_id}_0000.mha"
        label_path = labels_tr / f"{slice_id}.mha"
        write_2d_mha(image_path, item["image"], spacing_zy=item["slice_spacing"], is_label=False)
        write_2d_mha(label_path, item["mask"], spacing_zy=item["slice_spacing"], is_label=True)
        manifest_rows.append(
            {
                "nnunet_case_id": slice_id,
                "case_id": case_id,
                "file": item["file"],
                "domain": item["domain"],
                "sagittal_x_index": x_idx,
                "slice_position": int(item.get("slice_position", idx)),
                "foreground_ratio": float(item["foreground_ratio"]),
                "image_path": str(image_path),
                "label_path": str(label_path),
            }
        )

    dataset_json = {
        "channel_names": {"0": "MRI"},
        "labels": {"background": 0, "vertebrae": 1, "intervertebral_discs": 2},
        "numTraining": int(len(manifest_rows)),
        "file_ending": ".mha",
        "overwrite_image_reader_writer": "SimpleITKIO",
    }
    write_json(raw_dataset_dir / "dataset.json", dataset_json)
    write_csv(run_dir / "nnunet_train_manifest.csv", manifest_rows)

    metadata = {
        **nnunet_metadata(args, dataset_id, dataset_name),
        "source_domain": SOURCE_DOMAIN_NAME,
        "shot": int(shot),
        "seed": int(seed),
        "train_case_ids": [str(x) for x in train_dataset.selected_case_ids],
        "n_train_slices": int(len(train_dataset)),
        "slice_indices_by_case": train_dataset.slice_indices_by_case(),
        "raw_dataset_dir": str(raw_dataset_dir),
        "imagesTr": str(images_tr),
        "labelsTr": str(labels_tr),
        "dataset_json": str(raw_dataset_dir / "dataset.json"),
        "output_dir": str(run_dir),
    }
    write_json(run_dir / "dataset_metadata.json", metadata)
    return metadata


def command_to_string(cmd: Sequence[str]) -> str:
    return " ".join(str(x) for x in cmd)


def run_command(
    cmd: Sequence[str],
    *,
    env: Mapping[str, str],
    log_path: Path,
    cwd: Path = PROJECT_ROOT,
) -> Dict[str, Any]:
    ensure_dir(log_path.parent)
    started = time.time()
    with log_path.open("w", encoding="utf-8", errors="replace") as log:
        log.write(f"$ {command_to_string(cmd)}\n\n")
        log.flush()
        proc = subprocess.run(
            list(cmd),
            cwd=str(cwd),
            env=dict(env),
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
    elapsed = time.time() - started
    return {
        "command": command_to_string(cmd),
        "returncode": int(proc.returncode),
        "elapsed_sec": float(elapsed),
        "log_path": str(log_path),
    }


def run_nnunet_training(
    *,
    args: argparse.Namespace,
    env: Mapping[str, str],
    metadata: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    dataset_id = str(int(metadata["nnunet_dataset_id"]))
    config = str(args.nnunet_config)
    trainer = str(args.nnunet_trainer)
    fold = str(args.nnunet_fold)
    command_rows: List[Dict[str, Any]] = []

    plan_cmd = [
        "nnUNetv2_plan_and_preprocess",
        "-d",
        dataset_id,
        "--verify_dataset_integrity",
        "-c",
        config,
        "-np",
        str(int(args.nnunet_preprocess_np)),
    ]
    plan_env = nnunet_stage_env(env, n_proc_da=max(1, int(args.nnunet_n_proc_DA)))
    plan_row = run_command(plan_cmd, env=plan_env, log_path=run_dir / "nnunet_plan_and_preprocess.log")
    command_rows.append({"stage": "plan_and_preprocess", **plan_row})
    if plan_row["returncode"] != 0:
        write_csv(run_dir / "nnunet_commands.csv", command_rows)
        raise RuntimeError(f"nnU-Net plan/preprocess failed for dataset {dataset_id}; see {plan_row['log_path']}")

    train_cmd = ["nnUNetv2_train", dataset_id, config, fold, "-tr", trainer]
    train_env = nnunet_stage_env(env, n_proc_da=int(args.nnunet_n_proc_DA))
    train_row = run_command(train_cmd, env=train_env, log_path=run_dir / "nnunet_train.log")
    command_rows.append({"stage": "train", **train_row})
    write_csv(run_dir / "nnunet_commands.csv", command_rows)
    if train_row["returncode"] != 0:
        raise RuntimeError(f"nnU-Net training failed for dataset {dataset_id}; see {train_row['log_path']}")

    return {
        **metadata,
        "loss_mode": "official_nnunetv2_default",
        "train_status": "completed",
        "plan_command": plan_row["command"],
        "plan_returncode": plan_row["returncode"],
        "plan_elapsed_sec": plan_row["elapsed_sec"],
        "plan_log_path": plan_row["log_path"],
        "train_command": train_row["command"],
        "train_returncode": train_row["returncode"],
        "train_elapsed_sec": train_row["elapsed_sec"],
        "train_log_path": train_row["log_path"],
        "plan_n_proc_DA": int(plan_env["nnUNet_n_proc_DA"]),
        "train_n_proc_DA": int(train_env["nnUNet_n_proc_DA"]),
    }


def prepare_prediction_input(
    *,
    args: argparse.Namespace,
    parameter_root: Path,
    shot: int,
    seed: int,
    domain: str,
    train_case_ids: Sequence[str],
) -> Dict[str, Any]:
    dataset = build_eval_data(args, domain, train_case_ids)
    domain_slug = slugify(dataset.domain_path.name)
    input_dir = parameter_root / "prediction_inputs" / f"shot{int(shot)}" / f"Seed{int(seed)}" / domain_slug
    if input_dir.exists() and bool(args.overwrite):
        shutil.rmtree(input_dir)
    ensure_dir(input_dir)

    manifest_rows: List[Dict[str, Any]] = []
    for idx, item in enumerate(dataset.items):
        case_id = str(item["case_id"])
        x_idx = int(item["sagittal_x_index"])
        nnunet_case_id = f"{domain_slug}_case{slugify(case_id)}_x{x_idx:04d}_{idx:05d}"
        image_path = input_dir / f"{nnunet_case_id}_0000.mha"
        write_2d_mha(image_path, item["image"], spacing_zy=item["slice_spacing"], is_label=False)
        manifest_rows.append(
            {
                "nnunet_case_id": nnunet_case_id,
                "case_id": case_id,
                "file": item["file"],
                "domain": item["domain"],
                "sagittal_x_index": x_idx,
                "slice_position": int(item.get("slice_position", idx)),
                "foreground_ratio": float(item["foreground_ratio"]),
                "image_path": str(image_path),
            }
        )
    return {
        "dataset": dataset,
        "domain": dataset.domain_path.name,
        "domain_slug": domain_slug,
        "input_dir": input_dir,
        "manifest_rows": manifest_rows,
    }


def predictions_complete(output_dir: Path, manifest_rows: Sequence[Mapping[str, Any]]) -> bool:
    if not output_dir.exists():
        return False
    return all((output_dir / f"{row['nnunet_case_id']}.mha").exists() for row in manifest_rows)


def predict_domain(
    *,
    args: argparse.Namespace,
    env: Mapping[str, str],
    parameter_root: Path,
    metadata: Mapping[str, Any],
    prediction_info: Mapping[str, Any],
    run_dir: Path,
) -> Dict[str, Any]:
    dataset_id = str(int(metadata["nnunet_dataset_id"]))
    config = str(args.nnunet_config)
    trainer = str(args.nnunet_trainer)
    fold = str(args.nnunet_fold)
    domain_slug = str(prediction_info["domain_slug"])
    output_dir = parameter_root / "predictions" / f"shot{int(metadata['shot'])}" / f"Seed{int(metadata['seed'])}" / domain_slug
    if output_dir.exists() and bool(args.overwrite) and not bool(args.skip_existing_predictions):
        shutil.rmtree(output_dir)
    ensure_dir(output_dir)

    manifest_rows = prediction_info["manifest_rows"]
    if bool(args.skip_existing_predictions) and predictions_complete(output_dir, manifest_rows):
        return {
            "predict_status": "skipped_existing",
            "predict_command": "",
            "predict_returncode": 0,
            "predict_elapsed_sec": 0.0,
            "predict_log_path": "",
            "prediction_dir": str(output_dir),
        }

    cmd = [
        "nnUNetv2_predict",
        "-i",
        str(prediction_info["input_dir"]),
        "-o",
        str(output_dir),
        "-d",
        dataset_id,
        "-c",
        config,
        "-f",
        fold,
        "-tr",
        trainer,
        "-npp",
        str(int(args.nnunet_predict_npp)),
        "-nps",
        str(int(args.nnunet_predict_nps)),
    ]
    log_path = run_dir / f"nnunet_predict_{domain_slug}.log"
    predict_env = nnunet_stage_env(env, n_proc_da=max(1, int(args.nnunet_n_proc_DA)))
    row = run_command(cmd, env=predict_env, log_path=log_path)
    if row["returncode"] != 0:
        raise RuntimeError(f"nnU-Net predict failed for domain {prediction_info['domain']}; see {row['log_path']}")
    return {
        "predict_status": "completed",
        "predict_command": row["command"],
        "predict_returncode": row["returncode"],
        "predict_elapsed_sec": row["elapsed_sec"],
        "predict_log_path": row["log_path"],
        "prediction_dir": str(output_dir),
        "predict_n_proc_DA": int(predict_env["nnUNet_n_proc_DA"]),
    }


def _finite_metric_mean(values: Sequence[float]) -> float:
    xs = [float(x) for x in values if np.isfinite(float(x))]
    return float(np.mean(xs)) if xs else float("nan")


def _add_class_metric_columns(row: Dict[str, Any], prefix: str, values: Mapping[int, float]) -> None:
    for cls, value in values.items():
        name = FOREGROUND_CLASS_NAMES.get(int(cls), f"class_{int(cls)}")
        row[f"{prefix}_{name}"] = float(value)


def evaluate_predictions(
    *,
    prediction_info: Mapping[str, Any],
    predict_info: Mapping[str, Any],
    base_meta: Mapping[str, Any],
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    dataset = prediction_info["dataset"]
    manifest_by_id = {row["nnunet_case_id"]: row for row in prediction_info["manifest_rows"]}
    prediction_dir = Path(predict_info["prediction_dir"])
    pred_by_case: Dict[str, Dict[int, np.ndarray]] = {}
    for nnunet_case_id, row in manifest_by_id.items():
        pred = read_prediction_mha(prediction_dir / f"{nnunet_case_id}.mha")
        pred_by_case.setdefault(str(row["case_id"]), {})[int(row["sagittal_x_index"])] = pred

    case_rows: List[Dict[str, Any]] = []
    for case_id in dataset.grouped_case_ids():
        items = dataset.items_for_case(case_id)
        gt_stack = np.stack([item["mask"] for item in items], axis=0).astype(np.int64)
        pred_stack = np.stack(
            [pred_by_case[str(case_id)][int(item["sagittal_x_index"])] for item in items],
            axis=0,
        ).astype(np.int64)
        case_spacing = items[0]["case_spacing"]
        slice_spacing = items[0]["slice_spacing"]
        case_dice_cls = case_dice_by_class(pred_stack, gt_stack)
        case_hd95_cls = case_hd95_by_class(pred_stack, gt_stack, spacing=case_spacing)
        slice_dice_cls = slice_dice_by_class(pred_stack, gt_stack)
        slice_hd95_cls = slice_hd95_by_class(pred_stack, gt_stack, spacing=slice_spacing)
        row = {
            **base_meta,
            "domain": dataset.domain_path.name,
            "case_id": str(case_id),
            "n_slices": int(len(items)),
            "sagittal_x_indices": "|".join(str(int(item["sagittal_x_index"])) for item in items),
            "case_dice": _finite_metric_mean(list(case_dice_cls.values())),
            "case_hd95": _finite_metric_mean(list(case_hd95_cls.values())),
            "slice_dice": _finite_metric_mean(list(slice_dice_cls.values())),
            "slice_hd95": _finite_metric_mean(list(slice_hd95_cls.values())),
        }
        _add_class_metric_columns(row, "case_dice", case_dice_cls)
        _add_class_metric_columns(row, "case_hd95", case_hd95_cls)
        _add_class_metric_columns(row, "slice_dice", slice_dice_cls)
        _add_class_metric_columns(row, "slice_hd95", slice_hd95_cls)
        case_rows.append(row)

    summary = {
        "case_dice": _finite_metric_mean([row["case_dice"] for row in case_rows]),
        "case_hd95": _finite_metric_mean([row["case_hd95"] for row in case_rows]),
        "slice_dice": _finite_metric_mean([row["slice_dice"] for row in case_rows]),
        "slice_hd95": _finite_metric_mean([row["slice_hd95"] for row in case_rows]),
    }
    for prefix in ("case_dice", "case_hd95", "slice_dice", "slice_hd95"):
        for _cls, name in FOREGROUND_CLASS_NAMES.items():
            summary[f"{prefix}_{name}"] = _finite_metric_mean([row[f"{prefix}_{name}"] for row in case_rows])

    eval_row = {
        **base_meta,
        "domain": dataset.domain_path.name,
        "n_cases": int(len(case_rows)),
        "n_slices": int(len(dataset)),
        "excluded_case_ids": "|".join(base_meta.get("train_case_ids", [])),
        "prediction_dir": predict_info["prediction_dir"],
        "predict_status": predict_info["predict_status"],
        "predict_command": predict_info["predict_command"],
        "predict_log_path": predict_info["predict_log_path"],
    }
    eval_row.update(summary)
    return eval_row, case_rows


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
                "nnunet_config": rows[0].get("nnunet_config", ""),
                "nnunet_fold": rows[0].get("nnunet_fold", ""),
                "nnunet_trainer": rows[0].get("nnunet_trainer", ""),
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
    train_by_run = {(int(row["shot"]), int(row["seed"])): row for row in training_rows if "shot" in row and "seed" in row}
    groups: Dict[tuple[int, int], List[Mapping[str, Any]]] = {}
    for row in eval_rows:
        groups.setdefault((int(row["shot"]), int(row["seed"])), []).append(row)

    out: List[Dict[str, Any]] = []
    for (shot, seed), rows in sorted(groups.items()):
        train_row = train_by_run.get((shot, seed), {})
        target_rows = [row for row in rows if row.get("domain", "") != SOURCE_DOMAIN_NAME]
        out.append(
            {
                "method": METHOD,
                "sota_method": METHOD,
                "shot": int(shot),
                "seed": int(seed),
                "nnunet_dataset_id": train_row.get("nnunet_dataset_id", rows[0].get("nnunet_dataset_id", "")),
                "nnunet_dataset_name": train_row.get("nnunet_dataset_name", rows[0].get("nnunet_dataset_name", "")),
                "nnunet_config": rows[0].get("nnunet_config", ""),
                "nnunet_fold": rows[0].get("nnunet_fold", ""),
                "nnunet_trainer": rows[0].get("nnunet_trainer", ""),
                "train_status": train_row.get("train_status", ""),
                "plan_elapsed_sec": train_row.get("plan_elapsed_sec", float("nan")),
                "train_elapsed_sec": train_row.get("train_elapsed_sec", float("nan")),
                "mean_6domain_case_dice": finite_mean(row.get("case_dice") for row in rows),
                "mean_6domain_case_hd95": finite_mean(row.get("case_hd95") for row in rows),
                "mean_6domain_slice_dice": finite_mean(row.get("slice_dice") for row in rows),
                "mean_6domain_slice_hd95": finite_mean(row.get("slice_hd95") for row in rows),
                "mean_5target_case_dice": finite_mean(row.get("case_dice") for row in target_rows),
                "mean_5target_case_hd95": finite_mean(row.get("case_hd95") for row in target_rows),
                "mean_5target_slice_dice": finite_mean(row.get("slice_dice") for row in target_rows),
                "mean_5target_slice_hd95": finite_mean(row.get("slice_hd95") for row in target_rows),
                "model_parameters_saved": True,
                "checkpoint_source": "nnunetv2_results",
            }
        )
    return out


def write_analysis_report(result_root: Path, overall_rows: Sequence[Mapping[str, Any]]) -> None:
    lines = [
        "# nnU-Net v2 SoTA Results",
        "",
        "Training uses official nnU-Net v2 default `nnUNetTrainer` on 2D sagittal slices.",
        "Evaluation is aligned with the DSU/FDA/MixStyle slice protocol and reports helper-style case/slice Dice and HD95.",
        "Model parameters are saved under the configured `parameter_root`.",
        "",
        "## Overall 15-run Domain Summary",
        "",
        "| Domain | Dice | HD95 | n_rows | n_cases |",
        "|---|---:|---:|---:|---:|",
    ]
    for row in overall_rows:
        lines.append(
            "| "
            + " | ".join(
                [
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
            "- `training_curves.csv`: one row per shot/seed official nnU-Net run with command logs and status.",
            "- `eval_metrics.csv`: domain-level Dice/HD95.",
            "- `eval_case_metrics.csv`: case-level Dice/HD95.",
            "- `experiment_summary.csv`: one row per shot/seed run.",
            "- `domain_5seed_summary.csv`: per shot/domain 5-seed summary.",
            "- `overall_15run_domain_summary.csv`: per-domain summary over 3 shots x 5 seeds.",
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
    domain_5seed = summarize_eval_groups(eval_rows, ("shot", "domain"))
    overall = summarize_eval_groups(eval_rows, ("domain",))
    write_csv(result_root / "experiment_summary.csv", experiments)
    write_csv(result_root / "domain_5seed_summary.csv", domain_5seed)
    write_csv(result_root / "overall_15run_domain_summary.csv", overall)
    write_analysis_report(result_root, overall)


def run_one(
    *,
    args: argparse.Namespace,
    env: Mapping[str, str],
    parameter_root: Path,
    result_root: Path,
    shot: int,
    seed: int,
) -> tuple[Dict[str, Any] | None, List[Dict[str, Any]], List[Dict[str, Any]]]:
    run_dir = ensure_dir(run_dir_for(result_root, int(shot), int(seed)))
    metadata = prepare_nnunet_train_dataset(
        args=args,
        parameter_root=parameter_root,
        shot=int(shot),
        seed=int(seed),
        run_dir=run_dir,
    )
    write_json(run_dir / "run_config.json", {**vars(args), **metadata})
    print(
        f"[PREPARE] {METHOD} shot={shot} seed={seed} "
        f"dataset={metadata['nnunet_dataset_name']} slices={metadata['n_train_slices']} "
        f"cases={metadata['train_case_ids']}",
        flush=True,
    )
    if bool(args.prepare_only):
        return None, [], []

    train_row = run_nnunet_training(args=args, env=env, metadata=metadata, run_dir=run_dir)
    eval_rows: List[Dict[str, Any]] = []
    case_rows: List[Dict[str, Any]] = []
    for domain in DOMAIN_NAMES:
        prediction_info = prepare_prediction_input(
            args=args,
            parameter_root=parameter_root,
            shot=int(shot),
            seed=int(seed),
            domain=domain,
            train_case_ids=metadata["train_case_ids"],
        )
        domain_manifest_path = run_dir / f"nnunet_predict_manifest_{prediction_info['domain_slug']}.csv"
        write_csv(domain_manifest_path, prediction_info["manifest_rows"])
        predict_info = predict_domain(
            args=args,
            env=env,
            parameter_root=parameter_root,
            metadata=metadata,
            prediction_info=prediction_info,
            run_dir=run_dir,
        )
        base_meta = {
            **nnunet_metadata(args, int(metadata["nnunet_dataset_id"]), str(metadata["nnunet_dataset_name"])),
            "loss_mode": "official_nnunetv2_default",
            "shot": int(shot),
            "seed": int(seed),
            "train_case_ids": [str(x) for x in metadata["train_case_ids"]],
            "output_dir": str(run_dir),
        }
        eval_row, domain_case_rows = evaluate_predictions(
            prediction_info=prediction_info,
            predict_info=predict_info,
            base_meta=base_meta,
        )
        eval_rows.append(eval_row)
        case_rows.extend(domain_case_rows)
        print(
            f"[EVAL] {METHOD} shot={shot} seed={seed} domain={eval_row['domain']} "
            f"case_dice={float(eval_row['case_dice']):.6f} case_hd95={float(eval_row['case_hd95']):.6f} "
            f"slice_dice={float(eval_row['slice_dice']):.6f} slice_hd95={float(eval_row['slice_hd95']):.6f}",
            flush=True,
        )
    return train_row, eval_rows, case_rows


def run(args: argparse.Namespace) -> None:
    if bool(args.check_dependency_only):
        status = dependency_status()
        print(json.dumps(status, ensure_ascii=False, indent=2), flush=True)
        if not status["nnunetv2_importable"] or any(path is None for path in status["commands"].values()):
            raise SystemExit(1)
        return

    args.data_root = str(resolve_spider_data_root(args.data_root))
    result_root = reset_or_prepare_result_root(resolve_path(args.result_root), bool(args.overwrite))
    parameter_root = prepare_parameter_root(resolve_path(args.parameter_root), bool(args.overwrite))
    shots = parse_int_list(args.shots)
    seeds = parse_int_list(args.seeds)
    env = apply_nnunet_process_env(nnunet_env(parameter_root, str(args.device)), args)

    if not bool(args.prepare_only):
        require_nnunetv2()

    write_json(
        result_root / "run_config.json",
        {
            **vars(args),
            "method": METHOD,
            "training_mode": "official_nnunetv2_default",
            "model_parameters_saved": True,
            "parameter_root": str(parameter_root),
            "nnUNet_raw": env["nnUNet_raw"],
            "nnUNet_preprocessed": env["nnUNet_preprocessed"],
            "nnUNet_results": env["nnUNet_results"],
            "nnUNet_n_proc_DA": env["nnUNet_n_proc_DA"],
            "nnUNet_def_n_proc": env["nnUNet_def_n_proc"],
            "nnUNet_plan_n_proc_DA": max(1, int(args.nnunet_n_proc_DA)),
            "nnUNet_train_n_proc_DA": int(args.nnunet_n_proc_DA),
            "nnUNet_predict_n_proc_DA": max(1, int(args.nnunet_n_proc_DA)),
        },
    )

    all_training_rows: List[Dict[str, Any]] = []
    all_eval_rows: List[Dict[str, Any]] = []
    all_case_rows: List[Dict[str, Any]] = []

    print(f"[METHOD] {METHOD}", flush=True)
    print(f"[DATA_ROOT] {args.data_root}", flush=True)
    print(f"[RESULT_ROOT] {result_root}", flush=True)
    print(f"[PARAMETER_ROOT] {parameter_root}", flush=True)
    print(f"[NNUNET_RAW] {env['nnUNet_raw']}", flush=True)
    print(f"[NNUNET_PREPROCESSED] {env['nnUNet_preprocessed']}", flush=True)
    print(f"[NNUNET_RESULTS] {env['nnUNet_results']}", flush=True)
    print(
        f"[NNUNET_PROCESSES] n_proc_DA={env['nnUNet_n_proc_DA']} "
        f"def_n_proc={env['nnUNet_def_n_proc']} preprocess_np={args.nnunet_preprocess_np} "
        f"predict_npp={args.nnunet_predict_npp} predict_nps={args.nnunet_predict_nps}",
        flush=True,
    )
    print(f"[RUNS] shots={shots} seeds={seeds} config={args.nnunet_config} fold={args.nnunet_fold}", flush=True)

    for shot in shots:
        for seed in seeds:
            train_row, eval_rows, case_rows = run_one(
                args=args,
                env=env,
                parameter_root=parameter_root,
                result_root=result_root,
                shot=int(shot),
                seed=int(seed),
            )
            if train_row is not None:
                all_training_rows.append(train_row)
                all_eval_rows.extend(eval_rows)
                all_case_rows.extend(case_rows)
                write_all_outputs(
                    result_root,
                    training_rows=all_training_rows,
                    eval_rows=all_eval_rows,
                    case_rows=all_case_rows,
                )

    if bool(args.prepare_only):
        write_csv(result_root / "training_curves.csv", all_training_rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Official nnU-Net v2 on SPIDER 2D sagittal few-shot splits.")
    parser.add_argument(
        "--data_root",
        default=default_data_root(),
        help=(
            "SPIDER data root. Accepts either the Spine folder or the "
            "SPIDER_domain_strict_3Foreground folder. On the Imperial server, "
            "use /rds/general/user/hs225/ephemeral/TTA_Project_data/dataset/Spine."
        ),
    )
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--shots", default="3,4,5")
    parser.add_argument("--seeds", default="0,1,2,3,4")
    parser.add_argument("--result_root", default=str(DEFAULT_RESULT_ROOT))
    parser.add_argument("--parameter_root", default=str(DEFAULT_PARAMETER_ROOT))
    parser.add_argument("--dataset_id_base", type=int, default=910)
    parser.add_argument("--nnunet_config", default="2d")
    parser.add_argument("--nnunet_fold", default="all")
    parser.add_argument("--nnunet_trainer", default="nnUNetTrainer")
    parser.add_argument("--nnunet_n_proc_DA", type=int, default=0)
    parser.add_argument("--nnunet_def_n_proc", type=int, default=1)
    parser.add_argument("--nnunet_preprocess_np", type=int, default=1)
    parser.add_argument("--nnunet_predict_npp", type=int, default=1)
    parser.add_argument("--nnunet_predict_nps", type=int, default=1)
    parser.add_argument("--resize_hw", type=int, default=224)
    parser.add_argument("--min_fg_ratio", type=float, default=0.05)
    parser.add_argument("--filter_min_fg", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--slice_policy", default="all_filtered", choices=("center9", "all", "all_filtered"))
    parser.add_argument("--num_middle_slices", type=int, default=9)
    parser.add_argument("--eval_batch_size", type=int, default=4)
    parser.add_argument("--max_test_cases", type=int, default=0)
    parser.add_argument("--prepare_only", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--skip_existing_predictions", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--overwrite", action=argparse.BooleanOptionalAction, default=False)
    parser.add_argument("--check_dependency_only", action=argparse.BooleanOptionalAction, default=False)
    return parser.parse_args()


def main() -> None:
    try:
        run(parse_args())
    except RuntimeError as exc:
        print(f"[ERROR] {exc}", file=sys.stderr, flush=True)
        raise


if __name__ == "__main__":
    main()
