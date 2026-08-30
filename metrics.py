"""Three-dimensional metrics on the locked 256 x 256 evaluation grid."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


def dice_score(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    denominator = int(prediction.sum()) + int(target.sum())
    if denominator == 0:
        return 1.0
    return 2.0 * float(np.logical_and(prediction, target).sum()) / denominator


def _diagonal_penalty(shape: tuple[int, ...]) -> float:
    return float(np.sqrt(sum(max(0, length - 1) ** 2 for length in shape)))


def hd95_px(prediction: np.ndarray, target: np.ndarray) -> float:
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("HD95 expects equal-shape 3D arrays")
    pred_any, target_any = bool(prediction.any()), bool(target.any())
    if not pred_any and not target_any:
        return 0.0
    if pred_any != target_any:
        return _diagonal_penalty(prediction.shape)
    pred_surface = np.logical_xor(prediction, binary_erosion(prediction, border_value=0))
    target_surface = np.logical_xor(target, binary_erosion(target, border_value=0))
    distance_to_target = distance_transform_edt(~target_surface)
    distance_to_pred = distance_transform_edt(~pred_surface)
    distances = np.concatenate([distance_to_target[pred_surface], distance_to_pred[target_surface]])
    return float(np.percentile(distances, 95))


def evaluate_volume(
    prediction: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int] = (1, 2, 3),
    class_names: dict[int, str] | None = None,
) -> dict[str, float]:
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 3:
        raise ValueError("Prediction and target must have equal [Z,H,W] shape")
    names = class_names or {1: "rv", 2: "myo", 3: "lv"}
    result: dict[str, float] = {}
    dice_values, hd_values = [], []
    for class_id in classes:
        name = names[int(class_id)]
        pred_class, target_class = prediction == class_id, target == class_id
        dice = dice_score(pred_class, target_class)
        hd = hd95_px(pred_class, target_class)
        result[f"dice_{name}"] = dice
        result[f"hd95_px_{name}"] = hd
        dice_values.append(dice)
        hd_values.append(hd)
    result["dice_macro"] = float(np.mean(dice_values))
    result["hd95_px_macro"] = float(np.mean(hd_values))
    return result


def aggregate_results(
    records: list[dict[str, Any]],
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, dict[str, float]]:
    if not records:
        raise ValueError("Cannot aggregate an empty result list")
    by_patient: dict[str, list[dict[str, float]]] = defaultdict(list)
    for record in records:
        by_patient[str(record["patient_id"])].append(record["metrics"])
    metric_names = sorted(by_patient[next(iter(by_patient))][0])
    patients = sorted(by_patient)
    matrix = np.asarray([
        [np.mean([volume[name] for volume in by_patient[patient]]) for name in metric_names]
        for patient in patients
    ], dtype=np.float64)
    rng = np.random.default_rng(seed)
    boot = np.empty((bootstrap_resamples, matrix.shape[1]), dtype=np.float64)
    for index in range(bootstrap_resamples):
        sampled = rng.integers(0, matrix.shape[0], size=matrix.shape[0])
        boot[index] = matrix[sampled].mean(axis=0)
    return {
        name: {
            "mean": float(matrix[:, column].mean()),
            "ci95_low": float(np.percentile(boot[:, column], 2.5)),
            "ci95_high": float(np.percentile(boot[:, column], 97.5)),
            "n_patients": len(patients),
        }
        for column, name in enumerate(metric_names)
    }
