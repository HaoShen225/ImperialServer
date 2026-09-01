"""Three-dimensional metrics on the locked 256 x 256 evaluation grid."""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt


SLICE_METRIC_POLICY = {
    "metrics": ["dice", "hd95_2d_px"],
    "all_slices_absent_class": "both_absent_perfect_one_absent_diagonal_penalty",
    "views": ["all_slices", "foreground_present"],
    "point_estimate_weighting": "slice_equal",
    "confidence_interval": "patient_cluster_bootstrap",
}


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


def hd95_2d_px(prediction: np.ndarray, target: np.ndarray) -> float:
    """Symmetric 95th-percentile surface distance on one 2D pixel grid."""
    prediction = np.asarray(prediction, dtype=bool)
    target = np.asarray(target, dtype=bool)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("2D HD95 expects equal-shape 2D arrays")
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


def evaluate_slice(
    prediction: np.ndarray,
    target: np.ndarray,
    classes: Iterable[int] = (1, 2, 3),
    class_names: dict[int, str] | None = None,
) -> tuple[dict[str, float], dict[str, bool]]:
    """Evaluate one [H,W] slice and expose GT-presence for conditional reporting."""
    prediction = np.asarray(prediction)
    target = np.asarray(target)
    if prediction.shape != target.shape or prediction.ndim != 2:
        raise ValueError("Prediction and target must have equal [H,W] shape")
    names = class_names or {1: "rv", 2: "myo", 3: "lv"}
    result: dict[str, float] = {}
    gt_present: dict[str, bool] = {}
    dice_values, hd_values = [], []
    for class_id in classes:
        name = names[int(class_id)]
        pred_class, target_class = prediction == class_id, target == class_id
        dice = dice_score(pred_class, target_class)
        hd = hd95_2d_px(pred_class, target_class)
        result[f"dice_{name}"] = dice
        result[f"hd95_2d_px_{name}"] = hd
        gt_present[name] = bool(target_class.any())
        dice_values.append(dice)
        hd_values.append(hd)
    result["dice_macro"] = float(np.mean(dice_values))
    result["hd95_2d_px_macro"] = float(np.mean(hd_values))
    return result, gt_present


def _slice_bootstrap_statistics(
    values_by_patient: dict[str, list[float]],
    bootstrap_resamples: int,
    rng: np.random.Generator,
) -> tuple[float, float, float, int, int, int]:
    patients = sorted(values_by_patient)
    values = [value for patient in patients for value in values_by_patient[patient]]
    if not values:
        raise ValueError("Cannot aggregate a slice metric without eligible slices")
    sums = np.asarray([sum(values_by_patient[patient]) for patient in patients], dtype=np.float64)
    counts = np.asarray([len(values_by_patient[patient]) for patient in patients], dtype=np.int64)
    sampled = rng.integers(0, len(patients), size=(bootstrap_resamples, len(patients)))
    denominators = counts[sampled].sum(axis=1)
    valid = denominators > 0
    boot = sums[sampled].sum(axis=1)[valid] / denominators[valid]
    return (
        float(np.mean(values)),
        float(np.percentile(boot, 2.5)),
        float(np.percentile(boot, 97.5)),
        len(values),
        len(patients),
        int(np.count_nonzero(counts)),
    )


def aggregate_slice_results(
    records: list[dict[str, Any]],
    bootstrap_resamples: int = 10_000,
    seed: int = 42,
) -> dict[str, Any]:
    """Slice-weighted summaries with patient-clustered confidence intervals."""
    if not records:
        raise ValueError("Cannot aggregate an empty slice result list")
    metric_names = sorted(records[0]["metrics"])
    class_names = sorted(records[0]["gt_present"])
    patient_ids = sorted({str(record["patient_id"]) for record in records})

    def summarize(values_by_patient: dict[str, list[float]], metric_seed: int) -> dict[str, float | int]:
        mean, low, high, n_slices, n_patients, n_eligible_patients = _slice_bootstrap_statistics(
            values_by_patient,
            bootstrap_resamples,
            np.random.default_rng(metric_seed),
        )
        return {
            "mean": mean,
            "ci95_low": low,
            "ci95_high": high,
            "n_slices": n_slices,
            "n_patients": n_patients,
            "n_eligible_patients": n_eligible_patients,
        }

    all_slices: dict[str, dict[str, float | int]] = {}
    for metric_index, metric_name in enumerate(metric_names):
        grouped: dict[str, list[float]] = {patient: [] for patient in patient_ids}
        for record in records:
            grouped[str(record["patient_id"])].append(float(record["metrics"][metric_name]))
        all_slices[metric_name] = summarize(grouped, seed + metric_index)

    foreground: dict[str, dict[str, float | int]] = {}
    for class_index, class_name in enumerate(class_names):
        for family_index, family in enumerate(("dice", "hd95_2d_px")):
            metric_name = f"{family}_{class_name}"
            grouped = {patient: [] for patient in patient_ids}
            for record in records:
                if bool(record["gt_present"][class_name]):
                    grouped[str(record["patient_id"])].append(float(record["metrics"][metric_name]))
            metric_seed = seed + 100 + class_index * 10 + family_index
            foreground[metric_name] = summarize(grouped, metric_seed)

    for family in ("dice", "hd95_2d_px"):
        names = [f"{family}_{class_name}" for class_name in class_names]
        class_means = [float(foreground[name]["mean"]) for name in names]
        rng = np.random.default_rng(seed + (2000 if family == "dice" else 3000))
        sampled = rng.integers(
            0, len(patient_ids), size=(bootstrap_resamples, len(patient_ids))
        )
        class_boot = []
        for class_name, metric_name in zip(class_names, names):
            grouped = {patient: [] for patient in patient_ids}
            for record in records:
                if bool(record["gt_present"][class_name]):
                    grouped[str(record["patient_id"])].append(
                        float(record["metrics"][metric_name])
                    )
            sums = np.asarray([sum(grouped[patient]) for patient in patient_ids])
            counts = np.asarray([len(grouped[patient]) for patient in patient_ids])
            denominators = counts[sampled].sum(axis=1)
            values = np.full(bootstrap_resamples, np.nan, dtype=np.float64)
            valid = denominators > 0
            values[valid] = sums[sampled].sum(axis=1)[valid] / denominators[valid]
            class_boot.append(values)
        macro_boot = np.nanmean(np.stack(class_boot), axis=0)
        macro_boot = macro_boot[np.isfinite(macro_boot)]
        foreground[f"{family}_macro"] = {
            "mean": float(np.mean(class_means)),
            "ci95_low": float(np.percentile(macro_boot, 2.5)),
            "ci95_high": float(np.percentile(macro_boot, 97.5)),
            "n_class_slice_pairs": int(sum(int(foreground[name]["n_slices"]) for name in names)),
            "n_patients": len(patient_ids),
        }

    return {
        "aggregation_unit": "slice",
        "metric_policy": dict(SLICE_METRIC_POLICY),
        "point_estimate_weighting": SLICE_METRIC_POLICY["point_estimate_weighting"],
        "confidence_interval": SLICE_METRIC_POLICY["confidence_interval"],
        "all_slices": all_slices,
        "foreground_present": foreground,
    }


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
