from __future__ import annotations

import math

import numpy as np

from metrics import (
    aggregate_results,
    aggregate_slice_results,
    dice_score,
    evaluate_slice,
    evaluate_volume,
    hd95_2d_px,
    hd95_px,
)


def test_absent_class_conventions_and_diagonal_penalty():
    empty = np.zeros((3, 8, 8), dtype=bool)
    present = empty.copy()
    present[1, 3, 3] = True
    assert dice_score(empty, empty) == 1.0
    assert hd95_px(empty, empty) == 0.0
    assert dice_score(empty, present) == 0.0
    assert hd95_px(empty, present) == math.sqrt(2**2 + 7**2 + 7**2)


def test_hd95_is_in_isotropic_pixels():
    left = np.zeros((3, 16, 16), dtype=bool)
    right = np.zeros_like(left)
    left[1, 4:8, 4:8] = True
    right[1, 4:8, 5:9] = True
    assert hd95_px(left, right) == 1.0


def test_volume_metrics_and_patient_bootstrap():
    target = np.zeros((2, 8, 8), dtype=np.uint8)
    target[:, 1:3, 1:3] = 1
    target[:, 3:5, 3:5] = 2
    target[:, 5:7, 5:7] = 3
    metrics = evaluate_volume(target, target)
    assert metrics["dice_macro"] == 1.0
    assert metrics["hd95_px_macro"] == 0.0
    records = [
        {"patient_id": "p1", "metrics": metrics},
        {"patient_id": "p1", "metrics": metrics},
        {"patient_id": "p2", "metrics": metrics},
    ]
    summary = aggregate_results(records, bootstrap_resamples=20, seed=7)
    assert summary["dice_macro"]["mean"] == 1.0
    assert summary["dice_macro"]["n_patients"] == 2


def test_slice_metrics_use_two_dimensional_distance_and_presence():
    target = np.zeros((8, 8), dtype=np.uint8)
    target[2:5, 2:5] = 1
    prediction = target.copy()
    prediction[2:5, 2:5] = 0
    prediction[2:5, 3:6] = 1
    metrics, present = evaluate_slice(prediction, target)
    assert hd95_2d_px(prediction == 1, target == 1) == 1.0
    assert metrics["hd95_2d_px_rv"] == 1.0
    assert present == {"rv": True, "myo": False, "lv": False}
    assert metrics["dice_myo"] == 1.0
    assert metrics["hd95_2d_px_myo"] == 0.0


def test_slice_aggregation_reports_all_and_foreground_present_views():
    records = [
        {
            "patient_id": "p1",
            "metrics": {
                "dice_rv": 1.0, "dice_myo": 1.0, "dice_lv": 1.0,
                "dice_macro": 1.0,
                "hd95_2d_px_rv": 0.0, "hd95_2d_px_myo": 0.0,
                "hd95_2d_px_lv": 0.0, "hd95_2d_px_macro": 0.0,
            },
            "gt_present": {"rv": True, "myo": False, "lv": False},
        },
        {
            "patient_id": "p2",
            "metrics": {
                "dice_rv": 0.0, "dice_myo": 0.5, "dice_lv": 0.25,
                "dice_macro": 0.25,
                "hd95_2d_px_rv": 5.0, "hd95_2d_px_myo": 2.0,
                "hd95_2d_px_lv": 3.0, "hd95_2d_px_macro": 10.0 / 3.0,
            },
            "gt_present": {"rv": True, "myo": True, "lv": True},
        },
    ]
    summary = aggregate_slice_results(records, bootstrap_resamples=40, seed=7)
    assert summary["aggregation_unit"] == "slice"
    assert summary["all_slices"]["dice_macro"]["mean"] == 0.625
    assert summary["all_slices"]["dice_macro"]["n_slices"] == 2
    assert summary["foreground_present"]["dice_myo"]["mean"] == 0.5
    assert summary["foreground_present"]["dice_myo"]["n_slices"] == 1
    assert summary["foreground_present"]["dice_macro"]["mean"] == 5.0 / 12.0
