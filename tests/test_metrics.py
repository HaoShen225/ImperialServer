from __future__ import annotations

import math

import numpy as np

from metrics import aggregate_results, dice_score, evaluate_volume, hd95_px


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
