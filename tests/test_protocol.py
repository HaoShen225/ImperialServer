from __future__ import annotations

import inspect
import sys
from copy import deepcopy

import numpy as np
import torch
from torch.utils.data import DataLoader

import run_tta as run_tta_module
from data import MMSTargetSliceDataset, slice_order_sha256

from run_tta import (
    _probe_stage_counts,
    _validate_evaluation_target,
    attach_entropy_label_probe,
    run_random_slice_batch,
    run_slice_experiment,
    run_volume,
)
from tta_methods import build_method


def test_run_volume_signature_has_no_label_boundary():
    assert list(inspect.signature(run_volume).parameters) == ["method", "images", "batch_size", "device"]


def test_run_random_slice_batch_signature_has_no_label_boundary():
    assert list(inspect.signature(run_random_slice_batch).parameters) == [
        "method", "images", "device"
    ]


def test_cli_accepts_tta_batch_size_override(monkeypatch):
    monkeypatch.setattr(
        sys,
        "argv",
        [
            "run_tta.py",
            "--method",
            "tent",
            "--source-seed",
            "2022",
            "--stream-mode",
            "slice_random",
            "--batch-size",
            "8",
        ],
    )
    args = run_tta_module.parse_args()
    assert args.stream_mode == "slice_random"
    assert args.batch_size == 8


@__import__("pytest").mark.parametrize("method_name", ["source", "tent", "sar"])
def test_random_slice_batch_smoke_for_core_methods(
    config, tiny_model, images, method_name
):
    method_cfg = deepcopy(config["methods"][method_name])
    method = build_method(
        method_name,
        deepcopy(tiny_model),
        method_cfg,
        config["tta"],
        torch.device("cpu"),
    )
    prediction, adaptation, probe = run_random_slice_batch(
        method, images, torch.device("cpu")
    )
    assert prediction.shape == images.shape[:1] + images.shape[-2:]
    assert adaptation["arrival_batch_size"] == images.shape[0]
    assert "predicted_foreground_area" in adaptation
    if method_name == "sar":
        assert probe is not None


def test_slice_experiment_writes_slice_and_batch_results(
    config, tiny_model, tmp_path, monkeypatch
):
    cfg = deepcopy(config)
    cfg["tta"]["stream_mode"] = "slice_random"
    cfg["tta"]["results_dir"] = str(tmp_path / "results")
    cfg["evaluation"]["bootstrap_resamples"] = 20
    data_root = tmp_path / "arrays"
    data_root.mkdir()
    rows = []
    for index in range(4):
        image = np.full((16, 16), index / 4.0, dtype=np.float32)
        mask = np.zeros((16, 16), dtype=np.uint8)
        mask[1:4, 1:4] = 1
        mask[5:8, 5:8] = 2
        mask[9:12, 9:12] = 3
        image_name, mask_name = f"image_{index}.npy", f"mask_{index}.npy"
        np.save(data_root / image_name, image)
        np.save(data_root / mask_name, mask)
        rows.append({
            "image": image_name,
            "mask": mask_name,
            "slice_id": f"B/p{index}/ED/z000",
            "patient_id": f"p{index}",
            "phase": "ED",
            "vendor": "B",
            "z_index": 0,
            "slice_arrival_index": index,
        })
    order_hash = slice_order_sha256("B", 2022, [row["slice_id"] for row in rows])
    dataset = MMSTargetSliceDataset(
        rows,
        data_root,
        vendor="B",
        order_seed=2022,
        slice_order_sha256=order_hash,
    )
    loader = DataLoader(dataset, batch_size=4, shuffle=False)
    method_cfg = deepcopy(cfg["methods"]["source"])
    method = build_method(
        "source", deepcopy(tiny_model), method_cfg, cfg["tta"], torch.device("cpu")
    )
    checkpoint = tmp_path / "source.pt"
    checkpoint.write_bytes(b"test-checkpoint")
    monkeypatch.setattr(
        run_tta_module,
        "_prepare_experiment",
        lambda *_args, **_kwargs: (method, method_cfg, checkpoint, "stochastic", None),
    )
    monkeypatch.setattr(
        run_tta_module,
        "build_target_slice_loader",
        lambda *_args, **_kwargs: loader,
    )

    manifest = run_slice_experiment(
        cfg, "source", 2022, ["B"], torch.device("cpu")
    )
    root = tmp_path / "results" / "source" / "seed2022" / "slice_random_adapt_then_predict_vendor"
    assert manifest["stream_mode"] == "slice_random"
    assert manifest["target_orders"]["B"]["n_slices"] == 4
    assert len((root / "vendor_B.jsonl").read_text().splitlines()) == 4
    assert len((root / "vendor_B_batches.jsonl").read_text().splitlines()) == 1
    assert (root / "vendor_B_summary.json").is_file()
    assert (root / "run_manifest.json").is_file()


def test_target_label_permutation_cannot_change_prediction(config, tiny_model, images):
    method_cfg = dict(config["methods"]["source"])
    method = build_method("source", tiny_model, method_cfg, config["tta"], torch.device("cpu"))
    first, _ = run_volume(method, images, 4, torch.device("cpu"))
    arbitrary_target = torch.randint(0, 4, (4, 16, 16))
    arbitrary_target = arbitrary_target.flatten()[torch.randperm(arbitrary_target.numel())].reshape_as(arbitrary_target)
    assert arbitrary_target.shape == first.shape
    method.reset()
    second, _ = run_volume(method, images, 4, torch.device("cpu"))
    assert torch.equal(first, second)


def test_timing_modes_produce_distinct_first_batch(config, tiny_model, images):
    method_cfg = dict(config["methods"]["tent"])
    before_cfg = dict(config["tta"])
    before_cfg["timing"] = "predict_then_adapt"
    after_cfg = dict(config["tta"])
    after_cfg["timing"] = "adapt_then_predict"
    first = build_method("tent", tiny_model, method_cfg, before_cfg, torch.device("cpu"))
    second_model = __import__("copy").deepcopy(tiny_model)
    second = build_method("tent", second_model, method_cfg, after_cfg, torch.device("cpu"))
    logits_before, _ = first.process_batch(images)
    logits_after, _ = second.process_batch(images)
    assert not torch.equal(logits_before, logits_after)


def test_empty_evaluation_target_is_rejected():
    target = torch.zeros((4, 16, 16), dtype=torch.int64)
    with __import__("pytest").raises(ValueError, match="C/PATIENT/ED"):
        _validate_evaluation_target(target, [1, 2, 3], "C", "PATIENT", "ED")


def test_entropy_label_probe_counts_all_and_foreground_pixels():
    target = torch.tensor(
        [
            [[0, 1], [2, 0]],
            [[0, 3], [3, 0]],
        ]
    )
    payload = {
        "selected": torch.tensor([True, False]),
        "labels": torch.tensor(
            [
                [[0, 1], [0, 0]],
                [[3, 3], [3, 3]],
            ]
        ),
    }
    probe = _probe_stage_counts(payload, target)
    assert probe == {
        "seen_slices": 2,
        "selected_slices": 1,
        "selection_coverage": 0.5,
        "selected_pixels": 4,
        "correct_pixels": 3,
        "pixel_accuracy": 0.75,
        "gt_foreground_pixels": 2,
        "correct_gt_foreground_pixels": 1,
        "foreground_pixel_accuracy": 0.5,
    }


def test_entropy_label_probe_empty_selection_uses_null_accuracies():
    target = torch.ones((2, 2, 2), dtype=torch.int64)
    probe = _probe_stage_counts(
        {"selected": torch.tensor([False, False]), "labels": torch.zeros_like(target)},
        target,
    )
    assert probe["selected_slices"] == 0
    assert probe["selected_pixels"] == 0
    assert probe["pixel_accuracy"] is None
    assert probe["foreground_pixel_accuracy"] is None


def test_probe_targets_do_not_change_sar_state(config, tiny_model, images):
    method_cfg = dict(config["methods"]["sar"])
    method_cfg["entropy_margin_factor"] = 2.0
    method_cfg["recovery_threshold"] = -1.0
    method = build_method(
        "sar", tiny_model, method_cfg, config["tta"], torch.device("cpu")
    )
    prediction, records = run_volume(method, images, 4, torch.device("cpu"))
    state_before_probe = {
        name: value.detach().clone() for name, value in method.model.state_dict().items()
    }
    zero_target_records = deepcopy(records)
    foreground_target_records = deepcopy(records)
    zero_probe = attach_entropy_label_probe(
        zero_target_records, torch.zeros_like(prediction)
    )
    foreground_probe = attach_entropy_label_probe(
        foreground_target_records, torch.ones_like(prediction)
    )
    state_after_probe = method.model.state_dict()
    assert all(torch.equal(value, state_after_probe[name]) for name, value in state_before_probe.items())
    assert zero_probe != foreground_probe
    assert "_probe_payload" not in zero_target_records[0]
    assert "entropy_label_probe" in zero_target_records[0]
