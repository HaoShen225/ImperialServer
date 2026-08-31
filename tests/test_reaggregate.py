from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

import reaggregate_source_results as reaggregate


METRICS = {
    "dice_lv": 0.8,
    "dice_macro": 0.7,
    "dice_myo": 0.6,
    "dice_rv": 0.7,
    "hd95_px_lv": 2.0,
    "hd95_px_macro": 3.0,
    "hd95_px_myo": 3.0,
    "hd95_px_rv": 4.0,
}


def _record(patient_id: str, phase: str, updated: bool = False) -> dict:
    return {
        "method": "source",
        "prediction_source": "source_model",
        "trainable_parameters": [],
        "vendor": "C",
        "patient_id": patient_id,
        "phase": phase,
        "volume_id": f"{patient_id}_{phase}",
        "metrics": dict(METRICS),
        "adaptation": [{"updated": updated, "extras": {"parameter_drift": 0.0}}],
        "protocol_sha256": "old-protocol",
        "target_stream_sha256": "old-stream",
    }


def _fixture(tmp_path: Path, updated: bool = False) -> tuple[dict, Path]:
    protocol = tmp_path / "protocol.json"
    stream = tmp_path / "stream.json"
    protocol.write_text("{}\n", encoding="utf-8")
    stream.write_text("{}\n", encoding="utf-8")
    root = tmp_path / "results" / "source"
    cfg = {
        "experiment": {"target_vendors": ["C"], "source_seeds": [1, 2]},
        "data": {"protocol_file": str(protocol), "stream_file": str(stream)},
        "tta": {"timing": "adapt_then_predict", "reset": "vendor"},
        "evaluation": {"bootstrap_resamples": 20, "bootstrap_seed": 42},
    }
    records = [
        _record("U1", "ED"),
        _record("U1", "ES"),
        _record("P1", "ED", updated=updated),
        _record("P1", "ES"),
    ]
    for seed in cfg["experiment"]["source_seeds"]:
        run_root = root / f"seed{seed}" / "adapt_then_predict_vendor"
        run_root.mkdir(parents=True)
        (run_root / "vendor_C.jsonl").write_text(
            "".join(json.dumps(record) + "\n" for record in records), encoding="utf-8"
        )
        (run_root / "run_manifest.json").write_text(
            json.dumps({
                "method": "source",
                "protocol_sha256": "old-protocol",
                "target_stream_sha256": "old-stream",
                "summaries": {},
            }),
            encoding="utf-8",
        )
    return cfg, root


def test_source_reaggregation_filters_and_overwrites(tmp_path, monkeypatch):
    cfg, root = _fixture(tmp_path)
    volumes = [
        {"patient_id": "P1", "phase": "ED"},
        {"patient_id": "P1", "phase": "ES"},
    ]
    monkeypatch.setattr(
        reaggregate, "build_target_stream", lambda vendor, config: SimpleNamespace(volumes=volumes)
    )

    report = reaggregate.reaggregate_source_results(cfg, root, overwrite=True)

    assert report["overwritten"] is True
    assert report["seeds"]["1"]["C"]["removed_records"] == 2
    output_path = root / "seed1" / "adapt_then_predict_vendor" / "vendor_C.jsonl"
    output = [json.loads(line) for line in output_path.read_text().splitlines()]
    assert [(row["patient_id"], row["phase"]) for row in output] == [("P1", "ED"), ("P1", "ES")]
    assert output[0]["derivation"]["source_protocol_sha256"] == "old-protocol"
    summary = json.loads(
        (root / "seed1" / "adapt_then_predict_vendor" / "vendor_C_summary.json").read_text()
    )
    assert summary["dice_macro"]["n_patients"] == 1
    assert json.loads((root / "source_5seed_summary.json").read_text())["method"] == "source"


def test_source_reaggregation_rejects_adaptive_records(tmp_path, monkeypatch):
    cfg, root = _fixture(tmp_path, updated=True)
    monkeypatch.setattr(
        reaggregate,
        "build_target_stream",
        lambda vendor, config: SimpleNamespace(volumes=[{"patient_id": "P1", "phase": "ED"}]),
    )
    with pytest.raises(ValueError, match="Adaptive record"):
        reaggregate.prepare_reaggregation(cfg, root)
