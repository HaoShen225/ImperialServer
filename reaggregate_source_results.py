"""Reaggregate stateless source-only results after a target protocol change."""

from __future__ import annotations

import argparse
import csv
from copy import deepcopy
from io import StringIO
import json
import os
from pathlib import Path
import tempfile
from typing import Any

import numpy as np

from data import build_target_stream
from metrics import aggregate_results
from utils import file_sha256, load_config


DERIVATION_MODE = "offline_source_only_protocol_filter"


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ValueError(f"Expected a JSON object: {path}")
    return value


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if line.strip():
                value = json.loads(line)
                if not isinstance(value, dict):
                    raise ValueError(f"Expected a JSON object at {path}:{line_number}")
                records.append(value)
    return records


def _json_text(value: Any) -> str:
    return json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n"


def _jsonl_text(records: list[dict[str, Any]]) -> str:
    return "".join(json.dumps(record, sort_keys=True, allow_nan=False) + "\n" for record in records)


def _atomic_write_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{path.name}.", dir=path.parent, text=True)
    temporary = Path(temporary_name)
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(value)
            handle.flush()
            os.fsync(handle.fileno())
        temporary.replace(path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


def _validate_source_record(record: dict[str, Any], path: Path) -> None:
    if record.get("method") != "source" or record.get("prediction_source") != "source_model":
        raise ValueError(f"Offline reaggregation accepts only source-model records: {path}")
    if record.get("trainable_parameters"):
        raise ValueError(f"Source record unexpectedly contains trainable parameters: {path}")
    for batch in record.get("adaptation", []):
        drift = batch.get("extras", {}).get("parameter_drift", 0.0)
        if batch.get("updated") or float(drift) != 0.0:
            raise ValueError(f"Adaptive record cannot be reaggregated offline: {path}")


def _lineage(
    value: dict[str, Any], old_protocol_hash: str | None, old_stream_hash: str | None
) -> dict[str, Any]:
    existing = value.get("derivation")
    if isinstance(existing, dict) and existing.get("mode") == DERIVATION_MODE:
        return deepcopy(existing)
    return {
        "mode": DERIVATION_MODE,
        "inference_reused": True,
        "source_protocol_sha256": old_protocol_hash,
        "source_target_stream_sha256": old_stream_hash,
    }


def _five_seed_summary(
    per_seed: dict[int, dict[str, dict[str, dict[str, float]]]],
    vendors: list[str],
    seeds: list[int],
) -> dict[str, Any]:
    results: dict[str, Any] = {}
    for vendor in vendors:
        metrics = sorted(per_seed[seeds[0]][vendor])
        results[vendor] = {}
        for metric in metrics:
            values = {
                str(seed): float(per_seed[seed][vendor][metric]["mean"])
                for seed in seeds
            }
            array = np.asarray(list(values.values()), dtype=np.float64)
            results[vendor][metric] = {
                "mean": float(array.mean()),
                "per_seed": values,
                "std": float(array.std(ddof=1)),
                "std_ddof": 1,
            }
    return {"method": "source", "results": results}


def _five_seed_csv(summary: dict[str, Any], vendors: list[str], seeds: list[int]) -> str:
    output = StringIO(newline="")
    writer = csv.writer(output, lineterminator="\n")
    writer.writerow(
        ["vendor", "metric", *[f"seed_{seed}" for seed in seeds], "mean", "std", "std_ddof"]
    )
    for vendor in vendors:
        for metric, values in summary["results"][vendor].items():
            writer.writerow([
                vendor,
                metric,
                *[values["per_seed"][str(seed)] for seed in seeds],
                values["mean"],
                values["std"],
                values["std_ddof"],
            ])
    return output.getvalue()


def prepare_reaggregation(
    cfg: dict[str, Any], results_root: str | Path
) -> tuple[dict[Path, str], dict[str, Any]]:
    root = Path(results_root)
    vendors = list(cfg["experiment"]["target_vendors"])
    seeds = [int(seed) for seed in cfg["experiment"]["source_seeds"]]
    protocol_hash = file_sha256(cfg["data"]["protocol_file"])
    stream_hash = file_sha256(cfg["data"]["stream_file"])
    allowed_order = {
        vendor: [(volume["patient_id"], volume["phase"]) for volume in build_target_stream(vendor, cfg).volumes]
        for vendor in vendors
    }
    prepared: dict[Path, str] = {}
    per_seed: dict[int, dict[str, dict[str, dict[str, float]]]] = {}
    report: dict[str, Any] = {
        "mode": DERIVATION_MODE,
        "protocol_sha256": protocol_hash,
        "target_stream_sha256": stream_hash,
        "seeds": {},
    }

    for seed in seeds:
        run_root = root / f"seed{seed}" / f"{cfg['tta']['timing']}_{cfg['tta']['reset']}"
        manifest_path = run_root / "run_manifest.json"
        manifest = _read_json(manifest_path)
        if manifest.get("method") != "source":
            raise ValueError(f"Offline reaggregation accepts only source runs: {manifest_path}")
        old_protocol_hash = manifest.get("protocol_sha256")
        old_stream_hash = manifest.get("target_stream_sha256")
        seed_summaries: dict[str, dict[str, dict[str, float]]] = {}
        seed_report: dict[str, Any] = {}

        for vendor in vendors:
            records_path = run_root / f"vendor_{vendor}.jsonl"
            records = _read_jsonl(records_path)
            keyed: dict[tuple[str, str], dict[str, Any]] = {}
            for record in records:
                _validate_source_record(record, records_path)
                if record.get("vendor") != vendor:
                    raise ValueError(f"Unexpected vendor in {records_path}: {record.get('vendor')!r}")
                key = (str(record["patient_id"]), str(record["phase"]))
                if key in keyed:
                    raise ValueError(f"Duplicate source result {vendor}/{key[0]}/{key[1]}")
                keyed[key] = record

            expected = allowed_order[vendor]
            missing = [key for key in expected if key not in keyed]
            if missing:
                raise ValueError(f"Missing {len(missing)} required source results in {records_path}")
            kept = []
            for key in expected:
                record = deepcopy(keyed[key])
                record["derivation"] = _lineage(record, old_protocol_hash, old_stream_hash)
                record["protocol_sha256"] = protocol_hash
                record["target_stream_sha256"] = stream_hash
                kept.append(record)
            summary = aggregate_results(
                kept,
                bootstrap_resamples=int(cfg["evaluation"]["bootstrap_resamples"]),
                seed=int(cfg["evaluation"]["bootstrap_seed"]),
            )
            seed_summaries[vendor] = summary
            seed_report[vendor] = {
                "input_records": len(records),
                "output_records": len(kept),
                "removed_records": len(records) - len(kept),
                "n_patients": summary["dice_macro"]["n_patients"],
            }
            prepared[records_path] = _jsonl_text(kept)
            prepared[run_root / f"vendor_{vendor}_summary.json"] = _json_text(summary)

        manifest["protocol_sha256"] = protocol_hash
        manifest["target_stream_sha256"] = stream_hash
        manifest["resolved_config"] = deepcopy(cfg)
        manifest["summaries"] = seed_summaries
        manifest["derivation"] = _lineage(manifest, old_protocol_hash, old_stream_hash)
        manifest["derivation"]["record_counts"] = deepcopy(seed_report)
        prepared[manifest_path] = _json_text(manifest)
        per_seed[seed] = seed_summaries
        report["seeds"][str(seed)] = seed_report

    combined = _five_seed_summary(per_seed, vendors, seeds)
    combined["derivation"] = {
        "mode": DERIVATION_MODE,
        "inference_reused": True,
        "protocol_sha256": protocol_hash,
        "target_stream_sha256": stream_hash,
    }
    prepared[root / "source_5seed_summary.json"] = _json_text(combined)
    prepared[root / "source_5seed_summary.csv"] = _five_seed_csv(combined, vendors, seeds)
    return prepared, report


def reaggregate_source_results(
    cfg: dict[str, Any], results_root: str | Path, overwrite: bool = False
) -> dict[str, Any]:
    prepared, report = prepare_reaggregation(cfg, results_root)
    report["files_prepared"] = len(prepared)
    report["overwritten"] = bool(overwrite)
    if overwrite:
        for path, value in prepared.items():
            _atomic_write_text(path, value)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument("--results-root", default="results/source")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Atomically replace the existing source-only records and summaries after validation.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)
    report = reaggregate_source_results(cfg, args.results_root, overwrite=args.overwrite)
    print(json.dumps(report, indent=2, sort_keys=True, allow_nan=False))


if __name__ == "__main__":
    main()
