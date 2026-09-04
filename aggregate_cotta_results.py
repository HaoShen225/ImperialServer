"""Aggregate and compare five paired CoTTA seeds for one target-stream protocol."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from target_order_validation import require_distinct_seed_orders
from utils import file_sha256, load_config, save_json
from validate_cotta_results import LOCKED_COTTA, SEEDS, VENDORS, validate_run


BASELINES = ("source", "tent", "sar")
DIAGNOSTIC_KEYS = (
    "batches",
    "seen_slices",
    "augmented_slices",
    "augmentation_coverage",
    "restored_parameters",
    "mean_loss",
    "mean_anchor_confidence",
    "adaptation_seconds",
    "prediction_seconds",
    "final_parameter_drift",
)


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _seed_stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": fmean(values),
        "std": stdev(values),
        "std_ddof": 1,
        "per_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
    }


def _paired(candidate: list[float], baseline: list[float], baseline_name: str) -> dict[str, Any]:
    deltas = [left - right for left, right in zip(candidate, baseline)]
    return {
        "cotta_mean": fmean(candidate),
        f"{baseline_name}_mean": fmean(baseline),
        "delta_mean": fmean(deltas),
        "delta_std": stdev(deltas),
        "delta_std_ddof": 1,
        "per_seed": {
            str(seed): {"cotta": left, baseline_name: right, "delta": delta}
            for seed, left, right, delta in zip(SEEDS, candidate, baseline, deltas)
        },
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _baseline_path(results_root: Path, method: str, stream_mode: str) -> Path:
    suffix = "_slice_random_bs8" if stream_mode == "slice_random" else ""
    return results_root / method / f"{method}{suffix}_5seed_summary.json"


def _summary_path(results_root: Path, seed: int, stream_mode: str, vendor: str) -> Path:
    suffix = (
        "adapt_then_predict_vendor"
        if stream_mode == "patient_volume"
        else "slice_random_adapt_then_predict_vendor"
    )
    return results_root / "cotta" / f"seed{seed}" / suffix / f"vendor_{vendor}_summary.json"


def _baseline_checkpoint_hash(
    results_root: Path,
    baseline: dict[str, Any],
    method: str,
    seed: int,
    stream_mode: str,
) -> str:
    hashes = baseline.get("source_checkpoint_sha256")
    if isinstance(hashes, dict):
        return str(hashes[str(seed)])
    suffix = (
        "adapt_then_predict_vendor"
        if stream_mode == "patient_volume"
        else "slice_random_adapt_then_predict_vendor"
    )
    manifest = _read_json(
        results_root / method / f"seed{seed}" / suffix / "run_manifest.json"
    )
    return str(manifest["source_checkpoint_sha256"])


def _metric_groups(summary: dict[str, Any], stream_mode: str) -> dict[str, dict[str, Any]]:
    if stream_mode == "patient_volume":
        return {"patient_volume": summary}
    return {
        "all_slices": summary["all_slices"],
        "foreground_present": summary["foreground_present"],
    }


def _baseline_values(
    baseline: dict[str, Any],
    vendor: str,
    group: str,
    metric: str,
    stream_mode: str,
) -> list[float]:
    node = baseline["results"][vendor]
    if stream_mode == "slice_random":
        node = node[group]
    return [float(node[metric]["per_seed"][str(seed)]) for seed in SEEDS]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    parser.add_argument(
        "--stream-mode", choices=["patient_volume", "slice_random"], required=True
    )
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_root = Path(cfg["tta"]["results_dir"])
    stream_mode = args.stream_mode

    validated: dict[int, dict[str, Any]] = {
        seed: validate_run(seed, stream_mode, cfg, print_hashes=False) for seed in SEEDS
    }
    manifests = {seed: validated[seed]["manifest"] for seed in SEEDS}
    diagnostics = {seed: validated[seed]["diagnostics"] for seed in SEEDS}
    summaries = {
        seed: {
            vendor: _read_json(_summary_path(results_root, seed, stream_mode, vendor))
            for vendor in VENDORS
        }
        for seed in SEEDS
    }
    baselines = {
        method: _read_json(_baseline_path(results_root, method, stream_mode))
        for method in BASELINES
    }

    order_field = "target_order_sha256" if stream_mode == "patient_volume" else "slice_order_sha256"
    manifest_order_field = "target_order_sha256" if stream_mode == "patient_volume" else "slice_order_sha256"
    order_hashes: dict[int, dict[str, str]] = {
        seed: {
            vendor: str(manifests[seed]["target_orders"][vendor][manifest_order_field])
            for vendor in VENDORS
        }
        for seed in SEEDS
    }
    require_distinct_seed_orders(order_hashes, SEEDS, VENDORS)
    for seed in SEEDS:
        for method, baseline in baselines.items():
            baseline_checkpoint = _baseline_checkpoint_hash(
                results_root, baseline, method, seed, stream_mode
            )
            if baseline_checkpoint != manifests[seed]["source_checkpoint_sha256"]:
                raise RuntimeError(f"CoTTA and {method} use different seed-{seed} checkpoints")
            for vendor in VENDORS:
                baseline_order = baseline[order_field][str(seed)][vendor]
                if baseline_order != order_hashes[seed][vendor]:
                    raise RuntimeError(
                        f"CoTTA and {method} use different seed-{seed} Vendor-{vendor} orders"
                    )

    protocol_hashes = {manifest["protocol_sha256"] for manifest in manifests.values()}
    stream_hashes = {manifest["target_stream_sha256"] for manifest in manifests.values()}
    if len(protocol_hashes) != 1 or len(stream_hashes) != 1:
        raise RuntimeError("CoTTA seeds do not share one protocol and target-stream hash")

    batch_size = 4 if stream_mode == "patient_volume" else 8
    aggregate: dict[str, Any] = {
        "method": "cotta",
        "initialization_profile": "stochastic",
        "stream_mode": stream_mode,
        "batch_size": batch_size,
        "slice_filter": "manifest_has_fg_equals_1",
        "seeds": list(SEEDS),
        "vendors": list(VENDORS),
        "summary_statistic": "mean_and_sample_standard_deviation_across_paired_source_seeds",
        "settings": {**LOCKED_COTTA, "timing": "adapt_then_predict", "reset": "vendor"},
        "protocol_sha256": next(iter(protocol_hashes)),
        "target_stream_sha256": next(iter(stream_hashes)),
        "target_order_seed_source": "source_checkpoint_seed",
        order_field: {str(seed): order_hashes[seed] for seed in SEEDS},
        "source_checkpoint_sha256": {
            str(seed): manifests[seed]["source_checkpoint_sha256"] for seed in SEEDS
        },
        "delta_definition": (
            "cotta_minus_baseline; positive favors CoTTA for Dice and negative favors "
            "CoTTA for HD95"
        ),
        "results": {},
        **{f"comparison_to_{method}": {} for method in BASELINES},
        "diagnostics": {},
    }
    performance_rows: list[dict[str, Any]] = []
    diagnostic_rows: list[dict[str, Any]] = []

    for vendor in VENDORS:
        aggregate["results"][vendor] = {}
        aggregate["diagnostics"][vendor] = {}
        for method in BASELINES:
            aggregate[f"comparison_to_{method}"][vendor] = {}
        first_groups = _metric_groups(summaries[SEEDS[0]][vendor], stream_mode)
        for group, metrics in first_groups.items():
            aggregate["results"][vendor][group] = {}
            for method in BASELINES:
                aggregate[f"comparison_to_{method}"][vendor][group] = {}
            for metric in sorted(metrics):
                values = [
                    float(_metric_groups(summaries[seed][vendor], stream_mode)[group][metric]["mean"])
                    for seed in SEEDS
                ]
                aggregate["results"][vendor][group][metric] = _seed_stats(values)
                row: dict[str, Any] = {
                    "vendor": vendor,
                    "aggregation": group,
                    "metric": metric,
                    **{f"cotta_seed_{seed}": value for seed, value in zip(SEEDS, values)},
                    "cotta_mean": fmean(values),
                    "cotta_std": stdev(values),
                }
                for method, baseline in baselines.items():
                    baseline_values = _baseline_values(
                        baseline, vendor, group, metric, stream_mode
                    )
                    comparison = _paired(values, baseline_values, method)
                    aggregate[f"comparison_to_{method}"][vendor][group][metric] = comparison
                    deltas = [left - right for left, right in zip(values, baseline_values)]
                    row.update({
                        **{
                            f"{method}_seed_{seed}": value
                            for seed, value in zip(SEEDS, baseline_values)
                        },
                        **{
                            f"cotta_minus_{method}_seed_{seed}": value
                            for seed, value in zip(SEEDS, deltas)
                        },
                        f"{method}_mean": fmean(baseline_values),
                        f"cotta_minus_{method}_mean": fmean(deltas),
                        f"cotta_minus_{method}_std": stdev(deltas),
                    })
                performance_rows.append(row)

        for key in DIAGNOSTIC_KEYS:
            values = [float(diagnostics[seed][vendor][key]) for seed in SEEDS]
            aggregate["diagnostics"][vendor][key] = _seed_stats(values)
        for seed in SEEDS:
            diagnostic_rows.append({
                "vendor": vendor,
                "seed": seed,
                **diagnostics[seed][vendor],
            })

    method_root = results_root / "cotta"
    stem = "cotta" if stream_mode == "patient_volume" else "cotta_slice_random_bs8"
    json_path = method_root / f"{stem}_5seed_summary.json"
    csv_path = method_root / f"{stem}_5seed_summary.csv"
    diagnostic_path = method_root / f"{stem}_5seed_diagnostics.csv"
    save_json(aggregate, json_path)
    _write_csv(csv_path, performance_rows)
    _write_csv(diagnostic_path, diagnostic_rows)

    for path in (json_path, csv_path, diagnostic_path):
        print(file_sha256(path), path)
    for vendor in VENDORS:
        group = "patient_volume" if stream_mode == "patient_volume" else "all_slices"
        dice = aggregate["results"][vendor][group]["dice_macro"]
        print(
            f"[SUMMARY] stream={stream_mode} vendor={vendor} dice_macro="
            f"{dice['mean']:.6f}+/-{dice['std']:.6f}"
        )
    print(f"[AGGREGATE] stream={stream_mode} completed")


if __name__ == "__main__":
    main()
