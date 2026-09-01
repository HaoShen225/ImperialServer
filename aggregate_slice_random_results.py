"""Aggregate paired five-seed Source/TENT/SAR random-slice evaluations."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from target_order_validation import require_distinct_seed_orders, validate_target_slice_order
from utils import file_sha256, load_config, save_json
from validate_slice_random_results import PROBE_COUNT_KEYS, SEEDS, VENDORS


METHODS = ("source", "tent", "sar")
STRATA = ("all_slices", "foreground_present")
STAGES = ("first_filter", "second_filter")
PROBE_METRICS = ("selection_coverage", "pixel_accuracy", "foreground_pixel_accuracy")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]


def _run_root(results_root: Path, method: str, seed: int) -> Path:
    return results_root / method / f"seed{seed}" / "slice_random_adapt_then_predict_vendor"


def _seed_stats(values: list[float]) -> dict[str, Any]:
    return {
        "mean": fmean(values),
        "std": stdev(values),
        "std_ddof": 1,
        "per_seed": {str(seed): value for seed, value in zip(SEEDS, values)},
    }


def _paired_comparison(
    candidate_name: str,
    baseline_name: str,
    candidate: list[float],
    baseline: list[float],
) -> dict[str, Any]:
    deltas = [left - right for left, right in zip(candidate, baseline)]
    return {
        f"{candidate_name}_mean": fmean(candidate),
        f"{baseline_name}_mean": fmean(baseline),
        "delta_mean": fmean(deltas),
        "delta_std": stdev(deltas),
        "delta_std_ddof": 1,
        "per_seed": {
            str(seed): {
                candidate_name: left,
                baseline_name: right,
                "delta": delta,
            }
            for seed, left, right, delta in zip(SEEDS, candidate, baseline, deltas)
        },
    }


def _probe_metrics(counts: dict[str, int]) -> dict[str, float | None]:
    return {
        "selection_coverage": (
            counts["selected_slices"] / counts["seen_slices"]
            if counts["seen_slices"] else None
        ),
        "pixel_accuracy": (
            counts["correct_pixels"] / counts["selected_pixels"]
            if counts["selected_pixels"] else None
        ),
        "foreground_pixel_accuracy": (
            counts["correct_gt_foreground_pixels"] / counts["gt_foreground_pixels"]
            if counts["gt_foreground_pixels"] else None
        ),
    }


def _optional_stats(values: list[float | None]) -> dict[str, Any]:
    valid = [float(value) for value in values if value is not None]
    return {
        "mean": fmean(valid) if valid else None,
        "std": stdev(valid) if len(valid) > 1 else None,
        "std_ddof": 1,
        "n_valid_seeds": len(valid),
    }


def _load_probe_counts(path: Path) -> dict[str, dict[str, int]]:
    totals = {stage: {key: 0 for key in PROBE_COUNT_KEYS} for stage in STAGES}
    for batch in _read_jsonl(path):
        probe = batch.get("entropy_label_probe")
        if probe is None:
            raise RuntimeError(f"Missing SAR entropy-label probe in {path}")
        if probe != batch["adaptation"].get("entropy_label_probe"):
            raise RuntimeError(f"SAR batch/adaptation probe mismatch in {path}")
        if probe["second_filter"]["selected_slices"] > probe["first_filter"]["selected_slices"]:
            raise RuntimeError(f"SAR second filter is not a subset in {path}")
        for stage in STAGES:
            for key in PROBE_COUNT_KEYS:
                totals[stage][key] += int(probe[stage][key])
    return totals


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"Refusing to write an empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default="config.yaml")
    args = parser.parse_args()
    cfg = load_config(args.config)
    results_root = Path(cfg["tta"]["results_dir"])

    manifests: dict[str, dict[int, dict[str, Any]]] = {method: {} for method in METHODS}
    summaries: dict[str, dict[int, dict[str, dict[str, Any]]]] = {
        method: {} for method in METHODS
    }
    order_hashes: dict[str, dict[int, dict[str, str]]] = {
        method: {} for method in METHODS
    }
    probe_counts: dict[int, dict[str, dict[str, dict[str, int]]]] = {}

    for method in METHODS:
        for seed in SEEDS:
            root = _run_root(results_root, method, seed)
            manifest = _read_json(root / "run_manifest.json")
            if (
                manifest.get("method") != method
                or manifest.get("stream_mode") != "slice_random"
                or int(manifest["resolved_config"]["tta"]["batch_size"]) != 8
            ):
                raise RuntimeError(f"Invalid BS=8 {method} manifest for seed {seed}")
            manifests[method][seed] = manifest
            summaries[method][seed] = {}
            order_hashes[method][seed] = {}
            for vendor in VENDORS:
                records = _read_jsonl(root / f"vendor_{vendor}.jsonl")
                order_hashes[method][seed][vendor] = validate_target_slice_order(
                    manifest, records, cfg, vendor, seed
                )
                summaries[method][seed][vendor] = _read_json(
                    root / f"vendor_{vendor}_summary.json"
                )
            if method == "sar":
                probe_counts[seed] = {
                    vendor: _load_probe_counts(root / f"vendor_{vendor}_batches.jsonl")
                    for vendor in VENDORS
                }

    require_distinct_seed_orders(order_hashes["source"], SEEDS, VENDORS)
    for seed in SEEDS:
        checkpoint_hashes = {
            manifests[method][seed]["source_checkpoint_sha256"] for method in METHODS
        }
        if len(checkpoint_hashes) != 1:
            raise RuntimeError(f"Methods use different source checkpoints for seed {seed}")
        for vendor in VENDORS:
            hashes = {order_hashes[method][seed][vendor] for method in METHODS}
            if len(hashes) != 1:
                raise RuntimeError(
                    f"Methods use different target orders for seed {seed}, Vendor {vendor}"
                )

    protocol_hashes = {
        manifests[method][seed]["protocol_sha256"]
        for method in METHODS for seed in SEEDS
    }
    stream_hashes = {
        manifests[method][seed]["target_stream_sha256"]
        for method in METHODS for seed in SEEDS
    }
    if len(protocol_hashes) != 1 or len(stream_hashes) != 1:
        raise RuntimeError("Runs do not share one protocol and random-slice stream hash")

    common = {
        "initialization_profile": "stochastic",
        "stream_mode": "slice_random",
        "batch_size": 8,
        "seeds": list(SEEDS),
        "vendors": list(VENDORS),
        "strata": list(STRATA),
        "summary_statistic": "mean_and_sample_standard_deviation_across_paired_source_seeds",
        "protocol_sha256": next(iter(protocol_hashes)),
        "target_stream_sha256": next(iter(stream_hashes)),
        "target_order_seed_source": "source_checkpoint_seed",
        "slice_order_sha256": {
            str(seed): order_hashes["source"][seed] for seed in SEEDS
        },
        "source_checkpoint_sha256": {
            str(seed): manifests["source"][seed]["source_checkpoint_sha256"]
            for seed in SEEDS
        },
    }

    aggregates: dict[str, dict[str, Any]] = {}
    rows: dict[str, list[dict[str, Any]]] = {method: [] for method in METHODS}
    for method in METHODS:
        aggregate: dict[str, Any] = {
            **common,
            "method": method,
            "resolved_method_config": manifests[method][SEEDS[0]]["resolved_method_config"],
            "results": {},
        }
        if method != "source":
            aggregate["comparison_to_source"] = {}
            aggregate["delta_definition"] = (
                f"{method}_minus_baseline; positive favors {method.upper()} for Dice "
                f"and negative favors {method.upper()} for HD95"
            )
        if method == "sar":
            aggregate["comparison_to_tent"] = {}

        for vendor in VENDORS:
            aggregate["results"][vendor] = {}
            if method != "source":
                aggregate["comparison_to_source"][vendor] = {}
            if method == "sar":
                aggregate["comparison_to_tent"][vendor] = {}
            for stratum in STRATA:
                metric_names = sorted(summaries[method][SEEDS[0]][vendor][stratum])
                for seed in SEEDS:
                    if set(summaries[method][seed][vendor][stratum]) != set(metric_names):
                        raise RuntimeError(
                            f"Metric schema differs for {method}, seed {seed}, {vendor}/{stratum}"
                        )
                aggregate["results"][vendor][stratum] = {}
                if method != "source":
                    aggregate["comparison_to_source"][vendor][stratum] = {}
                if method == "sar":
                    aggregate["comparison_to_tent"][vendor][stratum] = {}
                for metric in metric_names:
                    values = [
                        float(summaries[method][seed][vendor][stratum][metric]["mean"])
                        for seed in SEEDS
                    ]
                    aggregate["results"][vendor][stratum][metric] = _seed_stats(values)
                    row: dict[str, Any] = {
                        "vendor": vendor,
                        "stratum": stratum,
                        "metric": metric,
                        **{f"{method}_seed_{seed}": value for seed, value in zip(SEEDS, values)},
                        f"{method}_mean": fmean(values),
                        f"{method}_std": stdev(values),
                    }
                    if method != "source":
                        source_values = [
                            float(summaries["source"][seed][vendor][stratum][metric]["mean"])
                            for seed in SEEDS
                        ]
                        comparison = _paired_comparison(method, "source", values, source_values)
                        aggregate["comparison_to_source"][vendor][stratum][metric] = comparison
                        deltas = [left - right for left, right in zip(values, source_values)]
                        row.update({
                            **{f"source_seed_{seed}": value for seed, value in zip(SEEDS, source_values)},
                            **{f"{method}_minus_source_seed_{seed}": value for seed, value in zip(SEEDS, deltas)},
                            "source_mean": fmean(source_values),
                            f"{method}_minus_source_mean": fmean(deltas),
                            f"{method}_minus_source_std": stdev(deltas),
                        })
                    if method == "sar":
                        tent_values = [
                            float(summaries["tent"][seed][vendor][stratum][metric]["mean"])
                            for seed in SEEDS
                        ]
                        comparison = _paired_comparison("sar", "tent", values, tent_values)
                        aggregate["comparison_to_tent"][vendor][stratum][metric] = comparison
                        deltas = [left - right for left, right in zip(values, tent_values)]
                        row.update({
                            **{f"tent_seed_{seed}": value for seed, value in zip(SEEDS, tent_values)},
                            **{f"sar_minus_tent_seed_{seed}": value for seed, value in zip(SEEDS, deltas)},
                            "tent_mean": fmean(tent_values),
                            "sar_minus_tent_mean": fmean(deltas),
                            "sar_minus_tent_std": stdev(deltas),
                        })
                    rows[method].append(row)
        aggregates[method] = aggregate

    probe_rows: list[dict[str, Any]] = []
    aggregates["sar"]["entropy_label_probe"] = {}
    for vendor in VENDORS:
        aggregates["sar"]["entropy_label_probe"][vendor] = {}
        for stage in STAGES:
            per_seed_counts = {
                str(seed): probe_counts[seed][vendor][stage] for seed in SEEDS
            }
            per_seed_metrics = {
                str(seed): _probe_metrics(probe_counts[seed][vendor][stage])
                for seed in SEEDS
            }
            pooled = {
                key: sum(probe_counts[seed][vendor][stage][key] for seed in SEEDS)
                for key in PROBE_COUNT_KEYS
            }
            stage_summary: dict[str, Any] = {
                "per_seed_counts": per_seed_counts,
                "per_seed": per_seed_metrics,
                "pooled_counts": pooled,
                "pooled": _probe_metrics(pooled),
            }
            probe_row: dict[str, Any] = {"vendor": vendor, "stage": stage, **pooled}
            for metric in PROBE_METRICS:
                values = [per_seed_metrics[str(seed)][metric] for seed in SEEDS]
                stage_summary[metric] = {
                    **_optional_stats(values),
                    "per_seed": dict(zip(map(str, SEEDS), values)),
                }
                probe_row.update({
                    **{f"{metric}_seed_{seed}": value for seed, value in zip(SEEDS, values)},
                    f"{metric}_mean": stage_summary[metric]["mean"],
                    f"{metric}_std": stage_summary[metric]["std"],
                    f"{metric}_pooled": stage_summary["pooled"][metric],
                })
            aggregates["sar"]["entropy_label_probe"][vendor][stage] = stage_summary
            probe_rows.append(probe_row)

    output_paths: list[Path] = []
    for method in METHODS:
        method_root = results_root / method
        json_path = method_root / f"{method}_slice_random_bs8_5seed_summary.json"
        csv_path = method_root / f"{method}_slice_random_bs8_5seed_summary.csv"
        save_json(aggregates[method], json_path)
        _write_csv(csv_path, rows[method])
        output_paths.extend((json_path, csv_path))
    probe_path = results_root / "sar" / "sar_slice_random_bs8_5seed_probe_summary.csv"
    _write_csv(probe_path, probe_rows)
    output_paths.append(probe_path)

    for path in output_paths:
        print(file_sha256(path), path)
    for vendor in VENDORS:
        source = aggregates["source"]["results"][vendor]["all_slices"]["dice_macro"]
        tent = aggregates["tent"]["results"][vendor]["all_slices"]["dice_macro"]
        sar = aggregates["sar"]["results"][vendor]["all_slices"]["dice_macro"]
        print(
            f"[SUMMARY] vendor={vendor} all_slices/dice_macro "
            f"source={source['mean']:.6f}+/-{source['std']:.6f} "
            f"tent={tent['mean']:.6f}+/-{tent['std']:.6f} "
            f"sar={sar['mean']:.6f}+/-{sar['std']:.6f}"
        )
    print("[AGGREGATE] completed")


if __name__ == "__main__":
    main()
