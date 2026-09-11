"""Aggregate the completed BS(8/4), LR=1e-3 comparison across five seeds."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from statistics import fmean, stdev
from typing import Any

from run_bs_lr_comparison import (
    BATCH_SIZES,
    DEFAULT_RESULTS_ROOT,
    LEARNING_RATE,
    METHODS,
    SOURCE_SEEDS,
    STREAM_MODES,
    run_root,
)
from target_order_validation import require_distinct_seed_orders
from utils import file_sha256, save_json


VENDORS = ("B", "C", "D")


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _metric_groups(summary: dict[str, Any], stream_mode: str) -> dict[str, dict[str, Any]]:
    if stream_mode == "patient_volume":
        return {"patient_volume": summary}
    return {
        "all_slices": summary["all_slices"],
        "foreground_present": summary["foreground_present"],
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError("Refusing to write an empty comparison CSV")
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def _load(results_root: Path) -> tuple[dict, dict]:
    manifests: dict[tuple[str, int, str], dict[str, Any]] = {}
    summaries: dict[tuple[str, int, str, str], dict[str, Any]] = {}
    for stream_mode in STREAM_MODES:
        for method_name in METHODS:
            for seed in SOURCE_SEEDS:
                root = run_root(results_root, method_name, seed, stream_mode)
                manifests[(method_name, seed, stream_mode)] = _read_json(
                    root / "run_manifest.json"
                )
                for vendor in VENDORS:
                    summaries[(method_name, seed, stream_mode, vendor)] = _read_json(
                        root / f"vendor_{vendor}_summary.json"
                    )
    return manifests, summaries


def _validate_pairing(manifests: dict) -> dict[str, dict[int, dict[str, str]]]:
    order_hashes: dict[str, dict[int, dict[str, str]]] = {}
    for stream_mode in STREAM_MODES:
        order_field = (
            "target_order_sha256"
            if stream_mode == "patient_volume"
            else "slice_order_sha256"
        )
        order_hashes[stream_mode] = {}
        for seed in SOURCE_SEEDS:
            reference = manifests[("source", seed, stream_mode)]
            order_hashes[stream_mode][seed] = {
                vendor: str(reference["target_orders"][vendor][order_field])
                for vendor in VENDORS
            }
            for method_name in METHODS[1:]:
                candidate = manifests[(method_name, seed, stream_mode)]
                if candidate["source_checkpoint_sha256"] != reference["source_checkpoint_sha256"]:
                    raise RuntimeError(
                        f"{method_name}/{stream_mode}/seed{seed} uses a different checkpoint"
                    )
                for vendor in VENDORS:
                    if (
                        str(candidate["target_orders"][vendor][order_field])
                        != order_hashes[stream_mode][seed][vendor]
                    ):
                        raise RuntimeError(
                            f"{method_name}/{stream_mode}/seed{seed}/Vendor-{vendor} "
                            "uses a different arrival order"
                        )
        require_distinct_seed_orders(
            order_hashes[stream_mode], SOURCE_SEEDS, VENDORS
        )
    return order_hashes


def aggregate(results_root: Path) -> tuple[Path, Path, Path]:
    manifests, summaries = _load(results_root)
    order_hashes = _validate_pairing(manifests)
    payload: dict[str, Any] = {
        "experiment": "dual_protocol_patient_bs8_slice_bs4_adaptive_lr_1e-3",
        "methods": list(METHODS),
        "seeds": list(SOURCE_SEEDS),
        "vendors": list(VENDORS),
        "batch_sizes": BATCH_SIZES,
        "adaptive_learning_rate": LEARNING_RATE,
        "non_optimizing_methods": ["source", "tbn"],
        "summary_statistic": "mean_and_sample_standard_deviation_across_paired_source_seeds",
        "target_order_hashes": order_hashes,
        "results": {},
    }
    rows: list[dict[str, Any]] = []
    for stream_mode in STREAM_MODES:
        payload["results"][stream_mode] = {}
        for vendor in VENDORS:
            payload["results"][stream_mode][vendor] = {}
            first = summaries[(METHODS[0], SOURCE_SEEDS[0], stream_mode, vendor)]
            for group, metrics in _metric_groups(first, stream_mode).items():
                payload["results"][stream_mode][vendor][group] = {}
                for metric in sorted(metrics):
                    payload["results"][stream_mode][vendor][group][metric] = {}
                    row: dict[str, Any] = {
                        "stream_mode": stream_mode,
                        "batch_size": BATCH_SIZES[stream_mode],
                        "vendor": vendor,
                        "aggregation": group,
                        "metric": metric,
                    }
                    for method_name in METHODS:
                        values = [
                            float(
                                _metric_groups(
                                    summaries[(method_name, seed, stream_mode, vendor)],
                                    stream_mode,
                                )[group][metric]["mean"]
                            )
                            for seed in SOURCE_SEEDS
                        ]
                        stats = {
                            "mean": fmean(values),
                            "std": stdev(values),
                            "std_ddof": 1,
                            "per_seed": {
                                str(seed): value
                                for seed, value in zip(SOURCE_SEEDS, values)
                            },
                        }
                        payload["results"][stream_mode][vendor][group][metric][method_name] = stats
                        row.update({
                            f"{method_name}_seed_{seed}": value
                            for seed, value in zip(SOURCE_SEEDS, values)
                        })
                        row[f"{method_name}_mean"] = stats["mean"]
                        row[f"{method_name}_std"] = stats["std"]
                    rows.append(row)

    json_path = results_root / "comparison_5seed_summary.json"
    csv_path = results_root / "comparison_5seed_summary.csv"
    markdown_path = results_root / "comparison_5seed_macro.md"
    save_json(payload, json_path)
    _write_csv(csv_path, rows)

    lines = [
        "# BS(患者=8、切片=4)，自适应学习率 1e-3：五方法对比",
        "",
        "Source 和 TBN 不含优化器，因此学习率记为 N/A。数值为五个配对 seed 的 mean ± sample SD。",
        "",
        "| 协议 | Vendor | 方法 | Dice macro | HD95 macro (px) |",
        "|---|---:|---|---:|---:|",
    ]
    for stream_mode in STREAM_MODES:
        group = "patient_volume" if stream_mode == "patient_volume" else "all_slices"
        hd95_name = "hd95_px_macro" if stream_mode == "patient_volume" else "hd95_2d_px_macro"
        for vendor in VENDORS:
            node = payload["results"][stream_mode][vendor][group]
            for method_name in METHODS:
                dice = node["dice_macro"][method_name]
                hd95 = node[hd95_name][method_name]
                lines.append(
                    f"| {stream_mode} (BS={BATCH_SIZES[stream_mode]}) | {vendor} | "
                    f"{method_name} | {dice['mean']:.6f} ± {dice['std']:.6f} | "
                    f"{hd95['mean']:.6f} ± {hd95['std']:.6f} |"
                )
    markdown_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return json_path, csv_path, markdown_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--results-root", type=Path, default=DEFAULT_RESULTS_ROOT)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = aggregate(args.results_root)
    for path in paths:
        print(file_sha256(path), path)
    print(f"[AGGREGATE] completed root={args.results_root}")


if __name__ == "__main__":
    main()
