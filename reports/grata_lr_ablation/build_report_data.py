#!/usr/bin/env python3
"""Validate and aggregate the GraTA learning-rate sweep into a report artifact."""

from __future__ import annotations

import glob
import json
import math
import sqlite3
import statistics
from collections import defaultdict
from datetime import datetime, timezone
from pathlib import Path


REPO = Path(__file__).resolve().parents[2]
RESULTS = REPO / "results/Stochastic_Ini_ForegroundOnly/grata_lr_sweep"
LOGS = REPO / "checkpoints/logs/Stochastic_Ini_ForegroundOnly/grata_lr_sweep"
OUT = Path(__file__).resolve().parent
AGGREGATION_SQL = OUT / "aggregation.sql"
LRS = (1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1.0)
SEEDS = tuple(range(2022, 2027))
VENDORS = ("B", "C", "D")
RUNS = {
    "patient_volume": "adapt_then_predict_vendor",
    "slice_random": "slice_random_adapt_then_predict_vendor",
}


def lr_tag(value: float) -> str:
    if value == 1.0:
        return "1"
    return f"{value:.0e}".replace("e-0", "e-")


def mean_sd(values: list[float]) -> tuple[float, float]:
    return statistics.mean(values), statistics.stdev(values)


def metric_string(value: float, sd: float) -> str:
    return f"{value:.4f} ± {sd:.4f}"


def load_summary(protocol: str, lr: float, seed: int, vendor: str) -> dict:
    path = (
        RESULTS
        / f"lr_{lr_tag(lr)}"
        / "grata"
        / f"seed{seed}"
        / RUNS[protocol]
        / f"vendor_{vendor}_summary.json"
    )
    with path.open() as handle:
        return json.load(handle)


def primary_metrics(protocol: str, summary: dict) -> tuple[float, float, int]:
    if protocol == "patient_volume":
        return (
            summary["dice_macro"]["mean"],
            summary["hd95_px_macro"]["mean"],
            summary["dice_macro"]["n_patients"],
        )
    return (
        summary["all_slices"]["dice_macro"]["mean"],
        summary["all_slices"]["hd95_2d_px_macro"]["mean"],
        summary["all_slices"]["dice_lv"]["n_slices"],
    )


def validate_manifests() -> dict:
    issues: list[str] = []
    seen: set[tuple[float, int, str]] = set()
    protocol_hashes: set[str] = set()
    commits: set[str] = set()
    source_hashes: dict[int, set[str]] = defaultdict(set)
    stream_hashes: dict[str, set[str]] = defaultdict(set)
    batch_sizes: dict[str, set[int]] = defaultdict(set)
    sample_sizes: dict[str, dict[str, set[int]]] = {
        protocol: {vendor: set() for vendor in VENDORS} for protocol in RUNS
    }

    for protocol, run_name in RUNS.items():
        for lr in LRS:
            for seed in SEEDS:
                run_dir = RESULTS / f"lr_{lr_tag(lr)}" / "grata" / f"seed{seed}" / run_name
                manifest_path = run_dir / "run_manifest.json"
                if not manifest_path.exists():
                    issues.append(f"missing manifest: {manifest_path.relative_to(REPO)}")
                    continue
                with manifest_path.open() as handle:
                    manifest = json.load(handle)
                seen.add((lr, seed, protocol))
                protocol_hashes.add(manifest["protocol_sha256"])
                commits.add(manifest["runtime"]["git_commit"])
                source_hashes[seed].add(manifest["source_checkpoint_sha256"])
                stream_hashes[protocol].add(manifest["target_stream_sha256"])
                batch_sizes[protocol].add(manifest["resolved_config"]["tta"]["batch_size"])

                expected = {
                    "method": "grata",
                    "stream_mode": protocol,
                    "source_seed": seed,
                    "target_order_seed": seed,
                }
                for key, value in expected.items():
                    if manifest.get(key) != value:
                        issues.append(
                            f"{manifest_path.relative_to(REPO)}: {key}={manifest.get(key)!r}, expected {value!r}"
                        )
                resolved_lr = float(manifest["resolved_method_config"]["lr"])
                if not math.isclose(resolved_lr, lr, rel_tol=1e-12):
                    issues.append(
                        f"{manifest_path.relative_to(REPO)}: lr={resolved_lr}, expected {lr}"
                    )

                for vendor in VENDORS:
                    summary_path = run_dir / f"vendor_{vendor}_summary.json"
                    if not summary_path.exists():
                        issues.append(f"missing summary: {summary_path.relative_to(REPO)}")
                        continue
                    with summary_path.open() as handle:
                        dice, hd95, n = primary_metrics(protocol, json.load(handle))
                    if not (math.isfinite(dice) and math.isfinite(hd95)):
                        issues.append(f"non-finite metric: {summary_path.relative_to(REPO)}")
                    sample_sizes[protocol][vendor].add(n)

    expected = {(lr, seed, protocol) for lr in LRS for seed in SEEDS for protocol in RUNS}
    for missing in sorted(expected - seen):
        issues.append(f"missing combination: {missing}")

    validated_logs = 0
    error_logs: list[str] = []
    for log_path in LOGS.rglob("*.log"):
        text = log_path.read_text(errors="replace")
        if "[VALIDATED]" in text:
            validated_logs += 1
        if any(marker in text for marker in ("Traceback", "RuntimeError", "[FAILED]")):
            error_logs.append(str(log_path.relative_to(REPO)))

    if validated_logs != 60:
        issues.append(f"validated task logs={validated_logs}, expected 60")
    if error_logs:
        issues.append(f"error-bearing logs: {error_logs}")
    if len(protocol_hashes) != 1:
        issues.append(f"protocol hashes are inconsistent: {sorted(protocol_hashes)}")
    if len(commits) != 1:
        issues.append(f"git commits are inconsistent: {sorted(commits)}")
    for seed, hashes in source_hashes.items():
        if len(hashes) != 1:
            issues.append(f"seed {seed} has inconsistent checkpoint hashes")
    for protocol, hashes in stream_hashes.items():
        if len(hashes) != 1:
            issues.append(f"{protocol} has inconsistent stream hashes")

    return {
        "issues": issues,
        "combination_count": len(seen),
        "summary_count": len(seen) * len(VENDORS),
        "validated_log_count": validated_logs,
        "protocol_sha256": sorted(protocol_hashes),
        "git_commits": sorted(commits),
        "batch_sizes": {key: sorted(value) for key, value in batch_sizes.items()},
        "sample_sizes": {
            protocol: {vendor: sorted(values) for vendor, values in by_vendor.items()}
            for protocol, by_vendor in sample_sizes.items()
        },
    }


def aggregate_metrics() -> tuple[dict, dict]:
    aggregated: dict[str, list[dict]] = {protocol: [] for protocol in RUNS}
    raw: dict[str, dict[str, dict[str, list[dict]]]] = {}
    sql_rows: list[tuple[str, float, int, str, float, float, int]] = []
    for protocol in RUNS:
        raw[protocol] = {}
        for lr in LRS:
            raw[protocol][lr_tag(lr)] = {}
            for seed in SEEDS:
                raw[protocol][lr_tag(lr)][str(seed)] = []
                for vendor in VENDORS:
                    summary = load_summary(protocol, lr, seed, vendor)
                    dice, hd95, n = primary_metrics(protocol, summary)
                    sql_rows.append((protocol, lr, seed, vendor, dice, hd95, n))
                    raw[protocol][lr_tag(lr)][str(seed)].append(
                        {"vendor": vendor, "dice": dice, "hd95": hd95, "n": n}
                    )

    connection = sqlite3.connect(":memory:")
    connection.execute(
        "CREATE TABLE grata_summary_metrics "
        "(protocol TEXT, lr REAL, seed INTEGER, vendor TEXT, dice REAL, hd95 REAL, n INTEGER)"
    )
    connection.executemany(
        "INSERT INTO grata_summary_metrics VALUES (?, ?, ?, ?, ?, ?, ?)",
        sql_rows,
    )
    connection.row_factory = sqlite3.Row
    query = AGGREGATION_SQL.read_text()
    for row in connection.execute(query):
        values = dict(row)
        protocol = values.pop("protocol")
        lr = values.pop("lr")
        aggregated[protocol].append(
            {
                "beta": lr,
                "beta_label": lr_tag(lr),
                "vendor_b_dice": metric_string(
                    values["vendor_b_dice_mean"], values["vendor_b_dice_sd"]
                ),
                "vendor_c_dice": metric_string(
                    values["vendor_c_dice_mean"], values["vendor_c_dice_sd"]
                ),
                "vendor_d_dice": metric_string(
                    values["vendor_d_dice_mean"], values["vendor_d_dice_sd"]
                ),
                "avg_dice": metric_string(values["avg_dice_mean"], values["avg_dice_sd"]),
                "avg_dice_mean": values["avg_dice_mean"],
                "avg_dice_sd": values["avg_dice_sd"],
                "avg_hd95": f"{values['avg_hd95_mean']:.2f} ± {values['avg_hd95_sd']:.2f}",
                "avg_hd95_mean": values["avg_hd95_mean"],
                "avg_hd95_sd": values["avg_hd95_sd"],
            }
        )
    connection.close()
    return aggregated, raw


def adaptation_diagnostics() -> list[dict]:
    output: list[dict] = []
    for protocol, run_name in RUNS.items():
        for lr in LRS:
            effective_lrs: list[float] = []
            cosines: list[float] = []
            endpoint_drifts: list[float] = []
            for seed in SEEDS:
                run_dir = RESULTS / f"lr_{lr_tag(lr)}" / "grata" / f"seed{seed}" / run_name
                for vendor in VENDORS:
                    suffix = f"vendor_{vendor}.jsonl" if protocol == "patient_volume" else f"vendor_{vendor}_batches.jsonl"
                    endpoint = None
                    with (run_dir / suffix).open() as handle:
                        for line in handle:
                            record = json.loads(line)
                            adaptation = record.get("adaptation")
                            adaptations = adaptation if isinstance(adaptation, list) else [adaptation]
                            for item in adaptations:
                                if not item:
                                    continue
                                extras = item["extras"]
                                effective_lrs.append(float(extras["effective_lr"]))
                                cosines.append(float(extras["gradient_cosine"]))
                                endpoint = float(extras["parameter_drift"])
                    if endpoint is not None:
                        endpoint_drifts.append(endpoint)
            output.append(
                {
                    "protocol": "病人随机" if protocol == "patient_volume" else "切片随机",
                    "protocol_key": protocol,
                    "beta": lr,
                    "beta_label": lr_tag(lr),
                    "n_updates": len(effective_lrs),
                    "effective_lr_mean": statistics.mean(effective_lrs),
                    "effective_lr_mean_label": f"{statistics.mean(effective_lrs):.3e}",
                    "effective_lr_fraction": statistics.mean(effective_lrs) / lr,
                    "effective_lr_fraction_label": f"{100 * statistics.mean(effective_lrs) / lr:.1f}%",
                    "gradient_cosine_mean": statistics.mean(cosines),
                    "gradient_cosine_mean_label": f"{statistics.mean(cosines):.3f}",
                    "endpoint_drift_mean": statistics.mean(endpoint_drifts),
                    "endpoint_drift_mean_label": f"{statistics.mean(endpoint_drifts):.3f}",
                    "endpoint_drift_max": max(endpoint_drifts),
                }
            )
    return output


def source_metadata(generated_at: str) -> dict:
    return {
        "id": "grata_lr_sweep_results",
        "label": "GraTA 学习率消融结果文件",
        "path": "reports/grata_lr_ablation/aggregation.sql",
        "query": {
            "engine": "sqlite",
            "language": "sql",
            "description": "Python loader 将 180 个 Vendor summary 的主指标装入内存表 grata_summary_metrics；该 SQL 按相同 seed 先跨 Vendor 等权平均，再汇总五个 seed。",
            "sql": AGGREGATION_SQL.read_text(),
            "executed_at": generated_at,
            "tables_used": ["grata_summary_metrics"],
            "filters": [
                "method=grata",
                "source_seeds=2022..2026",
                "target_vendors=B,C,D",
                "slice_filter=manifest_has_fg_equals_1",
                "learning_rates=1e-5,1e-4,1e-3,1e-2,1e-1,1",
            ],
            "metric_definitions": {
                "patient_volume Dice": "每个 Vendor 内按病人汇总的三前景类 macro Dice；报告值为五个 seed 的均值±样本标准差。",
                "slice_random Dice": "all_slices 视图下逐切片等权的三前景类 macro Dice；报告值为五个 seed 的均值±样本标准差。",
                "cross-vendor average": "每个 seed 内对 B/C/D 三个 Vendor 的 point estimate 等权平均，再对五个 seed 计算均值与样本标准差。",
                "HD95": "病人协议使用 hd95_px_macro；切片协议使用 all_slices.hd95_2d_px_macro；单位为 processed 256 网格上的像素。",
            },
        },
    }


def report_artifact(aggregated: dict, diagnostics: list[dict], validation: dict, generated_at: str) -> dict:
    patient = aggregated["patient_volume"]
    slices = aggregated["slice_random"]
    patient_best = max(patient, key=lambda row: row["avg_dice_mean"])
    slice_best = max(slices, key=lambda row: row["avg_dice_mean"])
    patient_reference = next(row for row in patient if row["beta"] == 1e-4)
    slice_reference = next(row for row in slices if row["beta"] == 1e-4)
    diag_best = [row for row in diagnostics if row["beta"] == 1e-3]
    lr_curve = []
    for patient_row, slice_row in zip(patient, slices):
        for protocol_label, row in (
            ("病人随机", patient_row),
            ("切片随机", slice_row),
        ):
            lr_curve.append(
                {
                    "log10_beta": math.log10(row["beta"]),
                    "beta": row["beta"],
                    "beta_label": row["beta_label"],
                    "protocol": protocol_label,
                    "dice": row["avg_dice_mean"],
                    "dice_sd": row["avg_dice_sd"],
                    "hd95": row["avg_hd95_mean"],
                }
            )

    source = source_metadata(generated_at)
    common_columns = [
        {"field": "beta", "label": "β（配置 LR）", "format": "number"},
        {"field": "vendor_b_dice", "label": "Vendor B Dice", "type": "text"},
        {"field": "vendor_c_dice", "label": "Vendor C Dice", "type": "text"},
        {"field": "vendor_d_dice", "label": "Vendor D Dice", "type": "text"},
        {"field": "avg_dice", "label": "B/C/D 等权 Dice", "type": "text"},
        {"field": "avg_hd95", "label": "B/C/D 等权 HD95 (px)", "type": "text"},
    ]
    return {
        "surface": "report",
        "manifest": {
            "version": 1,
            "surface": "report",
            "title": "GraTA 学习率消融结果",
            "description": "M&Ms A→B/C/D、五个随机初始化 backbone、病人随机与切片随机双协议的 GraTA 学习率扫描。",
            "generatedAt": generated_at,
            "cards": [
                {
                    "id": "patient_best",
                    "description": "病人随机协议下 B/C/D 等权平均的最佳五 seed Dice。",
                    "dataset": "headline",
                    "sourceId": source["id"],
                    "metrics": [{"label": "病人协议最佳 Dice（β=1e-3）", "field": "patient_best_dice", "format": "number"}],
                },
                {
                    "id": "slice_best",
                    "description": "切片随机协议 all_slices 口径下 B/C/D 等权平均的最佳五 seed Dice。",
                    "dataset": "headline",
                    "sourceId": source["id"],
                    "metrics": [{"label": "切片协议最佳 Dice（β=1e-3）", "field": "slice_best_dice", "format": "number"}],
                },
                {
                    "id": "coverage",
                    "description": "完整生成 run_manifest 且任务日志内通过验证的学习率×seed×协议组合。",
                    "dataset": "headline",
                    "sourceId": source["id"],
                    "metrics": [{"label": "验证完成的组合", "field": "validated_combinations", "format": "number"}],
                },
            ],
            "charts": [
                {
                    "id": "dice_lr_curve",
                    "title": "双协议跨 Vendor Dice 随学习率变化",
                    "subtitle": "五个 source seed；B/C/D 等权；横轴为 log10 β，精确均值与标准差见下表。",
                    "intent": "comparison",
                    "question": "哪一个 GraTA base/max learning rate 在两种协议下最稳健？",
                    "rationale": "对数学习率是有序连续变量；双折线同时显示共同峰值与高学习率失稳。",
                    "comparisonContext": {
                        "grain": "protocol × learning rate",
                        "unit": "macro Dice",
                        "normalization": "每个 seed 内 B/C/D 等权平均，再跨五 seed 求均值",
                    },
                    "type": "line",
                    "dataset": "lr_curve",
                    "sourceId": source["id"],
                    "encodings": {
                        "x": {"field": "log10_beta", "type": "quantitative", "label": "log10 β"},
                        "y": {
                            "field": "dice",
                            "type": "quantitative",
                            "label": "Macro Dice",
                            "format": "number",
                        },
                        "color": {"field": "protocol", "type": "nominal", "label": "测试流协议"},
                        "lineStyle": {"field": "protocol", "type": "nominal", "label": "测试流协议"},
                        "tooltip": [
                            {"field": "beta_label", "type": "text", "label": "β"},
                            {"field": "dice_sd", "type": "quantitative", "label": "跨 seed SD", "format": "number"},
                            {"field": "hd95", "type": "quantitative", "label": "跨 Vendor HD95", "format": "number"},
                        ],
                    },
                    "xAxisTitle": "log10 β",
                    "yAxisTitle": "Macro Dice",
                    "valueFormat": "number",
                    "layout": "full",
                    "labels": {"values": "endpoints"},
                    "legend": {"position": "bottom", "sort": "spec", "title": "测试流协议"},
                    "palette": {"kind": "categorical", "name": "blue-orange"},
                    "settings": {"showPoints": "always", "sort": "ascending"},
                    "surface": {"surface": "explorer", "interactiveLegend": True, "viewMode": "both"},
                }
            ],
            "tables": [
                {
                    "id": "patient_results",
                    "title": "病人随机协议学习率扫描",
                    "subtitle": "BS=4；每个单元格为五个 source seed 的均值 ± 样本标准差。",
                    "dataset": "patient_results",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "beta", "direction": "asc"},
                    "layout": "full",
                    "columns": common_columns,
                },
                {
                    "id": "slice_results",
                    "title": "切片随机协议学习率扫描",
                    "subtitle": "BS=8；all_slices 逐切片等权口径；每个单元格为五个 source seed 的均值 ± 样本标准差。",
                    "dataset": "slice_results",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "beta", "direction": "asc"},
                    "layout": "full",
                    "columns": common_columns,
                },
                {
                    "id": "effective_lr",
                    "title": "GraTA 动态学习率与参数漂移诊断",
                    "subtitle": "实际 LR 由梯度余弦动态缩放；漂移为每个 Vendor 流末尾相对初始 BN affine 的 L2 距离。",
                    "dataset": "diagnostics",
                    "sourceId": source["id"],
                    "defaultSort": {"field": "beta", "direction": "asc"},
                    "layout": "full",
                    "columns": [
                        {"field": "protocol", "label": "协议", "type": "text"},
                        {"field": "beta", "label": "β", "format": "number"},
                        {"field": "effective_lr_mean_label", "label": "平均实际 LR", "type": "text"},
                        {"field": "effective_lr_fraction_label", "label": "实际 LR / β", "type": "text"},
                        {"field": "gradient_cosine_mean_label", "label": "平均梯度余弦", "type": "text"},
                        {"field": "endpoint_drift_mean_label", "label": "平均末端漂移", "type": "text"},
                    ],
                },
            ],
            "sources": [source],
            "blocks": [
                {"id": "title", "type": "markdown", "body": "# GraTA 学习率消融结果"},
                {
                    "id": "technical_summary",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## 技术结论：β=1e-3 是双协议共同的稳健最优点\n\n"
                        f"病人随机协议的最佳 B/C/D 等权 Dice 为 **{patient_best['avg_dice']}**，HD95 为 **{patient_best['avg_hd95']} px**；"
                        f"切片随机协议的最佳 Dice 为 **{slice_best['avg_dice']}**，HD95 为 **{slice_best['avg_hd95']} px**。"
                        f"相对 β=1e-4，前者 Dice 提升 **{100*(patient_best['avg_dice_mean']-patient_reference['avg_dice_mean']):.2f} 个百分点**、HD95 降低 **{patient_reference['avg_hd95_mean']-patient_best['avg_hd95_mean']:.2f} px**；"
                        f"后者 Dice 提升 **{100*(slice_best['avg_dice_mean']-slice_reference['avg_dice_mean']):.2f} 个百分点**、HD95 降低 **{slice_reference['avg_hd95_mean']-slice_best['avg_hd95_mean']:.2f} px**。"
                        "β≥1e-2 后总体性能急剧下降，因此不应把某个单独 Vendor 的局部峰值当作全局配置。"
                    ),
                },
                {"id": "headline_metrics", "type": "metric-strip", "cardIds": ["patient_best", "slice_best", "coverage"]},
                {
                    "id": "curve_finding",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": "## 两种协议共享 1e-3 峰值，并在 1e-2 后快速分化或崩溃\n\n曲线用于观察整体形状：1e-5 到 1e-4 基本平坦，1e-3 带来一致改善；继续放大至 1e-2 后，病人协议整体失稳，切片协议也因 B/D 崩溃而明显下降。",
                },
                {"id": "curve", "type": "chart", "chartId": "dice_lr_curve", "layout": "full"},
                {
                    "id": "patient_finding",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": "## 病人随机协议在 1e-3 达峰，1e-2 已发生严重失稳\n\nβ=1e-3 在 B、C、D 三个 Vendor 上都取得各自最高的五 seed 平均 Dice。相比 β=1e-4，五个 seed 的跨 Vendor Dice 均改善；β=1e-2 时 B/D 首先崩溃，β=1e-1 与 1 接近失效。",
                },
                {"id": "patient_table", "type": "table", "tableId": "patient_results", "layout": "full"},
                {
                    "id": "slice_finding",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": "## 切片随机协议同样以 1e-3 最稳健，但 Vendor C 在 1e-2 有局部例外\n\n全局平均仍在 β=1e-3 达峰。Vendor C 在 β=1e-2 的 Dice 升至 0.7106 ± 0.0237，但同时 Vendor B/D 分别跌至 0.4462 与 0.4547；该点反映强烈的域依赖，不能作为统一超参数。",
                },
                {"id": "slice_table", "type": "table", "tableId": "slice_results", "layout": "full"},
                {
                    "id": "dynamic_finding",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": (
                        "## 配置 LR 不是实际步长：1e-3 的平均实际 LR 约为 β 的两成\n\n"
                        f"在 β=1e-3 时，病人协议平均实际 LR 为 **{diag_best[0]['effective_lr_mean_label']}**，"
                        f"切片协议为 **{diag_best[1]['effective_lr_mean_label']}**。GraTA 使用 "
                        "`effective_lr = β × (cos+1)^2 / 4`，所以扫描的是最大尺度 β；高 β 即使被动态门控，累积 BN affine 漂移仍会快速放大并导致分割退化。"
                    ),
                },
                {"id": "dynamic_table", "type": "table", "tableId": "effective_lr", "layout": "full"},
                {
                    "id": "scope",
                    "type": "markdown",
                    "body": "## 口径与数据范围\n\n- 源域：M&Ms Vendor A；目标域：Vendor B/C/D。\n- Backbone：随机初始化训练得到的 source seeds 2022–2026；目标病例/切片顺序 seed 与 checkpoint seed 相同。\n- 病人随机：BS=4，patient-volume 流，指标为病人级三前景类 macro Dice 与 3D HD95。\n- 切片随机：BS=8，all_slices 逐切片等权，指标为 2D macro Dice 与 2D HD95。\n- 两种协议都只加载 `manifest_has_fg_equals_1` 的切片。跨 Vendor 总分为 B/C/D 等权，不按样本量加权。",
                },
                {
                    "id": "methodology",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": "## 聚合与验证方法\n\n每个 LR×seed×协议先读取三个 Vendor 的 summary point estimate，并在 seed 内对 B/C/D 等权平均；随后对五个 seed 报告均值与样本标准差。独立核查 60/60 个 manifest、180 个 Vendor summary、协议/stream/checkpoint 哈希、LR 与 seed 字段，并扫描任务日志中的 `[VALIDATED]` 和错误标记。",
                },
                {
                    "id": "limitations",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": "## 限制与稳健性判断\n\n**可分享，但需保留口径限定。** 五个 backbone seed 给出了初始化不确定性，但本扫描没有为每个 checkpoint 重复多个独立 TTA augmentation seed，因此未单独量化 GraTA 强增强随机性的方差。PBS 的历史条目已从调度器中清理；任务完成性由 60 个落盘 manifest、180 个 summary 和 60 个 `[VALIDATED]` 日志交叉确认。切片协议与病人协议使用不同聚合粒度和 HD95 定义，不应直接把两张表的绝对分数解释成协议优劣。",
                },
                {
                    "id": "recommendation",
                    "type": "markdown",
                    "sourceId": source["id"],
                    "body": "## 建议\n\n1. 后续 GraTA 主实验统一采用 **β=1e-3**；在论文或配置中同时写明这是动态公式的 base/max LR，而非每步实际 LR。\n2. 若要做 Vendor-specific oracle，可单独记录切片协议 Vendor C 的 β=1e-2，但不能用于无监督统一模型选择。\n3. 正式 SoTA 对比应使用预先固定的 β=1e-3，并保留 β=1e-4 作为保守敏感性分析。",
                },
                {
                    "id": "further_questions",
                    "type": "markdown",
                    "body": "## 下一步问题\n\n- β=1e-3 相对 Source/TBN/TEnt/SAR/CoTTA 的增益是否超过对应五 seed 方差？\n- 将 method seed 与 checkpoint seed 解耦并重复 GraTA 强增强后，最优点是否仍稳定在 1e-3？",
                },
            ],
        },
        "snapshot": {
            "version": 1,
            "generatedAt": generated_at,
            "status": "ready",
            "datasets": {
                "headline": [
                    {
                        "patient_best_dice": round(patient_best["avg_dice_mean"], 4),
                        "slice_best_dice": round(slice_best["avg_dice_mean"], 4),
                        "validated_combinations": validation["combination_count"],
                    }
                ],
                "patient_results": patient,
                "slice_results": slices,
                "diagnostics": diagnostics,
                "lr_curve": lr_curve,
            },
        },
        "sources": [source],
    }


def main() -> None:
    generated_at = datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")
    validation = validate_manifests()
    if validation["issues"]:
        raise SystemExit("validation failed:\n- " + "\n- ".join(validation["issues"]))
    aggregated, raw = aggregate_metrics()
    diagnostics = adaptation_diagnostics()
    summary = {
        "generated_at": generated_at,
        "validation": validation,
        "aggregated": aggregated,
        "diagnostics": diagnostics,
        "raw_primary_metrics": raw,
        "chart_map": [
            {
                "section": "两种协议共享 1e-3 峰值",
                "question": "哪一个 GraTA base/max learning rate 在两种测试流协议下最稳健？",
                "family": "ordered relationship",
                "type": "line",
                "fields": ["log10_beta", "protocol", "dice", "dice_sd", "hd95"],
                "takeaway": "两种协议均在 beta=1e-3 达到跨 Vendor 平均 Dice 峰值，beta>=1e-2 后快速退化。",
                "palette": "blue solid for patient-volume; orange dashed for slice-random",
                "delivery": "reports/grata_lr_ablation/report.html",
                "source": "results/Stochastic_Ini_ForegroundOnly/grata_lr_sweep",
            }
        ],
    }
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / "aggregate_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n"
    )
    artifact = report_artifact(aggregated, diagnostics, validation, generated_at)
    (OUT / "artifact.json").write_text(
        json.dumps(artifact, indent=2, ensure_ascii=False) + "\n"
    )
    print(json.dumps({"status": "ready", "validation": validation}, ensure_ascii=False))


if __name__ == "__main__":
    main()
