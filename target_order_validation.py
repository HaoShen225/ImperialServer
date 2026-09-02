"""Validation helpers for checkpoint-seeded target-domain arrival orders."""

from __future__ import annotations

from typing import Any, Sequence

from data import (
    TARGET_ORDER_POLICY,
    TARGET_SLICE_ORDER_POLICY,
    MMSTargetSliceDataset,
    build_target_slice_loader,
    build_target_stream,
)


def validate_target_order(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    cfg: dict[str, Any],
    vendor: str,
    checkpoint_seed: int,
) -> str:
    """Rebuild and exactly validate one vendor stream; return its order hash."""
    seed = int(checkpoint_seed)
    expected = build_target_stream(vendor, cfg, order_seed=seed)
    expected_policy = {
        **TARGET_ORDER_POLICY,
        "vendor_order": list(manifest["vendors"]),
    }
    if int(manifest.get("source_seed", -1)) != seed:
        raise RuntimeError(f"Manifest source seed does not match checkpoint seed {seed}")
    if int(manifest.get("target_order_seed", -1)) != seed:
        raise RuntimeError(f"Manifest target order seed does not match checkpoint seed {seed}")
    if manifest.get("target_order_policy") != expected_policy:
        raise RuntimeError("Manifest target order policy differs from the locked policy")

    expected_manifest_order = {
        "order_seed": seed,
        "patient_ids": expected.patient_order,
        "target_order_sha256": expected.target_order_sha256,
        "target_content_sha256": expected.target_content_sha256,
        "n_slices": expected.n_slices,
        "slice_filter": expected.slice_filter,
    }
    actual_manifest_order = manifest.get("target_orders", {}).get(vendor)
    if actual_manifest_order != expected_manifest_order:
        raise RuntimeError(f"Manifest target order is invalid for Vendor {vendor}, seed {seed}")
    if len(records) != len(expected.volumes):
        raise RuntimeError(
            f"Vendor {vendor}, seed {seed} has {len(records)} records; "
            f"expected {len(expected.volumes)}"
        )

    for record, volume in zip(records, expected.volumes):
        actual = (
            record.get("vendor"),
            record.get("patient_id"),
            record.get("phase"),
            record.get("patient_arrival_index"),
            record.get("volume_arrival_index"),
            record.get("target_order_seed"),
            record.get("target_order_sha256"),
            record.get("target_content_sha256"),
            record.get("slice_filter"),
            record.get("n_slices"),
            record.get("slice_ids"),
            record.get("z_indices"),
        )
        wanted = (
            vendor,
            volume["patient_id"],
            volume["phase"],
            volume["patient_arrival_index"],
            volume["volume_arrival_index"],
            seed,
            expected.target_order_sha256,
            expected.target_content_sha256,
            expected.slice_filter,
            len(volume["slices"]),
            [row["slice_id"] for row in volume["slices"]],
            [int(row["z_index"]) for row in volume["slices"]],
        )
        if actual != wanted:
            raise RuntimeError(
                f"Target arrival sequence mismatch for Vendor {vendor}, seed {seed}, "
                f"volume index {volume['volume_arrival_index']}"
            )
    return str(expected.target_order_sha256)


def require_distinct_seed_orders(
    order_hashes: dict[int, dict[str, str]],
    seeds: Sequence[int],
    vendors: Sequence[str],
) -> None:
    """Reject an aggregate if five seeds did not resolve distinct per-vendor orders."""
    for vendor in vendors:
        hashes = [order_hashes[int(seed)][vendor] for seed in seeds]
        if len(set(hashes)) != len(hashes):
            raise RuntimeError(f"Vendor {vendor} does not have one distinct order per seed")


def validate_target_slice_order(
    manifest: dict[str, Any],
    records: Sequence[dict[str, Any]],
    cfg: dict[str, Any],
    vendor: str,
    checkpoint_seed: int,
) -> str:
    """Rebuild and exactly validate one Vendor-local random slice stream."""
    seed = int(checkpoint_seed)
    loader = build_target_slice_loader(vendor, cfg, order_seed=seed)
    expected = loader.dataset
    if not isinstance(expected, MMSTargetSliceDataset):
        raise TypeError("Target slice loader has an unexpected dataset type")
    expected_policy = {
        **TARGET_SLICE_ORDER_POLICY,
        "vendor_order": list(manifest["vendors"]),
    }
    if manifest.get("stream_mode") != "slice_random":
        raise RuntimeError("Manifest is not a slice_random run")
    if int(manifest.get("source_seed", -1)) != seed:
        raise RuntimeError(f"Manifest source seed does not match checkpoint seed {seed}")
    if int(manifest.get("target_order_seed", -1)) != seed:
        raise RuntimeError(f"Manifest target order seed does not match checkpoint seed {seed}")
    if manifest.get("target_order_policy") != expected_policy:
        raise RuntimeError("Manifest target slice order policy differs from the locked policy")

    expected_manifest_order = {
        "order_seed": seed,
        "n_slices": len(expected),
        "slice_order_sha256": expected.slice_order_sha256,
        "slice_filter": expected.slice_filter,
    }
    if manifest.get("target_orders", {}).get(vendor) != expected_manifest_order:
        raise RuntimeError(f"Manifest target slice order is invalid for Vendor {vendor}, seed {seed}")
    if len(records) != len(expected):
        raise RuntimeError(
            f"Vendor {vendor}, seed {seed} has {len(records)} slice records; "
            f"expected {len(expected)}"
        )
    for arrival_index, (record, expected_row) in enumerate(zip(records, expected.records)):
        actual = (
            record.get("vendor"),
            record.get("patient_id"),
            record.get("phase"),
            record.get("z_index"),
            record.get("slice_id"),
            record.get("slice_arrival_index"),
            record.get("target_order_seed"),
            record.get("slice_order_sha256"),
            record.get("slice_filter"),
        )
        wanted = (
            vendor,
            expected_row["patient_id"],
            expected_row["phase"],
            int(expected_row["z_index"]),
            expected_row["slice_id"],
            arrival_index,
            seed,
            expected.slice_order_sha256,
            expected.slice_filter,
        )
        if actual != wanted:
            raise RuntimeError(
                f"Target slice arrival sequence mismatch for Vendor {vendor}, seed {seed}, "
                f"slice index {arrival_index}"
            )
    return str(expected.slice_order_sha256)
