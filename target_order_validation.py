"""Validation helpers for checkpoint-seeded target-domain arrival orders."""

from __future__ import annotations

from typing import Any, Sequence

from data import TARGET_ORDER_POLICY, build_target_stream


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
        )
        wanted = (
            vendor,
            volume["patient_id"],
            volume["phase"],
            volume["patient_arrival_index"],
            volume["volume_arrival_index"],
            seed,
            expected.target_order_sha256,
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
