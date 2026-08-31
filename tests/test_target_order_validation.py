from __future__ import annotations

from copy import deepcopy

import pytest

from data import TARGET_ORDER_POLICY, build_target_stream
from target_order_validation import require_distinct_seed_orders, validate_target_order


def _manifest_and_records(config, vendor: str, seed: int):
    stream = build_target_stream(vendor, config, order_seed=seed)
    manifest = {
        "source_seed": seed,
        "target_order_seed": seed,
        "vendors": ["B", "C", "D"],
        "target_order_policy": {
            **TARGET_ORDER_POLICY,
            "vendor_order": ["B", "C", "D"],
        },
        "target_orders": {
            vendor: {
                "order_seed": seed,
                "patient_ids": stream.patient_order,
                "target_order_sha256": stream.target_order_sha256,
            }
        },
    }
    records = [
        {
            "vendor": vendor,
            "patient_id": volume["patient_id"],
            "phase": volume["phase"],
            "patient_arrival_index": volume["patient_arrival_index"],
            "volume_arrival_index": volume["volume_arrival_index"],
            "target_order_seed": seed,
            "target_order_sha256": stream.target_order_sha256,
        }
        for volume in stream.volumes
    ]
    return manifest, records, stream.target_order_sha256


def test_validate_target_order_accepts_exact_stream(config):
    manifest, records, expected_hash = _manifest_and_records(config, "C", 2022)
    assert validate_target_order(manifest, records, config, "C", 2022) == expected_hash


def test_validate_target_order_rejects_reordered_records(config):
    manifest, records, _ = _manifest_and_records(config, "C", 2022)
    invalid = deepcopy(records)
    invalid[0], invalid[2] = invalid[2], invalid[0]
    with pytest.raises(RuntimeError, match="arrival sequence mismatch"):
        validate_target_order(manifest, invalid, config, "C", 2022)


def test_require_distinct_seed_orders():
    hashes = {
        2022: {"B": "b1", "C": "c1"},
        2023: {"B": "b2", "C": "c2"},
    }
    require_distinct_seed_orders(hashes, [2022, 2023], ["B", "C"])
    hashes[2023]["C"] = "c1"
    with pytest.raises(RuntimeError, match="distinct order"):
        require_distinct_seed_orders(hashes, [2022, 2023], ["B", "C"])
