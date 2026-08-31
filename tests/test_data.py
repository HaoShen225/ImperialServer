from __future__ import annotations

import random

import pytest

from data import _source_records, build_target_stream, split_volume_into_batches


def test_locked_source_split_and_target_disjoint(config):
    train = {row["patient_id"] for row in _source_records(config, "train")}
    validation = {row["patient_id"] for row in _source_records(config, "val")}
    assert len(train) == 60
    assert len(validation) == 15
    assert train.isdisjoint(validation)
    targets = {}
    for vendor, expected in (("B", 125), ("C", 50), ("D", 50)):
        stream = build_target_stream(vendor, config, order_seed=2022)
        targets[vendor] = {volume["patient_id"] for volume in stream.volumes}
        assert len(targets[vendor]) == expected
        assert len(stream) == expected * 2
        assert all([int(row["z_index"]) for row in volume["slices"]] == sorted(int(row["z_index"]) for row in volume["slices"]) for volume in stream.volumes)
        if vendor == "C":
            assert {
                row["original_part"] for volume in stream.volumes for row in volume["slices"]
            } == {"Testing", "Validation"}
    all_source = train | validation
    assert all(all_source.isdisjoint(value) for value in targets.values())
    assert targets["B"].isdisjoint(targets["C"])
    assert targets["B"].isdisjoint(targets["D"])
    assert targets["C"].isdisjoint(targets["D"])


def test_volume_batching_keeps_partial_batch(images):
    expanded = images.new_zeros((10, 1, 16, 16))
    batches = list(split_volume_into_batches(expanded, 4))
    assert [batch.shape[0] for batch in batches] == [4, 4, 2]
    assert torch_equal(expanded, __import__("torch").cat(batches))


def torch_equal(left, right):
    return bool(__import__("torch").equal(left, right))


def test_target_mask_is_lazy(config):
    stream = build_target_stream("B", config, order_seed=2022)
    volume = stream[0]
    assert "mask" not in volume
    assert volume["image"].shape[0] == len(volume["mask_paths"])
    mask = stream.load_mask(volume)
    assert mask.shape == volume["image"].shape[:1] + volume["image"].shape[-2:]


@pytest.mark.parametrize("vendor", ["B", "C", "D"])
def test_target_order_is_seeded_reproducible_and_patient_atomic(config, vendor):
    first = build_target_stream(vendor, config, order_seed=2022)
    repeated = build_target_stream(vendor, config, order_seed=2022)
    different = build_target_stream(vendor, config, order_seed=2023)

    assert first.order_seed == 2022
    assert first.patient_order == repeated.patient_order
    assert first.target_order_sha256 == repeated.target_order_sha256
    assert first.patient_order != different.patient_order
    assert first.target_order_sha256 != different.target_order_sha256
    assert set(first.patient_order) == set(different.patient_order)

    for patient_index, patient_id in enumerate(first.patient_order):
        ed, es = first.volumes[2 * patient_index : 2 * patient_index + 2]
        assert (ed["patient_id"], ed["phase"]) == (patient_id, "ED")
        assert (es["patient_id"], es["phase"]) == (patient_id, "ES")
        assert ed["patient_arrival_index"] == es["patient_arrival_index"] == patient_index
        assert ed["volume_arrival_index"] == 2 * patient_index
        assert es["volume_arrival_index"] == 2 * patient_index + 1


def test_target_order_does_not_mutate_global_random_state(config):
    random.seed(991)
    state = random.getstate()
    build_target_stream("B", config, order_seed=2022)
    assert random.getstate() == state


def test_target_order_requires_explicit_integer_seed(config):
    with pytest.raises(TypeError, match="integer checkpoint seed"):
        build_target_stream("B", config, order_seed=True)
