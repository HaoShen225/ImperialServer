from __future__ import annotations

import random

import pytest

from data import (
    MMSTargetSliceDataset,
    _source_records,
    build_target_slice_loader,
    build_source_validation_volumes,
    build_target_stream,
    split_volume_into_batches,
)


def test_locked_source_split_and_target_disjoint(config):
    train_records = _source_records(config, "train")
    validation_records = _source_records(config, "val")
    train = {row["patient_id"] for row in train_records}
    validation = {row["patient_id"] for row in validation_records}
    assert len(train_records) == 1017
    assert len(validation_records) == 255
    assert all(row["has_fg"] == "1" and int(row["fg_pixels"]) > 0 for row in train_records)
    assert all(row["has_fg"] == "1" and int(row["fg_pixels"]) > 0 for row in validation_records)
    assert len(train) == 60
    assert len(validation) == 15
    assert train.isdisjoint(validation)
    targets = {}
    for vendor, expected in (("B", 125), ("C", 50), ("D", 50)):
        stream = build_target_stream(vendor, config, order_seed=2022)
        targets[vendor] = {volume["patient_id"] for volume in stream.volumes}
        assert len(targets[vendor]) == expected
        assert len(stream) == expected * 2
        assert all(
            row["has_fg"] == "1" and int(row["fg_pixels"]) > 0
            for volume in stream.volumes for row in volume["slices"]
        )
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
    assert volume["patient_arrival_index"] == 0
    assert volume["volume_arrival_index"] == 0
    assert volume["phase_arrival_index"] == 0


def test_source_validation_volume_remains_compatible_without_arrival_metadata(config):
    stream = build_source_validation_volumes(config)
    assert stream.n_slices == 255
    assert all(
        row["has_fg"] == "1" and int(row["fg_pixels"]) > 0
        for item in stream.volumes for row in item["slices"]
    )
    volume = stream[0]
    assert "patient_arrival_index" not in volume
    assert "volume_arrival_index" not in volume
    assert "phase_arrival_index" not in volume


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
    for volume_index in (0, 1, len(first) - 1):
        expected_volume = first.volumes[volume_index]
        loaded_volume = first[volume_index]
        for key in (
            "patient_arrival_index",
            "volume_arrival_index",
            "phase_arrival_index",
        ):
            assert loaded_volume[key] == expected_volume[key]


def test_target_order_does_not_mutate_global_random_state(config):
    random.seed(991)
    state = random.getstate()
    build_target_stream("B", config, order_seed=2022)
    assert random.getstate() == state


def test_target_order_requires_explicit_integer_seed(config):
    with pytest.raises(TypeError, match="integer checkpoint seed"):
        build_target_stream("B", config, order_seed=True)


@pytest.mark.parametrize(
    "vendor,expected,last_batch",
    [("B", 2049, 1), ("C", 806, 6), ("D", 835, 3)],
)
def test_random_slice_stream_has_exact_seeded_coverage(config, vendor, expected, last_batch):
    first_loader = build_target_slice_loader(vendor, config, order_seed=2022, batch_size=8)
    repeated_loader = build_target_slice_loader(vendor, config, order_seed=2022, batch_size=8)
    different_loader = build_target_slice_loader(vendor, config, order_seed=2023, batch_size=8)
    first = first_loader.dataset
    repeated = repeated_loader.dataset
    different = different_loader.dataset
    assert isinstance(first, MMSTargetSliceDataset)
    assert len(first) == expected
    assert len(first.slice_order) == len(set(first.slice_order))
    assert first.slice_order == repeated.slice_order
    assert first.slice_order_sha256 == repeated.slice_order_sha256
    assert first.slice_order != different.slice_order
    assert first.slice_order_sha256 != different.slice_order_sha256
    assert set(first.slice_order) == set(different.slice_order)
    assert [len(indices) for indices in first_loader.batch_sampler][-1] == last_batch
    assert all(row["has_fg"] == "1" and int(row["fg_pixels"]) > 0 for row in first.records)
    assert any(
        len({row["patient_id"] for row in first.records[start : start + 8]}) > 1
        for start in range(0, len(first), 8)
    )
    assert any(
        len({row["phase"] for row in first.records[start : start + 8]}) > 1
        for start in range(0, len(first), 8)
    )
    if vendor == "C":
        assert {row["original_part"] for row in first.records} == {"Testing", "Validation"}


def test_random_slice_stream_does_not_mutate_global_random_state(config):
    random.seed(913)
    state = random.getstate()
    build_target_slice_loader("B", config, order_seed=2022)
    assert random.getstate() == state


def test_random_slice_stream_requires_explicit_integer_seed(config):
    with pytest.raises(TypeError, match="integer checkpoint seed"):
        build_target_slice_loader("B", config, order_seed=False)
