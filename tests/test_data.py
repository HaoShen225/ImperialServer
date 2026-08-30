from __future__ import annotations

from data import _source_records, build_target_stream, split_volume_into_batches


def test_locked_source_split_and_target_disjoint(config):
    train = {row["patient_id"] for row in _source_records(config, "train")}
    validation = {row["patient_id"] for row in _source_records(config, "val")}
    assert len(train) == 60
    assert len(validation) == 15
    assert train.isdisjoint(validation)
    targets = {}
    for vendor, expected in (("B", 125), ("C", 75), ("D", 50)):
        stream = build_target_stream(vendor, config)
        targets[vendor] = {volume["patient_id"] for volume in stream.volumes}
        assert len(targets[vendor]) == expected
        assert len(stream) == expected * 2
        assert all([int(row["z_index"]) for row in volume["slices"]] == sorted(int(row["z_index"]) for row in volume["slices"]) for volume in stream.volumes)
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
    stream = build_target_stream("B", config)
    volume = stream[0]
    assert "mask" not in volume
    assert volume["image"].shape[0] == len(volume["mask_paths"])
    mask = stream.load_mask(volume)
    assert mask.shape == volume["image"].shape[:1] + volume["image"].shape[-2:]
