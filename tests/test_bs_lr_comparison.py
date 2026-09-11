from __future__ import annotations

from copy import deepcopy

import pytest

from run_bs_lr_comparison import (
    ADAPTIVE_METHODS,
    BATCH_SIZES,
    DEFAULT_RESULTS_ROOT,
    LEARNING_RATE,
    METHODS,
    PROFILE_KIND,
    SOURCE_SEEDS,
    STREAM_MODES,
    comparison_cells,
    configure_comparison,
    resolve_array_index,
    run_root,
)


@pytest.mark.parametrize(
    ("stream_mode", "batch_size"),
    [("patient_volume", 8), ("slice_random", 4)],
)
@pytest.mark.parametrize("method_name", METHODS)
def test_comparison_configuration_is_isolated(
    config, tmp_path, method_name, stream_mode, batch_size
):
    original = deepcopy(config)
    resolved = configure_comparison(config, method_name, stream_mode, tmp_path)
    assert config == original
    assert resolved["tta"]["stream_mode"] == stream_mode
    assert resolved["tta"]["batch_size"] == batch_size
    assert resolved["tta"]["results_dir"] == str(tmp_path)
    if method_name in ADAPTIVE_METHODS:
        assert resolved["methods"][method_name]["lr"] == pytest.approx(LEARNING_RATE)
        assert resolved["methods"][method_name]["profile_kind"] == PROFILE_KIND
    else:
        assert "lr" not in resolved["methods"][method_name]
        assert resolved["methods"][method_name] == original["methods"][method_name]


def test_comparison_array_contains_exactly_50_unique_cells():
    cells = comparison_cells()
    assert len(cells) == len(METHODS) * len(SOURCE_SEEDS) * len(STREAM_MODES) == 50
    assert len(set(cells)) == len(cells)
    assert tuple(resolve_array_index(index) for index in range(50)) == cells
    assert cells[0] == ("source", 2022, "patient_volume")
    assert cells[-1] == ("grata", 2026, "slice_random")
    with pytest.raises(ValueError, match="array index"):
        resolve_array_index(50)


@pytest.mark.parametrize("stream_mode", STREAM_MODES)
def test_comparison_result_root_is_protocol_specific(stream_mode):
    path = run_root(DEFAULT_RESULTS_ROOT, "tent", 2022, stream_mode)
    expected_suffix = (
        "adapt_then_predict_vendor"
        if stream_mode == "patient_volume"
        else "slice_random_adapt_then_predict_vendor"
    )
    assert path == DEFAULT_RESULTS_ROOT / "tent" / "seed2022" / expected_suffix
    assert BATCH_SIZES[stream_mode] in {4, 8}
