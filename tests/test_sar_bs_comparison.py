from __future__ import annotations

from copy import deepcopy

import pytest

from run_sar_bs_comparison import (
    BATCH_SIZES,
    DEFAULT_RESULTS_ROOT,
    LEARNING_RATE,
    PROFILE_KIND,
    SOURCE_SEEDS,
    STREAM_MODES,
    comparison_cells,
    configure_sar_comparison,
    resolve_array_index,
    run_root,
)


@pytest.mark.parametrize("stream_mode", STREAM_MODES)
def test_sar_comparison_configuration_is_isolated(config, tmp_path, stream_mode):
    original = deepcopy(config)
    resolved = configure_sar_comparison(config, stream_mode, tmp_path)
    assert config == original
    assert resolved["tta"]["stream_mode"] == stream_mode
    assert resolved["tta"]["batch_size"] == BATCH_SIZES[stream_mode]
    assert resolved["tta"]["timing"] == "adapt_then_predict"
    assert resolved["tta"]["reset"] == "vendor"
    assert resolved["tta"]["results_dir"] == str(tmp_path)
    assert resolved["methods"]["sar"]["lr"] == pytest.approx(LEARNING_RATE)
    assert resolved["methods"]["sar"]["profile_kind"] == PROFILE_KIND
    expected_method = deepcopy(original["methods"]["sar"])
    expected_method.update(lr=LEARNING_RATE, profile_kind=PROFILE_KIND)
    assert resolved["methods"]["sar"] == expected_method


def test_sar_array_contains_exactly_ten_unique_cells():
    cells = comparison_cells()
    assert len(cells) == len(SOURCE_SEEDS) * len(STREAM_MODES) == 10
    assert len(set(cells)) == len(cells)
    assert tuple(resolve_array_index(index) for index in range(10)) == cells
    assert cells[0] == (2022, "patient_volume")
    assert cells[-1] == (2026, "slice_random")
    with pytest.raises(ValueError, match="array index"):
        resolve_array_index(10)


@pytest.mark.parametrize("stream_mode", STREAM_MODES)
def test_sar_result_root_is_protocol_specific(stream_mode):
    path = run_root(DEFAULT_RESULTS_ROOT, 2022, stream_mode)
    suffix = (
        "adapt_then_predict_vendor"
        if stream_mode == "patient_volume"
        else "slice_random_adapt_then_predict_vendor"
    )
    assert path == DEFAULT_RESULTS_ROOT / "sar" / "seed2022" / suffix
