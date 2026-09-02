from __future__ import annotations

import ast
from pathlib import Path

import pytest

import utils
from utils import load_config


def test_method_import_hygiene():
    root = Path(__file__).resolve().parents[1] / "tta_methods"
    forbidden = {"data", "metrics", "run_tta"}
    for path in root.rglob("*.py"):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        imports = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                imports.update(alias.name.split(".")[0] for alias in node.names)
            elif isinstance(node, ast.ImportFrom) and node.module:
                imports.add(node.module.split(".")[0])
        assert not imports.intersection(forbidden), f"{path} imports {imports.intersection(forbidden)}"


def test_unknown_method_config_key_rejected(tmp_path):
    original = Path(__file__).resolve().parents[1] / "config.yaml"
    text = original.read_text(encoding="utf-8").replace("    steps: 1\n    update_scope: bn_affine", "    steps: 1\n    unexpected: true\n    update_scope: bn_affine", 1)
    path = tmp_path / "bad.yaml"
    path.write_text(text, encoding="utf-8")
    with pytest.raises(ValueError, match="Unknown configuration"):
        load_config(path)


def test_set_seed_uses_warn_only_determinism(monkeypatch):
    calls = []
    monkeypatch.setattr(
        utils.torch,
        "use_deterministic_algorithms",
        lambda mode, *, warn_only=False: calls.append((mode, warn_only)),
    )

    utils.set_seed(2022, deterministic=True)

    assert calls == [(True, True)]


def test_stochastic_artifact_namespaces_are_locked():
    cfg = load_config(Path(__file__).resolve().parents[1] / "config.yaml")
    assert cfg["source"]["checkpoint_dir"] == "checkpoints/Stochastic_Ini_ForegroundOnly"
    assert cfg["tta"]["results_dir"] == "results/Stochastic_Ini_ForegroundOnly"
    assert cfg["tta"]["stream_mode"] == "patient_volume"
    assert cfg["data"]["slice_stream_file"] == "splits/target_slice_streams.json"
    assert cfg["data"]["slice_filter"] == "manifest_has_fg_equals_1"
