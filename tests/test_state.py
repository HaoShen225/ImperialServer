from __future__ import annotations

import ast
from pathlib import Path

import pytest

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
