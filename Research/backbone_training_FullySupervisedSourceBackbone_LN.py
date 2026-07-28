#!/usr/bin/env python3
"""Train fully supervised MMS source backbones with the LayerNorm U-Net.

This entrypoint uses the same data matrix, supervised objective, optimization,
checkpointing, and resume behavior as
``backbone_training_FullySupervisedSourceBackbone.py``.  Only the backbone and
architecture provenance differ:

    A/seed0 .. A/seed4, B/seed0 .. B/seed4,
    C/seed0 .. C/seed4, D/seed0 .. D/seed4.

Checkpoints are stored below ``Research/backbone_params_cleanSource_LN``.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import torch

from helper import Clean_TTA_Protocol as protocol
from helper.backbones.UNet_with_LayerNorms import UNet


RESEARCH_ROOT = Path(__file__).resolve().parent
DEFAULT_OUTPUT_ROOT = RESEARCH_ROOT / "backbone_params_cleanSource_LN"
ARCHITECTURE = "UNetLayerNorm"
NORMALIZATION = "LayerNorm2d"
NORMALIZATION_AXES = "per_pixel_channels"
LAYER_NORM_EPS = 1e-5

_ORIGINAL_PROTOCOL_SIGNATURE = protocol.protocol_signature
_ORIGINAL_FINAL_CHECKPOINT_PAYLOAD = protocol._final_checkpoint_payload


def build_model() -> UNet:
    """Build the four-class LayerNorm U-Net used by this experiment."""
    return UNet(
        n_channels=1,
        n_classes=protocol.NUM_CLASSES,
        only_feature=False,
        bilinear=False,
    )


def backbone_id(vendor: str, seed: int) -> str:
    """Return a stable id that distinguishes LN and BN source backbones."""
    info = protocol.VENDOR_INFO[protocol.resolve_vendor(vendor)]
    return f"CleanMMS_{info['domain']}_Seed{int(seed)}_{ARCHITECTURE}"


def protocol_signature(
    args: Any,
    vendor: str,
    seed: int,
) -> Dict[str, Any]:
    """Add normalization provenance to resumable-state compatibility checks."""
    signature = _ORIGINAL_PROTOCOL_SIGNATURE(args, vendor, seed)
    signature.update(
        {
            "architecture": ARCHITECTURE,
            "normalization": NORMALIZATION,
            "normalization_axes": NORMALIZATION_AXES,
            "layer_norm_eps": LAYER_NORM_EPS,
        }
    )
    return signature


def final_checkpoint_payload(**kwargs: Any) -> Dict[str, Any]:
    """Add LayerNorm architecture provenance to the shared checkpoint schema."""
    payload = _ORIGINAL_FINAL_CHECKPOINT_PAYLOAD(**kwargs)
    metadata = payload.setdefault("metadata", {})
    if not isinstance(metadata, Mapping):
        raise TypeError("Checkpoint metadata must be a mapping.")
    metadata.update(
        {
            "architecture": ARCHITECTURE,
            "normalization": NORMALIZATION,
            "normalization_axes": NORMALIZATION_AXES,
            "layer_norm_eps": LAYER_NORM_EPS,
            "batch_independent_normalization": True,
        }
    )
    return payload


def install_layernorm_protocol_hooks() -> None:
    """Install LN-specific hooks only inside this Python process."""
    protocol.DEFAULT_OUTPUT_ROOT = DEFAULT_OUTPUT_ROOT
    protocol.build_model = build_model
    protocol.backbone_id = backbone_id
    protocol.protocol_signature = protocol_signature
    protocol._final_checkpoint_payload = final_checkpoint_payload


install_layernorm_protocol_hooks()

# Re-export the main protocol interfaces for callers and focused tests.
build_arg_parser = protocol.build_arg_parser
rebuild_run_summary = protocol.rebuild_run_summary
run_directory = protocol.run_directory
task_coordinates = protocol.task_coordinates
train_one_run = protocol.train_one_run


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fully supervised MMS LayerNorm source-backbone protocol."""
    protocol.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
