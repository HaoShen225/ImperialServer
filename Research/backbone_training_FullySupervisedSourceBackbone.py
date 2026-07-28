#!/usr/bin/env python3
"""Train the 4-domain x 5-seed fully supervised MMS source backbones.

This Research-level entrypoint exposes the implementation in
``helper.Clean_TTA_Protocol``.  With no task id it runs the complete matrix in
domain-major order.  ``--task-id 0..19`` selects one run for PBS arrays:

    A/seed0 .. A/seed4, B/seed0 .. B/seed4,
    C/seed0 .. C/seed4, D/seed0 .. D/seed4.

Checkpoints are stored below ``Research/backbone_params_cleanSource``.
"""

from __future__ import annotations

import sys
from typing import Sequence

from helper import Clean_TTA_Protocol as protocol


# Re-export the main protocol interfaces for callers and focused tests.
build_arg_parser = protocol.build_arg_parser
rebuild_run_summary = protocol.rebuild_run_summary
run_directory = protocol.run_directory
task_coordinates = protocol.task_coordinates
train_one_run = protocol.train_one_run


def main(argv: Sequence[str] | None = None) -> None:
    """Run the fully supervised MMS source-backbone protocol."""
    protocol.main(argv)


if __name__ == "__main__":
    main(sys.argv[1:])
