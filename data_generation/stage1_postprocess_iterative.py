#!/usr/bin/env python3
"""Compatibility wrapper for the packaged iterative stage1 postprocess entrypoint."""

from __future__ import annotations

import sys
from pathlib import Path


def _bootstrap_import() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    src_path = repo_root / "src"
    if str(src_path) not in sys.path:
        sys.path.insert(0, str(src_path))


def main(argv: list[str] | None = None) -> int:
    _bootstrap_import()
    from contextual_drag.data.stage1_postprocess_iterative_cli import Stage1PostprocessIterativeCLI

    return int(Stage1PostprocessIterativeCLI.main(argv=argv) or 0)


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
