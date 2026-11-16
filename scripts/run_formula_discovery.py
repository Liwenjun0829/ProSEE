#!/usr/bin/env python3
"""Run the ProSEE symbolic formula-discovery pipeline from a checkout."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SOURCE_ROOT = PROJECT_ROOT / "src"
if str(SOURCE_ROOT) not in sys.path:
    sys.path.insert(0, str(SOURCE_ROOT))

from prosee.symbolic.cli import main  # noqa: E402


if __name__ == "__main__":
    main()
