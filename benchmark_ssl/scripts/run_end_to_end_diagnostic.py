#!/usr/bin/env python3
"""Entry-point wrapper — see the imported module for documentation."""

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.analysis.end_to_end_diagnostic import main

if __name__ == "__main__":
    main()
