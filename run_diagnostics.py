#!/usr/bin/env python3
"""
Simple wrapper for advanced betting diagnostics

Usage: uv run run_diagnostics.py
"""
import sys
from pathlib import Path

# Add scripts to path
sys.path.insert(0, str(Path(__file__).parent / "scripts" / "validation"))

from advanced_diagnostics import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
