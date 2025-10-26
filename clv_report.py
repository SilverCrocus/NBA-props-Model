#!/usr/bin/env python3
"""
Simple wrapper for CLV validation report
Usage: uv run clv_report.py
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from utils.clv_tracker import CLVTracker  # noqa: E402

if __name__ == "__main__":
    tracker = CLVTracker()
    tracker.print_clv_report()
