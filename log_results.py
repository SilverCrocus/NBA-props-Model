#!/usr/bin/env python3
"""
Simple wrapper for results logger
Usage: uv run log_results.py 2025 - 10 - 24
"""
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the actual script
from scripts.clv.log_results import main  # noqa: E402

if __name__ == "__main__":
    sys.exit(main())
