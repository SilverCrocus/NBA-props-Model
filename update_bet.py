#!/usr/bin/env python3
"""
Simple wrapper for manual bet logger
Usage: uv run update_bet.py
"""
import sys
from pathlib import Path

from scripts.clv.log_manual_bet import main

# Add project root to path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

# Import and run the actual script

if __name__ == "__main__":
    sys.exit(main())
