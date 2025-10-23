#!/usr/bin/env python3
"""
Quick Bet Tracking Commands
============================
Simplified interface for bet tracking workflow.

Usage:
  # Record bets
  uv run track_bet.py record 2025-10-21 --top 5
  uv run track_bet.py record 2025-10-21 --all

  # Update results
  uv run track_bet.py update 2025-10-21
  uv run track_bet.py update --all

  # View dashboard
  uv run track_bet.py status
  uv run track_bet.py status --detailed
"""

import subprocess
import sys
from pathlib import Path

ROOT_DIR = Path(__file__).parent


def record_bets(args):
    """Record placed bets."""
    cmd = ["uv", "run", "python", "scripts/betting/record_bets.py"] + args
    subprocess.run(cmd)


def update_results(args):
    """Update bet results."""
    cmd = ["uv", "run", "python", "scripts/betting/update_results.py"] + args
    subprocess.run(cmd)


def show_status(args):
    """Show betting dashboard."""
    cmd = ["uv", "run", "python", "scripts/betting/betting_dashboard.py"] + args
    subprocess.run(cmd)


def main():
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(1)

    command = sys.argv[1]
    args = sys.argv[2:]

    if command == "record":
        record_bets(args)
    elif command == "update":
        update_results(args)
    elif command == "status":
        show_status(args)
    else:
        print(f"Unknown command: {command}")
        print(__doc__)
        sys.exit(1)


if __name__ == "__main__":
    main()
