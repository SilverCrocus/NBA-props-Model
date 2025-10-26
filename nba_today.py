#!/usr/bin/env python3
"""
Simple NBA Betting Recommendations
Usage: uv run nba_today.py [date] [bankroll] [--update]

Examples:
  uv run nba_today.py                      # Today's games, $1000 bankroll
  uv run nba_today.py 2025 - 10 - 21           # Specific date, $1000 bankroll
  uv run nba_today.py 2025 - 10 - 21 5000      # Specific date, $5000 bankroll
  uv run nba_today.py --update             # Update data first, then predict
  uv run nba_today.py 2025 - 10 - 21 --update  # Update + specific date

Note: To retrain model too, use: uv run update_and_predict.py
"""

import subprocess
import sys
from datetime import datetime

# Parse arguments
args = sys.argv[1:]
update_data = "--update" in args

# Remove --update from args if present
if update_data:
    args = [arg for arg in args if arg != "--update"]

# Get date and bankroll from remaining arguments
if len(args) >= 1:
    target_date = args[0]
else:
    target_date = datetime.now().strftime("%Y-%m-%d")

if len(args) >= 2:
    bankroll = args[1]
else:
    bankroll = "1000"

# Update game logs if requested
if update_data:
    print("=" * 80)
    print("UPDATING GAME LOGS...")
    print("=" * 80)
    update_cmd = ["uv", "run", "python", "scripts/update_latest_games.py"]
    result = subprocess.run(update_cmd)

    if result.returncode != 0:
        print("\n⚠️  WARNING: Game log update failed")
        print("   Continuing with existing data...")

# Run the main script
cmd = [
    "uv",
    "run",
    "python",
    "scripts/production/daily_betting_recommendations.py",
    "--date",
    target_date,
    "--bankroll",
    bankroll,
    "--save-html",
]

subprocess.run(cmd)
