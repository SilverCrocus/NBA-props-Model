#!/usr/bin/env python3
"""
Simple NBA Betting Recommendations
Usage: uv run nba_today.py [date] [bankroll]

Examples:
  uv run nba_today.py                    # Today's games, $1000 bankroll
  uv run nba_today.py 2025-10-21         # Specific date, $1000 bankroll
  uv run nba_today.py 2025-10-21 5000    # Specific date, $5000 bankroll
"""

import subprocess
import sys
from datetime import datetime

# Get date and bankroll from arguments
if len(sys.argv) >= 2:
    target_date = sys.argv[1]
else:
    target_date = datetime.now().strftime("%Y-%m-%d")

if len(sys.argv) >= 3:
    bankroll = sys.argv[2]
else:
    bankroll = "1000"

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
