#!/usr/bin/env python3
"""
Simple wrapper for daily betting recommendations.

Usage:
    uv run nba_today.py                    # Default: $1,000 bankroll, moderate strategy
    uv run nba_today.py 2000               # $2,000 bankroll
    uv run nba_today.py 2000 aggressive    # $2,000 bankroll, aggressive strategy
    uv run nba_today.py 500 conservative   # $500 bankroll, conservative strategy
"""

import sys
import subprocess

# Default values
bankroll = 1000
strategy = "moderate"

# Parse arguments
if len(sys.argv) > 1:
    try:
        bankroll = float(sys.argv[1])
    except ValueError:
        print(f"❌ Invalid bankroll: {sys.argv[1]}")
        print("Usage: uv run nba_today.py [bankroll] [strategy]")
        sys.exit(1)

if len(sys.argv) > 2:
    strategy = sys.argv[2].lower()
    if strategy not in ["conservative", "moderate", "aggressive", "maximum"]:
        print(f"❌ Invalid strategy: {strategy}")
        print("Valid strategies: conservative, moderate, aggressive, maximum")
        sys.exit(1)

# Build command
cmd = [
    "python",
    "scripts/production/daily_betting_recommendations.py",
    "--bankroll", str(bankroll),
    "--strategy", strategy
]

# Print what we're doing
print(f"🎯 Getting recommendations for today...")
print(f"   Bankroll: ${bankroll:,.0f}")
print(f"   Strategy: {strategy.upper()}")
print()

# Run the command
result = subprocess.run(cmd)
sys.exit(result.returncode)
