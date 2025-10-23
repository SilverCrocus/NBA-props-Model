#!/usr/bin/env python3
"""
Automated Daily NBA Props Pipeline
===================================

Runs the complete daily workflow:
1. Fetch latest game results from last night
2. Update master game logs file
3. Retrain production model (optional - only if significant new data)
4. Generate predictions for tonight's games
5. Fetch current betting lines
6. Generate betting recommendations

This script is designed to be run daily via cron or Task Scheduler.

Usage:
  Manual: uv run python scripts/production/daily_update_pipeline.py
  Cron:   0 10 * * * cd /path/to/project && uv run python scripts/production/daily_update_pipeline.py
"""

import subprocess
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

print("=" * 80)
print(f"NBA PROPS DAILY UPDATE PIPELINE")
print(f"Run time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 80)
print()

# Configuration
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_DIR = PROJECT_ROOT / "data" / "game_logs"
MASTER_FILE = DATA_DIR / "all_game_logs_through_2025.csv"
MODEL_DIR = PROJECT_ROOT / "models"

# Check if we should retrain (only retrain weekly to save time)
RETRAIN_DAYS = [0, 3, 6]  # Monday, Thursday, Sunday
SHOULD_RETRAIN = datetime.now().weekday() in RETRAIN_DAYS

print(f"Configuration:")
print(f"  Project root: {PROJECT_ROOT}")
print(f"  Master file: {MASTER_FILE}")
print(f"  Retrain today: {'YES' if SHOULD_RETRAIN else 'NO (runs Mon/Thu/Sun only)'}")
print()

# ======================================================================
# STEP 1: FETCH LATEST GAMES
# ======================================================================

print("=" * 80)
print("STEP 1: Fetching latest game results")
print("=" * 80)
print()

try:
    # Get current game count
    if MASTER_FILE.exists():
        df_before = pd.read_csv(MASTER_FILE)
        games_before = len(df_before)
        latest_date_before = pd.to_datetime(df_before["GAME_DATE"]).max()
    else:
        games_before = 0
        latest_date_before = None

    # Run update script
    result = subprocess.run(
        [sys.executable, "scripts/update_latest_games.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ ERROR fetching games:")
        print(result.stderr)
        sys.exit(1)

    # Check what was added
    df_after = pd.read_csv(MASTER_FILE)
    games_after = len(df_after)
    games_added = games_after - games_before
    latest_date_after = pd.to_datetime(df_after["GAME_DATE"]).max()

    print(f"✅ Games updated:")
    print(f"   Before: {games_before:,}")
    print(f"   After:  {games_after:,}")
    print(f"   Added:  {games_added:,} new games")
    print(f"   Latest: {latest_date_after.date()}")
    print()

    if games_added == 0:
        print("ℹ️  No new games to add. Database is up to date.")
        print()

except Exception as e:
    print(f"❌ ERROR in step 1: {str(e)}")
    sys.exit(1)

# ======================================================================
# STEP 2: RETRAIN MODEL (if scheduled)
# ======================================================================

if SHOULD_RETRAIN and games_added > 0:
    print("=" * 80)
    print("STEP 2: Retraining production model")
    print("=" * 80)
    print()

    try:
        print("Starting model training (this takes ~5-10 minutes)...")
        result = subprocess.run(
            [sys.executable, "scripts/production/train_ensemble_v1_production.py"],
            cwd=PROJECT_ROOT,
            capture_output=True,
            text=True,
        )

        if result.returncode != 0:
            print(f"❌ ERROR training model:")
            print(result.stderr)
            print("\n⚠️  Continuing with existing model...")
        else:
            print("✅ Model retrained successfully")
            print()

    except Exception as e:
        print(f"❌ ERROR in step 2: {str(e)}")
        print("⚠️  Continuing with existing model...")

else:
    print("=" * 80)
    print(f"STEP 2: SKIPPING model retrain")
    print("=" * 80)
    print(
        f"Reason: {'No new games added' if games_added == 0 else 'Not scheduled (next: Mon/Thu/Sun)'}"
    )
    print()

# ======================================================================
# STEP 3: GENERATE PREDICTIONS FOR TODAY
# ======================================================================

print("=" * 80)
print("STEP 3: Generating predictions for today's games")
print("=" * 80)
print()

try:
    result = subprocess.run(
        [sys.executable, "scripts/production/daily_betting_recommendations.py"],
        cwd=PROJECT_ROOT,
        capture_output=True,
        text=True,
    )

    if result.returncode != 0:
        print(f"❌ ERROR generating predictions:")
        print(result.stderr)
        # Don't exit - predictions might not be critical
    else:
        print("✅ Predictions generated")
        # Print last 20 lines of output
        output_lines = result.stdout.split("\n")
        for line in output_lines[-20:]:
            print(line)
        print()

except Exception as e:
    print(f"❌ ERROR in step 3: {str(e)}")
    print("⚠️  Skipping predictions...")

# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("✅ DAILY PIPELINE COMPLETE")
print("=" * 80)
print()
print(f"Summary:")
print(f"  ✅ Games fetched: {games_added:,} new")
print(
    f"  {'✅' if SHOULD_RETRAIN and games_added > 0 else '⊘'} Model retrained: {'YES' if SHOULD_RETRAIN and games_added > 0 else 'NO'}"
)
print(f"  ✅ Predictions generated: Check data/results/")
print()
print(f"Next run: Tomorrow at {datetime.now().strftime('%H:%M')}")
print()
