#!/usr/bin/env python3
"""
Update Latest NBA Game Logs
============================

Fetches the most recent NBA games and appends them to the master game logs file.
Use this daily during the season to keep your production model up-to-date.

Features:
- Fetches games from the last 7 days (configurable)
- Deduplicates against existing data
- Appends only new games
- Safe incremental updates

Usage: uv run python scripts/update_latest_games.py
"""

import argparse
import time
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

print("=" * 80)
print("UPDATE LATEST NBA GAME LOGS")
print("=" * 80)
print()

# ======================================================================
# CONFIGURATION
# ======================================================================

# Paths
DATA_DIR = Path("data/game_logs")
DATA_DIR.mkdir(parents=True, exist_ok=True)
MASTER_FILE = DATA_DIR / "all_game_logs_through_2025.csv"
BACKUP_FILE = DATA_DIR / f"all_game_logs_backup_{datetime.now().strftime('%Y%m%d')}.csv"

# Current season
CURRENT_SEASON = "2025-26"

# How many days back to fetch (default: 7 days to catch any games we missed)
DAYS_BACK = 7

print(f"Configuration:")
print(f"  Current season: {CURRENT_SEASON}")
print(f"  Master file: {MASTER_FILE}")
print(f"  Fetching games from last {DAYS_BACK} days")
print()

# ======================================================================
# 1. LOAD EXISTING DATA
# ======================================================================

print("STEP 1: Loading existing game logs...")

if not MASTER_FILE.exists():
    print(f"❌ ERROR: Master file not found: {MASTER_FILE}")
    print("   Please run fetch_all_game_logs.py first to create the master file")
    exit(1)

# Load existing data
df_existing = pd.read_csv(MASTER_FILE)
df_existing["GAME_DATE"] = pd.to_datetime(df_existing["GAME_DATE"])

print(f"✅ Loaded {len(df_existing):,} existing game logs")
print(
    f"   Date range: {df_existing['GAME_DATE'].min().date()} to {df_existing['GAME_DATE'].max().date()}"
)
print(f"   Latest game: {df_existing['GAME_DATE'].max().date()}")
print()

# Create backup
print("Creating backup...")
df_existing.to_csv(BACKUP_FILE, index=False)
print(f"✅ Backup saved to {BACKUP_FILE}")
print()

# ======================================================================
# 2. FETCH LATEST GAMES
# ======================================================================

print("=" * 80)
print(f"STEP 2: Fetching latest games from {CURRENT_SEASON} season")
print("=" * 80)
print()

all_new_games = []

print(f"Fetching Regular Season games...")
try:
    # Fetch league game log for current season
    # Use PlayerOrTeam='P' to get PLAYER game logs (not team logs)
    gamelog = leaguegamelog.LeagueGameLog(
        season=CURRENT_SEASON,
        season_type_all_star="Regular Season",
        player_or_team_abbreviation="P",  # P = Player, T = Team
    )

    df_new = gamelog.get_data_frames()[0]

    if len(df_new) > 0:
        # Convert GAME_DATE to datetime
        df_new["GAME_DATE"] = pd.to_datetime(df_new["GAME_DATE"])

        # Filter to recent games only (last N days)
        cutoff_date = datetime.now() - timedelta(days=DAYS_BACK)
        df_new = df_new[df_new["GAME_DATE"] >= cutoff_date].copy()

        print(f"✅ Found {len(df_new):,} games from last {DAYS_BACK} days")
        print(
            f"   Date range: {df_new['GAME_DATE'].min().date()} to {df_new['GAME_DATE'].max().date()}"
        )

        all_new_games.append(df_new)
    else:
        print(f"⚠️  No games found for {CURRENT_SEASON}")

    time.sleep(1)  # Rate limiting

except Exception as e:
    print(f"❌ Error fetching {CURRENT_SEASON} Regular Season: {str(e)}")

print()

# ======================================================================
# 3. DEDUPLICATE AND MERGE
# ======================================================================

print("=" * 80)
print("STEP 3: Deduplicating and merging new games")
print("=" * 80)
print()

if not all_new_games:
    print("⚠️  No new games to add")
    print(
        "   Either no games were played recently, or all recent games are already in the database"
    )
    print()
    exit(0)

# Combine all new games
df_new_all = pd.concat(all_new_games, ignore_index=True)

print(f"Total new games fetched: {len(df_new_all):,}")

# Check what columns are available in new data
print(f"Columns in new data: {df_new_all.columns.tolist()[:20]}...")  # Show first 20
print()

# Map column names if needed (NBA API sometimes uses different names)
# Common mappings: PLAYER_ID vs Player_ID, GAME_ID vs Game_ID
column_map = {}
for col in df_new_all.columns:
    lower_col = col.lower()
    if "player" in lower_col and "id" in lower_col:
        column_map[col] = "PLAYER_ID"
    elif "game" in lower_col and "id" in lower_col:
        column_map[col] = "GAME_ID"

if column_map:
    print(f"Renaming columns: {column_map}")
    df_new_all = df_new_all.rename(columns=column_map)
    print()

# Deduplicate against existing data
# Use GAME_ID and PLAYER_ID as unique identifier
if "PLAYER_ID" in df_new_all.columns and "GAME_ID" in df_new_all.columns:
    existing_keys = set(
        zip(df_existing["GAME_ID"].astype(str), df_existing["PLAYER_ID"].astype(str))
    )

    new_games_mask = ~df_new_all.apply(
        lambda row: (str(row["GAME_ID"]), str(row["PLAYER_ID"])) in existing_keys, axis=1
    )

    df_truly_new = df_new_all[new_games_mask].copy()
else:
    print("⚠️  WARNING: Could not find PLAYER_ID and GAME_ID columns")
    print(f"Available columns: {df_new_all.columns.tolist()}")
    print("Using all new games without deduplication")
    df_truly_new = df_new_all.copy()

print(f"After deduplication: {len(df_truly_new):,} NEW game logs to add")
print()

if len(df_truly_new) == 0:
    print("✅ All recent games are already in the database - nothing to update")
    print()
    exit(0)

# Show sample of new games
print("Sample of new games being added:")
if "PLAYER_NAME" in df_truly_new.columns:
    sample = df_truly_new[["PLAYER_NAME", "GAME_DATE", "PTS", "REB", "AST", "MIN"]].head(10)
else:
    sample = df_truly_new[["GAME_DATE", "PTS", "REB", "AST", "MIN"]].head(10)
print(sample.to_string())
print()

# ======================================================================
# 4. UPDATE MASTER FILE
# ======================================================================

print("=" * 80)
print("STEP 4: Updating master file")
print("=" * 80)
print()

# Ensure column alignment
# Get common columns
common_cols = list(set(df_existing.columns) & set(df_truly_new.columns))

# Combine old and new data
df_updated = pd.concat([df_existing[common_cols], df_truly_new[common_cols]], ignore_index=True)

# Sort by date
if "PLAYER_ID" in df_updated.columns:
    df_updated = df_updated.sort_values(["GAME_DATE", "PLAYER_ID"])
else:
    df_updated = df_updated.sort_values("GAME_DATE")

# Add PRA if not present
if "PRA" not in df_updated.columns:
    df_updated["PRA"] = df_updated["PTS"] + df_updated["REB"] + df_updated["AST"]

# Save updated file
df_updated.to_csv(MASTER_FILE, index=False)

print(f"✅ Updated master file: {MASTER_FILE}")
print()
print(f"Summary:")
print(f"  Previous total: {len(df_existing):,} games")
print(f"  New games added: {len(df_truly_new):,}")
print(f"  New total: {len(df_updated):,} games")
print(
    f"  Date range: {df_updated['GAME_DATE'].min().date()} to {df_updated['GAME_DATE'].max().date()}"
)
print()

# Show latest games
print("Latest 10 games in database:")
latest = df_updated.nlargest(10, "GAME_DATE")
if "PLAYER_NAME" in latest.columns:
    display = latest[["PLAYER_NAME", "GAME_DATE", "PTS", "REB", "AST", "PRA", "MIN"]]
else:
    display = latest[["GAME_DATE", "PTS", "REB", "AST", "PRA", "MIN"]]
print(display.to_string(index=False))
print()

print("=" * 80)
print("✅ UPDATE COMPLETE!")
print("=" * 80)
print()
print("Next steps:")
print("  1. Retrain production model with updated data:")
print("     uv run python scripts/production/train_ensemble_v1_production.py")
print()
print("  2. Make predictions for tonight's games:")
print("     uv run python scripts/production/daily_betting_recommendations.py")
print()
