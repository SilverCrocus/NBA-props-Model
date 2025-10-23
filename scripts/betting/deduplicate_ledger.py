#!/usr/bin/env python3
"""
Deduplicate Bet Ledger
======================
Removes duplicate bets (same player, same game, same line, same direction).
Keeps the first occurrence and removes subsequent duplicates.

Usage:
  uv run python scripts/betting/deduplicate_ledger.py
  uv run python scripts/betting/deduplicate_ledger.py --dry-run  # Preview only
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
BETTING_DIR = ROOT_DIR / "data" / "betting"
LEDGER_FILE = BETTING_DIR / "bet_ledger.csv"
BACKUP_FILE = BETTING_DIR / "bet_ledger_BACKUP.csv"


def deduplicate_ledger(dry_run=False):
    """Remove duplicate bets from ledger."""

    if not LEDGER_FILE.exists():
        print("❌ No bet ledger found")
        sys.exit(1)

    # Load ledger
    ledger = pd.read_csv(LEDGER_FILE)
    original_count = len(ledger)

    print(f"📊 Original ledger: {original_count} bets")

    # Identify duplicates based on player, teams, line, direction
    # (Same actual bet, regardless of when it was queried)
    ledger["_dedup_key"] = (
        ledger["player_name"].str.replace(" ", "_")
        + "_"
        + ledger["away_team"].str.replace(" ", "_")
        + "_vs_"
        + ledger["home_team"].str.replace(" ", "_")
        + "_"
        + ledger["line"].astype(str)
        + "_"
        + ledger["direction"]
    )

    # Find duplicates
    duplicates = ledger[ledger.duplicated(subset="_dedup_key", keep="first")]

    if duplicates.empty:
        print("✅ No duplicates found")
        return

    print(f"\n⚠️  Found {len(duplicates)} duplicate bets:")
    print("\nDuplicates to remove:")
    print("-" * 100)

    for idx, dup in duplicates.iterrows():
        print(f"  • {dup['player_name']} - {dup['direction']} {dup['line']}")
        print(f"    {dup['away_team']} @ {dup['home_team']}")
        print(f"    Bet ID: {dup['bet_id']}")
        print(f"    Placed: {dup['date_placed']}")
        print()

    if dry_run:
        print("🔍 DRY RUN - No changes made")
        print(f"   Would remove {len(duplicates)} duplicates")
        print(f"   Would keep {original_count - len(duplicates)} unique bets")
        return

    # Confirm
    print("\n" + "=" * 100)
    response = input(f"Remove {len(duplicates)} duplicates? (yes/no): ").strip().lower()

    if response != "yes":
        print("❌ Cancelled")
        return

    # Backup original
    ledger.to_csv(BACKUP_FILE, index=False)
    print(f"\n💾 Backup saved to: {BACKUP_FILE}")

    # Remove duplicates (keep first occurrence)
    clean_ledger = ledger.drop_duplicates(subset="_dedup_key", keep="first")

    # Drop the temporary dedup key
    clean_ledger = clean_ledger.drop(columns=["_dedup_key"])

    # Save cleaned ledger
    clean_ledger.to_csv(LEDGER_FILE, index=False)

    final_count = len(clean_ledger)
    removed_count = original_count - final_count

    print(f"\n✅ Removed {removed_count} duplicates")
    print(f"📊 Clean ledger: {final_count} unique bets")
    print(f"💾 Saved to: {LEDGER_FILE}")

    # Summary by game
    print("\n📋 Unique bets by game:")
    game_counts = (
        clean_ledger.groupby(["away_team", "home_team"]).size().sort_values(ascending=False)
    )
    for (away, home), count in game_counts.head(10).items():
        print(f"  {away} @ {home}: {count} bets")


def main():
    parser = argparse.ArgumentParser(description="Deduplicate bet ledger")
    parser.add_argument("--dry-run", action="store_true", help="Preview only, do not modify")
    args = parser.parse_args()

    deduplicate_ledger(dry_run=args.dry_run)


if __name__ == "__main__":
    main()
