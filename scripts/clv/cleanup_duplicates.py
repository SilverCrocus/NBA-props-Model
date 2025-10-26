#!/usr/bin/env python3
"""
CLV Ledger Duplicate Cleanup
=============================
Removes duplicate bets and keeps only the LATEST bet for each player/direction/date combination.

Use case: When betting lines change (e.g., Curry 38.5 → 34.5), you want to track only  # noqa: E501
the latest line you actually bet on, not all the historical line movements.

Usage:
    uv run scripts/clv/cleanup_duplicates.py
    uv run scripts/clv/cleanup_duplicates.py --dry-run  # Preview without saving  # noqa: E501

The script:
1. Groups bets by (date, player, side)
2. Keeps only the LATEST bet (most recent entry_time)
3. Removes all older bets
4. Saves cleaned ledger
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

# Add parent directory to path
sys.path.append(str(Path(__file__).parent.parent.parent))


def cleanup_duplicates(ledger_path: str = "data/clv_ledger.csv", dry_run: bool = False):
    """
    Remove duplicate bets, keeping only latest for each player/direction/date

    Args:
        ledger_path: Path to CLV ledger CSV
        dry_run: If True, preview changes without saving
    """

    print("=" * 80)
    print("🧹 CLV LEDGER DUPLICATE CLEANUP")
    print("=" * 80)
    print(f"\n📋 Ledger: {ledger_path}")
    if dry_run:
        print("🔍 Mode: DRY RUN (no changes will be saved)\n")
    else:
        print("⚠️  Mode: LIVE (will modify ledger)\n")

    # Load ledger
    if not Path(ledger_path).exists():
        print(f"❌ Ledger file not found: {ledger_path}")
        return 1

    ledger = pd.read_csv(ledger_path)
    original_count = len(ledger)

    print(f"📊 Original ledger: {original_count} bets")

    # Find duplicates (same date, player, side)
    ledger["entry_time"] = pd.to_datetime(ledger["entry_time"])

    # Group by (date, player, side) and keep only the latest (most recent
    # entry_time)
    duplicates = ledger.groupby(["date", "player", "side"]).size()
    duplicates = duplicates[duplicates > 1]

    if len(duplicates) == 0:
        print("\n✅ No duplicates found! Ledger is clean.")
        return 0

    print(
        f"\n🔍 Found {
            len(duplicates)} player/direction combinations with multiple bets:\n"
    )  # noqa: E501

    # Show duplicates
    for (date, player, side), count in duplicates.items():
        player_bets = ledger[
            (ledger["date"] == date)
            & (ledger["player"] == player)
            & (ledger["side"] == side)  # noqa: E501
        ].sort_values("entry_time")

        print(f"  📌 {date} | {player} {side} ({count} bets)")
        for idx, bet in player_bets.iterrows():
            marker = "  ✅ KEEP" if idx == player_bets.index[-1] else "  ❌ REMOVE"  # noqa: E501
            print(
                f"     {marker}: Line {
                    bet['line']} @ {
                    bet['entry_odds_dec']:.3f} ({
                    bet['entry_time']})"
            )

    # Remove duplicates, keeping only latest
    ledger_cleaned = (
        ledger.sort_values("entry_time").groupby(["date", "player", "side"]).tail(1).copy()
    )

    ledger_cleaned = ledger_cleaned.sort_values(["date", "player", "side"]).reset_index(drop=True)

    removed_count = original_count - len(ledger_cleaned)

    print("\n" + "=" * 80)
    print("📊 SUMMARY")
    print("=" * 80)
    print(f"\nOriginal bets:  {original_count}")
    print(f"Cleaned bets:   {len(ledger_cleaned)}")
    print(f"Removed:        {removed_count}")

    if dry_run:
        print("\n🔍 DRY RUN: No changes saved")
        print("   To apply changes, run without --dry-run flag")
    else:
        # Save cleaned ledger
        ledger_cleaned.to_csv(ledger_path, index=False)
        print(f"\n✅ Cleaned ledger saved to: {ledger_path}")
        print(f"   Removed {removed_count} duplicate bets")

        # Create backup of original
        backup_path = ledger_path.replace(".csv", "_backup_before_cleanup.csv")
        ledger.to_csv(backup_path, index=False)
        print(f"   Backup created: {backup_path}")

    print("\n" + "=" * 80)

    return 0


def main():
    parser = argparse.ArgumentParser(description="Clean up duplicate bets in CLV ledger")
    parser.add_argument(
        "--ledger-path",
        default="data/clv_ledger.csv",
        help="Path to CLV ledger (default: data/clv_ledger.csv)",
    )
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without saving")

    args = parser.parse_args()

    return cleanup_duplicates(ledger_path=args.ledger_path, dry_run=args.dry_run)


if __name__ == "__main__":
    sys.exit(main())
