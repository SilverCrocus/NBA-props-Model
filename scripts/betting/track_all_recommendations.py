#!/usr/bin/env python3
"""
Auto-Track All Recommendations
===============================
Automatically tracks ALL bets from the recommendations CSV.
This runs automatically when you run nba_today.py

Usage:
  uv run python scripts/betting/track_all_recommendations.py 2025-10-21
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
BETTING_DIR = ROOT_DIR / "data" / "betting"
LEDGER_FILE = BETTING_DIR / "bet_ledger.csv"


def load_recommendations(date: str) -> pd.DataFrame:
    """Load daily recommendations CSV."""
    rec_file = BETTING_DIR / f"recommendations_{date}.csv"
    if not rec_file.exists():
        print(f"❌ No recommendations found for {date}")
        sys.exit(1)

    return pd.read_csv(rec_file)


def load_or_create_ledger() -> pd.DataFrame:
    """Load existing bet ledger or create new one."""
    if LEDGER_FILE.exists():
        return pd.read_csv(LEDGER_FILE)
    else:
        # Create new ledger with schema
        return pd.DataFrame(
            columns=[
                "bet_id",
                "date_placed",
                "game_date",
                "player_name",
                "line",
                "direction",
                "bookmaker",
                "american_odds",
                "decimal_odds",
                "predicted_PRA",
                "calibrated_prob",
                "edge",
                "confidence",
                "recommended_bet_size",
                "actual_PRA",
                "result",
                "profit_loss",
                "away_team",
                "home_team",
            ]
        )


def track_all_recommendations(date: str):
    """Auto-track all recommendations for a date."""

    # Load recommendations
    recommendations = load_recommendations(date)
    print(f"📋 Found {len(recommendations)} recommendations for {date}")

    # Load existing ledger
    ledger = load_or_create_ledger()

    # Track new bets
    new_bets = []
    skipped = 0

    for idx, row in recommendations.iterrows():
        # Create unique bet_id using player, teams, line, and direction
        # This prevents the same game from being tracked multiple times
        player_slug = row["player_name"].replace(" ", "_")
        teams_slug = f"{row['away_team'].replace(' ', '_')}_vs_{row['home_team'].replace(' ', '_')}"
        bet_id = f"{player_slug}_{teams_slug}_{row['line']}_{row['direction']}"

        # Check if already tracked (same player, same game, same line, same direction)
        if bet_id in ledger["bet_id"].values:
            skipped += 1
            continue

        # Create bet record
        bet_record = {
            "bet_id": bet_id,
            "date_placed": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "game_date": date,
            "player_name": row["player_name"],
            "line": row["line"],
            "direction": row["direction"],
            "bookmaker": row["bookmaker"],
            "american_odds": row["american_odds"],
            "decimal_odds": row["decimal_odds"],
            "predicted_PRA": row["predicted_PRA"],
            "calibrated_prob": row["calibrated_prob"],
            "edge": row["edge"],
            "confidence": row["confidence"],
            "recommended_bet_size": row["bet_size"],
            "actual_PRA": None,
            "result": "PENDING",
            "profit_loss": None,
            "away_team": row["away_team"],
            "home_team": row["home_team"],
        }
        new_bets.append(bet_record)

    if not new_bets:
        print(f"✅ All {len(recommendations)} bets already tracked")
        if skipped > 0:
            print(f"   (Skipped {skipped} duplicates)")
        return

    # Append to ledger
    new_df = pd.DataFrame(new_bets)
    ledger = pd.concat([ledger, new_df], ignore_index=True)

    # Save
    ledger.to_csv(LEDGER_FILE, index=False)
    print(f"✅ Tracked {len(new_bets)} new bets")
    if skipped > 0:
        print(f"   (Skipped {skipped} duplicates)")

    # Summary
    total_edge = sum(b["edge"] for b in new_bets)
    avg_edge = total_edge / len(new_bets) if new_bets else 0

    print(f"\n📊 Summary:")
    print(f"   Total bets: {len(new_bets)}")
    print(f"   Avg edge: {avg_edge:.1%}")
    print(f"   Confidence levels:")

    for conf in ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]:
        count = sum(1 for b in new_bets if b["confidence"] == conf)
        if count > 0:
            print(f"      {conf}: {count} bets")

    print(f"\n💡 Next: Wait for games to finish, then run:")
    print(f"   uv run track_bet.py update {date}")


def main():
    parser = argparse.ArgumentParser(description="Auto-track all recommendations")
    parser.add_argument("date", help="Game date (YYYY-MM-DD)")
    args = parser.parse_args()

    track_all_recommendations(args.date)


if __name__ == "__main__":
    main()
