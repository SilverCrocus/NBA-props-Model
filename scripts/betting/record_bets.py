#!/usr/bin/env python3
"""
Record Placed Bets
==================
Records which bets from daily recommendations you actually placed.

Usage:
  # Record ALL recommended bets for a date
  uv run python scripts/betting/record_bets.py 2025-10-21 --all

  # Record specific bets (interactive selection)
  uv run python scripts/betting/record_bets.py 2025-10-21

  # Record top N bets
  uv run python scripts/betting/record_bets.py 2025-10-21 --top 5

  # Specify actual stake amounts (if different from recommended)
  uv run python scripts/betting/record_bets.py 2025-10-21 --top 5 --stakes 10,10,15,20,25
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Project root
ROOT_DIR = Path(__file__).parent.parent.parent
BETTING_DIR = ROOT_DIR / "data" / "betting"
LEDGER_FILE = BETTING_DIR / "bet_ledger.csv"


def load_recommendations(date: str) -> pd.DataFrame:
    """Load daily recommendations CSV."""
    rec_file = BETTING_DIR / f"recommendations_{date}.csv"
    if not rec_file.exists():
        print(f"❌ No recommendations found for {date}")
        print(f"   Expected file: {rec_file}")
        print(f"\n💡 Run: uv run nba_today.py {date} [bankroll]")
        sys.exit(1)

    df = pd.read_csv(rec_file)
    print(f"✅ Loaded {len(df)} recommendations for {date}")
    return df


def load_ledger() -> pd.DataFrame:
    """Load existing bet ledger or create new one."""
    if LEDGER_FILE.exists():
        return pd.read_csv(LEDGER_FILE)
    else:
        # Create new ledger with schema
        return pd.DataFrame(
            columns=[
                "bet_id",  # Unique ID: {date}_{player}_{direction}
                "date_placed",  # Date bet was placed
                "game_date",  # Date of actual game
                "player_name",
                "line",
                "direction",  # OVER or UNDER
                "bookmaker",
                "american_odds",
                "decimal_odds",
                "predicted_PRA",
                "calibrated_prob",
                "edge",
                "confidence",
                "recommended_bet_size",
                "actual_bet_size",  # What you actually wagered
                "actual_PRA",  # Filled in after game
                "result",  # WIN, LOSS, PUSH (filled in after game)
                "profit_loss",  # Filled in after game
                "notes",  # Optional notes
            ]
        )


def save_ledger(df: pd.DataFrame):
    """Save ledger to CSV."""
    df.to_csv(LEDGER_FILE, index=False)
    print(f"💾 Saved to: {LEDGER_FILE}")


def select_bets_interactive(recommendations: pd.DataFrame) -> pd.DataFrame:
    """Interactive bet selection."""
    print("\n" + "=" * 80)
    print("📋 RECOMMENDED BETS")
    print("=" * 80)

    for idx, row in recommendations.iterrows():
        edge_value = row.get("edge", row.get("prob_edge", 0))
        print(f"\n{idx+1}. {row['player_name']} - {row['direction']} {row['line']}")
        print(f"   Edge: {edge_value:.1%} | Confidence: {row['confidence']}")
        print(f"   Recommended: ${row['bet_size']:.2f}")

    print("\n" + "=" * 80)
    print("Enter bet numbers to record (e.g., '1,3,5' or 'all'):")
    selection = input("> ").strip().lower()

    if selection == "all":
        return recommendations

    try:
        indices = [int(x.strip()) - 1 for x in selection.split(",")]
        return recommendations.iloc[indices].copy()
    except Exception as e:
        print(f"❌ Invalid selection: {e}")
        sys.exit(1)


def record_bets(date: str, select_all: bool = False, top_n: int = None, custom_stakes: list = None):
    """Record bets to ledger."""

    # Load recommendations
    recommendations = load_recommendations(date)

    # Select which bets to record
    if select_all:
        selected = recommendations
        print(f"📝 Recording all {len(selected)} bets")
    elif top_n:
        selected = recommendations.head(top_n).copy()
        print(f"📝 Recording top {top_n} bets")
    else:
        selected = select_bets_interactive(recommendations)

    # Load existing ledger
    ledger = load_ledger()

    # Create bet records
    new_bets = []
    for idx, row in selected.iterrows():
        bet_id = f"{date}_{row['player_name'].replace(' ', '_')}_{row['direction']}"

        # Check if already recorded
        if bet_id in ledger["bet_id"].values:
            print(f"⚠️  Skipping {row['player_name']} - already recorded")
            continue

        # Determine actual stake
        if custom_stakes and len(custom_stakes) > len(new_bets):
            actual_stake = custom_stakes[len(new_bets)]
        else:
            actual_stake = row["bet_size"]

        # Handle both 'edge' and 'prob_edge' column names
        edge_value = row.get("edge", row.get("prob_edge", 0))

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
            "edge": edge_value,
            "confidence": row["confidence"],
            "recommended_bet_size": row["bet_size"],
            "actual_bet_size": actual_stake,
            "actual_PRA": None,
            "result": "PENDING",
            "profit_loss": None,
            "notes": "",
        }
        new_bets.append(bet_record)

    if not new_bets:
        print("\n✅ No new bets to record")
        return

    # Append to ledger
    ledger = pd.concat([ledger, pd.DataFrame(new_bets)], ignore_index=True)
    save_ledger(ledger)

    # Summary
    print("\n" + "=" * 80)
    print("✅ RECORDED BETS")
    print("=" * 80)
    for bet in new_bets:
        print(f"✓ {bet['player_name']} - {bet['direction']} {bet['line']}")
        print(f"  Stake: ${bet['actual_bet_size']:.2f} | Edge: {bet['edge']:.1%}")

    total_wagered = sum(b["actual_bet_size"] for b in new_bets)
    print(f"\n💰 Total wagered: ${total_wagered:.2f}")
    print(f"📊 Bets recorded: {len(new_bets)}")
    print(f"\n💡 Next: Wait for games to finish, then run:")
    print(f"   uv run python scripts/betting/update_results.py {date}")


def main():
    parser = argparse.ArgumentParser(description="Record placed bets")
    parser.add_argument("date", help="Game date (YYYY-MM-DD)")
    parser.add_argument("--all", action="store_true", help="Record all recommended bets")
    parser.add_argument("--top", type=int, help="Record top N bets")
    parser.add_argument("--stakes", help="Comma-separated actual stake amounts")

    args = parser.parse_args()

    custom_stakes = None
    if args.stakes:
        custom_stakes = [float(x.strip()) for x in args.stakes.split(",")]

    record_bets(date=args.date, select_all=args.all, top_n=args.top, custom_stakes=custom_stakes)


if __name__ == "__main__":
    main()
