#!/usr/bin/env python3
"""
Update Bet Results
==================
Fetches actual game results and updates bet outcomes in the ledger.

Usage:
  # Update results for a specific date
  uv run python scripts/betting/update_results.py 2025-10-21

  # Update all pending bets
  uv run python scripts/betting/update_results.py --all-pending

  # Force refresh even if results already recorded
  uv run python scripts/betting/update_results.py 2025-10-21 --force
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

from nba_api.stats.endpoints import leaguegamefinder
from nba_api.stats.static import players

BETTING_DIR = ROOT_DIR / "data" / "betting"
LEDGER_FILE = BETTING_DIR / "bet_ledger.csv"


def load_ledger() -> pd.DataFrame:
    """Load bet ledger."""
    if not LEDGER_FILE.exists():
        print("❌ No bet ledger found. Record some bets first:")
        print("   uv run python scripts/betting/record_bets.py [date]")
        sys.exit(1)

    return pd.read_csv(LEDGER_FILE)


def save_ledger(df: pd.DataFrame):
    """Save updated ledger."""
    df.to_csv(LEDGER_FILE, index=False)
    print(f"💾 Saved to: {LEDGER_FILE}")


def get_player_game_stats(player_name: str, game_date: str) -> dict:
    """
    Fetch actual game stats for a player on a specific date.
    Returns dict with PTS, REB, AST, and calculated PRA.
    """
    try:
        # Find player ID
        all_players = players.get_players()
        player = [p for p in all_players if p["full_name"].lower() == player_name.lower()]

        if not player:
            # Try fuzzy matching
            player = [p for p in all_players if player_name.lower() in p["full_name"].lower()]

        if not player:
            print(f"⚠️  Player not found: {player_name}")
            return None

        player_id = player[0]["id"]

        # Fetch games for this player
        gamefinder = leaguegamefinder.LeagueGameFinder(
            player_id_nullable=player_id, date_from_nullable=game_date, date_to_nullable=game_date
        )

        games = gamefinder.get_data_frames()[0]

        if games.empty:
            print(f"⚠️  No game found for {player_name} on {game_date}")
            return None

        # Get the game stats
        game = games.iloc[0]
        return {
            "PTS": game["PTS"],
            "REB": game["REB"],
            "AST": game["AST"],
            "PRA": game["PTS"] + game["REB"] + game["AST"],
        }

    except Exception as e:
        print(f"❌ Error fetching stats for {player_name}: {e}")
        return None


def calculate_result(actual_pra: float, line: float, direction: str) -> str:
    """Determine if bet won, lost, or pushed."""
    if actual_pra == line:
        return "PUSH"
    elif direction == "OVER":
        return "WIN" if actual_pra > line else "LOSS"
    else:  # UNDER
        return "WIN" if actual_pra < line else "LOSS"


def calculate_profit_loss(result: str, stake: float, decimal_odds: float) -> float:
    """Calculate profit/loss for a bet."""
    if result == "PUSH":
        return 0.0
    elif result == "WIN":
        return stake * (decimal_odds - 1)  # Profit
    else:  # LOSS
        return -stake


def update_results_for_date(date: str, force: bool = False):
    """Update results for all bets on a specific date."""

    ledger = load_ledger()

    # Filter bets for this date
    date_bets = ledger[ledger["game_date"] == date].copy()

    if date_bets.empty:
        print(f"❌ No bets found for {date}")
        return

    # Filter pending bets (unless force)
    if not force:
        date_bets = date_bets[date_bets["result"] == "PENDING"]

    if date_bets.empty:
        print(f"✅ All bets for {date} already settled")
        return

    print(f"\n🔄 Updating {len(date_bets)} bets for {date}...")

    updated_count = 0
    for idx, bet in date_bets.iterrows():
        print(f"\n📊 {bet['player_name']} - {bet['direction']} {bet['line']}")

        # Fetch actual stats
        stats = get_player_game_stats(bet["player_name"], bet["game_date"])

        if stats is None:
            print(f"   ⏳ Skipping - no data available yet")
            continue

        actual_pra = stats["PRA"]
        result = calculate_result(actual_pra, bet["line"], bet["direction"])
        profit_loss = calculate_profit_loss(result, bet["actual_bet_size"], bet["decimal_odds"])

        # Update ledger
        ledger.loc[idx, "actual_PRA"] = actual_pra
        ledger.loc[idx, "result"] = result
        ledger.loc[idx, "profit_loss"] = profit_loss

        # Print result
        emoji = "✅" if result == "WIN" else "❌" if result == "LOSS" else "〰️"
        print(f"   {emoji} Actual: {actual_pra:.1f} PRA → {result}")
        print(f"   💵 P/L: ${profit_loss:+.2f}")

        updated_count += 1

    if updated_count > 0:
        save_ledger(ledger)
        print(f"\n✅ Updated {updated_count} bets")
        print_summary(ledger, date)
    else:
        print(f"\n⏳ No results available yet for {date}")


def update_all_pending():
    """Update all pending bets."""
    ledger = load_ledger()
    pending = ledger[ledger["result"] == "PENDING"]

    if pending.empty:
        print("✅ No pending bets")
        return

    print(f"\n🔄 Found {len(pending)} pending bets")

    # Group by date and update
    for date in pending["game_date"].unique():
        update_results_for_date(date, force=False)


def print_summary(ledger: pd.DataFrame, date: str = None):
    """Print performance summary."""

    if date:
        df = ledger[ledger["game_date"] == date]
        title = f"RESULTS FOR {date}"
    else:
        df = ledger
        title = "OVERALL RESULTS"

    settled = df[df["result"].isin(["WIN", "LOSS", "PUSH"])]

    if settled.empty:
        print(f"\n📊 No settled bets yet")
        return

    wins = len(settled[settled["result"] == "WIN"])
    losses = len(settled[settled["result"] == "LOSS"])
    pushes = len(settled[settled["result"] == "PUSH"])

    total_wagered = settled["actual_bet_size"].sum()
    total_profit = settled["profit_loss"].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print("\n" + "=" * 80)
    print(title)
    print("=" * 80)
    print(f"📊 Record: {wins}W - {losses}L - {pushes}P")
    print(f"🎯 Win Rate: {win_rate:.1f}%")
    print(f"💰 Total Wagered: ${total_wagered:.2f}")
    print(f"💵 Total Profit: ${total_profit:+.2f}")
    print(f"📈 ROI: {roi:+.1f}%")

    # By confidence level
    print("\n📊 By Confidence Level:")
    for conf in settled["confidence"].unique():
        conf_bets = settled[settled["confidence"] == conf]
        conf_wins = len(conf_bets[conf_bets["result"] == "WIN"])
        conf_total = len(conf_bets[conf_bets["result"].isin(["WIN", "LOSS"])])
        conf_win_rate = conf_wins / conf_total * 100 if conf_total > 0 else 0
        conf_profit = conf_bets["profit_loss"].sum()

        print(
            f"   {conf}: {conf_wins}/{conf_total} ({conf_win_rate:.1f}%) | P/L: ${conf_profit:+.2f}"
        )


def main():
    parser = argparse.ArgumentParser(description="Update bet results")
    parser.add_argument("date", nargs="?", help="Game date (YYYY-MM-DD)")
    parser.add_argument("--all-pending", action="store_true", help="Update all pending bets")
    parser.add_argument("--force", action="store_true", help="Force refresh existing results")
    parser.add_argument("--summary", action="store_true", help="Show summary only")

    args = parser.parse_args()

    if args.summary:
        ledger = load_ledger()
        print_summary(ledger)
    elif args.all_pending:
        update_all_pending()
    elif args.date:
        update_results_for_date(args.date, args.force)
    else:
        print("❌ Provide a date or use --all-pending")
        parser.print_help()


if __name__ == "__main__":
    main()
