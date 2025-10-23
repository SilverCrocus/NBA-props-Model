#!/usr/bin/env python3
"""
Betting Performance Dashboard
==============================
View your betting performance and track ROI over time.

Usage:
  # Show overall summary
  uv run python scripts/betting/betting_dashboard.py

  # Show detailed breakdown
  uv run python scripts/betting/betting_dashboard.py --detailed

  # Export to CSV
  uv run python scripts/betting/betting_dashboard.py --export
"""

import argparse
import sys
from datetime import datetime
from pathlib import Path

import pandas as pd

ROOT_DIR = Path(__file__).parent.parent.parent
BETTING_DIR = ROOT_DIR / "data" / "betting"
LEDGER_FILE = BETTING_DIR / "bet_ledger.csv"


def load_ledger() -> pd.DataFrame:
    """Load bet ledger."""
    if not LEDGER_FILE.exists():
        print("❌ No bet ledger found. Start tracking bets with:")
        print("   uv run python scripts/betting/record_bets.py [date]")
        sys.exit(1)

    df = pd.read_csv(LEDGER_FILE)

    # Convert dates
    df["date_placed"] = pd.to_datetime(df["date_placed"])
    df["game_date"] = pd.to_datetime(df["game_date"])

    return df


def print_overall_summary(ledger: pd.DataFrame):
    """Print overall performance summary."""

    total_bets = len(ledger)
    pending = len(ledger[ledger["result"] == "PENDING"])
    settled = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])]

    print("\n" + "=" * 80)
    print("BETTING PERFORMANCE DASHBOARD")
    print("=" * 80)

    print(f"\n📊 OVERALL STATS")
    print(f"   Total Bets Placed: {total_bets}")
    print(f"   Settled: {len(settled)}")
    print(f"   Pending: {pending}")

    if settled.empty:
        print("\n⏳ No settled bets yet")
        return

    wins = len(settled[settled["result"] == "WIN"])
    losses = len(settled[settled["result"] == "LOSS"])
    pushes = len(settled[settled["result"] == "PUSH"])

    total_wagered = settled["actual_bet_size"].sum()
    total_profit = settled["profit_loss"].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0

    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n🎯 RESULTS")
    print(f"   Record: {wins}W - {losses}L - {pushes}P")
    print(f"   Win Rate: {win_rate:.1f}%")

    print(f"\n💰 FINANCIAL")
    print(f"   Total Wagered: ${total_wagered:.2f}")
    print(f"   Total Profit: ${total_profit:+.2f}")
    print(f"   ROI: {roi:+.1f}%")

    # Average bet size
    avg_bet = settled["actual_bet_size"].mean()
    print(f"   Avg Bet Size: ${avg_bet:.2f}")

    # Best/worst bets
    if not settled["profit_loss"].isna().all():
        best_bet = settled.loc[settled["profit_loss"].idxmax()]
        worst_bet = settled.loc[settled["profit_loss"].idxmin()]

        print(
            f"\n🏆 BEST BET: {best_bet['player_name']} ({best_bet['game_date'].strftime('%Y-%m-%d')})"
        )
        print(f"   {best_bet['direction']} {best_bet['line']} → ${best_bet['profit_loss']:+.2f}")

        print(
            f"\n💔 WORST BET: {worst_bet['player_name']} ({worst_bet['game_date'].strftime('%Y-%m-%d')})"
        )
        print(f"   {worst_bet['direction']} {worst_bet['line']} → ${worst_bet['profit_loss']:+.2f}")


def print_detailed_breakdown(ledger: pd.DataFrame):
    """Print detailed performance breakdown."""

    settled = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])]

    if settled.empty:
        print("\n⏳ No settled bets yet")
        return

    print("\n" + "=" * 80)
    print("DETAILED BREAKDOWN")
    print("=" * 80)

    # By confidence level
    print(f"\n📊 BY CONFIDENCE LEVEL:")
    print(f"{'Level':<12} {'Bets':<8} {'W-L-P':<12} {'Win%':<8} {'ROI%':<10} {'P/L':<10}")
    print("-" * 80)

    for conf in ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]:
        conf_bets = settled[settled["confidence"] == conf]
        if conf_bets.empty:
            continue

        wins = len(conf_bets[conf_bets["result"] == "WIN"])
        losses = len(conf_bets[conf_bets["result"] == "LOSS"])
        pushes = len(conf_bets[conf_bets["result"] == "PUSH"])

        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        wagered = conf_bets["actual_bet_size"].sum()
        profit = conf_bets["profit_loss"].sum()
        roi = (profit / wagered * 100) if wagered > 0 else 0

        print(
            f"{conf:<12} {len(conf_bets):<8} {wins}-{losses}-{pushes:<8} {win_rate:<7.1f}% {roi:<9.1f}% ${profit:+.2f}"
        )

    # By direction
    print(f"\n📊 BY BET TYPE:")
    print(f"{'Direction':<12} {'Bets':<8} {'W-L-P':<12} {'Win%':<8} {'ROI%':<10} {'P/L':<10}")
    print("-" * 80)

    for direction in ["OVER", "UNDER"]:
        dir_bets = settled[settled["direction"] == direction]
        if dir_bets.empty:
            continue

        wins = len(dir_bets[dir_bets["result"] == "WIN"])
        losses = len(dir_bets[dir_bets["result"] == "LOSS"])
        pushes = len(dir_bets[dir_bets["result"] == "PUSH"])

        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        wagered = dir_bets["actual_bet_size"].sum()
        profit = dir_bets["profit_loss"].sum()
        roi = (profit / wagered * 100) if wagered > 0 else 0

        print(
            f"{direction:<12} {len(dir_bets):<8} {wins}-{losses}-{pushes:<8} {win_rate:<7.1f}% {roi:<9.1f}% ${profit:+.2f}"
        )

    # By edge bucket
    print(f"\n📊 BY EDGE SIZE:")
    print(f"{'Edge Range':<12} {'Bets':<8} {'W-L-P':<12} {'Win%':<8} {'ROI%':<10} {'P/L':<10}")
    print("-" * 80)

    edge_buckets = [
        ("0-10%", 0, 0.10),
        ("10-20%", 0.10, 0.20),
        ("20-30%", 0.20, 0.30),
        ("30%+", 0.30, 1.0),
    ]

    for label, low, high in edge_buckets:
        edge_bets = settled[(settled["edge"] >= low) & (settled["edge"] < high)]
        if edge_bets.empty:
            continue

        wins = len(edge_bets[edge_bets["result"] == "WIN"])
        losses = len(edge_bets[edge_bets["result"] == "LOSS"])
        pushes = len(edge_bets[edge_bets["result"] == "PUSH"])

        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        wagered = edge_bets["actual_bet_size"].sum()
        profit = edge_bets["profit_loss"].sum()
        roi = (profit / wagered * 100) if wagered > 0 else 0

        print(
            f"{label:<12} {len(edge_bets):<8} {wins}-{losses}-{pushes:<8} {win_rate:<7.1f}% {roi:<9.1f}% ${profit:+.2f}"
        )

    # By bookmaker
    print(f"\n📊 BY BOOKMAKER:")
    print(f"{'Bookmaker':<12} {'Bets':<8} {'W-L-P':<12} {'Win%':<8} {'ROI%':<10} {'P/L':<10}")
    print("-" * 80)

    for book in settled["bookmaker"].unique():
        book_bets = settled[settled["bookmaker"] == book]

        wins = len(book_bets[book_bets["result"] == "WIN"])
        losses = len(book_bets[book_bets["result"] == "LOSS"])
        pushes = len(book_bets[book_bets["result"] == "PUSH"])

        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        wagered = book_bets["actual_bet_size"].sum()
        profit = book_bets["profit_loss"].sum()
        roi = (profit / wagered * 100) if wagered > 0 else 0

        print(
            f"{book:<12} {len(book_bets):<8} {wins}-{losses}-{pushes:<8} {win_rate:<7.1f}% {roi:<9.1f}% ${profit:+.2f}"
        )


def print_chronological_performance(ledger: pd.DataFrame):
    """Print performance over time."""

    settled = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])].copy()

    if settled.empty:
        return

    print("\n" + "=" * 80)
    print("PERFORMANCE OVER TIME")
    print("=" * 80)

    # Group by date
    settled = settled.sort_values("game_date")

    print(
        f"\n{'Date':<12} {'Bets':<6} {'W-L-P':<10} {'Win%':<8} {'P/L':<10} {'Cumulative P/L':<15}"
    )
    print("-" * 80)

    cumulative_pl = 0
    for date in settled["game_date"].unique():
        day_bets = settled[settled["game_date"] == date]

        wins = len(day_bets[day_bets["result"] == "WIN"])
        losses = len(day_bets[day_bets["result"] == "LOSS"])
        pushes = len(day_bets[day_bets["result"] == "PUSH"])

        total = wins + losses
        win_rate = wins / total * 100 if total > 0 else 0

        profit = day_bets["profit_loss"].sum()
        cumulative_pl += profit

        date_str = pd.to_datetime(date).strftime("%Y-%m-%d")
        print(
            f"{date_str:<12} {len(day_bets):<6} {wins}-{losses}-{pushes:<7} {win_rate:<7.1f}% ${profit:+8.2f}  ${cumulative_pl:+.2f}"
        )


def export_to_csv(ledger: pd.DataFrame):
    """Export performance summary to CSV."""

    settled = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])]

    if settled.empty:
        print("⏳ No settled bets to export")
        return

    # Create summary file
    export_file = BETTING_DIR / f"performance_summary_{datetime.now().strftime('%Y%m%d')}.csv"

    summary_data = []

    # Overall
    wins = len(settled[settled["result"] == "WIN"])
    losses = len(settled[settled["result"] == "LOSS"])
    pushes = len(settled[settled["result"] == "PUSH"])
    total_wagered = settled["actual_bet_size"].sum()
    total_profit = settled["profit_loss"].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    summary_data.append(
        {
            "category": "Overall",
            "subcategory": "All",
            "bets": len(settled),
            "wins": wins,
            "losses": losses,
            "pushes": pushes,
            "win_rate": win_rate,
            "total_wagered": total_wagered,
            "profit_loss": total_profit,
            "roi": roi,
        }
    )

    pd.DataFrame(summary_data).to_csv(export_file, index=False)
    print(f"\n✅ Exported to: {export_file}")


def main():
    parser = argparse.ArgumentParser(description="Betting performance dashboard")
    parser.add_argument("--detailed", action="store_true", help="Show detailed breakdown")
    parser.add_argument("--export", action="store_true", help="Export to CSV")
    parser.add_argument("--chronological", action="store_true", help="Show performance over time")

    args = parser.parse_args()

    ledger = load_ledger()

    print_overall_summary(ledger)

    if args.detailed:
        print_detailed_breakdown(ledger)

    if args.chronological:
        print_chronological_performance(ledger)

    if args.export:
        export_to_csv(ledger)


if __name__ == "__main__":
    main()
