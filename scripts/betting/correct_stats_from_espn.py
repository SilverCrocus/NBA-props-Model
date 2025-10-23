#!/usr/bin/env python3
"""
Correct Bet Stats from ESPN
============================
The NBA API has incorrect stats. This script will manually correct key players
based on ESPN box scores.
"""

from pathlib import Path

import pandas as pd

# Manually verified stats from ESPN for Oct 22, 2025
# Source: ESPN box scores
ESPN_STATS = {
    "Deni Avdija": {"PTS": 20, "REB": 7, "AST": 1, "PRA": 28},
    "Zach LaVine": {"PTS": 30, "REB": 2, "AST": 1, "PRA": 33},  # User confirmed
    # Add more as we verify them
}


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
        return stake * (decimal_odds - 1)
    else:  # LOSS
        return -stake


# Load ledger
ledger_file = Path("data/betting/bet_ledger.csv")
ledger = pd.read_csv(ledger_file)

print("=" * 80)
print("CORRECTING STATS FROM ESPN")
print("=" * 80)

corrections = []

for idx, bet in ledger.iterrows():
    player_name = bet["player_name"]

    if player_name in ESPN_STATS:
        espn_stats = ESPN_STATS[player_name]
        actual_pra = espn_stats["PRA"]
        old_pra = bet["actual_PRA"]

        if actual_pra != old_pra:
            # Calculate correct result
            result = calculate_result(actual_pra, bet["line"], bet["direction"])
            profit_loss = calculate_profit_loss(result, bet["actual_bet_size"], bet["decimal_odds"])

            # Update ledger
            ledger.loc[idx, "actual_PRA"] = actual_pra
            ledger.loc[idx, "result"] = result
            ledger.loc[idx, "profit_loss"] = profit_loss

            corrections.append(
                {
                    "player": player_name,
                    "old_pra": old_pra,
                    "new_pra": actual_pra,
                    "pts": espn_stats["PTS"],
                    "reb": espn_stats["REB"],
                    "ast": espn_stats["AST"],
                    "old_result": bet["result"],
                    "new_result": result,
                    "line": bet["line"],
                    "direction": bet["direction"],
                }
            )

            print(f"✓ {player_name}:")
            print(f"  Old: {old_pra} PRA → {bet['result']}")
            print(
                f"  New: {actual_pra} PRA ({espn_stats['PTS']}P + {espn_stats['REB']}R + {espn_stats['AST']}A) → {result}"
            )
            print(f"  Bet: {bet['direction']} {bet['line']}")
            print(f"  P/L: ${profit_loss:+.2f}")
            print()

if corrections:
    # Save corrected ledger
    ledger.to_csv(ledger_file, index=False)
    print(f"💾 Corrected ledger saved to {ledger_file}")

    # Print summary
    settled = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])]
    wins = len(settled[settled["result"] == "WIN"])
    losses = len(settled[settled["result"] == "LOSS"])
    pushes = len(settled[settled["result"] == "PUSH"])
    total_profit = settled["profit_loss"].sum()
    total_wagered = settled["actual_bet_size"].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print("\n" + "=" * 80)
    print("CORRECTED RESULTS FOR 2025-10-22")
    print("=" * 80)
    print(f"📊 Record: {wins}W - {losses}L - {pushes}P")
    print(f"🎯 Win Rate: {win_rate:.1f}%")
    print(f"💰 Total Wagered: ${total_wagered:.2f}")
    print(f"💵 Total Profit: ${total_profit:+.2f}")
    print(f"📈 ROI: {roi:+.1f}%")

    # Top 10 stats
    top_10 = ledger.head(10)
    top10_wins = len(top_10[top_10["result"] == "WIN"])
    top10_losses = len(top_10[top_10["result"] == "LOSS"])
    top10_profit = top_10["profit_loss"].sum()
    top10_wagered = top_10["actual_bet_size"].sum()
    top10_roi = (top10_profit / top10_wagered * 100) if top10_wagered > 0 else 0
    top10_wr = (
        top10_wins / (top10_wins + top10_losses) * 100 if (top10_wins + top10_losses) > 0 else 0
    )

    print("\n📊 TOP 10 BETS:")
    print(f"   Record: {top10_wins}W - {top10_losses}L ({top10_wr:.1f}% WR, {top10_roi:+.1f}% ROI)")

else:
    print("✅ No corrections needed - all stats match ESPN")
