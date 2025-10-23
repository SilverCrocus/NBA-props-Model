#!/usr/bin/env python3
"""
ESPN Verified Stats - October 22, 2025
=======================================
All stats manually verified from ESPN box scores.
"""

from pathlib import Path

import pandas as pd

# All 32 players' stats verified from ESPN box scores
ESPN_STATS_OCT22 = {
    # Cleveland Cavaliers @ New York Knicks (401809234)
    "Jarrett Allen": {"PTS": 4, "REB": 4, "AST": 1, "PRA": 9, "MIN": 28},
    "Evan Mobley": {"PTS": 22, "REB": 8, "AST": 3, "PRA": 33, "MIN": 36},
    "Mikal Bridges": {"PTS": 16, "REB": 5, "AST": 6, "PRA": 27, "MIN": 33},
    # Sacramento Kings @ Phoenix Suns (401809941)
    "Zach LaVine": {"PTS": 30, "REB": 2, "AST": 1, "PRA": 33, "MIN": 35},
    "Russell Westbrook": {"PTS": 6, "REB": 6, "AST": 1, "PRA": 13, "MIN": 19},
    "Malik Monk": {"PTS": 19, "REB": 1, "AST": 2, "PRA": 22, "MIN": 27},
    # Minnesota Timberwolves @ Portland Trail Blazers (401809942)
    "Anthony Edwards": {"PTS": 41, "REB": 7, "AST": 1, "PRA": 49, "MIN": 39},
    "Deni Avdija": {"PTS": 20, "REB": 7, "AST": 1, "PRA": 28, "MIN": 33},
    "Jrue Holiday": {"PTS": 14, "REB": 6, "AST": 7, "PRA": 27, "MIN": 33},
    # New Orleans Pelicans @ Memphis Grizzlies (401809938)
    "Zion Williamson": {"PTS": 27, "REB": 9, "AST": 5, "PRA": 41, "MIN": 33},
    "Jordan Poole": {"PTS": 17, "REB": 0, "AST": 2, "PRA": 19, "MIN": 31},
    "Ja Morant": {"PTS": 35, "REB": 3, "AST": 3, "PRA": 41, "MIN": 29},
    # Detroit Pistons @ Chicago Bulls (401809937)
    "Jalen Duren": {"PTS": 15, "REB": 6, "AST": 1, "PRA": 22, "MIN": 20},
    "Cade Cunningham": {"PTS": 23, "REB": 7, "AST": 10, "PRA": 40, "MIN": 34},
    "Ausar Thompson": {"PTS": 11, "REB": 9, "AST": 7, "PRA": 27, "MIN": 33},
    "Tobias Harris": {"PTS": 10, "REB": 9, "AST": 4, "PRA": 23, "MIN": 34},
    "Nikola Vucevic": {"PTS": 28, "REB": 14, "AST": 2, "PRA": 44, "MIN": 33},
    "Matas Buzelis": {"PTS": 21, "REB": 6, "AST": 1, "PRA": 28, "MIN": 34},
    # Brooklyn Nets @ Charlotte Hornets (401809933)
    "Cam Thomas": {"PTS": 15, "REB": 1, "AST": 2, "PRA": 18, "MIN": 24},
    "Brandon Miller": {"PTS": 25, "REB": 0, "AST": 7, "PRA": 32, "MIN": 31},
    "Terance Mann": {"PTS": 13, "REB": 1, "AST": 1, "PRA": 15, "MIN": 19},
    # Toronto Raptors @ Atlanta Hawks (401809935)
    "R.J. Barrett": {"PTS": 25, "REB": 8, "AST": 5, "PRA": 38, "MIN": 30},
    "Scottie Barnes": {"PTS": 22, "REB": 6, "AST": 9, "PRA": 37, "MIN": 32},
    "Trae Young": {"PTS": 22, "REB": 1, "AST": 5, "PRA": 28, "MIN": 34},
    "Brandon Ingram": {"PTS": 22, "REB": 7, "AST": 8, "PRA": 37, "MIN": 35},
    "Nickeil Alexander-Walker": {"PTS": 10, "REB": 4, "AST": 4, "PRA": 18, "MIN": 28},
    # Washington Wizards @ Milwaukee Bucks (401809939)
    "Alex Sarr": {"PTS": 10, "REB": 11, "AST": 3, "PRA": 24, "MIN": 26},
    "Khris Middleton": {
        "PTS": 23,
        "REB": 6,
        "AST": 3,
        "PRA": 32,
        "MIN": 27,
    },  # Fixed: was Giannis's stats
    "Bobby Portis": {"PTS": 2, "REB": 6, "AST": 1, "PRA": 9, "MIN": 17},
    # Los Angeles Clippers @ Utah Jazz (401809940)
    "Keyonte George": {"PTS": 16, "REB": 2, "AST": 9, "PRA": 27, "MIN": 30},
    # San Antonio Spurs @ Dallas Mavericks (401809235)
    "Devin Vassell": {"PTS": 13, "REB": 3, "AST": 4, "PRA": 20, "MIN": 31},
    # Miami Heat @ Orlando Magic (401809934)
    "Paolo Banchero": {"PTS": 24, "REB": 11, "AST": 1, "PRA": 36, "MIN": 35},
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


def update_ledger():
    """Update bet ledger with ESPN verified stats."""
    ledger_file = Path("data/betting/bet_ledger.csv")
    ledger = pd.read_csv(ledger_file)

    print("=" * 80)
    print("UPDATING LEDGER WITH ESPN VERIFIED STATS")
    print("=" * 80)
    print()

    updates = 0
    corrections = []

    for idx, bet in ledger.iterrows():
        player_name = bet["player_name"]

        if player_name in ESPN_STATS_OCT22:
            espn_stats = ESPN_STATS_OCT22[player_name]
            actual_pra = espn_stats["PRA"]
            old_pra = bet["actual_PRA"]

            # Calculate correct result
            result = calculate_result(actual_pra, bet["line"], bet["direction"])
            profit_loss = calculate_profit_loss(result, bet["actual_bet_size"], bet["decimal_odds"])

            # Update ledger
            ledger.loc[idx, "actual_PRA"] = actual_pra
            ledger.loc[idx, "result"] = result
            ledger.loc[idx, "profit_loss"] = profit_loss

            if actual_pra != old_pra or result != bet["result"]:
                corrections.append(
                    {
                        "player": player_name,
                        "old_pra": old_pra,
                        "new_pra": actual_pra,
                        "old_result": bet["result"],
                        "new_result": result,
                    }
                )
                emoji = "✅" if result == "WIN" else "❌" if result == "LOSS" else "〰️"
                print(f"{emoji} {player_name}")
                if old_pra != actual_pra:
                    print(
                        f"   PRA: {old_pra} → {actual_pra} ({espn_stats['PTS']}P + {espn_stats['REB']}R + {espn_stats['AST']}A)"
                    )
                if result != bet["result"]:
                    print(f"   Result: {bet['result']} → {result}")
                print(f"   Bet: {bet['direction']} {bet['line']} | P/L: ${profit_loss:+.2f}")
                print()

            updates += 1

    # Save updated ledger
    ledger.to_csv(ledger_file, index=False)
    print(f"💾 Updated {updates} players in ledger")

    if corrections:
        print(f"\n📝 Made {len(corrections)} corrections")

    # Print final summary
    print("\n" + "=" * 80)
    print("FINAL RESULTS - ESPN VERIFIED (OCT 22, 2025)")
    print("=" * 80)

    settled = ledger[ledger["result"].isin(["WIN", "LOSS", "PUSH"])]
    wins = len(settled[settled["result"] == "WIN"])
    losses = len(settled[settled["result"] == "LOSS"])
    pushes = len(settled[settled["result"] == "PUSH"])
    total_profit = settled["profit_loss"].sum()
    total_wagered = settled["actual_bet_size"].sum()
    roi = (total_profit / total_wagered * 100) if total_wagered > 0 else 0
    win_rate = wins / (wins + losses) * 100 if (wins + losses) > 0 else 0

    print(f"\n📊 ALL 32 BETS:")
    print(f"   Record: {wins}W - {losses}L - {pushes}P")
    print(f"   Win Rate: {win_rate:.1f}%")
    print(f"   Total Wagered: ${total_wagered:.2f}")
    print(f"   Total Profit: ${total_profit:+.2f}")
    print(f"   ROI: {roi:+.1f}%")

    # Top 10
    top_10 = ledger.head(10)
    top10_wins = len(top_10[top_10["result"] == "WIN"])
    top10_losses = len(top_10[top_10["result"] == "LOSS"])
    top10_profit = top_10["profit_loss"].sum()
    top10_wagered = top_10["actual_bet_size"].sum()
    top10_roi = (top10_profit / top10_wagered * 100) if top10_wagered > 0 else 0
    top10_wr = (
        top10_wins / (top10_wins + top10_losses) * 100 if (top10_wins + top10_losses) > 0 else 0
    )

    print(f"\n📊 TOP 10 BETS (Highest Edge):")
    print(f"   Record: {top10_wins}W - {top10_losses}L")
    print(f"   Win Rate: {top10_wr:.1f}%")
    print(f"   Total Wagered: ${top10_wagered:.2f}")
    print(f"   Total Profit: ${top10_profit:+.2f}")
    print(f"   ROI: {top10_roi:+.1f}%")


if __name__ == "__main__":
    update_ledger()
