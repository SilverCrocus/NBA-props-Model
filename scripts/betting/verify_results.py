#!/usr/bin/env python3
"""
Verify Bet Results
==================
Double-check all bet results against actual game stats.
"""

from pathlib import Path

import pandas as pd
from nba_api.stats.endpoints import leaguegamelog

# Fetch all games from Oct 22, 2025
print("Fetching all games from 2025-10-22...\n")
gamelog = leaguegamelog.LeagueGameLog(
    season="2025-26", season_type_all_star="Regular Season", player_or_team_abbreviation="P"
)

df = gamelog.get_data_frames()[0]
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

# Filter to Oct 22
oct22_games = df[df["GAME_DATE"] == "2025-10-22"].copy()
oct22_games["PRA"] = oct22_games["PTS"] + oct22_games["REB"] + oct22_games["AST"]

# Load bet ledger
ledger_file = Path("data/betting/bet_ledger.csv")
ledger = pd.read_csv(ledger_file)

# Check each player
print("Verifying player stats:\n")
print("=" * 80)

mismatches = []

for idx, bet in ledger.iterrows():
    player_name = bet["player_name"]

    # Find player in games (handle R.J. Barrett -> RJ Barrett)
    search_name = player_name.replace("R.J.", "RJ").replace(".", "")
    player_game = oct22_games[
        oct22_games["PLAYER_NAME"].str.contains(search_name, case=False, na=False)
    ]

    if player_game.empty:
        # Try exact match for special characters (Vucevic)
        player_game = oct22_games[
            oct22_games["PLAYER_NAME"]
            .str.lower()
            .str.contains(player_name.lower().split()[0], na=False)
        ]

    if player_game.empty:
        print(f"{player_name}: NOT FOUND IN GAMES")
        continue

    actual_pra = int(player_game.iloc[0]["PRA"])
    recorded_pra = bet["actual_PRA"]

    if actual_pra != recorded_pra:
        mismatches.append(
            {
                "player": player_name,
                "recorded": recorded_pra,
                "actual": actual_pra,
                "line": bet["line"],
                "direction": bet["direction"],
                "recorded_result": bet["result"],
                "idx": idx,
            }
        )
        print(f"❌ MISMATCH - {player_name}:")
        print(f"   Recorded: {recorded_pra} PRA | Actual: {actual_pra} PRA")
        print(f'   Bet: {bet["direction"]} {bet["line"]}')
        print(f'   Recorded Result: {bet["result"]}')

        # Calculate correct result
        if actual_pra == bet["line"]:
            correct_result = "PUSH"
        elif bet["direction"] == "OVER":
            correct_result = "WIN" if actual_pra > bet["line"] else "LOSS"
        else:  # UNDER
            correct_result = "WIN" if actual_pra < bet["line"] else "LOSS"

        print(f"   Correct Result: {correct_result}")
        print()

print("=" * 80)
print(f"\nTotal mismatches found: {len(mismatches)}")

if mismatches:
    print("\n" + "=" * 80)
    print("CORRECTING LEDGER...")
    print("=" * 80)

    for m in mismatches:
        idx = m["idx"]
        actual_pra = m["actual"]
        line = m["line"]
        direction = m["direction"]

        # Calculate correct result
        if actual_pra == line:
            result = "PUSH"
        elif direction == "OVER":
            result = "WIN" if actual_pra > line else "LOSS"
        else:  # UNDER
            result = "WIN" if actual_pra < line else "LOSS"

        # Calculate profit/loss
        stake = ledger.loc[idx, "actual_bet_size"]
        decimal_odds = ledger.loc[idx, "decimal_odds"]

        if result == "PUSH":
            profit_loss = 0.0
        elif result == "WIN":
            profit_loss = stake * (decimal_odds - 1)
        else:  # LOSS
            profit_loss = -stake

        # Update ledger
        ledger.loc[idx, "actual_PRA"] = actual_pra
        ledger.loc[idx, "result"] = result
        ledger.loc[idx, "profit_loss"] = profit_loss

        print(f'✓ Updated {m["player"]}: {actual_pra} PRA → {result} (P/L: ${profit_loss:+.2f})')

    # Save corrected ledger
    ledger.to_csv(ledger_file, index=False)
    print(f"\n💾 Corrected ledger saved to {ledger_file}")

    # Print final summary
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
else:
    print("\n✅ All results verified - no corrections needed!")
