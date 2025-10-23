#!/usr/bin/env python3
"""
Update Game Logs with ESPN Stats
=================================
Corrects the game logs file with ESPN verified stats for October 22, 2025.
The NBA API had incorrect stats for multiple players.
"""

import sys
from pathlib import Path

import pandas as pd

# Add project root to path
ROOT_DIR = Path(__file__).parent.parent.parent
sys.path.insert(0, str(ROOT_DIR))

# ESPN verified stats from October 22, 2025
ESPN_STATS_OCT22 = {
    # Cleveland Cavaliers @ New York Knicks (401809234)
    "Jarrett Allen": {"PTS": 4, "REB": 4, "AST": 1, "MIN": 28},
    "Evan Mobley": {"PTS": 22, "REB": 8, "AST": 3, "MIN": 36},
    "Mikal Bridges": {"PTS": 16, "REB": 5, "AST": 6, "MIN": 33},
    # Sacramento Kings @ Phoenix Suns (401809941)
    "Zach LaVine": {"PTS": 30, "REB": 2, "AST": 1, "MIN": 35},
    "Russell Westbrook": {"PTS": 6, "REB": 6, "AST": 1, "MIN": 19},
    "Malik Monk": {"PTS": 19, "REB": 1, "AST": 2, "MIN": 27},
    # Minnesota Timberwolves @ Portland Trail Blazers (401809942)
    "Anthony Edwards": {"PTS": 41, "REB": 7, "AST": 1, "MIN": 39},
    "Deni Avdija": {"PTS": 20, "REB": 7, "AST": 1, "MIN": 33},
    "Jrue Holiday": {"PTS": 14, "REB": 6, "AST": 7, "MIN": 33},
    # New Orleans Pelicans @ Memphis Grizzlies (401809938)
    "Zion Williamson": {"PTS": 27, "REB": 9, "AST": 5, "MIN": 33},
    "Jordan Poole": {"PTS": 17, "REB": 0, "AST": 2, "MIN": 31},
    "Ja Morant": {"PTS": 35, "REB": 3, "AST": 3, "MIN": 29},
    # Detroit Pistons @ Chicago Bulls (401809937)
    "Jalen Duren": {"PTS": 15, "REB": 6, "AST": 1, "MIN": 20},
    "Cade Cunningham": {"PTS": 23, "REB": 7, "AST": 10, "MIN": 34},
    "Ausar Thompson": {"PTS": 11, "REB": 9, "AST": 7, "MIN": 33},
    "Tobias Harris": {"PTS": 10, "REB": 9, "AST": 4, "MIN": 34},
    "Nikola Vucevic": {"PTS": 28, "REB": 14, "AST": 2, "MIN": 33},
    "Matas Buzelis": {"PTS": 21, "REB": 6, "AST": 1, "MIN": 34},
    # Brooklyn Nets @ Charlotte Hornets (401809933)
    "Cam Thomas": {"PTS": 15, "REB": 1, "AST": 2, "MIN": 24},
    "Brandon Miller": {"PTS": 25, "REB": 0, "AST": 7, "MIN": 31},
    "Terance Mann": {"PTS": 13, "REB": 1, "AST": 1, "MIN": 19},
    # Toronto Raptors @ Atlanta Hawks (401809935)
    "RJ Barrett": {"PTS": 25, "REB": 8, "AST": 5, "MIN": 30},  # Note: RJ not R.J.
    "Scottie Barnes": {"PTS": 22, "REB": 6, "AST": 9, "MIN": 32},
    "Trae Young": {"PTS": 22, "REB": 1, "AST": 5, "MIN": 34},
    "Brandon Ingram": {"PTS": 22, "REB": 7, "AST": 8, "MIN": 35},
    "Nickeil Alexander-Walker": {"PTS": 10, "REB": 4, "AST": 4, "MIN": 28},
    # Washington Wizards @ Milwaukee Bucks (401809939)
    "Alex Sarr": {"PTS": 10, "REB": 11, "AST": 3, "MIN": 26},
    "Khris Middleton": {"PTS": 23, "REB": 6, "AST": 3, "MIN": 27},  # Fixed: was Giannis's stats
    "Bobby Portis": {"PTS": 2, "REB": 6, "AST": 1, "MIN": 17},
    # Los Angeles Clippers @ Utah Jazz (401809940)
    "Keyonte George": {"PTS": 16, "REB": 2, "AST": 9, "MIN": 30},
    # San Antonio Spurs @ Dallas Mavericks (401809235)
    "Devin Vassell": {"PTS": 13, "REB": 3, "AST": 4, "MIN": 31},
    # Miami Heat @ Orlando Magic (401809934)
    "Paolo Banchero": {"PTS": 24, "REB": 11, "AST": 1, "MIN": 35},
}


def update_game_logs():
    """Update game logs file with ESPN verified stats."""

    game_logs_file = Path("data/game_logs/all_game_logs_through_2025.csv")

    print("=" * 80)
    print("UPDATING GAME LOGS WITH ESPN VERIFIED STATS")
    print("=" * 80)
    print()

    # Create backup first
    backup_file = Path("data/game_logs/all_game_logs_backup_before_espn_correction.csv")
    print(f"Creating backup: {backup_file}")

    # Load game logs
    df = pd.read_csv(game_logs_file)
    df.to_csv(backup_file, index=False)
    print(f"✅ Backup created\n")

    df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])

    # Filter to Oct 22, 2025
    oct22_games = df[df["GAME_DATE"] == "2025-10-22"].copy()

    print(f"Found {len(oct22_games)} game logs from October 22, 2025")
    print()

    corrections = []
    not_found = []

    # Update each player's stats
    for player_name, stats in ESPN_STATS_OCT22.items():
        # Find player in game logs
        # Handle RJ Barrett special case
        if player_name == "RJ Barrett":
            player_mask = (
                (df["GAME_DATE"] == "2025-10-22")
                & (df["PLAYER_NAME"].str.contains("Barrett", case=False, na=False))
                & (df["PLAYER_NAME"].str.contains("RJ", case=False, na=False))
            )
        else:
            player_mask = (df["GAME_DATE"] == "2025-10-22") & (df["PLAYER_NAME"] == player_name)

        player_games = df[player_mask]

        if player_games.empty:
            not_found.append(player_name)
            print(f"⚠️  {player_name} not found in game logs")
            continue

        # Get the index
        idx = player_games.index[0]

        # Get old stats
        old_pts = df.loc[idx, "PTS"]
        old_reb = df.loc[idx, "REB"]
        old_ast = df.loc[idx, "AST"]
        old_min = df.loc[idx, "MIN"]
        old_pra = old_pts + old_reb + old_ast

        # New stats from ESPN
        new_pts = stats["PTS"]
        new_reb = stats["REB"]
        new_ast = stats["AST"]
        new_min = stats["MIN"]
        new_pra = new_pts + new_reb + new_ast

        # Check if correction needed
        if (old_pts != new_pts) or (old_reb != new_reb) or (old_ast != new_ast):
            corrections.append(
                {
                    "player": player_name,
                    "old": f"{old_pts}P {old_reb}R {old_ast}A = {old_pra} PRA",
                    "new": f"{new_pts}P {new_reb}R {new_ast}A = {new_pra} PRA",
                    "diff": new_pra - old_pra,
                }
            )

            # Update the stats
            df.loc[idx, "PTS"] = new_pts
            df.loc[idx, "REB"] = new_reb
            df.loc[idx, "AST"] = new_ast
            df.loc[idx, "MIN"] = new_min

            # Update PRA if it exists
            if "PRA" in df.columns:
                df.loc[idx, "PRA"] = new_pra

            print(f"✓ {player_name}:")
            print(f"  Old: {old_pts}P + {old_reb}R + {old_ast}A = {old_pra} PRA")
            print(f"  New: {new_pts}P + {new_reb}R + {new_ast}A = {new_pra} PRA")
            print(f"  Difference: {new_pra - old_pra:+d} PRA")
            print()

    # Save updated file
    df.to_csv(game_logs_file, index=False)

    print("=" * 80)
    print("SUMMARY")
    print("=" * 80)
    print(f"✅ Updated {len(corrections)} players")
    print(f"⚠️  {len(not_found)} players not found in game logs")
    print(f"💾 Saved to: {game_logs_file}")
    print(f"📦 Backup at: {backup_file}")

    if corrections:
        print("\n" + "=" * 80)
        print("CORRECTIONS MADE")
        print("=" * 80)
        for c in corrections:
            print(f"{c['player']}:")
            print(f"  {c['old']} → {c['new']} ({c['diff']:+d} PRA)")

    if not_found:
        print("\n" + "=" * 80)
        print("PLAYERS NOT FOUND (may use different names in NBA API)")
        print("=" * 80)
        for p in not_found:
            print(f"  - {p}")

    print("\n✅ Game logs updated with ESPN verified stats!")


if __name__ == "__main__":
    update_game_logs()
