#!/usr/bin/env python3
"""
Scrape ESPN Box Scores
======================
Fetches actual player stats from ESPN for bet verification.
ESPN is more reliable than NBA API for recent games.
"""

import time
from pathlib import Path

import pandas as pd

# Known ESPN game IDs for Oct 22, 2025
# Format: "away_team @ home_team": game_id
ESPN_GAME_IDS = {
    "Minnesota Timberwolves @ Portland Trail Blazers": "401809942",
    # We'll need to find the others
}

# Map team names to ESPN abbreviations
TEAM_ABBREV = {
    "Cleveland Cavaliers": "CLE",
    "New York Knicks": "NYK",
    "Sacramento Kings": "SAC",
    "Phoenix Suns": "PHX",
    "Minnesota Timberwolves": "MIN",
    "Portland Trail Blazers": "POR",
    "New Orleans Pelicans": "NOP",
    "Memphis Grizzlies": "MEM",
    "Detroit Pistons": "DET",
    "Chicago Bulls": "CHI",
    "Brooklyn Nets": "BKN",
    "Charlotte Hornets": "CHA",
    "Toronto Raptors": "TOR",
    "Atlanta Hawks": "ATL",
    "Washington Wizards": "WAS",
    "Milwaukee Bucks": "MIL",
    "Los Angeles Clippers": "LAC",
    "Utah Jazz": "UTA",
    "San Antonio Spurs": "SAS",
    "Dallas Mavericks": "DAL",
    "Miami Heat": "MIA",
    "Orlando Magic": "ORL",
}


def get_players_by_team():
    """Get list of players we need stats for, organized by team."""
    ledger = pd.read_csv("data/betting/bet_ledger.csv")
    recs = pd.read_csv("data/betting/recommendations_2025-10-22.csv")

    # Merge to get team info
    players_teams = {}

    for idx, bet in ledger.iterrows():
        player = bet["player_name"]
        # Find in recommendations
        rec = recs[recs["player_name"] == player]
        if not rec.empty:
            rec = rec.iloc[0]
            matchup = f"{rec['away_team']} @ {rec['home_team']}"
            if matchup not in players_teams:
                players_teams[matchup] = []
            players_teams[matchup].append(player)

    return players_teams


if __name__ == "__main__":
    players_by_game = get_players_by_team()

    print("=" * 80)
    print("PLAYERS BY GAME")
    print("=" * 80)

    for matchup, players in players_by_game.items():
        print(f"\n{matchup}:")
        for p in players:
            print(f"  - {p}")

    print(f"\nTotal games: {len(players_by_game)}")
    print(f"Total players: {sum(len(p) for p in players_by_game.values())}")
