"""
Check what markets are available from The Odds API

Author: NBA Props Model
Date: October 15, 2025
"""

import os
import sys
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

api_key = os.getenv("ODDS_API_KEY") or os.getenv("THEODDSAPI_KEY")

if not api_key:
    print("ERROR: No API key found")
    sys.exit(1)

# Get sports list
url = "https://api.the-odds-api.com/v4/sports"
response = requests.get(url, params={"apiKey": api_key})

print("Available sports:")
print(response.json())

# Get NBA odds to see what markets are available
url = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"
params = {"apiKey": api_key, "regions": "us"}

response = requests.get(url, params=params)

print("\nNBA Odds response:")
if response.status_code == 200:
    data = response.json()
    if data and len(data) > 0:
        print(f"Found {len(data)} games")

        # Check first game for available markets
        game = data[0]
        print(f"\nGame: {game['away_team']} @ {game['home_team']}")

        if "bookmakers" in game and len(game["bookmakers"]) > 0:
            bookmaker = game["bookmakers"][0]
            print(f"\nBookmaker: {bookmaker['key']}")
            print("\nAvailable markets:")
            for market in bookmaker.get("markets", []):
                print(f"  - {market['key']}")
    else:
        print("No games available")
else:
    print(f"Error: {response.status_code}")
    print(response.text)
