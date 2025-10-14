"""
Find available player prop markets for NBA
"""

import requests
from dotenv import load_dotenv
import os

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = "https://api.the-odds-api.com/v4"

print("="*80)
print("FINDING AVAILABLE PLAYER PROP MARKETS")
print("="*80)

# Common player prop market names to try
markets_to_test = [
    'player_points',
    'player_rebounds',
    'player_assists',
    'player_threes',
    'player_blocks',
    'player_steals',
    'player_turnovers',
    'player_points_rebounds_assists',  # PRA combined
    'player_points_rebounds',
    'player_points_assists',
    'player_rebounds_assists',
    'h2h',  # Head to head (game winner)
    'spreads',
    'totals'
]

print(f"\nTesting {len(markets_to_test)} market types...\n")

working_markets = []
failed_markets = []

for market in markets_to_test:
    url = f"{BASE_URL}/sports/basketball_nba/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': market,
        'oddsFormat': 'american'
    }

    try:
        response = requests.get(url, params=params)

        if response.status_code == 200:
            data = response.json()
            print(f"✅ {market:<40} - WORKS ({len(data)} games)")
            working_markets.append(market)
        elif response.status_code == 422:
            error_data = response.json()
            print(f"❌ {market:<40} - Not supported")
            failed_markets.append(market)
        else:
            print(f"⚠️  {market:<40} - Status {response.status_code}")

    except Exception as e:
        print(f"❌ {market:<40} - Error: {str(e)[:50]}")

    # Check rate limit
    remaining = response.headers.get('x-requests-remaining', 'N/A')
    if isinstance(remaining, str) and remaining != 'N/A':
        if float(remaining) < 10:
            print(f"\n⚠️  Low on requests: {remaining} remaining")
            break

print("\n" + "="*80)
print("RESULTS")
print("="*80)

print(f"\n✅ Working Markets ({len(working_markets)}):")
for market in working_markets:
    print(f"  - {market}")

print(f"\n❌ Failed Markets ({len(failed_markets)}):")
for market in failed_markets[:5]:
    print(f"  - {market}")

if working_markets:
    # Get detailed info for first working market
    print("\n" + "="*80)
    print(f"SAMPLE DATA - {working_markets[0]}")
    print("="*80)

    url = f"{BASE_URL}/sports/basketball_nba/odds"
    params = {
        'apiKey': API_KEY,
        'regions': 'us',
        'markets': working_markets[0],
        'oddsFormat': 'american'
    }

    response = requests.get(url, params=params)
    if response.status_code == 200:
        data = response.json()

        if len(data) > 0:
            import json
            print(json.dumps(data[0], indent=2)[:2000])  # First game, truncated

print(f"\nRequests remaining: {response.headers.get('x-requests-remaining', 'N/A')}")
