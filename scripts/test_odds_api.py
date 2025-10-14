"""
Test TheOddsAPI connection and explore available data
"""

import requests
import json
from datetime import datetime
from dotenv import load_dotenv
import os

# Load API key
load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')

print("="*80)
print("THEODDSAPI - CONNECTION TEST & DATA EXPLORATION")
print("="*80)

BASE_URL = "https://api.the-odds-api.com/v4"

# Test 1: Check sports available
print("\n1. Testing API connection...")
url = f"{BASE_URL}/sports"
params = {'apiKey': API_KEY}

try:
    response = requests.get(url, params=params)
    response.raise_for_status()
    sports = response.json()

    print(f"✅ API connection successful!")
    print(f"Requests remaining: {response.headers.get('x-requests-remaining', 'N/A')}")
    print(f"Requests used: {response.headers.get('x-requests-used', 'N/A')}")

    # Find NBA
    nba_sports = [s for s in sports if 'basketball_nba' in s.get('key', '')]
    if nba_sports:
        print(f"\n✅ NBA sport found:")
        for sport in nba_sports:
            print(f"  Key: {sport['key']}")
            print(f"  Title: {sport['title']}")
            print(f"  Active: {sport.get('active', 'N/A')}")
    else:
        print("\n❌ NBA sport not found in available sports")
        print("Available sports:")
        for sport in sports[:5]:
            print(f"  - {sport.get('key')}: {sport.get('title')}")

except Exception as e:
    print(f"❌ API connection failed: {e}")
    exit(1)

# Test 2: Check available markets
print("\n" + "="*80)
print("2. Checking available markets for NBA...")
print("="*80)

url = f"{BASE_URL}/sports/basketball_nba/odds"
params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'player_points_rebounds_assists',  # PRA market
    'oddsFormat': 'american'
}

try:
    response = requests.get(url, params=params)
    print(f"\nStatus Code: {response.status_code}")
    print(f"Requests remaining: {response.headers.get('x-requests-remaining', 'N/A')}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Live odds fetched successfully!")
        print(f"Games available: {len(data)}")

        if len(data) > 0:
            print(f"\nSample game:")
            game = data[0]
            print(f"  Home: {game.get('home_team')}")
            print(f"  Away: {game.get('away_team')}")
            print(f"  Commence: {game.get('commence_time')}")
            print(f"  Bookmakers: {len(game.get('bookmakers', []))}")

            if game.get('bookmakers'):
                bookmaker = game['bookmakers'][0]
                print(f"\n  Sample bookmaker: {bookmaker.get('title')}")
                print(f"  Markets available: {len(bookmaker.get('markets', []))}")

                if bookmaker.get('markets'):
                    market = bookmaker['markets'][0]
                    print(f"\n  Market: {market.get('key')}")
                    print(f"  Outcomes: {len(market.get('outcomes', []))}")

                    # Show first few player props
                    for i, outcome in enumerate(market.get('outcomes', [])[:3]):
                        print(f"\n  Player {i+1}:")
                        print(f"    Name: {outcome.get('description', 'N/A')}")
                        print(f"    Line: {outcome.get('point', 'N/A')}")
                        print(f"    Price (Over): {outcome.get('price', 'N/A')}")
        else:
            print("\n⚠️  No live games currently available")
            print("This is normal if NBA season hasn't started or is between games")

    else:
        print(f"❌ Failed to fetch odds: {response.text[:500]}")

except Exception as e:
    print(f"❌ Error fetching live odds: {e}")

# Test 3: Check historical odds availability
print("\n" + "="*80)
print("3. Checking HISTORICAL odds availability...")
print("="*80)

# TheOddsAPI historical endpoint (if available)
historical_url = f"{BASE_URL}/historical/sports/basketball_nba/odds"
params = {
    'apiKey': API_KEY,
    'regions': 'us',
    'markets': 'player_points_rebounds_assists',
    'date': '2024-06-17T00:00:00Z',  # NBA Finals Game 5
    'oddsFormat': 'american'
}

try:
    response = requests.get(historical_url, params=params)
    print(f"Status Code: {response.status_code}")
    print(f"Requests remaining: {response.headers.get('x-requests-remaining', 'N/A')}")

    if response.status_code == 200:
        data = response.json()
        print(f"✅ Historical odds endpoint works!")
        print(f"Data returned: {type(data)}")
        print(f"Length: {len(data) if isinstance(data, list) else 'N/A'}")

        if data:
            print(f"\n✅ HISTORICAL DATA AVAILABLE!")
            print(f"We can backtest with real odds!")
    elif response.status_code == 401:
        print(f"❌ Historical odds require higher API plan")
        print(f"Current plan may not include historical data access")
    elif response.status_code == 404:
        print(f"❌ Historical endpoint not available")
        print(f"Response: {response.text[:200]}")
    else:
        print(f"⚠️  Status {response.status_code}")
        print(f"Response: {response.text[:500]}")

except Exception as e:
    print(f"❌ Error checking historical: {e}")

# Test 4: Alternative - Check events endpoint
print("\n" + "="*80)
print("4. Checking event history endpoint...")
print("="*80)

events_url = f"{BASE_URL}/sports/basketball_nba/events"
params = {
    'apiKey': API_KEY,
    'dateFrom': '2024-06-01T00:00:00Z',
    'dateTo': '2024-06-20T00:00:00Z'
}

try:
    response = requests.get(events_url, params=params)
    print(f"Status Code: {response.status_code}")

    if response.status_code == 200:
        events = response.json()
        print(f"✅ Events endpoint works!")
        print(f"Events found: {len(events) if isinstance(events, list) else 'N/A'}")
    else:
        print(f"⚠️  Status {response.status_code}: {response.text[:200]}")

except Exception as e:
    print(f"❌ Error: {e}")

# Summary
print("\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
API Key: {'✅ Valid' if API_KEY else '❌ Missing'}
Connection: ✅ Working
Live Odds: Check above
Historical Odds: Check above

Next Steps:
1. If historical available → Fetch 2023-24 season odds
2. If NOT available → Options:
   a) Upgrade API plan
   b) Use alternative data source
   c) Track prospectively from today

API Usage:
- Requests remaining: Check above
- Be mindful of rate limits!
""")
