"""
Fetch Historical NBA PRA Odds from TheOddsAPI for 2023-24 Season

This script fetches historical player prop odds (PRA - Points + Rebounds + Assists)
for the entire 2023-24 NBA season to enable proper backtesting.
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time
import json
from pathlib import Path

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = "https://api.the-odds-api.com/v4"

print("="*80)
print("FETCHING HISTORICAL NBA PRA ODDS - 2023-24 SEASON")
print("="*80)

# Configuration
SEASON_START = datetime(2023, 10, 24)  # 2023-24 season opener
SEASON_END = datetime(2024, 6, 17)     # 2023-24 Finals Game 5
OUTPUT_DIR = Path("data/historical_odds")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

print(f"\nConfiguration:")
print(f"  Season: {SEASON_START.date()} to {SEASON_END.date()}")
print(f"  Output: {OUTPUT_DIR}")
print(f"  API Key: {API_KEY[:8]}...")

# Check API quota before starting
print(f"\nChecking API quota...")
test_url = f"{BASE_URL}/sports"
test_response = requests.get(test_url, params={"apiKey": API_KEY})
remaining = test_response.headers.get('x-requests-remaining', 'Unknown')
print(f"  Requests remaining: {remaining}")

if isinstance(remaining, str) and remaining != 'Unknown':
    if float(remaining) < 100:
        print(f"\n⚠️  WARNING: Low quota ({remaining} requests)")
        print(f"  Historical data collection needs ~500-1000 requests")
        response = input("Continue anyway? (y/n): ")
        if response.lower() != 'y':
            exit(0)

# Step 1: Get all historical events for the season
print(f"\n" + "="*80)
print("STEP 1: Fetching Historical Events")
print("="*80)

all_events = []
current_date = SEASON_START

# TheOddsAPI historical events endpoint
# Note: We'll need to request events in chunks (by date)
while current_date <= SEASON_END:
    # Request events for a week at a time
    date_from = current_date.strftime("%Y-%m-%dT00:00:00Z")
    date_to = (current_date + timedelta(days=7)).strftime("%Y-%m-%dT00:00:00Z")

    events_url = f"{BASE_URL}/sports/basketball_nba/events"
    params = {
        "apiKey": API_KEY,
        "dateFrom": date_from,
        "dateTo": date_to
    }

    print(f"\nFetching events: {current_date.date()} to {(current_date + timedelta(days=7)).date()}")

    try:
        response = requests.get(events_url, params=params)

        if response.status_code == 200:
            events = response.json()
            all_events.extend(events)
            print(f"  ✅ Found {len(events)} events")

            remaining = response.headers.get('x-requests-remaining')
            print(f"  Requests remaining: {remaining}")

        elif response.status_code == 404:
            # No events for this period (expected during off-season)
            print(f"  ℹ️  No events found")

        else:
            print(f"  ❌ Error {response.status_code}: {response.text[:200]}")

        # Rate limiting
        time.sleep(1)

    except Exception as e:
        print(f"  ❌ Exception: {e}")

    current_date += timedelta(days=7)

print(f"\n✅ Total events found: {len(all_events)}")

# Save events list
events_file = OUTPUT_DIR / "2023-24_events.json"
with open(events_file, 'w') as f:
    json.dump(all_events, f, indent=2)
print(f"✅ Saved events to {events_file}")

# Step 2: Fetch historical odds for each event
print(f"\n" + "="*80)
print("STEP 2: Fetching Historical Player Props for Each Event")
print("="*80)

all_props_data = []
errors = []

print(f"\nNote: This will use ~{len(all_events)} API requests")
print(f"Processing {len(all_events)} events...")

for i, event in enumerate(all_events, 1):
    event_id = event['id']
    event_date = event['commence_time']
    home_team = event['home_team']
    away_team = event['away_team']

    if i % 10 == 0:
        print(f"\nProgress: {i}/{len(all_events)} ({i/len(all_events)*100:.1f}%)")

    # Fetch historical odds for this specific event
    # Historical endpoint syntax
    props_url = f"{BASE_URL}/historical/sports/basketball_nba/events/{event_id}/odds"

    params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "player_points_rebounds_assists",
        "date": event_date,  # Get odds as of game time
        "oddsFormat": "american"
    }

    try:
        response = requests.get(props_url, params=params)

        if response.status_code == 200:
            odds_data = response.json()

            # Store the data
            all_props_data.append({
                "event_id": event_id,
                "event_date": event_date,
                "home_team": home_team,
                "away_team": away_team,
                "odds_data": odds_data
            })

            print(f"  ✅ {i}: {away_team} @ {home_team} - {event_date[:10]}")

        elif response.status_code == 401:
            print(f"\n❌ AUTHENTICATION ERROR: Historical data may require upgraded API plan")
            print(f"Response: {response.text[:300]}")
            break

        elif response.status_code == 404:
            # No props data available for this event
            errors.append({
                "event_id": event_id,
                "event_date": event_date,
                "teams": f"{away_team} @ {home_team}",
                "error": "No props data (404)"
            })
            print(f"  ⚠️  {i}: {away_team} @ {home_team} - No props data")

        else:
            errors.append({
                "event_id": event_id,
                "event_date": event_date,
                "teams": f"{away_team} @ {home_team}",
                "error": f"Status {response.status_code}"
            })
            print(f"  ❌ {i}: Error {response.status_code}")

        # Check quota
        remaining = response.headers.get('x-requests-remaining')
        if remaining and float(remaining) < 10:
            print(f"\n⚠️  Low quota ({remaining} remaining) - stopping")
            break

        # Rate limiting (1 request per second to be safe)
        time.sleep(1.5)

    except Exception as e:
        errors.append({
            "event_id": event_id,
            "event_date": event_date,
            "teams": f"{away_team} @ {home_team}",
            "error": str(e)
        })
        print(f"  ❌ {i}: Exception - {str(e)[:100]}")
        time.sleep(2)

print(f"\n✅ Successfully fetched props for {len(all_props_data)} events")
print(f"⚠️  Errors for {len(errors)} events")

# Save raw props data
props_file = OUTPUT_DIR / "2023-24_raw_props.json"
with open(props_file, 'w') as f:
    json.dump(all_props_data, f, indent=2)
print(f"\n✅ Saved raw props data to {props_file}")

# Save errors log
if errors:
    errors_file = OUTPUT_DIR / "2023-24_fetch_errors.json"
    with open(errors_file, 'w') as f:
        json.dump(errors, f, indent=2)
    print(f"⚠️  Saved errors log to {errors_file}")

# Step 3: Parse and structure the data
print(f"\n" + "="*80)
print("STEP 3: Parsing Player Props Data")
print("="*80)

parsed_props = []

for event_data in all_props_data:
    event_id = event_data['event_id']
    event_date = event_data['event_date']
    home_team = event_data['home_team']
    away_team = event_data['away_team']
    odds_data = event_data['odds_data']

    # Parse bookmakers and player props
    if 'bookmakers' in odds_data:
        for bookmaker in odds_data['bookmakers']:
            bookmaker_name = bookmaker['title']

            for market in bookmaker.get('markets', []):
                if market['key'] == 'player_points_rebounds_assists':

                    for outcome in market.get('outcomes', []):
                        parsed_props.append({
                            'event_id': event_id,
                            'event_date': event_date,
                            'home_team': home_team,
                            'away_team': away_team,
                            'bookmaker': bookmaker_name,
                            'player_name': outcome.get('description', ''),
                            'pra_line': outcome.get('point', None),
                            'over_price': outcome.get('price', None) if outcome.get('name') == 'Over' else None,
                            'under_price': outcome.get('price', None) if outcome.get('name') == 'Under' else None,
                            'last_update': market.get('last_update', '')
                        })

# Create DataFrame
if parsed_props:
    df_props = pd.DataFrame(parsed_props)

    # Save to CSV
    csv_file = OUTPUT_DIR / "2023-24_pra_odds.csv"
    df_props.to_csv(csv_file, index=False)
    print(f"✅ Saved parsed props to {csv_file}")

    print(f"\nDataset Summary:")
    print(f"  Total prop lines: {len(df_props):,}")
    print(f"  Unique players: {df_props['player_name'].nunique()}")
    print(f"  Unique events: {df_props['event_id'].nunique()}")
    print(f"  Bookmakers: {df_props['bookmaker'].nunique()}")
    print(f"  Date range: {df_props['event_date'].min()} to {df_props['event_date'].max()}")

    print(f"\nSample props:")
    print(df_props[['player_name', 'pra_line', 'over_price', 'bookmaker', 'event_date']].head(10))

else:
    print(f"⚠️  No props data to parse")

# Final summary
print(f"\n" + "="*80)
print("SUMMARY")
print("="*80)

print(f"""
✅ Events fetched: {len(all_events)}
✅ Props data collected: {len(all_props_data)} events
⚠️  Errors: {len(errors)} events
✅ Total prop lines: {len(parsed_props):,} (if successful)

Files saved to: {OUTPUT_DIR}/
  - 2023-24_events.json (all events)
  - 2023-24_raw_props.json (raw API responses)
  - 2023-24_pra_odds.csv (parsed props data)
  - 2023-24_fetch_errors.json (error log)

Next Steps:
1. Check if historical data requires upgraded API plan (if auth errors)
2. Match odds to your model predictions
3. Re-run backtest with real betting lines
4. Calculate true ROI and CLV
""")

final_quota = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY})
final_remaining = final_quota.headers.get('x-requests-remaining', 'Unknown')
print(f"Final API quota: {final_remaining} requests remaining")

print("="*80)
print("✅ HISTORICAL ODDS FETCH COMPLETE")
print("="*80)
