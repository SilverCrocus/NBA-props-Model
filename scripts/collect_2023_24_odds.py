"""
Production Historical Odds Collector for 2023-24 NBA Season

Fetches all PRA betting lines for the entire 2023-24 season with:
- Progress tracking & resume capability
- Incremental saves
- Rate limiting
- Error handling
"""

import requests
import pandas as pd
from datetime import datetime, timedelta
from dotenv import load_dotenv
import os
import time
import json
from pathlib import Path
from tqdm import tqdm

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = "https://api.the-odds-api.com/v4"

# Configuration
SEASON_START = datetime(2023, 10, 24)
SEASON_END = datetime(2024, 6, 17)
OUTPUT_DIR = Path("data/historical_odds/2023-24")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROGRESS_FILE = OUTPUT_DIR / "fetch_progress.json"
PROPS_FILE = OUTPUT_DIR / "pra_odds.csv"

print("="*80)
print("2023-24 NBA SEASON - HISTORICAL PRA ODDS COLLECTOR")
print("="*80)
print(f"\nSeason: {SEASON_START.date()} to {SEASON_END.date()}")
print(f"Output: {OUTPUT_DIR}")

# Load progress if exists
def load_progress():
    if PROGRESS_FILE.exists():
        with open(PROGRESS_FILE, 'r') as f:
            return json.load(f)
    return {"completed_dates": [], "total_events": 0, "total_props": 0}

def save_progress(progress):
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f, indent=2)

progress = load_progress()
print(f"\nProgress: {len(progress['completed_dates'])} dates already fetched")
print(f"Total events collected: {progress['total_events']}")
print(f"Total props collected: {progress['total_props']}")

# Check API quota
print(f"\nChecking API quota...")
test = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY})
remaining = float(test.headers.get('x-requests-remaining', 0))
print(f"Requests remaining: {remaining:,.0f}")

# Estimate total requests needed
total_days = (SEASON_END - SEASON_START).days + 1
days_remaining = total_days - len(progress['completed_dates'])
estimated_requests = days_remaining * 5  # ~5 games per day average

print(f"\nEstimated requests needed: ~{estimated_requests}")
if remaining < estimated_requests:
    print(f"⚠️  WARNING: May not have enough quota ({remaining} < {estimated_requests})")
    cont = input("Continue anyway? (y/n): ")
    if cont.lower() != 'y':
        exit(0)

# Fetch historical odds day by day
print(f"\n" + "="*80)
print("FETCHING HISTORICAL ODDS")
print("="*80)

all_props = []
if PROPS_FILE.exists():
    # Load existing props
    existing = pd.read_csv(PROPS_FILE)
    all_props = existing.to_dict('records')
    print(f"Loaded {len(all_props):,} existing props from {PROPS_FILE}")

current_date = SEASON_START
pbar = tqdm(total=total_days, desc="Collecting odds", unit="day")
pbar.update(len(progress['completed_dates']))

while current_date <= SEASON_END:
    date_str = current_date.strftime("%Y-%m-%d")

    # Skip if already completed
    if date_str in progress['completed_dates']:
        current_date += timedelta(days=1)
        continue

    # Step 1: Get events for this date
    events_url = f"{BASE_URL}/historical/sports/basketball_nba/events"
    events_params = {
        "apiKey": API_KEY,
        "date": current_date.strftime("%Y-%m-%dT12:00:00Z")  # Noon UTC
    }

    try:
        events_resp = requests.get(events_url, params=events_params)

        if events_resp.status_code != 200:
            pbar.write(f"❌ {date_str}: Error fetching events ({events_resp.status_code})")
            time.sleep(2)
            current_date += timedelta(days=1)
            continue

        events_data = events_resp.json()
        all_events = events_data.get('data', []) if isinstance(events_data, dict) else events_data

        # FILTER: Only games on this specific date
        # Make date_start timezone-aware (UTC)
        from datetime import timezone
        date_start = current_date.replace(hour=0, minute=0, second=0, tzinfo=timezone.utc)
        date_end = date_start + timedelta(days=1)

        events = []
        for event in all_events:
            try:
                commence_dt = datetime.fromisoformat(event['commence_time'].replace('Z', '+00:00'))
                # Check if game is on this date (UTC)
                if date_start <= commence_dt < date_end:
                    events.append(event)
            except Exception:
                # Skip events with invalid dates
                continue

        if not events:
            # No games this day
            progress['completed_dates'].append(date_str)
            save_progress(progress)
            pbar.update(1)
            current_date += timedelta(days=1)
            time.sleep(0.5)
            continue

        pbar.write(f"📅 {date_str}: {len(events)} games (filtered from {len(all_events)} total)")

        # Step 2: Fetch props for each event
        for event in events:
            event_id = event['id']
            commence_time = event['commence_time']

            # Get odds for this event
            odds_url = f"{BASE_URL}/historical/sports/basketball_nba/events/{event_id}/odds"
            odds_params = {
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "player_points_rebounds_assists",
                "date": commence_time,
                "oddsFormat": "american"
            }

            try:
                odds_resp = requests.get(odds_url, params=odds_params)

                if odds_resp.status_code == 200:
                    odds_data = odds_resp.json()

                    # Parse props
                    data = odds_data.get('data', {}) if isinstance(odds_data, dict) else {}
                    bookmakers = data.get('bookmakers', [])

                    props_count = 0
                    for bookmaker in bookmakers:
                        for market in bookmaker.get('markets', []):
                            if market['key'] == 'player_points_rebounds_assists':
                                # Group Over/Under pairs
                                outcomes_dict = {}
                                for outcome in market.get('outcomes', []):
                                    player = outcome.get('description', '')
                                    if player not in outcomes_dict:
                                        outcomes_dict[player] = {
                                            'player_name': player,
                                            'event_id': event_id,
                                            'event_date': commence_time[:10],
                                            'home_team': event['home_team'],
                                            'away_team': event['away_team'],
                                            'bookmaker': bookmaker['title'],
                                            'line': outcome.get('point'),
                                            'timestamp': odds_data.get('timestamp', '')
                                        }

                                    if outcome.get('name') == 'Over':
                                        outcomes_dict[player]['over_price'] = outcome.get('price')
                                    elif outcome.get('name') == 'Under':
                                        outcomes_dict[player]['under_price'] = outcome.get('price')

                                # Add to all_props
                                for prop in outcomes_dict.values():
                                    if prop.get('line') is not None:  # Only if line exists
                                        all_props.append(prop)
                                        props_count += 1

                    pbar.write(f"  ✅ {event['away_team']} @ {event['home_team']}: {props_count} props")
                    progress['total_events'] += 1
                    progress['total_props'] += props_count

                else:
                    pbar.write(f"  ⚠️  {event['away_team']} @ {event['home_team']}: No props ({odds_resp.status_code})")

                # Rate limit
                time.sleep(0.8)

            except Exception as e:
                pbar.write(f"  ❌ Error: {str(e)[:100]}")
                time.sleep(2)

        # Mark date complete
        progress['completed_dates'].append(date_str)
        save_progress(progress)

        # Save props incrementally (every day)
        if all_props:
            df = pd.DataFrame(all_props)
            df.to_csv(PROPS_FILE, index=False)

        pbar.update(1)
        current_date += timedelta(days=1)
        time.sleep(1)  # Rate limit between dates

    except Exception as e:
        pbar.write(f"❌ {date_str}: Exception - {str(e)[:100]}")
        time.sleep(5)
        current_date += timedelta(days=1)

pbar.close()

# Final save
if all_props:
    df = pd.DataFrame(all_props)
    df.to_csv(PROPS_FILE, index=False)

    print(f"\n" + "="*80)
    print("COLLECTION COMPLETE")
    print("="*80)
    print(f"\n✅ Total props collected: {len(df):,}")
    print(f"✅ Unique players: {df['player_name'].nunique()}")
    print(f"✅ Unique events: {df['event_id'].nunique()}")
    print(f"✅ Bookmakers: {df['bookmaker'].nunique()}")
    print(f"✅ Date range: {df['event_date'].min()} to {df['event_date'].max()}")

    print(f"\n📁 Saved to: {PROPS_FILE}")

    # Show sample
    print(f"\nSample props:")
    print(df[['player_name', 'line', 'over_price', 'under_price', 'bookmaker', 'event_date']].head(10))

else:
    print(f"\n⚠️  No props collected")

# Check final quota
final = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY})
final_remaining = final.headers.get('x-requests-remaining', 'Unknown')
print(f"\nFinal API quota: {final_remaining} requests remaining")
print("="*80)
