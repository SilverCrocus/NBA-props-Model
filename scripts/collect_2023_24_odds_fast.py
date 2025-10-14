"""
FAST Historical Odds Collector - Parallel Requests

Speed improvements:
- Concurrent requests (10x faster)
- Filtered events (90% fewer API calls)
- Batch processing
- Resume capability
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
from concurrent.futures import ThreadPoolExecutor, as_completed
import threading

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

# Parallelization settings
MAX_WORKERS = 10  # Concurrent requests (adjust based on rate limits)
RATE_LIMIT_DELAY = 0.1  # Seconds between requests

print("="*80)
print("FAST 2023-24 NBA SEASON - HISTORICAL PRA ODDS COLLECTOR")
print("="*80)
print(f"\nSettings:")
print(f"  Concurrent workers: {MAX_WORKERS}")
print(f"  Season: {SEASON_START.date()} to {SEASON_END.date()}")
print(f"  Output: {OUTPUT_DIR}")

# Thread-safe counter for rate limiting
request_lock = threading.Lock()
last_request_time = [time.time()]

def rate_limited_request(url, params):
    """Make request with rate limiting."""
    with request_lock:
        # Ensure minimum delay between requests
        elapsed = time.time() - last_request_time[0]
        if elapsed < RATE_LIMIT_DELAY:
            time.sleep(RATE_LIMIT_DELAY - elapsed)
        last_request_time[0] = time.time()

    return requests.get(url, params=params)

# Load progress
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

# Check quota
test = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY})
remaining = float(test.headers.get('x-requests-remaining', 0))
print(f"API requests remaining: {remaining:,.0f}")

# Load existing props
all_props = []
if PROPS_FILE.exists():
    existing = pd.read_csv(PROPS_FILE)
    all_props = existing.to_dict('records')
    print(f"Loaded {len(all_props):,} existing props")

# Thread-safe props list
props_lock = threading.Lock()

def fetch_event_odds(event, date_str):
    """Fetch odds for a single event."""
    event_id = event['id']
    commence_time = event['commence_time']
    home_team = event['home_team']
    away_team = event['away_team']

    odds_url = f"{BASE_URL}/historical/sports/basketball_nba/events/{event_id}/odds"
    odds_params = {
        "apiKey": API_KEY,
        "regions": "us",
        "markets": "player_points_rebounds_assists",
        "date": commence_time,
        "oddsFormat": "american"
    }

    try:
        odds_resp = rate_limited_request(odds_url, odds_params)

        if odds_resp.status_code == 200:
            odds_data = odds_resp.json()
            data = odds_data.get('data', {}) if isinstance(odds_data, dict) else {}
            bookmakers = data.get('bookmakers', [])

            event_props = []
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
                                    'home_team': home_team,
                                    'away_team': away_team,
                                    'bookmaker': bookmaker['title'],
                                    'line': outcome.get('point'),
                                    'timestamp': odds_data.get('timestamp', '')
                                }

                            if outcome.get('name') == 'Over':
                                outcomes_dict[player]['over_price'] = outcome.get('price')
                            elif outcome.get('name') == 'Under':
                                outcomes_dict[player]['under_price'] = outcome.get('price')

                        # Add valid props
                        for prop in outcomes_dict.values():
                            if prop.get('line') is not None:
                                event_props.append(prop)

            return {
                'success': True,
                'props': event_props,
                'game': f"{away_team} @ {home_team}",
                'count': len(event_props)
            }
        else:
            return {
                'success': False,
                'props': [],
                'game': f"{away_team} @ {home_team}",
                'status': odds_resp.status_code
            }

    except Exception as e:
        return {
            'success': False,
            'props': [],
            'game': f"{away_team} @ {home_team}",
            'error': str(e)[:100]
        }

# Main collection loop
print(f"\n" + "="*80)
print("FETCHING HISTORICAL ODDS (PARALLEL)")
print("="*80)

total_days = (SEASON_END - SEASON_START).days + 1
current_date = SEASON_START

with tqdm(total=total_days, desc="Collecting", unit="day") as pbar:
    pbar.update(len(progress['completed_dates']))

    while current_date <= SEASON_END:
        date_str = current_date.strftime("%Y-%m-%d")

        # Skip completed dates
        if date_str in progress['completed_dates']:
            current_date += timedelta(days=1)
            continue

        # Fetch events for this date
        events_url = f"{BASE_URL}/historical/sports/basketball_nba/events"
        events_params = {
            "apiKey": API_KEY,
            "date": current_date.strftime("%Y-%m-%dT12:00:00Z")
        }

        try:
            events_resp = requests.get(events_url, params=events_params)

            if events_resp.status_code != 200:
                pbar.write(f"❌ {date_str}: Error fetching events")
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
                    if date_start <= commence_dt < date_end:
                        events.append(event)
                except:
                    continue

            if not events:
                progress['completed_dates'].append(date_str)
                save_progress(progress)
                pbar.update(1)
                current_date += timedelta(days=1)
                continue

            pbar.write(f"📅 {date_str}: {len(events)} games")

            # Fetch odds in parallel
            day_props = []
            with ThreadPoolExecutor(max_workers=MAX_WORKERS) as executor:
                # Submit all tasks
                future_to_event = {
                    executor.submit(fetch_event_odds, event, date_str): event
                    for event in events
                }

                # Collect results
                for future in as_completed(future_to_event):
                    result = future.result()

                    if result['success']:
                        day_props.extend(result['props'])
                        if result['count'] > 0:
                            pbar.write(f"  ✅ {result['game']}: {result['count']} props")
                    else:
                        if result.get('status') == 404:
                            pbar.write(f"  ⚠️  {result['game']}: No props (404)")
                        elif result.get('error'):
                            pbar.write(f"  ❌ {result['game']}: {result['error']}")

            # Add to master list
            with props_lock:
                all_props.extend(day_props)
                progress['total_events'] += len(events)
                progress['total_props'] += len(day_props)

            # Mark complete and save
            progress['completed_dates'].append(date_str)
            save_progress(progress)

            # Save props incrementally
            if all_props:
                df = pd.DataFrame(all_props)
                df.to_csv(PROPS_FILE, index=False)

            pbar.update(1)
            current_date += timedelta(days=1)

        except Exception as e:
            pbar.write(f"❌ {date_str}: {str(e)[:100]}")
            current_date += timedelta(days=1)
            time.sleep(2)

# Final save and summary
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

    # Sample
    print(f"\nSample props:")
    sample = df[['player_name', 'line', 'over_price', 'under_price', 'bookmaker', 'event_date']].head(10)
    print(sample.to_string())

    print(f"\n📁 Saved to: {PROPS_FILE}")

# Final quota check
final = requests.get(f"{BASE_URL}/sports", params={"apiKey": API_KEY})
final_remaining = final.headers.get('x-requests-remaining', 'Unknown')
used = remaining - float(final_remaining) if final_remaining != 'Unknown' else 'Unknown'
print(f"\nAPI usage:")
print(f"  Started with: {remaining:,.0f}")
print(f"  Remaining: {final_remaining}")
print(f"  Used: {used if used == 'Unknown' else f'{used:,.0f}'}")
print("="*80)
