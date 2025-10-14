"""
Quick test to verify historical odds access for 2023-24 season
"""

import requests
from dotenv import load_dotenv
import os
import json

load_dotenv()
API_KEY = os.getenv('ODDS_API_KEY')
BASE_URL = "https://api.the-odds-api.com/v4"

print("="*80)
print("TESTING HISTORICAL ODDS ACCESS - Single Game")
print("="*80)

# Test with 2023-24 season opener (October 24, 2023) - we know this game happened
test_date_from = "2023-10-24T00:00:00Z"
test_date_to = "2023-10-25T00:00:00Z"

print(f"\nStep 1: Get HISTORICAL events for {test_date_from[:10]}")
print("-"*80)

# For historical events, we need to use historical endpoint
events_url = f"{BASE_URL}/historical/sports/basketball_nba/events"
params = {
    "apiKey": API_KEY,
    "date": test_date_from,  # Historical uses single date parameter
}

try:
    response = requests.get(events_url, params=params)
    print(f"Status: {response.status_code}")
    print(f"Requests remaining: {response.headers.get('x-requests-remaining')}")

    if response.status_code == 200:
        data = response.json()
        print(f"Response type: {type(data)}")
        print(f"Raw response: {json.dumps(data, indent=2)[:1000]}")

        # Historical endpoint returns dict with 'data' field
        if isinstance(data, dict) and 'data' in data:
            events = data['data'] if isinstance(data['data'], list) else [data['data']]
        elif isinstance(data, list):
            events = data
        else:
            events = [data]

        print(f"✅ Found {len(events)} events")

        if len(events) > 0:
            event = events[0] if isinstance(events[0], dict) else None
            if not event:
                print(f"⚠️  Unable to parse event data")
                print("="*80)
                exit(1)
            event_id = event['id']
            print(f"\nEvent details:")
            print(f"  ID: {event_id}")
            print(f"  Home: {event['home_team']}")
            print(f"  Away: {event['away_team']}")
            print(f"  Date: {event['commence_time']}")

            # Step 2: Try to fetch historical odds for this event
            print(f"\nStep 2: Fetch historical PRA odds")
            print("-"*80)

            # Try historical endpoint
            historical_url = f"{BASE_URL}/historical/sports/basketball_nba/events/{event_id}/odds"
            hist_params = {
                "apiKey": API_KEY,
                "regions": "us",
                "markets": "player_points_rebounds_assists",
                "date": event['commence_time'],
                "oddsFormat": "american"
            }

            print(f"URL: {historical_url}")

            hist_response = requests.get(historical_url, params=hist_params)
            print(f"Status: {hist_response.status_code}")
            print(f"Requests remaining: {hist_response.headers.get('x-requests-remaining')}")

            if hist_response.status_code == 200:
                odds = hist_response.json()
                print(f"\n✅ HISTORICAL ODDS FETCHED SUCCESSFULLY!")
                print(f"Response type: {type(odds)}")
                print(f"Response keys: {odds.keys() if isinstance(odds, dict) else 'N/A'}")

                # Show sample data
                print(f"\n{json.dumps(odds, indent=2)[:2000]}")

            elif hist_response.status_code == 401:
                print(f"\n❌ AUTHENTICATION ERROR")
                print(f"Historical data requires upgraded API plan")
                print(f"Response: {hist_response.text}")

            elif hist_response.status_code == 404:
                print(f"\n⚠️  NOT FOUND (404)")
                print(f"Historical data may not be available for this date")
                print(f"Response: {hist_response.text}")

            else:
                print(f"\n❌ ERROR {hist_response.status_code}")
                print(f"Response: {hist_response.text[:500]}")

        else:
            print(f"\n⚠️  No events found for this date")
            print(f"Response: {response.text[:500]}")

    else:
        print(f"❌ Error fetching events: {response.text[:500]}")

except Exception as e:
    print(f"❌ Exception: {e}")

print("\n" + "="*80)
print("TEST COMPLETE")
print("="*80)
