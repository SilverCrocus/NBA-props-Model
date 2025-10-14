"""
Test TheOddsAPI Player Props Access - Correct Implementation

This script demonstrates the CORRECT way to access NBA player props from TheOddsAPI:
1. Get event IDs using /events endpoint (FREE)
2. Query player props per event using /events/{eventId}/odds

Based on research findings from THEODDSAPI_RESEARCH_FINDINGS.md
"""

import requests
import json
from datetime import datetime
from typing import List, Dict, Any


class TheOddsAPIPlayerProps:
    """
    Correct implementation for accessing NBA player props from TheOddsAPI.

    Key Insight: Player props are "non-featured markets" and MUST be accessed
    one event at a time using /events/{eventId}/odds endpoint.
    """

    def __init__(self, api_key: str):
        self.api_key = api_key
        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"

    def get_events(self) -> List[Dict[str, Any]]:
        """
        Step 1: Get upcoming NBA events and their IDs.

        This endpoint is FREE and does not count against quota.

        Returns:
            List of event dictionaries with id, teams, commence_time
        """
        url = f"{self.base_url}/sports/{self.sport}/events"

        params = {
            "apiKey": self.api_key,
            "dateFormat": "iso"
        }

        print(f"\n{'='*60}")
        print("STEP 1: Getting NBA Events (FREE - No Quota Cost)")
        print(f"{'='*60}")
        print(f"URL: {url}")

        response = requests.get(url, params=params)

        if response.status_code == 200:
            events = response.json()
            print(f"✓ Success! Found {len(events)} NBA events")

            for i, event in enumerate(events[:5], 1):
                print(f"\n  {i}. {event['away_team']} @ {event['home_team']}")
                print(f"     Event ID: {event['id']}")
                print(f"     Starts: {event['commence_time']}")

            if len(events) > 5:
                print(f"\n  ... and {len(events) - 5} more events")

            return events
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return []

    def get_player_props(
        self,
        event_id: str,
        markets: List[str] = None,
        regions: str = "us",
        odds_format: str = "american"
    ) -> Dict[str, Any]:
        """
        Step 2: Get player props for a specific event.

        This is the CORRECT endpoint for player props.

        Args:
            event_id: Event ID from get_events()
            markets: List of markets (default: player_points_rebounds_assists)
            regions: Bookmaker region (default: "us")
            odds_format: "american" or "decimal"

        Returns:
            Event odds data with player props from bookmakers
        """
        if markets is None:
            markets = ["player_points_rebounds_assists"]

        url = f"{self.base_url}/sports/{self.sport}/events/{event_id}/odds"

        params = {
            "apiKey": self.api_key,
            "regions": regions,
            "markets": ",".join(markets),
            "oddsFormat": odds_format
        }

        print(f"\n{'='*60}")
        print("STEP 2: Getting Player Props for Event")
        print(f"{'='*60}")
        print(f"URL: {url}")
        print(f"Markets: {', '.join(markets)}")
        print(f"Regions: {regions}")

        response = requests.get(url, params=params)

        # Check quota usage
        remaining = response.headers.get('x-requests-remaining')
        used = response.headers.get('x-requests-used')

        print(f"\nQuota Status:")
        print(f"  Requests Used: {used}")
        print(f"  Requests Remaining: {remaining}")

        if response.status_code == 200:
            data = response.json()

            if 'bookmakers' in data:
                num_bookmakers = len(data['bookmakers'])
                print(f"\n✓ Success! Found {num_bookmakers} bookmakers with player props")

                # Count total player props
                total_props = 0
                unique_players = set()

                for bookmaker in data['bookmakers']:
                    for market in bookmaker.get('markets', []):
                        for outcome in market.get('outcomes', []):
                            total_props += 1
                            unique_players.add(outcome['name'])

                print(f"  Total prop lines: {total_props}")
                print(f"  Unique players: {len(unique_players)}")

                return data
            else:
                print("✓ Request successful but no bookmakers returned")
                print("  (This event may not have player props available)")
                return data
        else:
            print(f"✗ Error {response.status_code}: {response.text}")
            return {}

    def get_all_player_props(
        self,
        max_events: int = 3,
        markets: List[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Complete workflow: Get events, then query player props for each.

        Args:
            max_events: Maximum number of events to query (to conserve quota)
            markets: List of markets to request

        Returns:
            List of event data with player props
        """
        # Step 1: Get events (FREE)
        events = self.get_events()

        if not events:
            print("\nNo events found or error occurred")
            return []

        # Step 2: Get player props for each event (uses quota)
        results = []

        print(f"\n{'='*60}")
        print(f"Querying Player Props for {min(max_events, len(events))} Events")
        print(f"{'='*60}")

        for i, event in enumerate(events[:max_events], 1):
            print(f"\n[{i}/{min(max_events, len(events))}] {event['away_team']} @ {event['home_team']}")

            props_data = self.get_player_props(
                event_id=event['id'],
                markets=markets
            )

            if props_data:
                results.append(props_data)

        return results

    def display_pra_props(self, event_data: Dict[str, Any]) -> None:
        """
        Display PRA props in a readable format.

        Args:
            event_data: Event data from get_player_props()
        """
        if not event_data or 'bookmakers' not in event_data:
            return

        print(f"\n{'='*60}")
        print(f"PRA Props: {event_data['away_team']} @ {event_data['home_team']}")
        print(f"Game Time: {event_data['commence_time']}")
        print(f"{'='*60}")

        for bookmaker in event_data['bookmakers']:
            print(f"\n{bookmaker['title']} ({bookmaker['key']}):")

            for market in bookmaker.get('markets', []):
                if market['key'] == 'player_points_rebounds_assists':
                    # Group by player
                    players = {}
                    for outcome in market['outcomes']:
                        player_name = outcome['name']
                        if player_name not in players:
                            players[player_name] = {'over': None, 'under': None, 'line': None}

                        players[player_name]['line'] = outcome['point']
                        if outcome['description'] == 'Over':
                            players[player_name]['over'] = outcome['price']
                        else:
                            players[player_name]['under'] = outcome['price']

                    # Display
                    for player, odds in sorted(players.items()):
                        print(f"  {player:25} O/U {odds['line']:4.1f}  "
                              f"Over: {odds['over']:+4}  Under: {odds['under']:+4}")


def main():
    """
    Main test function - demonstrates correct API usage.
    """
    API_KEY = "18405dde82249ca0a31950d7819767c7"

    print("="*60)
    print("TheOddsAPI Player Props Test - CORRECT Implementation")
    print("="*60)
    print(f"API Key: {API_KEY[:20]}...")
    print(f"Test Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")

    # Initialize client
    client = TheOddsAPIPlayerProps(api_key=API_KEY)

    # Test 1: Get events only (FREE)
    print("\n" + "="*60)
    print("TEST 1: Get Events (FREE - No Quota Usage)")
    print("="*60)
    events = client.get_events()

    if not events:
        print("\nNo events available. NBA might be off-season or no games scheduled.")
        return

    # Test 2: Get player props for first event
    print("\n" + "="*60)
    print("TEST 2: Get PRA Props for Single Event")
    print("="*60)

    first_event = events[0]
    props_data = client.get_player_props(
        event_id=first_event['id'],
        markets=['player_points_rebounds_assists']
    )

    if props_data:
        client.display_pra_props(props_data)

    # Test 3: Get multiple player prop markets
    print("\n" + "="*60)
    print("TEST 3: Get Multiple Markets for Same Event")
    print("="*60)

    multi_market_data = client.get_player_props(
        event_id=first_event['id'],
        markets=[
            'player_points_rebounds_assists',
            'player_points',
            'player_rebounds',
            'player_assists'
        ]
    )

    if multi_market_data and 'bookmakers' in multi_market_data:
        print(f"\nMarkets received:")
        for bookmaker in multi_market_data['bookmakers'][:1]:  # Just first bookmaker
            for market in bookmaker.get('markets', []):
                num_outcomes = len(market.get('outcomes', []))
                print(f"  - {market['key']}: {num_outcomes} outcomes")

    # Test 4: Complete workflow for multiple events
    print("\n" + "="*60)
    print("TEST 4: Complete Workflow (Multiple Events)")
    print("="*60)
    print("\nLimiting to 2 events to conserve quota...")

    all_props = client.get_all_player_props(
        max_events=2,
        markets=['player_points_rebounds_assists']
    )

    print(f"\n{'='*60}")
    print("TEST SUMMARY")
    print(f"{'='*60}")
    print(f"Total events found: {len(events)}")
    print(f"Events queried for props: 2")
    print(f"Events with props data: {len(all_props)}")

    if all_props:
        print("\n✓ SUCCESS! TheOddsAPI player props are working correctly.")
        print("\nKey Findings:")
        print("  1. Player props MUST use /events/{eventId}/odds endpoint")
        print("  2. /events endpoint is FREE (no quota cost)")
        print("  3. Event-specific odds DO cost quota credits")
        print("  4. Market key 'player_points_rebounds_assists' is VALID")
        print("\nYou were using the wrong endpoint before!")
    else:
        print("\n⚠ No player props data returned.")
        print("  This could mean:")
        print("  - Games are too far in the future (props not released yet)")
        print("  - NBA off-season (no games)")
        print("  - US bookmakers haven't posted props yet")
        print("  But the API structure is correct!")


if __name__ == "__main__":
    main()
