"""
Fetch Real NBA Player Prop Betting Lines from The Odds API

This script fetches actual historical betting lines for NBA player props
(Points + Rebounds + Assists) from The Odds API.

API Documentation: https://the-odds-api.com/liveapi/guides/v4/

Author: NBA Props Model - Phase 4 Week 1
Date: October 15, 2025
"""

import json
import logging
import os
import sys
import time
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent))

logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


class TheOddsAPIFetcher:
    """Fetch betting lines from The Odds API."""

    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize The Odds API fetcher.

        Args:
            api_key: The Odds API key (will check env if not provided)
        """
        self.api_key = api_key or os.getenv("ODDS_API_KEY") or os.getenv("THEODDSAPI_KEY")
        if not self.api_key:
            raise ValueError(
                "No API key found. Set ODDS_API_KEY or THEODDSAPI_KEY environment variable.\n"
                "Get your API key from: https://the-odds-api.com"
            )

        self.base_url = "https://api.the-odds-api.com/v4"
        self.sport = "basketball_nba"
        self.requests_used = 0
        self.requests_remaining = None

    def check_quota(self):
        """Check API quota/usage."""
        url = f"{self.base_url}/sports"
        params = {"apiKey": self.api_key}

        response = requests.get(url, params=params)

        if "x-requests-remaining" in response.headers:
            self.requests_remaining = int(float(response.headers["x-requests-remaining"]))
            logger.info(f"API requests remaining: {self.requests_remaining}")

        if "x-requests-used" in response.headers:
            self.requests_used = int(float(response.headers["x-requests-used"]))

        return response.status_code == 200

    def fetch_historical_odds(
        self, date: str, markets: List[str] = None, bookmakers: List[str] = None
    ) -> Optional[Dict]:
        """
        Fetch historical odds for a specific date.

        Args:
            date: ISO date string (YYYY-MM-DD)
            markets: List of markets (default: ['player_points_rebounds_assists'])
            bookmakers: List of bookmakers to include

        Returns:
            Dict with odds data or None if error
        """
        if markets is None:
            markets = ["player_points_rebounds_assists"]

        # Historical endpoint
        url = f"{self.base_url}/historical/sports/{self.sport}/odds"

        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
            "date": date + "T12:00:00Z",  # Noon UTC on that date
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        try:
            response = requests.get(url, params=params, timeout=30)

            # Update quota info
            if "x-requests-remaining" in response.headers:
                self.requests_remaining = int(float(response.headers["x-requests-remaining"]))

            if response.status_code == 200:
                data = response.json()
                logger.info(f"  {date}: Found {len(data.get('data', []))} games")
                return data
            elif response.status_code == 401:
                logger.error("Invalid API key")
                return None
            elif response.status_code == 422:
                logger.warning(f"  {date}: No data available (game may not have odds yet)")
                return None
            else:
                logger.error(f"  {date}: Error {response.status_code}: {response.text}")
                return None

        except requests.exceptions.Timeout:
            logger.error(f"  {date}: Request timed out")
            return None
        except Exception as e:
            logger.error(f"  {date}: Error - {str(e)}")
            return None

    def fetch_recent_odds(
        self, markets: List[str] = None, bookmakers: List[str] = None
    ) -> Optional[Dict]:
        """
        Fetch recent/upcoming odds (last 3 days + upcoming).

        Args:
            markets: List of markets
            bookmakers: List of bookmakers

        Returns:
            Dict with odds data
        """
        if markets is None:
            markets = ["player_points_rebounds_assists"]

        url = f"{self.base_url}/sports/{self.sport}/odds"

        params = {
            "apiKey": self.api_key,
            "regions": "us",
            "markets": ",".join(markets),
            "oddsFormat": "american",
        }

        if bookmakers:
            params["bookmakers"] = ",".join(bookmakers)

        try:
            response = requests.get(url, params=params, timeout=30)

            if "x-requests-remaining" in response.headers:
                self.requests_remaining = int(float(response.headers["x-requests-remaining"]))

            if response.status_code == 200:
                return response.json()
            else:
                logger.error(f"Error {response.status_code}: {response.text}")
                return None

        except Exception as e:
            logger.error(f"Error fetching recent odds: {str(e)}")
            return None

    def parse_player_props(self, odds_data: Dict) -> pd.DataFrame:
        """
        Parse player props from odds API response.

        Args:
            odds_data: Response from The Odds API

        Returns:
            DataFrame with player prop lines
        """
        rows = []

        games = odds_data.get("data", [])

        for game in games:
            game_id = game.get("id")
            commence_time = game.get("commence_time")
            home_team = game.get("home_team")
            away_team = game.get("away_team")

            # Parse game date
            if commence_time:
                game_date = datetime.fromisoformat(commence_time.replace("Z", "+00:00")).date()
            else:
                continue

            # Get bookmakers
            bookmakers = game.get("bookmakers", [])

            for bookmaker in bookmakers:
                bookmaker_name = bookmaker.get("key")

                # Get markets
                markets = bookmaker.get("markets", [])

                for market in markets:
                    if market.get("key") != "player_points_rebounds_assists":
                        continue

                    # Get outcomes (each player's over/under)
                    outcomes = market.get("outcomes", [])

                    # Group by player
                    player_lines = {}
                    for outcome in outcomes:
                        player_name = outcome.get("description")
                        line_value = outcome.get("point")
                        outcome_type = outcome.get("name")  # 'Over' or 'Under'
                        odds = outcome.get("price")

                        if player_name not in player_lines:
                            player_lines[player_name] = {}

                        player_lines[player_name][outcome_type] = {"line": line_value, "odds": odds}

                    # Create rows for each player
                    for player_name, lines in player_lines.items():
                        # Get line (should be same for over/under)
                        over_line = lines.get("Over", {}).get("line")
                        under_line = lines.get("Under", {}).get("line")
                        betting_line = over_line or under_line

                        if betting_line is not None:
                            rows.append(
                                {
                                    "game_date": game_date,
                                    "game_id": game_id,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "bookmaker": bookmaker_name,
                                    "player_name": player_name,
                                    "betting_line": betting_line,
                                    "over_odds": lines.get("Over", {}).get("odds"),
                                    "under_odds": lines.get("Under", {}).get("odds"),
                                }
                            )

        if rows:
            return pd.DataFrame(rows)
        else:
            return pd.DataFrame()


def main():
    logger.info("=" * 70)
    logger.info("FETCHING REAL BETTING LINES FROM THE ODDS API")
    logger.info("=" * 70)

    # Initialize API
    logger.info("\n1. Initializing The Odds API...")
    try:
        api = TheOddsAPIFetcher()
        logger.info(f"   API key found: {api.api_key[:10]}...")
    except ValueError as e:
        logger.error(f"   ERROR: {str(e)}")
        return

    # Check quota
    logger.info("\n2. Checking API quota...")
    if api.check_quota():
        logger.info(f"   API accessible, {api.requests_remaining} requests remaining")
    else:
        logger.error("   Cannot access API")
        return

    # Load our predictions to know what dates we need
    logger.info("\n3. Loading predictions to determine required dates...")
    predictions_df = pd.read_csv("data/results/walk_forward_advanced_features_2024_25.csv")
    logger.info(f"   Predictions: {len(predictions_df):,}")

    unique_dates = sorted(predictions_df["GAME_DATE"].unique())
    logger.info(f"   Unique game dates: {len(unique_dates)}")
    logger.info(f"   Date range: {unique_dates[0]} to {unique_dates[-1]}")

    # Estimate API calls needed
    logger.info(f"\n   WARNING: Will need ~{len(unique_dates)} API calls")
    logger.info(f"   Current remaining: {api.requests_remaining}")

    if api.requests_remaining and api.requests_remaining < len(unique_dates):
        logger.warning(
            f"   Insufficient quota! Need {len(unique_dates)}, have {api.requests_remaining}"
        )
        logger.info(f"   Consider: Fetching subset of dates or upgrading plan")

        # Ask to proceed with available quota
        max_dates = min(api.requests_remaining - 10, len(unique_dates))  # Leave 10 buffer
        logger.info(f"   Will fetch first {max_dates} dates only")
        unique_dates = unique_dates[:max_dates]

    # Fetch historical odds
    logger.info(f"\n4. Fetching historical odds for {len(unique_dates)} dates...")
    logger.info(f"   This may take a while (rate limiting applied)...")

    all_lines = []

    for i, date in enumerate(unique_dates, 1):
        logger.info(f"   [{i}/{len(unique_dates)}] Fetching {date}...")

        odds_data = api.fetch_historical_odds(date)

        if odds_data:
            lines_df = api.parse_player_props(odds_data)
            if not lines_df.empty:
                all_lines.append(lines_df)
                logger.info(f"      Found {len(lines_df)} player prop lines")

        # Rate limiting (be nice to the API)
        if i < len(unique_dates):
            time.sleep(1)  # 1 second between requests

    # Combine all lines
    if all_lines:
        logger.info("\n5. Combining betting lines...")
        all_lines_df = pd.concat(all_lines, ignore_index=True)
        logger.info(f"   Total lines fetched: {len(all_lines_df):,}")
        logger.info(f"   Unique players: {all_lines_df['player_name'].nunique()}")
        logger.info(f"   Unique bookmakers: {all_lines_df['bookmaker'].nunique()}")

        # Save raw data
        raw_output = "data/results/theoddsapi_raw_lines.csv"
        all_lines_df.to_csv(raw_output, index=False)
        logger.info(f"   Saved raw data: {raw_output}")

        # Match to predictions
        logger.info("\n6. Matching lines to predictions...")
        # Implementation continues...

    else:
        logger.error("\n   No betting lines fetched!")
        logger.info("\n   Possible reasons:")
        logger.info("   1. Historical data requires specific API plan")
        logger.info("   2. Dates are too far in past")
        logger.info("   3. Player props not available for these dates")
        logger.info("\n   Check: https://the-odds-api.com/historical-odds-data")

    logger.info("\n" + "=" * 70)
    logger.info("COMPLETE")
    logger.info("=" * 70)
    logger.info(f"\nAPI requests used: {api.requests_used}")
    logger.info(f"API requests remaining: {api.requests_remaining}")


if __name__ == "__main__":
    main()
