#!/usr/bin/env python3
"""
Daily Betting Recommendations - Production Script

Fetches upcoming NBA games, gets prop odds, makes predictions using
production ensemble models, and recommends bets with confidence levels.

Usage:
    uv run python scripts/production/daily_betting_recommendations.py
    uv run python scripts/production/daily_betting_recommendations.py --date 2025-10-22
    uv run python scripts/production/daily_betting_recommendations.py --strategy moderate

Requirements:
    - Production ensemble models (production_fold_1.pkl, production_fold_2.pkl, production_fold_3.pkl, production_meta.pkl)
    - API key for odds provider (set in environment or config)
    - Recent game logs for feature calculation

Output:
    - Console: Top betting recommendations with confidence levels
    - CSV: data/betting/recommendations_YYYY_MM_DD.csv
    - HTML: data/betting/recommendations_YYYY_MM_DD.html (optional)
"""

import argparse
import os
import pickle
import sys
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.utils.fast_feature_builder import FastFeatureBuilder

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration (loaded from .env file)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# NBA API for game schedule
NBA_API_URL = "https://stats.nba.com/stats/scoreboardv2"

# Strategy configurations
# Updated with optimal 3-7 point threshold based on backtest analysis
# Thresholds are in POINTS (prediction - line difference), not probability
# Max threshold prevents betting on predictions >7 pts from line (27.8% win rate)
STRATEGIES = {
    "conservative": {"kelly_fraction": 0.20, "min_edge": 4.0, "max_edge": 7.0},  # 4-7 pts edge
    "moderate": {
        "kelly_fraction": 0.25,
        "min_edge": 3.0,
        "max_edge": 7.0,
    },  # RECOMMENDED 3-7 pts edge - 64.9% WR, 21.85% ROI
    "aggressive": {"kelly_fraction": 0.30, "min_edge": 2.0, "max_edge": 7.0},  # 2-7 pts edge
    "maximum": {
        "kelly_fraction": 0.25,
        "min_edge": 0.0,
        "max_edge": 7.0,
    },  # 0-7 pts edge (avoid big misses)
}

# Preferred bookmaker (for single-book filtering)
PREFERRED_BOOKMAKER = "DraftKings"  # Change to "FanDuel" or "BetMGM" if desired

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================


def get_upcoming_games(target_date=None):
    """
    Fetch upcoming NBA games for a given date.

    Args:
        target_date (str): Date in YYYY-MM-DD format. Defaults to today.

    Returns:
        pd.DataFrame: Games with home/away teams and game IDs
    """
    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n📅 Fetching NBA games for {target_date}...")

    try:
        headers = {
            "User-Agent": "Mozilla/5.0",
            "Referer": "https://www.nba.com/",
        }

        params = {"GameDate": target_date.replace("-", ""), "LeagueID": "00"}

        response = requests.get(NBA_API_URL, headers=headers, params=params, timeout=10)
        response.raise_for_status()

        data = response.json()
        games = data["resultSets"][0]["rowSet"]

        if len(games) == 0:
            print(f"⚠️  No games scheduled for {target_date}")
            return pd.DataFrame()

        games_df = pd.DataFrame(
            games,
            columns=[
                "GAME_DATE_EST",
                "GAME_SEQUENCE",
                "GAME_ID",
                "GAME_STATUS_ID",
                "GAME_STATUS_TEXT",
                "GAMECODE",
                "HOME_TEAM_ID",
                "VISITOR_TEAM_ID",
                "SEASON",
                "LIVE_PERIOD",
                "LIVE_PC_TIME",
                "NATL_TV_BROADCASTER_ABBREVIATION",
                "HOME_TV_BROADCASTER_ABBREVIATION",
                "AWAY_TV_BROADCASTER_ABBREVIATION",
                "LIVE_PERIOD_TIME_BCAST",
                "ARENA_NAME",
                "WH_STATUS",
            ],
        )

        print(f"✅ Found {len(games_df)} games")
        return games_df

    except Exception as e:
        print(f"❌ Error fetching NBA games: {e}")
        return pd.DataFrame()


def get_prop_odds(target_date=None, api_key=None):
    """
    Fetch PRA prop odds from odds API.

    Args:
        target_date (str): Date in YYYY-MM-DD format
        api_key (str): API key for odds provider

    Returns:
        pd.DataFrame: Odds data with player, line, bookmaker, prices
    """
    if api_key is None:
        api_key = ODDS_API_KEY

    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n💰 Fetching prop odds for {target_date}...")

    try:
        # Step 1: Get events for target date
        events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"
        params = {"apiKey": api_key}

        response = requests.get(events_url, params=params, timeout=30)
        response.raise_for_status()

        all_events = response.json()

        # Filter for target date (including games that start late and go into next UTC day)
        from datetime import datetime, timedelta

        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        next_day = (target_date_obj + timedelta(days=1)).strftime("%Y-%m-%d")

        target_events = [
            e for e in all_events if e["commence_time"][:10] in [target_date, next_day]
        ]

        if len(target_events) == 0:
            print(f"⚠️  No games scheduled for {target_date}")
            return pd.DataFrame()

        print(f"✅ Found {len(target_events)} games on {target_date}")

        # Step 2: Fetch player props for each event
        odds_data = []
        for event in target_events:
            event_id = event["id"]
            event_date = event["commence_time"][:10]
            home_team = event["home_team"]
            away_team = event["away_team"]

            print(f"   Fetching props: {away_team} @ {home_team}...")

            # Use event-specific endpoint for player props
            event_url = (
                f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"
            )
            params = {
                "apiKey": api_key,
                "regions": "us",
                "markets": "player_points_rebounds_assists",
                "oddsFormat": "american",
            }

            response = requests.get(event_url, params=params, timeout=30)

            if response.status_code != 200:
                print(f"   ⚠️  No props available for this game")
                continue

            event_data = response.json()

            for bookmaker in event_data.get("bookmakers", []):
                bookmaker_name = bookmaker["title"]

                for market in bookmaker.get("markets", []):
                    if market["key"] == "player_points_rebounds_assists":
                        for outcome in market["outcomes"]:
                            player_name = outcome.get("description", outcome.get("name"))
                            line = outcome.get("point")
                            direction = outcome.get("name")  # Over or Under
                            price = outcome.get("price")

                            odds_data.append(
                                {
                                    "player_name": player_name,
                                    "event_id": event_id,
                                    "event_date": event_date,
                                    "home_team": home_team,
                                    "away_team": away_team,
                                    "bookmaker": bookmaker_name,
                                    "line": line,
                                    "direction": direction,
                                    "price": price,
                                }
                            )

        odds_df = pd.DataFrame(odds_data)

        # Filter to single preferred bookmaker to avoid duplicates
        odds_df = odds_df[odds_df["bookmaker"] == PREFERRED_BOOKMAKER]

        if len(odds_df) == 0:
            print(f"⚠️  No odds from {PREFERRED_BOOKMAKER}")
            print("   Trying alternate bookmakers...")

            # Fallback to other bookmakers if preferred not available
            odds_df = pd.DataFrame(odds_data)
            preferred_bookmakers = ["DraftKings", "FanDuel", "BetMGM"]
            odds_df = odds_df[odds_df["bookmaker"].isin(preferred_bookmakers)]

            if len(odds_df) == 0:
                print("⚠️  No odds from any preferred bookmakers")
                return pd.DataFrame()

            # Use the bookmaker with most lines
            best_book = odds_df.groupby("bookmaker").size().idxmax()
            odds_df = odds_df[odds_df["bookmaker"] == best_book]
            print(f"   Using {best_book} instead ({len(odds_df)} lines)")

        # Pivot to get over/under prices in same row
        odds_pivot = (
            odds_df.pivot_table(
                index=[
                    "player_name",
                    "event_id",
                    "event_date",
                    "home_team",
                    "away_team",
                    "bookmaker",
                    "line",
                ],
                columns="direction",
                values="price",
                aggfunc="first",
            )
            .reset_index()
            .rename(columns={"Over": "over_price", "Under": "under_price"})
        )

        bookmaker_used = odds_pivot["bookmaker"].iloc[0] if len(odds_pivot) > 0 else "Unknown"
        print(f"✅ Found {len(odds_pivot)} prop lines from {bookmaker_used}")
        return odds_pivot

    except Exception as e:
        print(f"❌ Error fetching odds: {e}")
        print("   Using local historical odds data as fallback...")

        # Fallback: Load latest historical odds
        try:
            odds_df = pd.read_csv("data/historical_odds/2024-25/pra_odds.csv")
            odds_df["event_date"] = pd.to_datetime(odds_df["event_date"])
            latest_date = odds_df["event_date"].max()
            odds_df = odds_df[odds_df["event_date"] == latest_date]
            print(f"   ✅ Loaded {len(odds_df)} lines from {latest_date.date()}")
            return odds_df
        except Exception as fallback_error:
            print(f"   ❌ Fallback also failed: {fallback_error}")
            return pd.DataFrame()


def normalize_name(name):
    """Normalize player name for fuzzy matching."""
    import re
    import unicodedata

    # Convert special characters to ASCII equivalents (č → c, ñ → n, etc.)
    name = unicodedata.normalize("NFKD", name).encode("ascii", "ignore").decode("ascii")

    # Remove Jr., Jr, Sr., III, etc.
    name = re.sub(r"\s+(Jr\.?|Sr\.?|III|II|IV)$", "", name, flags=re.IGNORECASE)

    # Remove extra whitespace
    name = " ".join(name.split())

    return name.strip()


def fuzzy_match_player(api_name, db_names):
    """
    Match API player name to database name using fuzzy matching.

    Args:
        api_name: Player name from odds API
        db_names: List of player names from database

    Returns:
        Matched database name or None
    """
    # Normalize API name
    normalized_api = normalize_name(api_name)

    # First try exact match on normalized names
    normalized_db = {normalize_name(name): name for name in db_names}

    if normalized_api in normalized_db:
        return normalized_db[normalized_api]

    # If no exact match, try fuzzy matching
    from difflib import get_close_matches

    matches = get_close_matches(normalized_api, normalized_db.keys(), n=1, cutoff=0.85)

    if matches:
        return normalized_db[matches[0]]

    return None


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def make_predictions(players_today, historical_df, models, feature_cols):
    """
    Make PRA predictions for today's players using ensemble models.

    Args:
        players_today (list): Player names playing today
        historical_df (pd.DataFrame): Historical game logs
        models (list): List of trained XGBoost models (3-fold ensemble)
        feature_cols (list): Feature column names

    Returns:
        pd.DataFrame: Predictions with player names and predicted PRA
    """
    print(f"\n🤖 Generating predictions for {len(players_today)} players...")

    # Get unique player names from database
    db_player_names = historical_df["PLAYER_NAME"].unique()

    # Match players using fuzzy matching
    player_mapping = {}
    matched_players = []

    for api_player in players_today:
        matched_db_name = fuzzy_match_player(api_player, db_player_names)

        if matched_db_name:
            player_mapping[api_player] = matched_db_name
            matched_players.append(matched_db_name)
            if api_player != matched_db_name:
                print(f"   ✓ Matched '{api_player}' → '{matched_db_name}'")
        else:
            print(f"   ⚠️  No history for {api_player} - skipping")

    # Create dummy "today" games for feature calculation
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_games = []

    for player_name in matched_players:
        # Get player's recent games
        player_hist = historical_df[historical_df["PLAYER_NAME"] == player_name]

        if len(player_hist) == 0:
            continue

        # Create dummy game entry with required columns
        player_id = player_hist["PLAYER_ID"].iloc[0]

        # Extract team from MATCHUP column (format: "BOS vs. LAL" or "BOS @ LAL")
        last_matchup = player_hist["MATCHUP"].iloc[-1] if "MATCHUP" in player_hist.columns else ""
        last_team = last_matchup.split()[0] if last_matchup else "UNK"

        # Create minimal game row with all required columns from historical data
        last_game = player_hist.iloc[-1].to_dict()

        # Determine current season (format: "2024-25")
        today_dt = datetime.strptime(today_date, "%Y-%m-%d")
        if today_dt.month >= 10:  # Oct-Dec
            season = f"{today_dt.year}-{str(today_dt.year + 1)[-2:]}"
        else:  # Jan-Sep
            season = f"{today_dt.year - 1}-{str(today_dt.year)[-2:]}"

        today_game = {
            "PLAYER_ID": player_id,
            "PLAYER_NAME": player_name,
            "GAME_DATE": today_date,
            "SEASON": season,
            "PRA": np.nan,  # To be predicted
        }

        # Copy over necessary columns from last game (for feature calculation)
        for col in [
            "MIN",
            "FGA",
            "FG_PCT",
            "FG3A",
            "FG3_PCT",
            "FTA",
            "FT_PCT",
            "OREB",
            "DREB",
            "STL",
            "BLK",
            "TOV",
            "PF",
        ]:
            if col in last_game:
                today_game[col] = 0  # Dummy values (will be replaced by lag features)

        today_games.append(today_game)

    if len(today_games) == 0:
        print("   ❌ No valid players to predict")
        return pd.DataFrame()

    today_df = pd.DataFrame(today_games)
    today_df["GAME_DATE"] = pd.to_datetime(today_df["GAME_DATE"])

    # Build features
    print("   Building features...")
    builder = FastFeatureBuilder()
    full_df = builder.build_features(historical_df, today_df, verbose=False)

    # Get today's games with features
    predictions_df = full_df[full_df["GAME_DATE"] == today_date].copy()

    # Make ensemble predictions (average predictions from all 3 models)
    X = predictions_df[feature_cols].fillna(0)

    raw_predictions = []
    for model in models:
        pred = model.predict(X)
        raw_predictions.append(pred)

    # Average the predictions
    predictions = np.mean(raw_predictions, axis=0)
    predictions = np.maximum(0, predictions)  # Clip to non-negative

    predictions_df["predicted_PRA"] = predictions

    # Convert database names back to API names using reverse mapping
    reverse_mapping = {v: k for k, v in player_mapping.items()}
    predictions_df["PLAYER_NAME"] = predictions_df["PLAYER_NAME"].map(
        lambda x: reverse_mapping.get(x, x)
    )

    print(f"   ✅ Generated {len(predictions_df)} predictions (ensemble of {len(models)} models)")
    return predictions_df[["PLAYER_NAME", "predicted_PRA"]]


def calculate_edges(predictions_df, odds_df, calibrators):
    """
    Calculate calibrated edges for each bet opportunity using ensemble calibrators.

    Args:
        predictions_df (pd.DataFrame): Predictions with player names and predicted PRA
        odds_df (pd.DataFrame): Odds data with lines and prices
        calibrators (list): List of isotonic regression calibrators

    Returns:
        pd.DataFrame: Betting opportunities with edges
    """
    print(f"\n📊 Calculating edges...")

    # Merge predictions with odds
    merged_df = odds_df.merge(
        predictions_df, left_on="player_name", right_on="PLAYER_NAME", how="inner"
    )

    if len(merged_df) == 0:
        print("   ❌ No matches between predictions and odds")
        return pd.DataFrame()

    print(f"   ✅ Matched {len(merged_df)} predictions with odds")

    # Calculate difference (prediction - line) - this is our "edge" in points
    merged_df["point_edge"] = merged_df["predicted_PRA"] - merged_df["line"]

    # Use sigmoid function to convert difference to probability
    # Scale based on model MAE (~5.86 points)
    scale = 5.86  # Use CV MAE as scale parameter
    merged_df["prob_over_calibrated"] = 1 / (1 + np.exp(-merged_df["point_edge"] / scale))
    print(f"   ✅ Calculated probabilities using sigmoid (scale={scale:.2f})")

    # Calculate edges for both over and under
    opportunities = []

    for idx, row in merged_df.iterrows():
        # Over bet
        over_decimal = american_to_decimal(row["over_price"])
        over_implied_prob = 1 / over_decimal
        over_prob_edge = row["prob_over_calibrated"] - over_implied_prob

        opportunities.append(
            {
                "player_name": row["player_name"],
                "predicted_PRA": row["predicted_PRA"],
                "line": row["line"],
                "bookmaker": row["bookmaker"],
                "direction": "OVER",
                "american_odds": row["over_price"],
                "decimal_odds": over_decimal,
                "implied_prob": over_implied_prob,
                "calibrated_prob": row["prob_over_calibrated"],
                "prob_edge": over_prob_edge,
                "point_edge": row["point_edge"],  # Edge in points (prediction - line)
                "away_team": row["away_team"],
                "home_team": row["home_team"],
            }
        )

        # Under bet
        under_decimal = american_to_decimal(row["under_price"])
        under_implied_prob = 1 / under_decimal
        under_prob_edge = (1 - row["prob_over_calibrated"]) - under_implied_prob

        opportunities.append(
            {
                "player_name": row["player_name"],
                "predicted_PRA": row["predicted_PRA"],
                "line": row["line"],
                "bookmaker": row["bookmaker"],
                "direction": "UNDER",
                "american_odds": row["under_price"],
                "decimal_odds": under_decimal,
                "implied_prob": under_implied_prob,
                "calibrated_prob": 1 - row["prob_over_calibrated"],
                "prob_edge": under_prob_edge,
                "point_edge": -row["point_edge"],  # Negative for under bets
                "away_team": row["away_team"],
                "home_team": row["home_team"],
            }
        )

    opportunities_df = pd.DataFrame(opportunities)
    print(f"   ✅ Found {len(opportunities_df)} betting opportunities")

    return opportunities_df


def calculate_confidence_level(edge, calibrated_prob):
    """
    Calculate confidence level for a bet.

    Confidence based on:
    1. Edge size (larger edge = higher confidence)
    2. Probability magnitude (avoid 50/50 bets)
    3. Combined score

    Returns:
        str: Confidence level (VERY HIGH, HIGH, MEDIUM, LOW)
    """
    # Edge contribution (0-50 points)
    edge_score = min(edge * 500, 50)  # 10% edge = 50 points

    # Probability magnitude contribution (0-50 points)
    # Penalize probabilities near 50% (low confidence)
    prob_distance_from_50 = abs(calibrated_prob - 0.5)
    prob_score = prob_distance_from_50 * 100  # Max 50 points when prob = 0 or 1

    # Total score (0-100)
    total_score = edge_score + prob_score

    # Thresholds
    if total_score >= 75:
        return "VERY HIGH"
    elif total_score >= 60:
        return "HIGH"
    elif total_score >= 40:
        return "MEDIUM"
    else:
        return "LOW"


def calculate_bet_size(edge, bankroll, kelly_fraction, max_bet_fraction):
    """
    Calculate bet size using fractional Kelly criterion.

    Args:
        edge (float): Edge on this bet
        bankroll (float): Current bankroll
        kelly_fraction (float): Fraction of Kelly to use (e.g., 0.25)
        max_bet_fraction (float): Max bet as fraction of bankroll (e.g., 0.10)

    Returns:
        float: Bet size in dollars
    """
    kelly_bet = edge * kelly_fraction * bankroll
    max_bet = max_bet_fraction * bankroll
    bet_size = min(kelly_bet, max_bet)
    return max(bet_size, 0)


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate daily betting recommendations using calibrated model"
    )
    parser.add_argument(
        "--date", type=str, default=None, help="Target date (YYYY-MM-DD). Defaults to today."
    )
    parser.add_argument(
        "--strategy",
        type=str,
        default="moderate",
        choices=["conservative", "moderate", "aggressive", "maximum"],
        help="Betting strategy (default: moderate)",
    )
    parser.add_argument(
        "--bankroll", type=float, default=1000.0, help="Current bankroll (default: $1,000)"
    )
    parser.add_argument("--api-key", type=str, default=None, help="Odds API key (overrides config)")
    parser.add_argument(
        "--save-html", action="store_true", help="Save HTML report in addition to CSV"
    )

    args = parser.parse_args()

    target_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")
    strategy_name = args.strategy
    bankroll = args.bankroll

    print("=" * 80)
    print("DAILY BETTING RECOMMENDATIONS")
    print("=" * 80)
    print(f"\n📅 Date: {target_date}")
    print(f"💼 Strategy: {strategy_name.upper()}")
    print(f"💰 Bankroll: ${bankroll:,.2f}")

    # Load strategy config
    strategy = STRATEGIES[strategy_name]
    kelly_fraction = strategy["kelly_fraction"]
    min_edge = strategy["min_edge"]
    max_edge = strategy["max_edge"]
    max_bet_fraction = 0.10 if strategy_name in ["aggressive", "maximum"] else 0.05

    print(f"   Kelly fraction: {kelly_fraction:.0%}")
    print(f"   Edge range: {min_edge:.1f} to {max_edge:.1f} pts (|prediction - line|)")
    print(f"   Max bet: {max_bet_fraction:.0%} of bankroll")

    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: LOADING PRODUCTION ENSEMBLE MODEL")
    print("=" * 80)

    try:
        # Load production ensemble models (3-fold CV)
        models_dir = Path("models")
        models = []
        calibrators = []
        feature_cols = None

        # Load metadata
        meta_path = models_dir / "production_meta.pkl"
        with open(meta_path, "rb") as f:
            meta = pickle.load(f)

        print(f"✅ Loaded metadata")

        # Load each fold model
        for i in range(1, 4):
            model_path = models_dir / f"production_fold_{i}.pkl"

            with open(model_path, "rb") as f:
                fold_dict = pickle.load(f)

            models.append(fold_dict["model"])
            calibrators.append(fold_dict["calibrator"])

            if feature_cols is None:
                feature_cols = fold_dict["feature_cols"]

            print(f"   ✅ Loaded Fold {i}")

        print(f"\n✅ Loaded production ensemble: {len(models)} models")
        print(f"   Features: {len(feature_cols)}")
        cv_mae = np.mean([r["mae_calibrated"] for r in meta["fold_results"]])
        print(f"   CV MAE: {cv_mae:.2f} points")
        print(f"   Test MAE: {meta.get('test_mae', 0):.2f} points")
        print(f"   Calibrators: ✅ Available")

    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return 1

    # ========================================================================
    # 2. LOAD HISTORICAL DATA
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 2: LOADING HISTORICAL GAME LOGS")
    print("=" * 80)

    try:
        # Load historical data (through 2023-24)
        df_historical = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
        df_historical["GAME_DATE"] = pd.to_datetime(df_historical["GAME_DATE"])

        # Load 2024-25 season data
        df_2024_25 = pd.read_csv("data/game_logs/game_logs_2024_25_preprocessed.csv")
        df_2024_25["GAME_DATE"] = pd.to_datetime(df_2024_25["GAME_DATE"])

        # Combine both datasets
        historical_df = pd.concat([df_historical, df_2024_25], ignore_index=True)
        historical_df = historical_df.sort_values(["PLAYER_ID", "GAME_DATE"])

        # Add PRA if missing
        if "PRA" not in historical_df.columns:
            historical_df["PRA"] = (
                historical_df["PTS"] + historical_df["REB"] + historical_df["AST"]
            )

        # Only keep data BEFORE target date (no leakage)
        cutoff_date = pd.to_datetime(target_date)
        historical_df = historical_df[historical_df["GAME_DATE"] < cutoff_date]

        print(f"✅ Loaded {len(historical_df):,} historical games")
        print(
            f"   Date range: {historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}"
        )
        print(f"   Players: {historical_df['PLAYER_ID'].nunique():,}")

    except Exception as e:
        print(f"❌ Error loading historical data: {e}")
        return 1

    # ========================================================================
    # 3. FETCH ODDS
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 3: FETCHING PROP ODDS")
    print("=" * 80)

    api_key = args.api_key if args.api_key else ODDS_API_KEY
    odds_df = get_prop_odds(target_date, api_key)

    if len(odds_df) == 0:
        print("❌ No odds available - cannot generate recommendations")
        return 1

    # Get unique players from odds
    players_today = odds_df["player_name"].unique().tolist()
    print(f"\n   Players with odds: {len(players_today)}")

    # ========================================================================
    # 4. GENERATE PREDICTIONS
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 4: GENERATING PREDICTIONS")
    print("=" * 80)

    predictions_df = make_predictions(players_today, historical_df, models, feature_cols)

    if len(predictions_df) == 0:
        print("❌ No predictions generated")
        return 1

    # ========================================================================
    # 5. CALCULATE EDGES
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 5: CALCULATING EDGES")
    print("=" * 80)

    opportunities_df = calculate_edges(predictions_df, odds_df, calibrators)

    if len(opportunities_df) == 0:
        print("❌ No betting opportunities found")
        return 1

    # ========================================================================
    # 6. FILTER AND RANK
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 6: FILTERING AND RANKING BETS")
    print("=" * 80)

    # Filter to bets within edge range (using absolute value of point_edge)
    opportunities_df["abs_point_edge"] = opportunities_df["point_edge"].abs()

    # Apply both min and max thresholds
    bets_df = opportunities_df[
        (opportunities_df["abs_point_edge"] >= min_edge)
        & (opportunities_df["abs_point_edge"] <= max_edge)
    ].copy()

    if len(bets_df) == 0:
        print(f"❌ No bets meet {min_edge:.1f}-{max_edge:.1f} point edge range")
        print(f"   Best point edge available: {opportunities_df['abs_point_edge'].max():.1f} pts")
        return 1

    # For each player/line, only keep the side with positive point_edge (matching direction)
    # Filter to only bets where prediction actually supports the direction
    # OVER bets: prediction > line (point_edge > 0)
    # UNDER bets: prediction < line (point_edge > 0, since we negated it)
    bets_df = bets_df[bets_df["point_edge"] > 0].copy()

    # Group by player and line, keep only the bet with highest point_edge
    bets_df = bets_df.loc[bets_df.groupby(["player_name", "line"])["point_edge"].idxmax()]

    print(
        f"✅ {len(bets_df)} bets meet {min_edge:.1f}-{max_edge:.1f} pt range (duplicates removed)"
    )
    print(f"   All bets from single bookmaker: {bets_df['bookmaker'].iloc[0]}")

    # Sort by absolute point edge (best bets first)
    bets_df = bets_df.sort_values("abs_point_edge", ascending=False)

    # Calculate confidence levels
    bets_df["confidence"] = bets_df.apply(
        lambda row: calculate_confidence_level(row["prob_edge"], row["calibrated_prob"]), axis=1
    )

    # Calculate bet sizes using probability edge
    bets_df["bet_size"] = bets_df["prob_edge"].apply(
        lambda edge: calculate_bet_size(edge, bankroll, kelly_fraction, max_bet_fraction)
    )

    # Re-sort by absolute point edge (best bets first) after adding new columns
    bets_df = bets_df.sort_values("abs_point_edge", ascending=False)

    # ========================================================================
    # 7. DISPLAY RECOMMENDATIONS
    # ========================================================================

    print("\n" + "=" * 80)
    print("📋 BETTING RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n🎯 TOP 10 BETS:")
    print("-" * 80)

    for i, (idx, row) in enumerate(bets_df.head(10).iterrows(), 1):
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(f"\n{i}. {row['player_name']} - {row['direction']} {row['line']}")
        print(f"   Game: {matchup}")
        print(f"   Bookmaker: {row['bookmaker']}")
        print(f"   Prediction: {row['predicted_PRA']:.1f} PRA")
        print(f"   Point Edge: {row['point_edge']:+.1f} pts | Prob Edge: {row['prob_edge']:.1%}")
        print(f"   Confidence: {row['confidence']}")
        print(f"   Odds: {row['american_odds']:+.0f} (Decimal: {row['decimal_odds']:.2f})")
        print(f"   Win Probability: {row['calibrated_prob']:.1%}")
        print(f"   💵 Recommended Bet: ${row['bet_size']:.2f}")

    # ========================================================================
    # 7b. SUGGESTED ACTION SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("💡 SUGGESTED ACTION")
    print("=" * 80)

    # Determine recommended subset based on edge thresholds
    top5_df = bets_df.head(5)
    top5_total = top5_df["bet_size"].sum()
    top5_avg_point_edge = top5_df["abs_point_edge"].mean()
    top5_avg_prob_edge = top5_df["prob_edge"].mean()

    all_total = bets_df["bet_size"].sum()
    all_avg_point_edge = bets_df["abs_point_edge"].mean()
    all_avg_prob_edge = bets_df["prob_edge"].mean()

    print(f"\n🎯 RECOMMENDED: Take Top 5 Bets (Highest Point Edges)")
    print(f"   Total Wager: ${top5_total:.2f}")
    print(f"   Avg Point Edge: {top5_avg_point_edge:.1f} pts")
    print(f"   Avg Prob Edge: {top5_avg_prob_edge:.1%}")
    print(f"   Risk Level: Lower (concentrated on best opportunities)")

    print(f"\n📊 ALTERNATIVE: Take All {len(bets_df)} Bets (Diversified)")
    print(f"   Total Wager: ${all_total:.2f}")
    print(f"   Avg Point Edge: {all_avg_point_edge:.1f} pts")
    print(f"   Avg Prob Edge: {all_avg_prob_edge:.1%}")
    print(f"   Risk Level: Higher (more exposure, more variance)")

    print(f"\n⚠️  IMPORTANT:")
    print(f"   • Check injury reports before placing bets")
    print(f"   • Verify player starting lineups 1-2 hours before games")
    print(f"   • Skip any bet if player status is uncertain")

    # ========================================================================
    # 8. SAVE RESULTS
    # ========================================================================

    print("\n" + "=" * 80)
    print("💾 SAVING RESULTS")
    print("=" * 80)

    # Save CSV
    output_dir = Path("data/betting")
    output_dir.mkdir(parents=True, exist_ok=True)

    csv_path = output_dir / f"recommendations_{target_date}.csv"
    bets_df.to_csv(csv_path, index=False)
    print(f"✅ Saved CSV: {csv_path}")

    # Save HTML if requested
    if args.save_html:
        html_path = output_dir / f"recommendations_{target_date}.html"

        html = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Betting Recommendations - {target_date}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        h1 {{ color: #2c3e50; }}
        table {{ border-collapse: collapse; width: 100%; }}
        th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
        th {{ background-color: #3498db; color: white; }}
        .very-high {{ background-color: #27ae60; color: white; }}
        .high {{ background-color: #2ecc71; }}
        .medium {{ background-color: #f39c12; }}
        .low {{ background-color: #e74c3c; color: white; }}
    </style>
</head>
<body>
    <h1>🎯 Betting Recommendations for {target_date}</h1>
    <p><strong>Strategy:</strong> {strategy_name.upper()}</p>
    <p><strong>Bankroll:</strong> ${bankroll:,.2f}</p>
    <p><strong>Total Bets:</strong> {len(bets_df)}</p>

    <h2>Top Recommendations</h2>
    {bets_df.to_html(classes='table', index=False)}
</body>
</html>
        """

        with open(html_path, "w") as f:
            f.write(html)

        print(f"✅ Saved HTML: {html_path}")

    # ========================================================================
    # SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("✅ RECOMMENDATIONS COMPLETE")
    print("=" * 80)

    print(f"\nSummary:")
    print(f"  📊 Total opportunities: {len(opportunities_df)}")
    print(f"  ✅ Bets meeting {min_edge:.1f}-{max_edge:.1f} pt threshold: {len(bets_df)}")
    print(f"  💰 Total to wager: ${bets_df['bet_size'].sum():,.2f}")
    print(f"  📈 Avg point edge: {bets_df['abs_point_edge'].mean():.1f} pts")
    print(f"  📈 Avg prob edge: {bets_df['prob_edge'].mean():.1%}")
    print(f"  🎯 Confidence distribution:")

    for conf in ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]:
        count = (bets_df["confidence"] == conf).sum()
        if count > 0:
            print(f"     {conf}: {count} bets")

    print(f"\n📂 Results saved to: {csv_path}")

    # ========================================================================
    # AUTO-TRACK BETS
    # ========================================================================
    print("\n" + "=" * 80)
    print("📋 AUTO-TRACKING BETS")
    print("=" * 80)

    try:
        # Import the tracking function
        from scripts.betting.track_all_recommendations import track_all_recommendations

        track_all_recommendations(target_date)
    except Exception as e:
        print(f"⚠️  Auto-tracking failed: {e}")
        print(f"   You can manually track later with:")
        print(f"   uv run python scripts/betting/track_all_recommendations.py {target_date}")

    print("\n🎰 Good luck!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
