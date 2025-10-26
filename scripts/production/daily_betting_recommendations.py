#!/usr/bin/env python3
"""
Daily Betting Recommendations - Production Script

Fetches upcoming NBA games, gets prop odds, makes predictions using
production ensemble models, and recommends bets with confidence levels.

Usage:
    uv run python scripts/production/daily_betting_recommendations.py
    uv run python scripts/production/daily_betting_recommendations.py --date 2025 - 10 - 22  # noqa: E501
    uv run python scripts/production/daily_betting_recommendations.py --strategy moderate  # noqa: E501

IMPORTANT - Timezone Behavior:
    All dates are interpreted as US EASTERN TIME (NBA game schedule timezone).
    Example: "--date 2025 - 10 - 23" fetches games scheduled for Oct 23 US ET,
    regardless of your local timezone (Sydney, London, etc.).

    This ensures you get the correct games for that NBA "game day".

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
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Dict
from zoneinfo import ZoneInfo

import numpy as np
import pandas as pd
import requests
from dotenv import load_dotenv
from scipy.stats import norm

from scripts.utils.fast_feature_builder import FastFeatureBuilder
from utils.clv_tracker import CLVTracker
from utils.edge_calculator import calculate_kelly_fraction, remove_vig
from utils.player_variance import get_player_variance_calculator

# Load environment variables
load_dotenv()

sys.path.append(str(Path(__file__).parent.parent.parent))

# ============================================================================
# CONFIGURATION
# ============================================================================

# API Configuration (loaded from .env file)
ODDS_API_KEY = os.getenv("ODDS_API_KEY")
ODDS_API_URL = "https://api.the-odds-api.com/v4/sports/basketball_nba/odds"

# NBA API for game schedule
NBA_API_URL = "https://stats.nba.com/stats/scoreboardv2"

# Timezone configuration
# All target dates are interpreted as US Eastern Time (NBA game schedule timezone)  # noqa: E501
# This ensures consistent game fetching regardless of user's local timezone
US_EASTERN = ZoneInfo("America/New_York")  # Handles EST/EDT automatically

# Strategy configurations
# Updated with optimal 3 - 7 point threshold based on backtest analysis
# Strategy configurations - EV-ONLY filtering
# Point-edge thresholds REMOVED to avoid overfitting to backtest heuristics
# EV already accounts for: σ (player variance), odds, vig, loss risk
STRATEGIES = {
    "conservative": {"kelly_fraction": 0.20, "min_ev": 0.03},  # 3% minimum EV
    "moderate": {
        "kelly_fraction": 0.25,
        "min_ev": 0.02,
    },  # RECOMMENDED: 2% minimum EV
    "aggressive": {"kelly_fraction": 0.30, "min_ev": 0.01},  # 1% minimum EV
    "maximum": {
        "kelly_fraction": 0.25,
        "min_ev": 0.005,
    },  # 0.5% minimum EV (test only)
}

# Preferred bookmaker (for single-book filtering)
# Note: Bet365 may not be available in The Odds API US region
# If Bet365 not available, script will fallback to DraftKings/FanDuel
# Options: "DraftKings", "FanDuel", "BetMGM", "Bet365"
PREFERRED_BOOKMAKER = "DraftKings"

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
        target_date (str): Date in YYYY-MM-DD format (interpreted as US Eastern Time - NBA game day)
        api_key (str): API key for odds provider

    Returns:
        pd.DataFrame: Odds data with player, line, bookmaker, prices

    Note:
        Target date is interpreted as US Eastern Time to match NBA game schedules.  # noqa: E501
        Example: "2025 - 10 - 23" fetches games scheduled for Oct 23 ET,
        regardless of user's local timezone.
    """
    if api_key is None:
        api_key = ODDS_API_KEY

    if target_date is None:
        target_date = datetime.now().strftime("%Y-%m-%d")

    print(f"\n💰 Fetching prop odds for {target_date}...")

    try:
        # Step 1: Get events for target date
        events_url = "https://api.the-odds-api.com/v4/sports/basketball_nba/events"  # noqa: E501
        params = {"apiKey": api_key}

        response = requests.get(events_url, params=params, timeout=30)
        response.raise_for_status()

        all_events = response.json()

        # Filter for target date using US Eastern Time (NBA game schedule timezone)  # noqa: E501
        # This ensures we get games scheduled for the target date in ET,
        # regardless of user's local timezone

        # Parse target date as US ET midnight
        target_date_obj = datetime.strptime(target_date, "%Y-%m-%d")
        et_start = target_date_obj.replace(hour=0, minute=0, second=0, tzinfo=US_EASTERN)
        et_end = et_start + timedelta(days=1)

        # Convert ET boundaries to UTC for API filtering
        utc_start = et_start.astimezone(timezone.utc)
        utc_end = et_end.astimezone(timezone.utc)

        # Filter events using timezone-aware datetime comparison
        target_events = []
        for event in all_events:
            try:
                # Parse UTC timestamp from API (format: "2025 - 10 -
                # 23T02:30:00Z")
                commence_dt = datetime.fromisoformat(event["commence_time"].replace("Z", "+00:00"))

                # Check if game is within target ET date
                if utc_start <= commence_dt < utc_end:
                    target_events.append(event)
            except (KeyError, ValueError):  # noqa: F841
                # Skip malformed events
                continue

        if len(target_events) == 0:
            print(f"⚠️  No games scheduled for {target_date} (US Eastern Time)")
            return pd.DataFrame()

        print(f"✅ Found {len(target_events)} games on {target_date} (US ET)")

        # Step 2: Fetch player props for each event
        odds_data = []
        for event in target_events:
            event_id = event["id"]
            event_date = event["commence_time"][:10]
            home_team = event["home_team"]
            away_team = event["away_team"]

            print(f"   Fetching props: {away_team} @ {home_team}...")

            # Use event-specific endpoint for player props
            event_url = f"https://api.the-odds-api.com/v4/sports/basketball_nba/events/{event_id}/odds"  # noqa: E501
            params = {
                "apiKey": api_key,
                "regions": "us",
                "markets": "player_points_rebounds_assists",
                "oddsFormat": "american",
            }

            response = requests.get(event_url, params=params, timeout=30)

            if response.status_code != 200:
                print("   ⚠️  No props available for this game")
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
            odds_df = pd.read_csv("data/historical_odds/2024 - 25/pra_odds.csv")
            odds_df["event_date"] = pd.to_datetime(odds_df["event_date"])
            latest_date = odds_df["event_date"].max()
            odds_df = odds_df[odds_df["event_date"] == latest_date]
            print(
                f"   ✅ Loaded {
                    len(odds_df)} lines from {
                    latest_date.date()}"
            )
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

        # Extract team from MATCHUP column (format: "BOS vs. LAL" or "BOS @
        # LAL")
        last_matchup = player_hist["MATCHUP"].iloc[-1] if "MATCHUP" in player_hist.columns else ""
        __last_team = last_matchup.split()[0] if last_matchup else "UNK"  # noqa: F841

        # Create minimal game row with all required columns from historical
        # data
        last_game = player_hist.iloc[-1].to_dict()

        # Determine current season (format: "2024 - 25")
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
                # Dummy values (will be replaced by lag features)
                today_game[col] = 0

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

    print(
        f"   ✅ Generated {
            len(predictions_df)} predictions (ensemble of {
            len(models)} models)"
    )
    return predictions_df[["PLAYER_NAME", "predicted_PRA"]]


def load_isotonic_calibrator(calibrator_path="models/isotonic_calibrator.pkl"):
    """Load isotonic calibrator if available"""
    calibrator_path = Path(calibrator_path)

    if calibrator_path.exists():
        with open(calibrator_path, "rb") as f:
            calibrator = pickle.load(f)
        print(f"   ✅ Loaded isotonic calibrator from {calibrator_path}")
        return calibrator
    else:
        print(f"   ⚠️  Isotonic calibrator not found at {calibrator_path}")
        print(
            "      Using raw predictions (run scripts/calibration/apply_isotonic_calibration.py)"
        )  # noqa: E501
        return None


def load_beta_calibrator(calibrator_path="models/beta_calibrator.pkl"):
    """
    Load side-specific beta calibrator (TIER 2 FIX)

    Returns:
        tuple: (calibrators, metadata) where calibrators is dict with 'OVER' and 'UNDER' keys  # noqa: E501
               Returns (None, None) if calibrator not found
    """
    calibrator_path = Path(calibrator_path)

    if calibrator_path.exists():
        with open(calibrator_path, "rb") as f:
            cal_pkg = pickle.load(f)

        calibrators = cal_pkg["calibrators"]
        metadata = cal_pkg["metadata"]

        # Calculate shrinkage weights for display (SIDE-SPECIFIC)
        # OVER: K=50, UNDER: K=100 (Expert Recommendation #2)
        K_over = 50
        K_under = 100
        n_over = metadata.get("n_over", 0)
        n_under = metadata.get("n_under", 0)
        w_over = n_over / (n_over + K_over) if n_over > 0 else 0.0
        w_under = n_under / (n_under + K_under) if n_under > 0 else 0.0

        print(f"   ✅ Loaded beta calibrator from {calibrator_path}")
        print(
            f"      ECE: {
                metadata['ece_train_cal']:.4f} (OVER: {
                metadata['calibrator_type_over']}, UNDER: {
                metadata['calibrator_type_under']})"
        )
        print("      BLENDED CALIBRATION ACTIVE (Side-Specific):")
        print(
            f"         OVER:  n={
                n_over:2d}, K={
                K_over:3d}, weight={
                w_over:.2f} ({
                    int(
                        (1 -
                         w_over) *
                        100)}% raw, {
                            int(
                                w_over *
                                100)}% beta, cap=±10pp)"
        )
        print(
            f"         UNDER: n={
                n_under:2d}, K={
                K_under:3d}, weight={
                w_under:.2f} ({
                    int(
                        (1 -
                         w_under) *
                        100)}% raw, {
                            int(
                                w_under *
                                100)}% beta, cap=±5pp)"
        )
        return calibrators, metadata
    else:
        print(f"   ⚠️  Beta calibrator not found at {calibrator_path}")
        print("      Falling back to isotonic calibration")
        return None, None


def apply_beta_calibration(prob_raw, side, beta_calibrators, beta_metadata=None):
    """
    Apply side-specific beta calibration with shrinkage to raw probability

    BLENDED CALIBRATION (Option B):
    - Uses sample-size-based shrinkage to prevent overfitting on small samples
    - Shrinks toward raw probability when n_side is small (< 50)
    - Caps calibration adjustments at ±0.10 to prevent extreme jumps

    Formula:
        w_side = n_side / (n_side + K) where K=50
        p_blend = w_side * p_calibrated + (1 - w_side) * p_raw
        p_final = clip(p_blend, p_raw ± 0.10)

    Args:
        prob_raw: Raw probability from normal CDF
        side: 'OVER' or 'UNDER'
        beta_calibrators: Dict with calibrators for each side
        beta_metadata: Dict with calibrator metadata (includes sample sizes)

    Returns:
        float: Blended calibrated probability
    """
    if beta_calibrators is None or side not in beta_calibrators:
        return prob_raw  # Fallback to uncalibrated

    cal_type, cal_model = beta_calibrators[side]

    # Get calibrated probability from beta/platt model
    if cal_type == "beta":
        # Beta calibration
        prob_cal = cal_model.predict([prob_raw])[0]
    elif cal_type == "platt":
        # Platt scaling (logistic regression)
        prob_clipped = np.clip(prob_raw, 1e-6, 1 - 1e-6)
        logit = np.log(prob_clipped / (1 - prob_clipped))
        prob_cal = cal_model.predict_proba([[logit]])[0, 1]
    else:
        prob_cal = prob_raw  # Unknown calibrator type

    # BLENDED CALIBRATION: Apply shrinkage based on sample size
    if beta_metadata is not None:
        # Get sample size for this side
        n_side_key = f"n_{side.lower()}"
        n_side = beta_metadata.get(n_side_key, 0)

        # SIDE-SPECIFIC SHRINKAGE PARAMETERS (Expert Recommendation #2)
        # UNDER: More conservative (K=100, ±5pp cap) due to OT tail risk + 88% bias  # noqa: E501
        # OVER: Standard (K=50, ±10pp cap) for stability
        if side == "UNDER":
            K = 100  # Slower to trust UNDER calibrator (more shrinkage)
            # Tighter cap (±5pp) to prevent overconfidence
            MAX_ADJUSTMENT = 0.05
        else:  # OVER
            K = 50  # Standard shrinkage
            MAX_ADJUSTMENT = 0.10  # Standard cap (±10pp)

        # Calculate shrinkage weight (James-Stein style)
        # w=0 when n=0 (use raw), w=1 when n→∞ (use calibrated)
        w_side = n_side / (n_side + K) if n_side > 0 else 0.0

        # Blend calibrated and raw probabilities
        prob_blend = w_side * prob_cal + (1 - w_side) * prob_raw

        # CAP: Prevent extreme calibration adjustments (safety guard)
        prob_capped = np.clip(prob_blend, prob_raw - MAX_ADJUSTMENT, prob_raw + MAX_ADJUSTMENT)

        # Log diagnostic info if large adjustment detected
        adjustment = abs(prob_capped - prob_raw)
        if adjustment > 0.05:  # Log if adjustment > 5%
            import warnings

            warnings.warn(
                f"{side} calibration: raw={
                    prob_raw:.3f} → blend={
                    prob_capped:.3f} "
                f"(n={n_side}, w={
                    w_side:.2f}, adjustment={
                    adjustment:+.3f})",
                UserWarning,
            )

        return prob_capped
    else:
        # No metadata available - use full calibrated probability (legacy behavior)  # noqa: E501
        # Cap adjustment as safety guard even without metadata
        MAX_ADJUSTMENT = 0.10
        prob_capped = np.clip(prob_cal, prob_raw - MAX_ADJUSTMENT, prob_raw + MAX_ADJUSTMENT)
        return prob_capped


def calculate_edges(predictions_df, odds_df, calibrators):
    """
    Calculate probabilistic edges using player-specific variance and no-vig odds.  # noqa: E501

    MAJOR UPDATE (TIER 2): Now uses side-specific beta calibration for improved accuracy.
    - Uses player-specific σ (variance) with shrinkage
    - Uses side-specific beta calibration (OVER/UNDER separate)
    - Falls back to isotonic calibration if beta not available
    - Calculates probabilities via normal CDF
    - Removes vig from market odds for fair comparison
    - Calculates proper EV: P(win) × (odds - 1) - P(lose)

    Args:
        predictions_df (pd.DataFrame): Predictions with player names and predicted PRA
        odds_df (pd.DataFrame): Odds data with lines and prices
        calibrators (list): List of isotonic regression calibrators (legacy, not used)

    Returns:
        pd.DataFrame: Betting opportunities with probabilities, EV, and edges
    """
    print("\n📊 Calculating probabilistic edges...")

    # TIER 2: Load beta calibrator (preferred) or fallback to isotonic
    beta_calibrators, beta_metadata = load_beta_calibrator()

    if beta_calibrators is None:
        # Fallback to isotonic calibrator (Tier 1 behavior)
        iso_calibrator = load_isotonic_calibrator()
    else:
        iso_calibrator = None  # Don't use isotonic if beta is available

    # Initialize player variance calculator (singleton)
    variance_calc = get_player_variance_calculator()
    print(
        f"   ✅ Loaded player variance calculator ({
            variance_calc.historical_df.shape[0]:,    } predictions)"
    )

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

    # Calculate player-specific probabilities using normal CDF
    print("   🔄 Calculating player-specific probabilities...")
    opportunities = []

    for idx, row in merged_df.iterrows():
        player_name = row["player_name"]
        prediction_raw = row["predicted_PRA"]

        # Apply PRA-level isotonic calibration if available (legacy behavior)
        if iso_calibrator is not None:
            prediction = iso_calibrator.predict([prediction_raw])[0]
        else:
            prediction = prediction_raw

        line = row["line"]

        # Get player-specific variance (with shrinkage)
        sigma = variance_calc.get_player_variance(player_name, prediction)

        # Calculate RAW probabilities using normal CDF
        # P(actual > line) = 1 - CDF(line | μ=prediction, σ=sigma)
        prob_over_raw = 1 - norm.cdf(line, loc=prediction, scale=sigma)
        prob_under_raw = 1 - prob_over_raw

        # TIER 2: Apply side-specific beta calibration with shrinkage to
        # probabilities
        if beta_calibrators is not None:
            prob_over = apply_beta_calibration(
                prob_over_raw, "OVER", beta_calibrators, beta_metadata
            )
            prob_under = apply_beta_calibration(
                prob_under_raw, "UNDER", beta_calibrators, beta_metadata
            )
        else:
            # Use uncalibrated probabilities if no beta calibrator
            prob_over = prob_over_raw
            prob_under = prob_under_raw

        # ================================================================
        # PHASE A+ FIX #1 & #2: PROBABILITY CAPS + UNDER OT PENALTY
        # ================================================================
        # Expert guidance: Win probs of 75 - 80% at -110 odds create inflated EVs (30 - 45%)  # noqa: E501
        # Solution: Cap at 78% for EV calculation + apply UNDER OT penalty
        #
        # Keep unclipped prob_over/prob_under for metrics (ECE/Brier)
        # Use prob_for_EV_over/prob_for_EV_under for EV calculation only

        # PHASE A+++ FIX #1: Dual probability caps
        # - Decision cap at 0.74: Determines which bets qualify
        # - Stake cap at 0.72: Reduces Kelly sizing on hot edges (15 - 22% EV band)  # noqa: E501
        # This tones down bet size on strongest bets without changing bet
        # selection
        PROB_CAP_DECISION = 0.74  # For bet qualification (unchanged from A++)
        PROB_CAP_STAKE = 0.72  # For Kelly sizing (NEW - Phase A+++)

        prob_for_decision_over = min(prob_over, PROB_CAP_DECISION)
        prob_for_decision_under = min(prob_under, PROB_CAP_DECISION)

        # PHASE A++ FIX #2: Global shrinkage toward 0.5 (reliability-based)
        # Shrink all probabilities toward 50% based on calibration confidence
        # Formula: p' = 0.5 + s * (p - 0.5) where s=0.85 (reliability factor)
        RELIABILITY_FACTOR = 0.85  # Later: make function of rolling ECE by side  # noqa: E501
        prob_for_decision_over = 0.5 + RELIABILITY_FACTOR * (prob_for_decision_over - 0.5)
        prob_for_decision_under = 0.5 + RELIABILITY_FACTOR * (prob_for_decision_under - 0.5)

        # PHASE A+++ FIX #2: Dynamic OT penalty based on game total/pace
        # OT rates vary by game pace: Low-total games ~5%, High-total games ~7%
        # Use predicted PRA as proxy for game pace (higher PRA = faster pace)
        # Totals <220 → 5% OT, 220 - 235 → 6% OT, 235+ → 7% OT
        # Since we don't have game totals, estimate from player PRA predictions

        # Proxy: Assume average team PRA ~110, use player's proportion
        # High PRA players → likely higher scoring game → higher OT risk
        estimated_game_pace_factor = prediction / 30.0  # Normalize by typical PRA  # noqa: E501

        if estimated_game_pace_factor < 0.85:  # Low-pace game proxy
            P_OT = 0.05  # 5% OT rate for slow games
        elif estimated_game_pace_factor < 1.15:  # Average-pace game
            P_OT = 0.06  # 6% OT rate (baseline)
        else:  # High-pace game proxy
            P_OT = 0.07  # 7% OT rate for fast games

        ALPHA_OT = 0.8  # Penalty weight (unchanged)
        prob_for_decision_under = prob_for_decision_under * (1 - ALPHA_OT * P_OT)  # Dynamic penalty

        # Create stake probabilities with lower cap for Kelly sizing
        prob_for_stake_over = min(prob_over, PROB_CAP_STAKE)
        prob_for_stake_under = min(prob_under, PROB_CAP_STAKE)

        # Apply same shrinkage and OT penalty to stake probabilities
        prob_for_stake_over = 0.5 + RELIABILITY_FACTOR * (prob_for_stake_over - 0.5)
        prob_for_stake_under = 0.5 + RELIABILITY_FACTOR * (prob_for_stake_under - 0.5)
        prob_for_stake_under = prob_for_stake_under * (1 - ALPHA_OT * P_OT)

        # Get market odds and calculate implied probabilities
        over_decimal = american_to_decimal(row["over_price"])
        under_decimal = american_to_decimal(row["under_price"])

        over_implied_vigged = 1 / over_decimal
        under_implied_vigged = 1 / under_decimal

        # Remove vig to get fair market probabilities
        no_vig_over, no_vig_under = remove_vig(over_implied_vigged, under_implied_vigged)

        # Calculate EV: P(win) × (odds - 1) - P(lose)
        # Use prob_for_decision (0.74 cap) for bet qualification
        # Will use prob_for_stake (0.72 cap) later for Kelly sizing
        over_ev = prob_for_decision_over * (over_decimal - 1) - (1 - prob_for_decision_over)
        under_ev = prob_for_decision_under * (under_decimal - 1) - (1 - prob_for_decision_under)

        # Calculate probability edges (vs no-vig market)
        over_prob_edge = prob_for_decision_over - no_vig_over
        under_prob_edge = prob_for_decision_under - no_vig_under

        # OVER bet opportunity
        opportunities.append(
            {
                "player_name": player_name,
                "predicted_PRA": prediction,
                "line": line,
                "bookmaker": row["bookmaker"],
                "direction": "OVER",
                "american_odds": row["over_price"],
                "decimal_odds": over_decimal,
                "implied_prob": over_implied_vigged,
                "no_vig_prob": no_vig_over,
                "prob_raw": prob_over_raw,  # Raw from normal CDF
                # After beta calibration (unclipped, for metrics)
                "calibrated_prob": prob_over,
                # 0.74 cap (for decision/EV)
                "prob_for_EV": prob_for_decision_over,
                # 0.72 cap (for Kelly sizing)
                "prob_for_stake": prob_for_stake_over,
                "player_sigma": sigma,
                "prob_edge": over_prob_edge,
                "ev": over_ev,
                "point_edge": row["point_edge"],
                "away_team": row["away_team"],
                "home_team": row["home_team"],
            }
        )

        # UNDER bet opportunity
        opportunities.append(
            {
                "player_name": player_name,
                "predicted_PRA": prediction,
                "line": line,
                "bookmaker": row["bookmaker"],
                "direction": "UNDER",
                "american_odds": row["under_price"],
                "decimal_odds": under_decimal,
                "implied_prob": under_implied_vigged,
                "no_vig_prob": no_vig_under,
                "prob_raw": prob_under_raw,  # Raw from normal CDF
                # After beta calibration (unclipped, for metrics)
                "calibrated_prob": prob_under,
                # 0.74 cap + OT penalty (for decision/EV)
                "prob_for_EV": prob_for_decision_under,
                # 0.72 cap + OT penalty (for Kelly sizing)
                "prob_for_stake": prob_for_stake_under,
                "player_sigma": sigma,
                "prob_edge": under_prob_edge,
                "ev": under_ev,
                "point_edge": -row["point_edge"],
                "away_team": row["away_team"],
                "home_team": row["home_team"],
            }
        )

    opportunities_df = pd.DataFrame(opportunities)
    print(f"   ✅ Found {len(opportunities_df)} betting opportunities")
    print(
        f"   📊 Sigma range: [{
            opportunities_df['player_sigma'].min():.2f}, {
            opportunities_df['player_sigma'].max():.2f}]"
    )
    print(
        f"   📊 Mean EV: {
            opportunities_df['ev'].mean():.4f} (filter by EV >= 0.02)"
    )

    return opportunities_df


def calculate_confidence_level(edge, calibrated_prob):
    """
    Calculate confidence level for a bet.

    Confidence based on:
    1. Edge size (larger edge = higher confidence)
    2. Probability magnitude (avoid 50 / 50 bets)
    3. Combined score

    Returns:
        str: Confidence level (VERY HIGH, HIGH, MEDIUM, LOW)
    """
    # Edge contribution (0 - 50 points)
    edge_score = min(edge * 500, 50)  # 10% edge = 50 points

    # Probability magnitude contribution (0 - 50 points)
    # Penalize probabilities near 50% (low confidence)
    prob_distance_from_50 = abs(calibrated_prob - 0.5)
    prob_score = prob_distance_from_50 * 100  # Max 50 points when prob = 0 or 1  # noqa: E501

    # Total score (0 - 100)
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


def create_bet_id(date: str, player_name: str, line: float, direction: str) -> str:
    """
    Generate unique bet ID for CLV tracking

    Args:
        date: Game date (YYYY-MM-DD)
        player_name: Player name
        line: Betting line (e.g., 35.5)
        direction: OVER or UNDER

    Returns:
        Unique bet ID (e.g., "2025 - 10 - 23_lebron_james_35.5_OVER")
    """
    # Normalize player name: lowercase, replace spaces, remove special chars
    player_slug = (
        player_name.lower()
        .replace(" ", "_")
        .replace("'", "")
        .replace(".", "")
        # Remove accents (e.g., Nikola Jokić → nikola_jokic)
        .encode("ascii", "ignore")
        .decode("ascii")
    )
    return f"{date}_{player_slug}_{line}_{direction}"


def log_bets_to_clv_tracker(
    display_bets: pd.DataFrame, target_date: str, clv_tracker: CLVTracker
) -> Dict[str, int]:
    """
    Log all recommended bets to CLV tracker for performance validation

    Handles line movement by removing old bets and keeping only the latest for each player/direction.
    Example: If line moves from Curry 38.5 → 34.5, removes old bet and logs new one.  # noqa: E501

    Args:
        display_bets: DataFrame of bets with stake > 0
        target_date: Game date (YYYY-MM-DD)
        clv_tracker: Initialized CLVTracker instance

    Returns:
        Dict with counts: {'logged': int, 'skipped': int, 'replaced': int, 'failed': int}
    """
    logged_count = 0
    skipped_count = 0
    replaced_count = 0
    failed_count = 0

    # Get existing bets for this date
    existing_ledger = clv_tracker.ledger
    existing_for_date = existing_ledger[existing_ledger["date"] == target_date].copy()  # noqa: E501

    for idx, row in display_bets.iterrows():
        try:
            player_name = row["player_name"]
            direction = row["direction"]
            line = row["line"]

            # Generate unique bet ID (includes line)
            bet_id = create_bet_id(
                date=target_date, player_name=player_name, line=line, direction=direction
            )

            # Check if exact same bet exists (same player, line, direction)
            if bet_id in existing_ledger["bet_id"].values:
                skipped_count += 1
                continue

            # Check if there's an OLDER bet for same player/direction (line has moved)  # noqa: E501
            # Remove old bet if it exists (keep only latest odds)
            old_bets = existing_for_date[
                (existing_for_date["player"] == player_name)
                & (existing_for_date["side"] == direction)
            ]

            if not old_bets.empty:
                # Remove old bets for this player/direction
                for old_bet_id in old_bets["bet_id"].values:
                    clv_tracker.ledger = clv_tracker.ledger[
                        clv_tracker.ledger["bet_id"] != old_bet_id
                    ].copy()
                    replaced_count += 1
                    print(
                        f"   🔄 Updated bet: {player_name} {direction} (line moved)"
                    )  # noqa: E501

            # Create game_id for grouping
            game_id = f"{row['away_team']} @ {row['home_team']}"

            # Get current timestamp
            entry_time = datetime.now().isoformat()

            # Log bet entry
            clv_tracker.log_bet_entry(
                bet_id=bet_id,
                date=target_date,
                player=player_name,
                market="PRA",
                side=direction,
                line=line,
                entry_odds_dec=row["decimal_odds"],
                entry_time=entry_time,
                game_id=game_id,
                predicted_pra=row["predicted_PRA"],
                player_sigma=row["player_sigma"],
                ev=row["ev"],
                stake=row["bet_size"],
            )
            logged_count += 1

        except Exception as e:
            # Log individual bet failure, continue with others
            failed_count += 1
            print(f"   ⚠️  Failed to log bet for {row['player_name']}: {e}")
            continue

    return {
        "logged": logged_count,
        "skipped": skipped_count,
        "replaced": replaced_count,
        "failed": failed_count,
    }


# ============================================================================
# MAIN SCRIPT
# ============================================================================


def main():
    parser = argparse.ArgumentParser(
        description="Generate daily betting recommendations using calibrated model"
    )  # noqa: E501
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

    target_date = args.date if args.date else datetime.now().strftime("%Y-%m-%d")  # noqa: E501
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
    min_ev = strategy["min_ev"]
    max_bet_fraction = 0.10 if strategy_name in ["aggressive", "maximum"] else 0.05

    print(f"   Kelly fraction: {kelly_fraction:.0%}")
    print(f"   Minimum EV: {min_ev:.1%} (Expected Value threshold)")
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

        print("✅ Loaded metadata")

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
        print("   Calibrators: ✅ Available")

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
        # Load historical data (through 2023 - 24)
        df_historical = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
        df_historical["GAME_DATE"] = pd.to_datetime(df_historical["GAME_DATE"])

        # Load 2024 - 25 season data
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
            f"   Date range: {
                historical_df['GAME_DATE'].min().date()} to {
                historical_df['GAME_DATE'].max().date()}"
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

    # Calculate absolute point edge for analysis (not filtering)
    opportunities_df["abs_point_edge"] = opportunities_df["point_edge"].abs()

    # ========================================================================
    # GUARD 1 & 2: SIDE-SPECIFIC EV THRESHOLDS (Expert Recommendation #1)
    # ========================================================================
    # OVER bets: require ≥4% EV (small sample size, n=25)
    # UNDER bets: require ≥8% EV (raised from 6% due to OT/tail risk + 88%
    # bias)
    EV_THRESHOLD_OVER = 0.04
    # RAISED to 12% (Phase A++: Final tuning for hot EVs)
    EV_THRESHOLD_UNDER = 0.12

    print("\n🛡️  APPLYING SIDE-SPECIFIC EV GUARDS:")
    print(
        f"   OVER bets:  EV ≥ {
            EV_THRESHOLD_OVER:.1%} (small sample protection)"
    )
    print(
        f"   UNDER bets: EV ≥ {
            EV_THRESHOLD_UNDER:.1%} (OT/tail risk + calibration protection)"
    )

    # Apply side-specific EV filters
    bets_df = opportunities_df[
        (
            (
                (opportunities_df["direction"] == "OVER")
                & (opportunities_df["ev"] >= EV_THRESHOLD_OVER)
            )
            | (
                (opportunities_df["direction"] == "UNDER")
                & (opportunities_df["ev"] >= EV_THRESHOLD_UNDER)
            )
        )
        & (opportunities_df["point_edge"] > 0)  # Match direction only
    ].copy()

    if len(bets_df) == 0:
        print("❌ No bets meet side-specific EV criteria")
        print(
            f"   Best OVER EV: {opportunities_df[opportunities_df['direction'] == 'OVER']['ev'].max():.3f}"  # noqa: E501
        )
        print(
            f"   Best UNDER EV: {opportunities_df[opportunities_df['direction'] == 'UNDER']['ev'].max():.3f}"  # noqa: E501
        )
        return 1

    # Show how many bets passed for each side
    over_count = (bets_df["direction"] == "OVER").sum()
    under_count = (bets_df["direction"] == "UNDER").sum()
    print(
        f"   ✅ Passed: {over_count} OVER bets (EV ≥ {
            EV_THRESHOLD_OVER:.1%}), {under_count} UNDER bets (EV ≥ {
            EV_THRESHOLD_UNDER:.1%})"
    )

    # Group by player and line, keep only the bet with highest EV
    bets_df = bets_df.loc[bets_df.groupby(["player_name", "line"])["ev"].idxmax()]

    # ========================================================================
    # GUARD 3: PROGRESSIVE SLATE IMBALANCE GUARD (Expert Recommendation #1)
    # ========================================================================
    # Progressive penalty function scales EV requirement with slate imbalance
    # Formula: extra_pp = 0.02 + 0.10 * max(0, ratio - 0.70)
    # 70%: +0.0pp, 80%: +1.0pp, 88%: +3.8pp, 95%: +5.5pp
    # Start penalizing at 70 / 30 (Expert Recommendation)
    IMBALANCE_START_THRESHOLD = 0.70

    over_pct = (bets_df["direction"] == "OVER").sum() / len(bets_df)
    under_pct = 1 - over_pct

    print("\n🛡️  PROGRESSIVE SLATE IMBALANCE CHECK:")
    print(
        f"   Direction split: {int(under_pct *
                                     100)}% UNDER / {int(over_pct *
                                                         100)}% OVER"
    )

    # Calculate progressive penalty based on imbalance severity
    if over_pct >= IMBALANCE_START_THRESHOLD:
        # OVER imbalance - apply progressive penalty
        extra_penalty_pp = 0.02 + 0.10 * max(0, over_pct - IMBALANCE_START_THRESHOLD)
        adjusted_threshold_over = EV_THRESHOLD_OVER + extra_penalty_pp

        print(f"   ⚠️  OVER imbalance detected ({int(over_pct * 100)}%)")
        print(
            f"      Progressive penalty: +{
                extra_penalty_pp:.1%} (base {
                EV_THRESHOLD_OVER:.1%} → {
                adjusted_threshold_over:.1%})"
        )

        bets_df = bets_df[
            (bets_df["direction"] != "OVER")  # Keep all UNDER bets
            | (
                (bets_df["direction"] == "OVER") & (bets_df["ev"] >= adjusted_threshold_over)
            )  # noqa: E501
        ].copy()
        print(
            f"   ✅ Filtered to {
                len(bets_df)} bets after progressive imbalance guard"
        )

    elif under_pct >= IMBALANCE_START_THRESHOLD:
        # UNDER imbalance - apply progressive penalty
        extra_penalty_pp = 0.02 + 0.10 * max(0, under_pct - IMBALANCE_START_THRESHOLD)
        adjusted_threshold_under = EV_THRESHOLD_UNDER + extra_penalty_pp

        print(f"   ⚠️  UNDER imbalance detected ({int(under_pct * 100)}%)")
        print(
            f"      Progressive penalty: +{
                extra_penalty_pp:.1%} (base {
                EV_THRESHOLD_UNDER:.1%} → {
                adjusted_threshold_under:.1%})"
        )

        bets_df = bets_df[
            (bets_df["direction"] != "UNDER")  # Keep all OVER bets
            | (
                (bets_df["direction"] == "UNDER") & (bets_df["ev"] >= adjusted_threshold_under)
            )  # noqa: E501
        ].copy()
        print(
            f"   ✅ Filtered to {
                len(bets_df)} bets after progressive imbalance guard"
        )
    else:
        print(
            f"   ✅ Balanced slate (<{int(IMBALANCE_START_THRESHOLD * 100)}% either side, no penalty)"  # noqa: E501
        )

    if len(bets_df) == 0:
        print("\n❌ No bets passed progressive imbalance guard")
        return 1

    # ========================================================================
    # GUARD 5: PER-SIDE BET CAP (Expert Recommendation #1)
    # ========================================================================
    # Limit exposure to one side during stabilization period
    # Max 5 UNDER bets OR 40% of total bets (whichever is more restrictive)
    MAX_UNDER_BETS = 5
    MAX_SIDE_PCT = 0.40  # 40% of total bets

    under_bets = bets_df[bets_df["direction"] == "UNDER"].copy()
    over_bets = bets_df[bets_df["direction"] == "OVER"].copy()

    print("\n🛡️  PER-SIDE BET CAP (Stabilization):")
    print(f"   Current: {len(under_bets)} UNDER / {len(over_bets)} OVER")

    # Check if UNDER bets exceed cap
    total_bets = len(bets_df)
    max_under_by_count = MAX_UNDER_BETS
    max_under_by_pct = int(total_bets * MAX_SIDE_PCT)
    max_under_allowed = max(max_under_by_count, max_under_by_pct)

    if len(under_bets) > max_under_allowed:
        print(
            f"   ⚠️  UNDER count exceeds cap ({
                len(under_bets)} > {max_under_allowed})"
        )
        print(f"      Keeping top {max_under_allowed} UNDER bets by EV")

        # Keep top N UNDER bets by EV
        under_bets_sorted = under_bets.nlargest(max_under_allowed, "ev")

        # Reconstruct bets_df with capped UNDER bets
        bets_df = pd.concat([over_bets, under_bets_sorted], ignore_index=True)
        print(
            f"   ✅ Capped to {
                len(bets_df)} total bets ({
                len(under_bets_sorted)} UNDER / {
                len(over_bets)} OVER)"
        )
    else:
        print(f"   ✅ Within cap (max {max_under_allowed} UNDER allowed)")

    if len(bets_df) == 0:
        print("\n❌ No bets passed per-side bet cap")
        return 1

    # ========================================================================
    # GUARD 6: WIN PROBABILITY RANGE FILTER (Tiny Safety)
    # ========================================================================
    # Skip bets with extreme win probabilities (likely miscalibration)
    # - Too low (<55%): Insufficient edge for Kelly sizing
    # - Too high (>85%): Likely overconfident/miscalibrated
    print("\n🛡️  GUARD 6: WIN PROBABILITY RANGE FILTER")
    print("   Requiring: 55% ≤ Win Prob ≤ 85%")

    before_prob_filter = len(bets_df)
    bets_df = bets_df[
        (bets_df["calibrated_prob"] >= 0.55) & (bets_df["calibrated_prob"] <= 0.85)
    ].copy()
    after_prob_filter = len(bets_df)

    filtered_out = before_prob_filter - after_prob_filter
    if filtered_out > 0:
        print(f"   ⚠️  Filtered out {filtered_out} bets with extreme probabilities")  # noqa: E501
    else:
        print(f"   ✅ All {after_prob_filter} bets within probability range")

    if len(bets_df) == 0:
        print("\n❌ No bets passed probability range filter")
        return 1

    # Add edge category for ANALYSIS (not filtering)
    bets_df["edge_category"] = pd.cut(
        bets_df["abs_point_edge"],
        bins=[0, 3, 5, 7, 10, 100],
        labels=["0 - 3", "3 - 5", "5 - 7", "7 - 10", "10+"],
    )

    print(f"\n✅ {len(bets_df)} bets passed all EV guards")
    print(f"   All bets from single bookmaker: {bets_df['bookmaker'].iloc[0]}")
    print(
        f"   Edge distribution: {
            bets_df['edge_category'].value_counts().to_dict()}"
    )

    # Sort by EV (best bets first) - NOT point-edge
    bets_df = bets_df.sort_values("ev", ascending=False)

    # Calculate confidence levels
    bets_df["confidence"] = bets_df.apply(
        lambda row: calculate_confidence_level(row["prob_edge"], row["calibrated_prob"]), axis=1
    )

    # Calculate bet sizes using Kelly criterion
    # Use EV and decimal odds for proper Kelly calculation
    # TIER 2 FIX: Use calibrator ECE if available, otherwise use Tier 1 static
    # ECE
    beta_calibrators_main, beta_metadata_main = load_beta_calibrator()

    if beta_metadata_main is not None:
        CURRENT_ECE = beta_metadata_main["ece_train_cal"]
        print(
            f"\n✅ TIER 2 ACTIVE: Using beta calibrator ECE = {
                CURRENT_ECE:.4f}"
        )
        print(
            f"   This is a {((0.208 -
                                CURRENT_ECE) /
                               0.208 *
                               100):.1f}% improvement over Tier 1 ECE (0.208)"
        )
    else:
        CURRENT_ECE = 0.208  # Tier 1 fallback
        print(
            f"\n⚠️  TIER 1 ACTIVE: Using static ECE = {
                CURRENT_ECE:.3f} (will block bets)"
        )
        print(
            "   Run: uv run python scripts/calibration/fit_beta_calibrator.py --train-only"
        )  # noqa: E501

    # ========================================================================
    # GUARD 4: CONSERVATIVE KELLY FRACTION (Stabilization Period)
    # ========================================================================
    # Use 0.1× Kelly (10% of Kelly) during calibrator stabilization
    # Check sample sizes - if either side has n < 50, apply conservative sizing
    n_over = beta_metadata_main.get("n_over", 0) if beta_metadata_main else 0
    n_under = beta_metadata_main.get("n_under", 0) if beta_metadata_main else 0
    min_sample_size = min(n_over, n_under) if (n_over > 0 and n_under > 0) else 0

    if min_sample_size < 50:
        # Apply extra conservative Kelly fraction
        STABILIZATION_KELLY_MULT = (
            # 40% of configured Kelly (e.g., 0.25 × 0.4 = 0.10 = 10% Kelly)
            0.4
        )
        kelly_fraction_adj = kelly_fraction * STABILIZATION_KELLY_MULT
        print("\n🛡️  GUARD 4: CONSERVATIVE KELLY SIZING (Stabilization)")
        print(f"   Sample sizes: OVER n={n_over}, UNDER n={n_under}")
        print(f"   Applying {STABILIZATION_KELLY_MULT:.0%} Kelly multiplier")
        print(
            f"   Effective Kelly: {
                kelly_fraction_adj:.2f} ({
                kelly_fraction_adj *
                100:.0f}% of full Kelly)"
        )
    else:
        kelly_fraction_adj = kelly_fraction
        print(f"\n✅ Using standard Kelly fraction: {kelly_fraction:.0%}")

    # PHASE A+++ FIX: Use prob_for_stake (0.72 cap) for Kelly sizing
    # This reduces bet size on hot edges without changing qualification
    # Recalculate EV using stake probability (lower than decision probability)
    def calculate_stake_ev(row):
        """Calculate EV for stake sizing using lower probability cap"""
        prob_stake = row["prob_for_stake"]
        decimal_odds = row["decimal_odds"]
        return prob_stake * (decimal_odds - 1) - (1 - prob_stake)

    bets_df["ev_for_stake"] = bets_df.apply(calculate_stake_ev, axis=1)

    bets_df["bet_size"] = bets_df.apply(
        lambda row: calculate_kelly_fraction(
            ev=row["ev_for_stake"],
            # Use stake EV (0.72 cap) not decision EV (0.74 cap)
            decimal_odds=row["decimal_odds"],
            bankroll=bankroll,
            kelly_fraction=kelly_fraction_adj,  # Use adjusted Kelly fraction
            max_bet_pct=max_bet_fraction,
            ece=CURRENT_ECE,
            # Tier 2: ~0.0076 (good), Tier 1: 0.208 (blocks bets)
        ),
        axis=1,
    )

    # Re-sort by EV (best bets first) after adding new columns
    bets_df = bets_df.sort_values("ev", ascending=False)

    # ========================================================================
    # PER-GAME EXPOSURE CAPS (Risk Management)
    # ========================================================================
    # Cap total stake per game at 3% of bankroll to avoid concentration risk
    MAX_GAME_EXPOSURE = 0.03  # 3% of bankroll per game

    bets_df["game_id"] = bets_df["away_team"] + " @ " + bets_df["home_team"]
    game_exposure = bets_df.groupby("game_id")["bet_size"].sum()

    # Identify games exceeding exposure limit
    over_exposed_games = game_exposure[game_exposure > bankroll * MAX_GAME_EXPOSURE]

    if len(over_exposed_games) > 0:
        print("\n⚠️  PER-GAME EXPOSURE CAPS:")
        for game_id, total_stake in over_exposed_games.items():
            scale_factor = (bankroll * MAX_GAME_EXPOSURE) / total_stake
            print(
                f"   {game_id}: ${
                    total_stake:.2f} → ${
                    bankroll *
                    MAX_GAME_EXPOSURE:.2f} (scaled {
                    scale_factor:.1%})"
            )

            # Scale down all bets in this game proportionally
            game_mask = bets_df["game_id"] == game_id
            bets_df.loc[game_mask, "bet_size"] *= scale_factor
            bets_df.loc[game_mask, "exposure_capped"] = True

    # Mark bets that weren't capped
    if "exposure_capped" not in bets_df.columns:
        bets_df["exposure_capped"] = False
    else:
        bets_df["exposure_capped"] = bets_df["exposure_capped"].fillna(False)

    # ========================================================================
    # PORTFOLIO DIVERSIFICATION (Multi-Criteria Selection)
    # ========================================================================
    # Apply portfolio diversification to reduce correlation risk
    # Filters: sensitivity testing, game concentration limits, CLV ranking,
    # EV-band randomization
    from src.betting.portfolio_selection import BetPortfolioSelector

    print("\n" + "=" * 80)
    print("🎯 PORTFOLIO DIVERSIFICATION")
    print("=" * 80)

    # Initialize portfolio selector
    portfolio_selector = BetPortfolioSelector(
        max_bets_per_game=2,  # Conservative: max 2 bets per game
        ev_band_width=0.02,  # 2 percentage point randomization band
        line_stress=0.5,  # ±0.5 point line movement stress
        sigma_stress=0.10,  # +10% sigma inflation stress
        seed_date=target_date,  # Reproducible randomization
    )

    # Select diversified portfolio (target ~15 bets for recommendations)
    # This replaces naive "top N by EV" with multi-criteria selection
    if len(bets_df) > 0:
        __bets_before_portfolio = len(bets_df)  # noqa: F841
        bets_df, portfolio_diag = portfolio_selector.select_portfolio(
            bets_df=bets_df,
            target_count=15,
            # Target number of bets (will be further filtered for display)
            min_ev=None,  # Already filtered by EV guards
        )

        print("\n✅ Portfolio selection complete:")
        print(
            f"   Input bets:                  {
                portfolio_diag['input_count']}"
        )
        print(
            f"   Sensitivity filtered:        -{
                portfolio_diag['sensitivity_filtered']} (failed stress tests)"
        )
        print(
            f"   Game concentration filtered: -{
                portfolio_diag['game_concentration_filtered']} (>2 per game)"
        )
        print(
            f"   Final selected:              {
                portfolio_diag['final_selected']}"
        )
        print(
            f"   Games represented:           {
                portfolio_diag['games_represented']} games"
        )
        print(
            f"   Max bets per game:           {
                portfolio_diag['max_bets_per_game']}"
        )
        print(
            f"   Direction split:             {
                portfolio_diag['over_under_split']['OVER']} OVER / {
                portfolio_diag['over_under_split']['UNDER']} UNDER"
        )
        print(
            f"   Average EV:                  {
                portfolio_diag['avg_ev']:.2%}"
        )

        # Alert if too few games represented
        if portfolio_diag["games_represented"] < 3:
            print(
                f"   ⚠️  WARNING: Low game count ({
                    portfolio_diag['games_represented']} games)"
            )
            print("      Consider relaxing filters if this persists")

        # Alert if too many bets filtered by sensitivity
        sensitivity_pct = portfolio_diag["sensitivity_filtered"] / max(
            1, portfolio_diag["input_count"]
        )
        if sensitivity_pct > 0.50:
            print(
                f"   ⚠️  WARNING: {
                    sensitivity_pct:.0%} of bets failed sensitivity tests"
            )
            print("      This may indicate model calibration issues")
    else:
        print("\n⚠️  No bets available for portfolio selection")
        portfolio_diag = {"input_count": 0, "final_selected": 0, "games_represented": 0}

    # ========================================================================
    # 7. DISPLAY RECOMMENDATIONS
    # ========================================================================

    print("\n" + "=" * 80)
    print("📋 BETTING RECOMMENDATIONS")
    print("=" * 80)

    # Filter to only bets with stake > 0 for display (keep all in CSV for
    # audit)
    display_bets = bets_df[bets_df["bet_size"] > 0].copy()

    if len(display_bets) == 0:
        # Calculate what the minimum bet would have been
        min_bet = max(1.0, bankroll * 0.005)

        print("\n⚠️  No bets meet minimum size threshold")
        print(
            f"   Found {
                len(bets_df)} opportunities with positive EV, but all bet sizes are below minimum"
        )  # noqa: E501
        print(
            f"   Minimum bet: ${
                min_bet:.2f} (0.5% of ${
                bankroll:,.0f} bankroll)"
        )
        if len(bets_df) > 0:
            max_bet = bets_df["bet_size"].max() if "bet_size" in bets_df.columns else 0
            if max_bet > 0:
                print(f"   Highest Kelly-sized bet: ${max_bet:.2f}")
        print("\n💡 To see betting recommendations, try:")
        print("   • Increase bankroll (larger bankroll = larger bet sizes)")
        print("   • Use aggressive strategy: --strategy aggressive")
        return 1

    print(f"\n🎯 TOP {min(10, len(display_bets))} BETS:")
    print("-" * 80)

    for i, (idx, row) in enumerate(display_bets.head(10).iterrows(), 1):
        matchup = f"{row['away_team']} @ {row['home_team']}"
        print(f"\n{i}. {row['player_name']} - {row['direction']} {row['line']}")
        print(f"   Game: {matchup}")
        print(f"   Bookmaker: {row['bookmaker']}")
        print(
            f"   Prediction: {
                row['predicted_PRA']:.1f} ± {
                row['player_sigma']:.1f} PRA"
        )
        print(
            f"   Point Edge: {row['point_edge']:+.1f} pts | Prob Edge: {row['prob_edge']:.1%}"
        )  # noqa: E501
        print(
            f"   Expected Value: {row['ev']:.1%} | No-Vig Market: {row['no_vig_prob']:.1%}"
        )  # noqa: E501
        print(f"   Confidence: {row['confidence']}")
        print(
            f"   Odds: {row['american_odds']:+.0f} (Decimal: {row['decimal_odds']:.2f})"
        )  # noqa: E501
        print(f"   Win Probability: {row['calibrated_prob']:.1%}")
        print(f"   💵 Recommended Bet: ${row['bet_size']:.2f}")

    # ========================================================================
    # 7b. SUGGESTED ACTION SUMMARY
    # ========================================================================

    print("\n" + "=" * 80)
    print("💡 SUGGESTED ACTION")
    print("=" * 80)

    # Determine recommended subset based on edge thresholds (only bets with
    # stake > 0)
    top5_df = display_bets.head(5)
    top5_total = top5_df["bet_size"].sum()
    top5_avg_point_edge = top5_df["abs_point_edge"].mean()
    top5_avg_prob_edge = top5_df["prob_edge"].mean()

    all_total = display_bets["bet_size"].sum()
    all_avg_point_edge = display_bets["abs_point_edge"].mean()
    all_avg_prob_edge = display_bets["prob_edge"].mean()

    print(
        f"\n🎯 RECOMMENDED: Take Top {min(5,
                                           len(display_bets))} Bets (Highest EV)"
    )  # noqa: E501
    print(f"   Total Wager: ${top5_total:.2f}")
    print(f"   Avg Point Edge: {top5_avg_point_edge:.1f} pts")
    print(f"   Avg Prob Edge: {top5_avg_prob_edge:.1%}")
    print("   Risk Level: Lower (concentrated on best opportunities)")

    print(f"\n📊 ALTERNATIVE: Take All {len(display_bets)} Bets (Diversified)")
    print(f"   Total Wager: ${all_total:.2f}")
    print(f"   Avg Point Edge: {all_avg_point_edge:.1f} pts")
    print(f"   Avg Prob Edge: {all_avg_prob_edge:.1%}")
    print("   Risk Level: Higher (more exposure, more variance)")

    print("\n⚠️  IMPORTANT:")
    print("   • Check injury reports before placing bets")
    print("   • Verify player starting lineups 1 - 2 hours before games")
    print("   • Skip any bet if player status is uncertain")

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

    # ========================================================================
    # 8b. CLV TRACKING - AUTOMATIC BET ENTRY LOGGING
    # ========================================================================

    print("\n" + "=" * 80)
    print("📊 CLV TRACKING")
    print("=" * 80)

    try:
        # Initialize CLV tracker
        clv_tracker = CLVTracker(ledger_path="data/clv_ledger.csv")

        # Log all recommended bets
        if len(display_bets) > 0:
            print("\n📝 Logging bet entries to CLV ledger...")
            results = log_bets_to_clv_tracker(display_bets, target_date, clv_tracker)

            # Print summary
            print(
                f"   ✅ Logged {
                    results['logged']} new bets to data/clv_ledger.csv"
            )
            if results["replaced"] > 0:
                print(
                    f"   🔄 Replaced {
                        results['replaced']} bets (line/odds changed)"
                )
            if results["skipped"] > 0:
                print(
                    f"   ⏭️  Skipped {
                        results['skipped']} duplicate bets (already in ledger)"
                )  # noqa: E501
            if results["failed"] > 0:
                print(
                    f"   ❌ Failed to log {
                        results['failed']} bets (see warnings above)"
                )

            # Get current CLV ledger status
            total_tracked = len(clv_tracker.ledger)
            bets_with_closing = (~clv_tracker.ledger["clv_pct"].isna()).sum()
            bets_with_results = (~clv_tracker.ledger["won"].isna()).sum()

            print("\n📊 CLV Ledger Status:")
            print(f"   Total bets tracked: {total_tracked}")
            print(f"   Bets with closing lines: {bets_with_closing}")
            print(f"   Bets with results: {bets_with_results}")

            # Validation readiness
            if bets_with_closing >= 50:
                print("   ✅ READY FOR VALIDATION (50+ bets with closing lines)")  # noqa: E501
                print("\n   Run this to get CLV report:")
                print(
                    '   python -c "from utils.clv_tracker import CLVTracker; CLVTracker().print_clv_report()"'  # noqa: E501
                )
            else:
                needed = 50 - bets_with_closing
                print(
                    f"   ⏳ Need {needed} more bets with closing lines for validation"
                )  # noqa: E501

            print("\n💡 Next Steps:")
            print("   1. ✅ Bet entries logged (DONE)")
            print(
                "   2. ⏳ Capture closing lines T-2 min before games (manual/script)"
            )  # noqa: E501
            print("   3. ⏳ Log results after games complete (manual/script)")
            print("   4. ⏳ Run validation report after 50+ bets")

        else:
            print("\n⚠️  No bets to log (all had stake = $0)")

    except Exception as e:
        # Non-blocking error: CLV tracking is optional
        print(f"\n⚠️  CLV tracking failed: {e}")
        print("   This does not affect your betting recommendations.")
        print(f"   Recommendations saved successfully to: {csv_path}")
        print("   You can manually log bets later using utils/clv_tracker.py")

    # Save HTML if requested
    if args.save_html:
        html_path = output_dir / f"recommendations_{target_date}.html"

        html = """
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

    print("\nSummary:")
    print(f"  📊 Total opportunities: {len(opportunities_df)}")
    print(f"  ✅ Bets meeting EV >= {min_ev:.1%} threshold: {len(bets_df)}")
    print(f"  💰 Total to wager: ${bets_df['bet_size'].sum():,.2f}")
    print(f"  📈 Avg EV: {bets_df['ev'].mean():.1%}")
    print(
        f"  📈 Avg point edge: {
            bets_df['abs_point_edge'].mean():.1f} pts (metadata)"
    )
    print(f"  📈 Avg prob edge: {bets_df['prob_edge'].mean():.1%}")

    # UNDER/OVER split tracking (detect regression-to-mean bias)
    under_count = (bets_df["direction"] == "UNDER").sum()
    over_count = (bets_df["direction"] == "OVER").sum()
    print(f"  📊 Direction split: {under_count} UNDER / {over_count} OVER")
    if under_count > 0 and over_count == 0:
        print(
            "     ⚠️  WARNING: All bets are UNDER - possible regression-to-mean bias"
        )  # noqa: E501
    elif over_count > 0 and under_count == 0:
        print("     ⚠️  WARNING: All bets are OVER - verify calibration")

    print("  🎯 Confidence distribution:")

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
        from scripts.betting.track_all_recommendations import (  # noqa: E501
            track_all_recommendations,
        )

        track_all_recommendations(target_date)
    except Exception as e:
        print(f"⚠️  Auto-tracking failed: {e}")
        print("   You can manually track later with:")
        print(
            f"   uv run python scripts/betting/track_all_recommendations.py {target_date}"
        )  # noqa: E501

    print("\n🎰 Good luck!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
