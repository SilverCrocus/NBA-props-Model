#!/usr/bin/env python3
"""
Daily Betting Recommendations - Production Script

Fetches upcoming NBA games, gets prop odds, makes predictions,
and recommends bets with confidence levels.

Usage:
    uv run python scripts/production/daily_betting_recommendations.py
    uv run python scripts/production/daily_betting_recommendations.py --date 2025-10-22
    uv run python scripts/production/daily_betting_recommendations.py --strategy moderate

Requirements:
    - Calibrated model trained (production_model_v2.0_CLEAN_CALIBRATED_latest.pkl)
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
STRATEGIES = {
    "conservative": {"kelly_fraction": 0.10, "edge_threshold": 0.07},
    "moderate": {"kelly_fraction": 0.20, "edge_threshold": 0.05},
    "aggressive": {"kelly_fraction": 0.25, "edge_threshold": 0.05},
    "maximum": {"kelly_fraction": 0.35, "edge_threshold": 0.05},
}

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
        params = {
            "apiKey": api_key,
            "regions": "us",
            "markets": "player_points_rebounds_assists",
            "oddsFormat": "american",
        }

        response = requests.get(ODDS_API_URL, params=params, timeout=30)
        response.raise_for_status()

        events = response.json()

        if len(events) == 0:
            print(f"⚠️  No odds available for {target_date}")
            return pd.DataFrame()

        # Parse odds into dataframe
        odds_data = []
        for event in events:
            event_id = event["id"]
            event_date = event["commence_time"][:10]
            home_team = event["home_team"]
            away_team = event["away_team"]

            for bookmaker in event.get("bookmakers", []):
                bookmaker_name = bookmaker["title"]

                for market in bookmaker.get("markets", []):
                    if market["key"] == "player_points_rebounds_assists":
                        for outcome in market["outcomes"]:
                            player_name = outcome["description"]
                            line = outcome["point"]
                            direction = outcome["name"]  # Over or Under
                            price = outcome["price"]

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

        print(f"✅ Found {len(odds_pivot)} prop lines from {odds_pivot['bookmaker'].nunique()} bookmakers")
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


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return 1 + (american_odds / 100)
    else:
        return 1 + (100 / abs(american_odds))


def make_predictions(players_today, historical_df, model, feature_cols):
    """
    Make PRA predictions for today's players.

    Args:
        players_today (list): Player names playing today
        historical_df (pd.DataFrame): Historical game logs
        model: Trained XGBoost model
        feature_cols (list): Feature column names

    Returns:
        pd.DataFrame: Predictions with player names and predicted PRA
    """
    print(f"\n🤖 Generating predictions for {len(players_today)} players...")

    # Create dummy "today" games for feature calculation
    today_date = datetime.now().strftime("%Y-%m-%d")
    today_games = []

    for player_name in players_today:
        # Get player's recent games
        player_hist = historical_df[historical_df["PLAYER_NAME"] == player_name]

        if len(player_hist) == 0:
            print(f"   ⚠️  No history for {player_name} - skipping")
            continue

        # Create dummy game entry with required columns
        player_id = player_hist["PLAYER_ID"].iloc[0]

        # Extract team from MATCHUP column (format: "BOS vs. LAL" or "BOS @ LAL")
        last_matchup = player_hist["MATCHUP"].iloc[-1] if "MATCHUP" in player_hist.columns else ""
        last_team = last_matchup.split()[0] if last_matchup else "UNK"

        # Create minimal game row with all required columns from historical data
        last_game = player_hist.iloc[-1].to_dict()
        today_game = {
            "PLAYER_ID": player_id,
            "PLAYER_NAME": player_name,
            "GAME_DATE": today_date,
            "PRA": np.nan,  # To be predicted
        }

        # Copy over necessary columns from last game (for feature calculation)
        for col in ["MIN", "FGA", "FG_PCT", "FG3A", "FG3_PCT", "FTA", "FT_PCT",
                    "OREB", "DREB", "STL", "BLK", "TOV", "PF", "SEASON"]:
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

    # Make predictions
    X = predictions_df[feature_cols].fillna(0)
    predictions = model.predict(X)
    predictions = np.maximum(0, predictions)  # Clip to non-negative

    predictions_df["predicted_PRA"] = predictions

    print(f"   ✅ Generated {len(predictions_df)} predictions")
    return predictions_df[["PLAYER_NAME", "predicted_PRA"]]


def calculate_edges(predictions_df, odds_df, calibrator):
    """
    Calculate calibrated edges for each bet opportunity.

    Args:
        predictions_df (pd.DataFrame): Predictions with player names and predicted PRA
        odds_df (pd.DataFrame): Odds data with lines and prices
        calibrator: Isotonic regression calibrator

    Returns:
        pd.DataFrame: Betting opportunities with edges
    """
    print(f"\n📊 Calculating edges...")

    # Merge predictions with odds
    merged_df = odds_df.merge(predictions_df, left_on="player_name", right_on="PLAYER_NAME", how="inner")

    if len(merged_df) == 0:
        print("   ❌ No matches between predictions and odds")
        return pd.DataFrame()

    print(f"   ✅ Matched {len(merged_df)} predictions with odds")

    # Calculate raw probabilities
    merged_df["difference"] = merged_df["predicted_PRA"] - merged_df["line"]
    scale = 5.0  # MAE-based scale
    merged_df["prob_over_raw"] = 1 / (1 + np.exp(-merged_df["difference"] / scale))

    # Calibrate probabilities
    if calibrator is not None:
        merged_df["prob_over_calibrated"] = calibrator.predict(merged_df["prob_over_raw"])
    else:
        print("   ⚠️  No calibrator found - using raw probabilities")
        merged_df["prob_over_calibrated"] = merged_df["prob_over_raw"]

    # Calculate edges for both over and under
    opportunities = []

    for idx, row in merged_df.iterrows():
        # Over bet
        over_decimal = american_to_decimal(row["over_price"])
        over_implied_prob = 1 / over_decimal
        over_edge = row["prob_over_calibrated"] - over_implied_prob

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
                "edge": over_edge,
            }
        )

        # Under bet
        under_decimal = american_to_decimal(row["under_price"])
        under_implied_prob = 1 / under_decimal
        under_edge = (1 - row["prob_over_calibrated"]) - under_implied_prob

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
                "edge": under_edge,
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
    parser.add_argument("--bankroll", type=float, default=1000.0, help="Current bankroll (default: $1,000)")
    parser.add_argument(
        "--api-key", type=str, default=None, help="Odds API key (overrides config)"
    )
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
    edge_threshold = strategy["edge_threshold"]
    max_bet_fraction = 0.10 if strategy_name in ["aggressive", "maximum"] else 0.05

    print(f"   Kelly fraction: {kelly_fraction:.0%}")
    print(f"   Edge threshold: ≥{edge_threshold:.0%}")
    print(f"   Max bet: {max_bet_fraction:.0%} of bankroll")

    # ========================================================================
    # 1. LOAD MODEL
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 1: LOADING MODEL")
    print("=" * 80)

    model_path = "models/production_model_v2.0_PRODUCTION_CALIBRATED_latest.pkl"

    try:
        with open(model_path, "rb") as f:
            model_dict = pickle.load(f)

        model = model_dict["model"]
        feature_cols = model_dict["feature_cols"]
        calibrator = model_dict.get("calibrator", None)

        print(f"✅ Loaded model: {model_dict['version']}")
        print(f"   Features: {len(feature_cols)}")
        print(f"   Val MAE: {model_dict.get('val_mae', 0):.2f} points")
        print(f"   Calibrator: {'✅ Available' if calibrator else '❌ Not found'}")

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
        historical_df = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
        historical_df["GAME_DATE"] = pd.to_datetime(historical_df["GAME_DATE"])
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
        print(f"   Date range: {historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}")
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

    predictions_df = make_predictions(players_today, historical_df, model, feature_cols)

    if len(predictions_df) == 0:
        print("❌ No predictions generated")
        return 1

    # ========================================================================
    # 5. CALCULATE EDGES
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 5: CALCULATING EDGES")
    print("=" * 80)

    opportunities_df = calculate_edges(predictions_df, odds_df, calibrator)

    if len(opportunities_df) == 0:
        print("❌ No betting opportunities found")
        return 1

    # ========================================================================
    # 6. FILTER AND RANK
    # ========================================================================

    print("\n" + "=" * 80)
    print("STEP 6: FILTERING AND RANKING BETS")
    print("=" * 80)

    # Filter to positive edges above threshold
    bets_df = opportunities_df[opportunities_df["edge"] >= edge_threshold].copy()

    if len(bets_df) == 0:
        print(f"❌ No bets meet {edge_threshold:.0%} edge threshold")
        print(f"   Best edge available: {opportunities_df['edge'].max():.1%}")
        return 1

    print(f"✅ {len(bets_df)} bets meet {edge_threshold:.0%} threshold")

    # Calculate confidence levels
    bets_df["confidence"] = bets_df.apply(
        lambda row: calculate_confidence_level(row["edge"], row["calibrated_prob"]), axis=1
    )

    # Calculate bet sizes
    bets_df["bet_size"] = bets_df["edge"].apply(
        lambda edge: calculate_bet_size(edge, bankroll, kelly_fraction, max_bet_fraction)
    )

    # Sort by edge (best bets first)
    bets_df = bets_df.sort_values("edge", ascending=False)

    # ========================================================================
    # 7. DISPLAY RECOMMENDATIONS
    # ========================================================================

    print("\n" + "=" * 80)
    print("📋 BETTING RECOMMENDATIONS")
    print("=" * 80)

    print(f"\n🎯 TOP 10 BETS:")
    print("-" * 80)

    for idx, row in bets_df.head(10).iterrows():
        print(f"\n{idx+1}. {row['player_name']} - {row['direction']} {row['line']}")
        print(f"   Bookmaker: {row['bookmaker']}")
        print(f"   Prediction: {row['predicted_PRA']:.1f} PRA")
        print(f"   Edge: {row['edge']:.1%} ({row['confidence']} CONFIDENCE)")
        print(f"   Odds: {row['american_odds']:+.0f} (Decimal: {row['decimal_odds']:.2f})")
        print(f"   Win Probability: {row['calibrated_prob']:.1%}")
        print(f"   💵 Recommended Bet: ${row['bet_size']:.2f}")

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
    print(f"  ✅ Bets meeting threshold: {len(bets_df)}")
    print(f"  💰 Total to wager: ${bets_df['bet_size'].sum():,.2f}")
    print(f"  📈 Average edge: {bets_df['edge'].mean():.1%}")
    print(f"  🎯 Confidence distribution:")

    for conf in ["VERY HIGH", "HIGH", "MEDIUM", "LOW"]:
        count = (bets_df["confidence"] == conf).sum()
        if count > 0:
            print(f"     {conf}: {count} bets")

    print(f"\n📂 Results saved to: {csv_path}")
    print("\n🎰 Good luck!")

    return 0


if __name__ == "__main__":
    sys.exit(main())
