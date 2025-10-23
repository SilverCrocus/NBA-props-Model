#!/usr/bin/env python3
"""
Backtest Ensemble on 2024-25 Season with Real DraftKings Odds
==============================================================

Backtests the production ensemble model on the 2024-25 season with:
- Walk-forward predictions (no temporal leakage)
- Real DraftKings betting lines and odds
- Kelly Criterion betting strategy
- 4-point edge threshold
- Real bankroll simulation

This script uses:
- Production models trained on 2003-2024 data only
- Actual DraftKings odds from data/historical_odds/2024-25/pra_odds.csv

Usage: uv run python scripts/validation/backtest_ensemble_2024_25_real_odds.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error

# Add scripts/utils to path
sys.path.append("scripts/utils")
from fast_feature_builder import FastFeatureBuilder

print("=" * 80)
print("BACKTEST ENSEMBLE ON 2024-25 SEASON (REAL DRAFTKINGS ODDS)")
print("=" * 80)
print()

# ======================================================================
# CONFIGURATION
# ======================================================================

# Kelly Criterion betting configuration
# EDGE_THRESHOLD = 4.0  # Only bet when we have 4+ point edge vs the line
EDGE_THRESHOLD = 0.0  # NO FILTER - Analyze all bets to find optimal strategy
KELLY_FRACTION = 0.25  # Quarter Kelly (conservative)
STARTING_BANKROLL = 10000
MAX_BET_PCT = 0.05  # Max 5% of bankroll per bet

# 2024-25 season dates
BACKTEST_START = "2024-10-01"
BACKTEST_END = "2025-06-30"

# Bookmaker to use
BOOKMAKER = "DraftKings"

# Model paths
MODEL_PATHS = [
    "models/production_fold_1.pkl",
    "models/production_fold_2.pkl",
    "models/production_fold_3.pkl",
]

# Historical odds file
ODDS_FILE = "data/historical_odds/2024-25/pra_odds.csv"

# ======================================================================
# HELPER FUNCTIONS
# ======================================================================


def american_to_decimal(american_odds):
    """Convert American odds to decimal odds."""
    if american_odds > 0:
        return (american_odds / 100) + 1
    else:
        return (100 / abs(american_odds)) + 1


def calculate_profit(bet_amount, american_odds):
    """Calculate profit from a winning bet with American odds."""
    if american_odds > 0:
        return bet_amount * (american_odds / 100)
    else:
        return bet_amount * (100 / abs(american_odds))


def normalize_player_name(name):
    """Normalize player name for matching."""
    return name.strip().lower()


# ======================================================================
# 1. LOAD MODELS
# ======================================================================

print("STEP 1: Loading ensemble models...")
print()

models = []
calibrators = []
feature_cols = None

for i, model_path in enumerate(MODEL_PATHS, 1):
    print(f"Loading Fold {i}: {model_path}")

    with open(model_path, "rb") as f:
        fold_dict = pickle.load(f)

    models.append(fold_dict["model"])
    calibrators.append(fold_dict["calibrator"])

    if feature_cols is None:
        feature_cols = fold_dict["feature_cols"]

    print(f"   Train period: {fold_dict['train_period']}")
    print(f"   Val period: {fold_dict['val_period']}")
    print(f"   Val MAE: {fold_dict['mae_calibrated']:.2f} pts")
    print()

print(f"✅ Loaded {len(models)} models with {len(feature_cols)} features")
print()

# ======================================================================
# 2. LOAD HISTORICAL ODDS
# ======================================================================

print("STEP 2: Loading historical DraftKings odds...")
print()

odds_df = pd.read_csv(ODDS_FILE)
odds_df["event_date"] = pd.to_datetime(odds_df["event_date"])

# Filter to DraftKings only
odds_df = odds_df[odds_df["bookmaker"] == BOOKMAKER].copy()

# Normalize player names
odds_df["player_name_normalized"] = odds_df["player_name"].apply(normalize_player_name)

print(f"✅ Loaded {len(odds_df):,} DraftKings odds")
print(
    f"   Date range: {odds_df['event_date'].min().date()} to {odds_df['event_date'].max().date()}"
)
print(f"   Unique players: {odds_df['player_name'].nunique()}")
print(f"   Unique dates: {odds_df['event_date'].nunique()}")
print()

# ======================================================================
# 3. LOAD GAME DATA
# ======================================================================

print("STEP 3: Loading 2024-25 season game data...")
print()

# Load game logs
game_logs_df = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
game_logs_df["GAME_DATE"] = pd.to_datetime(game_logs_df["GAME_DATE"])

# Filter to 2024-25 season
game_logs_df = game_logs_df[
    (game_logs_df["GAME_DATE"] >= BACKTEST_START) & (game_logs_df["GAME_DATE"] <= BACKTEST_END)
].copy()

# Add PRA if not present
if "PRA" not in game_logs_df.columns:
    game_logs_df["PRA"] = game_logs_df["PTS"] + game_logs_df["REB"] + game_logs_df["AST"]

# Filter to MIN >= 25
game_logs_df = game_logs_df[game_logs_df["MIN"] >= 25].copy()

# Normalize player names for matching
game_logs_df["player_name_normalized"] = game_logs_df["PLAYER_NAME"].apply(normalize_player_name)

# Sort by date
game_logs_df = game_logs_df.sort_values(["PLAYER_ID", "GAME_DATE"])

print(f"✅ Loaded {len(game_logs_df):,} games from 2024-25 season")
print(
    f"   Date range: {game_logs_df['GAME_DATE'].min().date()} to {game_logs_df['GAME_DATE'].max().date()}"
)
print(f"   Unique players: {game_logs_df['PLAYER_ID'].nunique()}")
print(f"   Unique game dates: {game_logs_df['GAME_DATE'].nunique()}")
print()

# ======================================================================
# 4. BUILD FEATURES
# ======================================================================

print("STEP 4: Building features...")
print()

builder = FastFeatureBuilder()
full_df = builder.build_features(game_logs_df, pd.DataFrame(), verbose=True)

print(f"✅ Features built: {full_df.shape}")
print()

# ======================================================================
# 5. WALK-FORWARD PREDICTIONS
# ======================================================================

print("=" * 80)
print("STEP 5: WALK-FORWARD PREDICTIONS")
print("=" * 80)
print()

# Get unique game dates
unique_dates = sorted(full_df["GAME_DATE"].unique())
print(f"Making predictions for {len(unique_dates)} game dates...")
print()

# Storage for predictions
all_predictions = []

for date_idx, game_date in enumerate(unique_dates, 1):
    # Games on this date
    games_today = full_df[full_df["GAME_DATE"] == game_date].copy()

    if len(games_today) == 0:
        continue

    # Prepare features
    X_today = games_today[feature_cols].fillna(0)

    # Get ensemble predictions
    fold_predictions = []
    for model in models:
        pred = model.predict(X_today)
        fold_predictions.append(pred)

    # Average raw predictions
    ensemble_pred = np.mean(fold_predictions, axis=0)

    # Apply calibration (averaged across folds)
    calibrated_preds = []
    for calibrator in calibrators:
        pred_calibrated = calibrator.transform(ensemble_pred)
        calibrated_preds.append(pred_calibrated)

    # Final prediction (average of calibrated predictions)
    final_pred = np.mean(calibrated_preds, axis=0)

    # Store predictions
    games_today["prediction"] = final_pred
    games_today["actual"] = games_today["PRA"]
    games_today["error"] = games_today["actual"] - games_today["prediction"]
    games_today["abs_error"] = np.abs(games_today["error"])

    all_predictions.append(games_today)

    if date_idx % 20 == 0:
        print(f"   Processed {date_idx}/{len(unique_dates)} dates...")

print()
print("✅ Predictions complete")
print()

# Combine all predictions
predictions_df = pd.concat(all_predictions, ignore_index=True)

# Add normalized player name
predictions_df["player_name_normalized"] = predictions_df["PLAYER_NAME"].apply(
    normalize_player_name
)

# ======================================================================
# 6. MERGE WITH ODDS
# ======================================================================

print("=" * 80)
print("STEP 6: MERGING PREDICTIONS WITH ODDS")
print("=" * 80)
print()

# Merge predictions with odds
predictions_with_odds = predictions_df.merge(
    odds_df[["player_name_normalized", "event_date", "line", "over_price", "under_price"]],
    left_on=["player_name_normalized", "GAME_DATE"],
    right_on=["player_name_normalized", "event_date"],
    how="inner",
)

print(f"✅ Matched {len(predictions_with_odds):,} predictions with DraftKings odds")
print(f"   Match rate: {len(predictions_with_odds) / len(predictions_df) * 100:.1f}%")
print()

# ======================================================================
# 7. PREDICTION ACCURACY
# ======================================================================

print("=" * 80)
print("PREDICTION ACCURACY (GAMES WITH ODDS)")
print("=" * 80)
print()

mae = mean_absolute_error(predictions_with_odds["actual"], predictions_with_odds["prediction"])
print(f"Mean Absolute Error: {mae:.2f} pts")
print(f"Total predictions: {len(predictions_with_odds):,}")
print()

# Error distribution
print("Error Distribution:")
print(
    f"   < 3 pts: {(predictions_with_odds['abs_error'] < 3).sum():,} ({(predictions_with_odds['abs_error'] < 3).mean()*100:.1f}%)"
)
print(
    f"   < 5 pts: {(predictions_with_odds['abs_error'] < 5).sum():,} ({(predictions_with_odds['abs_error'] < 5).mean()*100:.1f}%)"
)
print(
    f"   < 7 pts: {(predictions_with_odds['abs_error'] < 7).sum():,} ({(predictions_with_odds['abs_error'] < 7).mean()*100:.1f}%)"
)
print(
    f"   < 10 pts: {(predictions_with_odds['abs_error'] < 10).sum():,} ({(predictions_with_odds['abs_error'] < 10).mean()*100:.1f}%)"
)
print()

# ======================================================================
# 8. BETTING SIMULATION WITH REAL ODDS
# ======================================================================

print("=" * 80)
print("BETTING SIMULATION (REAL DRAFTKINGS ODDS)")
print("=" * 80)
print()

print(f"Strategy: Kelly Criterion ({KELLY_FRACTION*100:.0f}% Kelly)")
print(f"Edge threshold: {EDGE_THRESHOLD} pts")
print(f"Starting bankroll: ${STARTING_BANKROLL:,.0f}")
print(f"Max bet: {MAX_BET_PCT*100:.0f}% of bankroll")
print()

# Simulate bets
bankroll = STARTING_BANKROLL
bankroll_history = [bankroll]
bet_results = []

for idx, row in predictions_with_odds.iterrows():
    prediction = row["prediction"]
    actual = row["actual"]
    line = row["line"]
    over_price = row["over_price"]
    under_price = row["under_price"]

    # Calculate edge
    pred_over_line = prediction - line

    # Determine if we have an edge (prediction is significantly different from line)
    if abs(pred_over_line) < EDGE_THRESHOLD:
        continue  # No edge

    # Determine bet side
    if prediction > line + EDGE_THRESHOLD:
        # Bet OVER
        bet_side = "over"
        odds = over_price
        is_correct = actual > line
    elif prediction < line - EDGE_THRESHOLD:
        # Bet UNDER
        bet_side = "under"
        odds = under_price
        is_correct = actual < line
    else:
        continue  # No edge

    # Calculate Kelly bet size
    # For American odds, convert to implied probability
    decimal_odds = american_to_decimal(odds)
    implied_prob = 1 / decimal_odds

    # Estimate our win probability based on our model's confidence
    # Higher edge = higher confidence
    edge_magnitude = abs(pred_over_line)
    our_prob = 0.5 + (edge_magnitude / 100)  # Simplified conversion
    our_prob = min(max(our_prob, 0.51), 0.99)  # Clamp to reasonable range

    # Kelly criterion: f = (bp - q) / b
    # where b = decimal_odds - 1, p = our_prob, q = 1 - our_prob
    b = decimal_odds - 1
    kelly_bet_pct = (b * our_prob - (1 - our_prob)) / b
    kelly_bet_pct = max(0, kelly_bet_pct)  # Can't be negative

    # Apply Kelly fraction and max bet limit
    bet_pct = min(kelly_bet_pct * KELLY_FRACTION, MAX_BET_PCT)
    bet_amount = bankroll * bet_pct

    if bet_amount < 1:  # Skip tiny bets
        continue

    # Calculate profit/loss
    if is_correct:
        profit = calculate_profit(bet_amount, odds)
    else:
        profit = -bet_amount

    bankroll += profit
    bankroll_history.append(bankroll)

    bet_results.append(
        {
            "date": row["GAME_DATE"],
            "player": row["PLAYER_NAME"],
            "prediction": prediction,
            "line": line,
            "actual": actual,
            "edge": pred_over_line,
            "bet_side": bet_side,
            "odds": odds,
            "bet_amount": bet_amount,
            "profit": profit,
            "bankroll": bankroll,
            "is_correct": is_correct,
        }
    )

# ======================================================================
# 9. BETTING RESULTS
# ======================================================================

print("=" * 80)
print("BETTING RESULTS")
print("=" * 80)
print()

if len(bet_results) == 0:
    print("⚠️  No bets met the edge threshold")
    print()
else:
    bets_df = pd.DataFrame(bet_results)

    total_bets = len(bets_df)
    wins = bets_df["is_correct"].sum()
    losses = total_bets - wins
    win_rate = wins / total_bets if total_bets > 0 else 0

    total_wagered = bets_df["bet_amount"].sum()
    total_profit = bets_df["profit"].sum()
    roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0
    roi_on_bankroll = (total_profit / STARTING_BANKROLL) * 100

    final_bankroll = bankroll_history[-1]

    print(f"Total bets: {total_bets:,}")
    print(f"Wins: {wins:,} ({win_rate*100:.1f}%)")
    print(f"Losses: {losses:,}")
    print()
    print(f"Total wagered: ${total_wagered:,.2f}")
    print(f"Total profit: ${total_profit:,.2f}")
    print(f"ROI: {roi:.2f}%")
    print()
    print(f"Starting bankroll: ${STARTING_BANKROLL:,.0f}")
    print(f"Final bankroll: ${final_bankroll:,.2f}")
    print(f"Return on bankroll: {roi_on_bankroll:.2f}%")
    print()

    # Breakdown by bet side
    print("Breakdown by Bet Side:")
    side_stats = (
        bets_df.groupby("bet_side")
        .agg({"is_correct": ["sum", "count"], "profit": "sum", "bet_amount": "sum"})
        .reset_index()
    )
    side_stats.columns = ["bet_side", "wins", "bets", "profit", "wagered"]
    side_stats["win_rate"] = side_stats["wins"] / side_stats["bets"]
    side_stats["roi"] = (side_stats["profit"] / side_stats["wagered"]) * 100
    print(side_stats.to_string(index=False))
    print()

    # Monthly breakdown
    bets_df["month"] = pd.to_datetime(bets_df["date"]).dt.to_period("M")
    monthly_stats = (
        bets_df.groupby("month")
        .agg({"profit": "sum", "is_correct": ["sum", "count"], "bet_amount": "sum"})
        .reset_index()
    )
    monthly_stats.columns = ["month", "profit", "wins", "bets", "wagered"]
    monthly_stats["win_rate"] = monthly_stats["wins"] / monthly_stats["bets"]
    monthly_stats["roi"] = (monthly_stats["profit"] / monthly_stats["wagered"]) * 100

    print("Monthly Breakdown:")
    print(monthly_stats.to_string(index=False))
    print()

    # Save results
    bets_df.to_csv("data/results/backtest_ensemble_2024_25_real_odds.csv", index=False)
    print("✅ Saved betting results to data/results/backtest_ensemble_2024_25_real_odds.csv")
    print()

# ======================================================================
# 10. SAVE PREDICTIONS WITH ODDS
# ======================================================================

print("Saving predictions with odds...")
predictions_with_odds.to_csv("data/results/predictions_ensemble_2024_25_with_odds.csv", index=False)
print("✅ Saved predictions to data/results/predictions_ensemble_2024_25_with_odds.csv")
print()

# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("✅ BACKTEST COMPLETE!")
print("=" * 80)
print()
print("Summary:")
print(f"   - Prediction MAE: {mae:.2f} pts")
print(f"   - Total predictions with odds: {len(predictions_with_odds):,}")
if len(bet_results) > 0:
    print(f"   - Total bets: {total_bets:,}")
    print(f"   - Win rate: {win_rate*100:.1f}%")
    print(f"   - ROI on wagered: {roi:.2f}%")
    print(f"   - Final bankroll: ${final_bankroll:,.2f}")
    print(f"   - Profit: ${total_profit:,.2f} ({roi_on_bankroll:+.2f}%)")
print()
