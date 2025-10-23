#!/usr/bin/env python3
"""
Comprehensive Backtest with 4pt Threshold and Monte Carlo Simulation

Tests model on 2024-25 season with:
1. Train on data up to 2023-24 season
2. Test on 2024-25 season with 4pt edge threshold
3. Analyze over/under distribution and win rates
4. Run Monte Carlo simulation for betting strategy
"""

import pickle
from datetime import datetime
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

# ======================================================================
# CONFIGURATION
# ======================================================================

# Model path (will be trained up to 2023-24)
MODEL_PATH = "models/model_4pt_threshold_2024_25.pkl"

# Data paths
GAME_LOGS_PATH = "data/game_logs/all_game_logs_through_2025.csv"

# Output paths
OUTPUT_DIR = Path("data/results")
OUTPUT_DIR.mkdir(exist_ok=True)

PREDICTIONS_PATH = OUTPUT_DIR / "backtest_4pt_threshold_predictions.csv"
BETTING_RESULTS_PATH = OUTPUT_DIR / "backtest_4pt_threshold_betting.csv"
MONTE_CARLO_PATH = OUTPUT_DIR / "monte_carlo_simulation_4pt_threshold.csv"

# Betting strategy
MIN_EDGE = 4.0  # 4 point threshold
MAX_EDGE = 15.0  # Cap at 15 to avoid extreme predictions

# Monte Carlo parameters
NUM_SIMULATIONS = 10000
BANKROLL = 10000  # Starting bankroll
BET_SIZE = 100  # Fixed bet size per game
ODDS = -110  # Standard American odds

print("=" * 80)
print("COMPREHENSIVE BACKTEST: 4PT THRESHOLD + MONTE CARLO")
print("=" * 80)
print()
print(f"Configuration:")
print(f"  Min Edge: {MIN_EDGE} pts")
print(f"  Max Edge: {MAX_EDGE} pts")
print(f"  Monte Carlo Simulations: {NUM_SIMULATIONS:,}")
print(f"  Starting Bankroll: ${BANKROLL:,}")
print(f"  Bet Size: ${BET_SIZE}")
print()

# ======================================================================
# STEP 1: TRAIN MODEL ON DATA UP TO 2023-24
# ======================================================================

print("=" * 80)
print("STEP 1: TRAIN MODEL (2003-2024)")
print("=" * 80)
print()

print("Loading game logs...")
df_all = pd.read_csv(GAME_LOGS_PATH)
df_all["GAME_DATE"] = pd.to_datetime(df_all["GAME_DATE"])

# Split data
train_data = df_all[df_all["GAME_DATE"] < "2024-10-22"].copy()
test_data = df_all[
    (df_all["GAME_DATE"] >= "2024-10-22") & (df_all["GAME_DATE"] <= "2025-04-30")
].copy()

print(
    f"Training data: {len(train_data):,} games ({train_data['GAME_DATE'].min()} to {train_data['GAME_DATE'].max()})"
)
print(
    f"Test data: {len(test_data):,} games ({test_data['GAME_DATE'].min()} to {test_data['GAME_DATE'].max()})"
)
print()

# Feature engineering on training data
print("Engineering features on training data...")

# Sort by player and date
train_data = train_data.sort_values(["PLAYER_ID", "GAME_DATE"])

# Create lag features
for player_id, group in train_data.groupby("PLAYER_ID"):
    idx = group.index
    pra = group["PRA"].values

    # Lag features (shifted by 1 to prevent leakage)
    train_data.loc[idx, "PRA_lag1"] = np.roll(pra, 1)
    train_data.loc[idx, "PRA_lag3"] = np.roll(pra, 3)
    train_data.loc[idx, "PRA_lag5"] = np.roll(pra, 5)

    # Rolling means (using last N games, excluding current)
    train_data.loc[idx, "PRA_L5_mean"] = (
        pd.Series(pra).shift(1).rolling(5, min_periods=1).mean().values
    )
    train_data.loc[idx, "PRA_L10_mean"] = (
        pd.Series(pra).shift(1).rolling(10, min_periods=1).mean().values
    )
    train_data.loc[idx, "PRA_L20_mean"] = (
        pd.Series(pra).shift(1).rolling(20, min_periods=1).mean().values
    )

    # Rolling std
    train_data.loc[idx, "PRA_L5_std"] = (
        pd.Series(pra).shift(1).rolling(5, min_periods=1).std().values
    )
    train_data.loc[idx, "PRA_L10_std"] = (
        pd.Series(pra).shift(1).rolling(10, min_periods=1).std().values
    )

    # EWMA
    train_data.loc[idx, "PRA_ewma5"] = (
        pd.Series(pra).shift(1).ewm(span=5, min_periods=1).mean().values
    )
    train_data.loc[idx, "PRA_ewma10"] = (
        pd.Series(pra).shift(1).ewm(span=10, min_periods=1).mean().values
    )

    # Trend
    l5_mean = pd.Series(pra).shift(1).rolling(5, min_periods=1).mean().values
    l20_mean = pd.Series(pra).shift(1).rolling(20, min_periods=1).mean().values
    train_data.loc[idx, "PRA_trend"] = l5_mean - l20_mean

# Minutes projected (L5 average)
if "MIN" in train_data.columns:
    for player_id, group in train_data.groupby("PLAYER_ID"):
        idx = group.index
        minutes = group["MIN"].values
        train_data.loc[idx, "Minutes_Projected"] = (
            pd.Series(minutes).shift(1).rolling(5, min_periods=1).mean().values
        )

# Rest days
for player_id, group in train_data.groupby("PLAYER_ID"):
    idx = group.index
    dates = group["GAME_DATE"].values

    days_rest = []
    for i in range(len(dates)):
        if i == 0:
            days_rest.append(7)  # Default for first game
        else:
            rest = int((dates[i] - dates[i - 1]) / np.timedelta64(1, "D"))
            days_rest.append(min(rest, 7))  # Cap at 7

    train_data.loc[idx, "Days_Rest"] = days_rest
    train_data.loc[idx, "Is_BackToBack"] = [1 if d <= 1 else 0 for d in days_rest]

print(f"Features engineered: {train_data.shape[1]} columns")
print()

# Define feature columns
feature_cols = [
    "PRA_lag1",
    "PRA_lag3",
    "PRA_lag5",
    "PRA_L5_mean",
    "PRA_L10_mean",
    "PRA_L20_mean",
    "PRA_L5_std",
    "PRA_L10_std",
    "PRA_ewma5",
    "PRA_ewma10",
    "PRA_trend",
    "Minutes_Projected",
    "Days_Rest",
    "Is_BackToBack",
]

# Filter to rows with enough history (at least 5 games)
train_data = train_data.dropna(subset=["PRA_lag5"])

print(f"Training samples after filtering: {len(train_data):,}")
print()

# Prepare training data
X_train = train_data[feature_cols].fillna(0)
y_train = train_data["PRA"]

# Train XGBoost model
print("Training XGBoost model...")
from xgboost import XGBRegressor

model = XGBRegressor(
    n_estimators=200,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model.fit(X_train, y_train)

train_preds = model.predict(X_train)
train_mae = np.mean(np.abs(train_preds - y_train))

print(f"✅ Model trained")
print(f"   Training MAE: {train_mae:.2f} pts")
print()

# Save model
model_dict = {
    "model": model,
    "feature_cols": feature_cols,
    "train_mae": train_mae,
    "train_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

with open(MODEL_PATH, "wb") as f:
    pickle.dump(model_dict, f)

print(f"✅ Model saved to {MODEL_PATH}")
print()

# ======================================================================
# STEP 2: WALK-FORWARD VALIDATION ON 2024-25 SEASON
# ======================================================================

print("=" * 80)
print("STEP 2: WALK-FORWARD VALIDATION (2024-25)")
print("=" * 80)
print()

unique_dates = sorted(test_data["GAME_DATE"].unique())
all_predictions = []

for i, pred_date in enumerate(unique_dates, 1):
    games_today = test_data[test_data["GAME_DATE"] == pred_date].copy()

    if len(games_today) == 0:
        continue

    # Historical data up to (but not including) today
    past_games = df_all[df_all["GAME_DATE"] < pred_date].copy()

    # For each player in today's games
    for _, game in games_today.iterrows():
        player_id = game["PLAYER_ID"]
        player_name = game["PLAYER_NAME"]

        # Get player's past games
        player_past = past_games[past_games["PLAYER_ID"] == player_id].sort_values("GAME_DATE")

        if len(player_past) < 5:
            continue

        # Calculate features
        pra_values = player_past["PRA"].values

        features = {}
        features["PRA_lag1"] = pra_values[-1] if len(pra_values) >= 1 else 0
        features["PRA_lag3"] = pra_values[-3] if len(pra_values) >= 3 else 0
        features["PRA_lag5"] = pra_values[-5] if len(pra_values) >= 5 else 0

        features["PRA_L5_mean"] = np.mean(pra_values[-5:]) if len(pra_values) >= 5 else 0
        features["PRA_L10_mean"] = np.mean(pra_values[-10:]) if len(pra_values) >= 10 else 0
        features["PRA_L20_mean"] = np.mean(pra_values[-20:]) if len(pra_values) >= 20 else 0

        features["PRA_L5_std"] = np.std(pra_values[-5:]) if len(pra_values) >= 5 else 0
        features["PRA_L10_std"] = np.std(pra_values[-10:]) if len(pra_values) >= 10 else 0

        if len(pra_values) >= 5:
            features["PRA_ewma5"] = pd.Series(pra_values).ewm(span=5).mean().iloc[-1]
        else:
            features["PRA_ewma5"] = 0

        if len(pra_values) >= 10:
            features["PRA_ewma10"] = pd.Series(pra_values).ewm(span=10).mean().iloc[-1]
        else:
            features["PRA_ewma10"] = 0

        if len(pra_values) >= 20:
            features["PRA_trend"] = np.mean(pra_values[-5:]) - np.mean(pra_values[-20:])
        else:
            features["PRA_trend"] = 0

        # Minutes projected
        if "MIN" in player_past.columns and len(player_past) >= 5:
            features["Minutes_Projected"] = player_past["MIN"].iloc[-5:].mean()
        else:
            features["Minutes_Projected"] = 0

        # Rest days
        if len(player_past) >= 1:
            days_rest = (pred_date - player_past["GAME_DATE"].iloc[-1]).days
            features["Days_Rest"] = min(days_rest, 7)
            features["Is_BackToBack"] = 1 if days_rest <= 1 else 0
        else:
            features["Days_Rest"] = 7
            features["Is_BackToBack"] = 0

        # Create feature vector
        X_pred = pd.DataFrame([features])[feature_cols]

        # Make prediction
        pred_pra = model.predict(X_pred)[0]
        actual_pra = game["PRA"]

        result = {
            "GAME_DATE": pred_date,
            "PLAYER_ID": player_id,
            "PLAYER_NAME": player_name,
            "TEAM": game.get("TEAM_ABBREVIATION", ""),
            "MATCHUP": game.get("MATCHUP", ""),
            "predicted_PRA": pred_pra,
            "actual_PRA": actual_pra,
            "error": abs(pred_pra - actual_pra),
            "L5_mean": features["PRA_L5_mean"],
        }

        all_predictions.append(result)

    if (i % 10) == 0 or i == len(unique_dates):
        print(f"   Processed {i}/{len(unique_dates)} dates ({len(all_predictions):,} predictions)")

print()
print(f"✅ Walk-forward validation complete")
print(f"   Total predictions: {len(all_predictions):,}")
print()

df_pred = pd.DataFrame(all_predictions)

# Overall MAE
mae = df_pred["error"].mean()
print(f"📊 Overall MAE: {mae:.2f} pts")
print()

# Save predictions
df_pred.to_csv(PREDICTIONS_PATH, index=False)
print(f"✅ Predictions saved to {PREDICTIONS_PATH}")
print()

# ======================================================================
# STEP 3: LOAD REAL DRAFTKINGS ODDS AND MERGE
# ======================================================================

print("=" * 80)
print("STEP 3: LOAD REAL DRAFTKINGS ODDS")
print("=" * 80)
print()

# Load historical DraftKings odds
ODDS_PATH = "data/historical_odds/2024-25/pra_odds.csv"
print(f"Loading odds from: {ODDS_PATH}")
df_odds = pd.read_csv(ODDS_PATH)
df_odds["event_date"] = pd.to_datetime(df_odds["event_date"])

print(f"Loaded {len(df_odds):,} DraftKings odds lines")
print()

# Merge predictions with real odds
print("Merging predictions with DraftKings odds...")
df_pred_with_odds = df_pred.merge(
    df_odds[["player_name", "event_date", "line"]],
    left_on=["PLAYER_NAME", "GAME_DATE"],
    right_on=["player_name", "event_date"],
    how="inner",
)

print(f"Matched {len(df_pred_with_odds):,} predictions to DraftKings odds")
print(f"Unmatched predictions: {len(df_pred) - len(df_pred_with_odds):,}")
print()

# Calculate edge using REAL odds
df_pred_with_odds["edge"] = df_pred_with_odds["predicted_PRA"] - df_pred_with_odds["line"]
df_pred_with_odds["abs_edge"] = df_pred_with_odds["edge"].abs()

# ======================================================================
# STEP 4: SIMULATE BETTING WITH 4PT THRESHOLD
# ======================================================================

print("=" * 80)
print("STEP 4: BETTING SIMULATION (4PT THRESHOLD + REAL ODDS)")
print("=" * 80)
print()

# Filter bets with 4pt+ edge
bets = df_pred_with_odds[
    (df_pred_with_odds["abs_edge"] >= MIN_EDGE) & (df_pred_with_odds["abs_edge"] <= MAX_EDGE)
].copy()

print(f"Total predictions with odds: {len(df_pred_with_odds):,}")
print(f"Bets with {MIN_EDGE}+ pt edge: {len(bets):,}")
print()

# Determine bet type
bets["bet_type"] = bets["edge"].apply(lambda x: "OVER" if x > 0 else "UNDER")

# Determine bet outcome
bets["bet_correct"] = ((bets["bet_type"] == "OVER") & (bets["actual_PRA"] > bets["line"])) | (
    (bets["bet_type"] == "UNDER") & (bets["actual_PRA"] < bets["line"])
)

# Calculate win rate
total_bets = len(bets)
wins = bets["bet_correct"].sum()
losses = total_bets - wins
win_rate = wins / total_bets if total_bets > 0 else 0

print("=" * 80)
print("BETTING RESULTS")
print("=" * 80)
print()
print(f"Total Bets: {total_bets:,}")
print(f"Wins: {wins:,}")
print(f"Losses: {losses:,}")
print(f"Win Rate: {win_rate*100:.2f}%")
print()

# Over/Under distribution
over_bets = bets[bets["bet_type"] == "OVER"]
under_bets = bets[bets["bet_type"] == "UNDER"]

over_wins = over_bets["bet_correct"].sum()
under_wins = under_bets["bet_correct"].sum()

over_win_rate = over_wins / len(over_bets) if len(over_bets) > 0 else 0
under_win_rate = under_wins / len(under_bets) if len(under_bets) > 0 else 0

print("Over/Under Distribution:")
print(f"  OVER bets: {len(over_bets):,} ({len(over_bets)/total_bets*100:.1f}%)")
print(f"    Wins: {over_wins:,}")
print(f"    Win Rate: {over_win_rate*100:.2f}%")
print()
print(f"  UNDER bets: {len(under_bets):,} ({len(under_bets)/total_bets*100:.1f}%)")
print(f"    Wins: {under_wins:,}")
print(f"    Win Rate: {under_win_rate*100:.2f}%")
print()

# ROI calculation
profit_per_win = BET_SIZE * (100 / 110)  # $90.91 per $100 bet at -110 odds
loss_per_loss = -BET_SIZE
total_profit = (wins * profit_per_win) + (losses * loss_per_loss)
total_wagered = total_bets * BET_SIZE
roi = (total_profit / total_wagered) * 100 if total_wagered > 0 else 0

print(f"Financial Performance:")
print(f"  Total Wagered: ${total_wagered:,.2f}")
print(f"  Total Profit: ${total_profit:,.2f}")
print(f"  ROI: {roi:.2f}%")
print()

# Save betting results
bets.to_csv(BETTING_RESULTS_PATH, index=False)
print(f"✅ Betting results saved to {BETTING_RESULTS_PATH}")
print()

# ======================================================================
# STEP 5: MONTE CARLO SIMULATION
# ======================================================================

print("=" * 80)
print("STEP 5: MONTE CARLO SIMULATION")
print("=" * 80)
print()

print(f"Running {NUM_SIMULATIONS:,} simulations...")
print(f"Starting Bankroll: ${BANKROLL:,}")
print(f"Bet Size: ${BET_SIZE}")
print()

# Create bet outcome array (1 for win, 0 for loss)
bet_outcomes = bets["bet_correct"].values.astype(int)

simulation_results = []

for sim in range(NUM_SIMULATIONS):
    # Shuffle bet order to simulate different sequences
    shuffled_outcomes = np.random.permutation(bet_outcomes)

    # Simulate betting
    bankroll = BANKROLL
    max_bankroll = BANKROLL
    min_bankroll = BANKROLL

    for outcome in shuffled_outcomes:
        if outcome == 1:
            bankroll += profit_per_win
        else:
            bankroll += loss_per_loss

        max_bankroll = max(max_bankroll, bankroll)
        min_bankroll = min(min_bankroll, bankroll)

        # Stop if bankrupt
        if bankroll <= 0:
            break

    final_profit = bankroll - BANKROLL
    roi_sim = (final_profit / BANKROLL) * 100

    simulation_results.append(
        {
            "simulation": sim + 1,
            "final_bankroll": bankroll,
            "final_profit": final_profit,
            "roi": roi_sim,
            "max_bankroll": max_bankroll,
            "min_bankroll": min_bankroll,
            "went_bankrupt": 1 if bankroll <= 0 else 0,
        }
    )

    if (sim + 1) % 1000 == 0:
        print(f"   Completed {sim + 1:,}/{NUM_SIMULATIONS:,} simulations")

print()

df_sim = pd.DataFrame(simulation_results)

# Calculate statistics
mean_profit = df_sim["final_profit"].mean()
median_profit = df_sim["final_profit"].median()
std_profit = df_sim["final_profit"].std()
min_profit = df_sim["final_profit"].min()
max_profit = df_sim["final_profit"].max()

mean_roi = df_sim["roi"].mean()
median_roi = df_sim["roi"].median()

bankruptcy_rate = df_sim["went_bankrupt"].mean()

# Percentiles
p5 = df_sim["final_profit"].quantile(0.05)
p25 = df_sim["final_profit"].quantile(0.25)
p75 = df_sim["final_profit"].quantile(0.75)
p95 = df_sim["final_profit"].quantile(0.95)

print("=" * 80)
print("MONTE CARLO RESULTS")
print("=" * 80)
print()
print(f"Simulations Run: {NUM_SIMULATIONS:,}")
print()
print(f"Final Profit Statistics:")
print(f"  Mean: ${mean_profit:,.2f}")
print(f"  Median: ${median_profit:,.2f}")
print(f"  Std Dev: ${std_profit:,.2f}")
print(f"  Min: ${min_profit:,.2f}")
print(f"  Max: ${max_profit:,.2f}")
print()
print(f"Percentiles:")
print(f"  5th: ${p5:,.2f}")
print(f"  25th: ${p25:,.2f}")
print(f"  75th: ${p75:,.2f}")
print(f"  95th: ${p95:,.2f}")
print()
print(f"ROI Statistics:")
print(f"  Mean: {mean_roi:.2f}%")
print(f"  Median: {median_roi:.2f}%")
print()
print(f"Risk Metrics:")
print(f"  Bankruptcy Rate: {bankruptcy_rate*100:.2f}%")
print(f"  Probability of Profit: {(df_sim['final_profit'] > 0).mean()*100:.2f}%")
print()

# Save simulation results
df_sim.to_csv(MONTE_CARLO_PATH, index=False)
print(f"✅ Monte Carlo results saved to {MONTE_CARLO_PATH}")
print()

# ======================================================================
# SUMMARY
# ======================================================================

print("=" * 80)
print("FINAL SUMMARY")
print("=" * 80)
print()
print(f"📊 Model Performance:")
print(f"   Training MAE: {train_mae:.2f} pts")
print(f"   Test MAE: {mae:.2f} pts")
print()
print(f"💰 Betting Performance (4pt threshold):")
print(f"   Total Bets: {total_bets:,}")
print(f"   Win Rate: {win_rate*100:.2f}%")
print(f"   ROI: {roi:.2f}%")
print()
print(f"   Over Bets: {len(over_bets):,} ({over_win_rate*100:.2f}% win rate)")
print(f"   Under Bets: {len(under_bets):,} ({under_win_rate*100:.2f}% win rate)")
print()
print(f"🎲 Monte Carlo Simulation:")
print(f"   Expected Profit: ${mean_profit:,.2f}")
print(f"   Expected ROI: {mean_roi:.2f}%")
print(f"   Probability of Profit: {(df_sim['final_profit'] > 0).mean()*100:.2f}%")
print(f"   Bankruptcy Risk: {bankruptcy_rate*100:.2f}%")
print()
print("=" * 80)
