"""
Walk-Forward Validation for 2023-24 Season

Re-validates 2023-24 to check if the 79.66% win rate was due to temporal leakage.
Expected result: ~52% win rate (matching 2024-25 walk-forward results).

This validates that val.parquet had temporal leakage in its lag features.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import xgboost as xgb
from sklearn.metrics import mean_absolute_error
from tqdm import tqdm
import pickle
from datetime import datetime, timedelta

print("="*80)
print("WALK-FORWARD VALIDATION - 2023-24 SEASON")
print("="*80)

# Load historical training data (2003-2023 ONLY, excluding 2023-24)
print("\n1. Loading historical training data (2003-2023)...")
train_df = pd.read_parquet('data/processed/train.parquet')

print(f"✅ Loaded historical data:")
print(f"   Train: {len(train_df):,} rows")
print(f"   Season range: {train_df['SEASON'].min()} to {train_df['SEASON'].max()}")

# Load 2023-24 RAW game logs (without lag features)
print("\n2. Loading 2023-24 raw game logs...")
raw_gamelogs = pd.read_csv('data/game_logs/game_logs_2023-24.csv')
raw_gamelogs['GAME_DATE'] = pd.to_datetime(raw_gamelogs['GAME_DATE'])
raw_gamelogs = raw_gamelogs.sort_values('GAME_DATE')

# Add PRA if not present
if 'PRA' not in raw_gamelogs.columns:
    raw_gamelogs['PRA'] = raw_gamelogs['PTS'] + raw_gamelogs['REB'] + raw_gamelogs['AST']

print(f"✅ Loaded {len(raw_gamelogs):,} raw game logs")
print(f"   Date range: {raw_gamelogs['GAME_DATE'].min()} to {raw_gamelogs['GAME_DATE'].max()}")
print(f"   Unique dates: {raw_gamelogs['GAME_DATE'].nunique()}")

# Define features (exclude target and metadata)
exclude_cols = ['PRA', 'PTS', 'REB', 'AST', 'PLAYER_NAME', 'PLAYER_ID', 'GAME_ID',
                'GAME_DATE', 'SEASON', 'SEASON_TYPE', 'TEAM_ABBREVIATION', 'OPPONENT',
                'MATCHUP', 'WL', 'VIDEO_AVAILABLE', 'SEASON_ID']

core_stats = ['MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT',
              'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF', 'PLUS_MINUS']

# Train model on historical data
print("\n3. Training XGBoost model on historical data (2003-2023)...")

# Get lag features from training data
lag_features = [col for col in train_df.columns if any(x in col for x in ['_lag', '_L5_', '_L10_', '_L20_', '_ewma', '_trend'])]
feature_cols = core_stats + lag_features
feature_cols = [col for col in feature_cols if col in train_df.columns and col not in exclude_cols]

print(f"   Using {len(feature_cols)} features")

X_train = train_df[feature_cols].fillna(0)
y_train = train_df['PRA']

model = xgb.XGBRegressor(
    n_estimators=500,
    max_depth=6,
    learning_rate=0.05,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
    verbosity=0
)

model.fit(X_train, y_train, verbose=False)

# Calculate training MAE
train_mae = mean_absolute_error(y_train, model.predict(X_train))
print(f"✅ Model trained (Training MAE: {train_mae:.2f})")

# Walk-forward validation
print("\n4. Running walk-forward validation...")
print("   For each date, only using games BEFORE that date for lag features")
print("   This ensures NO temporal leakage\n")

# Get unique dates in 2023-24
unique_dates = sorted(raw_gamelogs['GAME_DATE'].unique())
print(f"   Total prediction dates: {len(unique_dates)}")

# Initialize storage
all_predictions = []

# Helper function to calculate lag features for a player up to a specific date
def calculate_lag_features(player_history, lags=[1, 3, 5, 7, 10], windows=[5, 10, 20]):
    """Calculate lag features using only historical games"""
    features = {}

    if len(player_history) == 0:
        return features

    # Sort by date (most recent first for lag calculation)
    history = player_history.sort_values('GAME_DATE', ascending=False)

    # Lag features
    for lag in lags:
        if len(history) >= lag:
            features[f'PRA_lag{lag}'] = history.iloc[lag-1]['PRA']
        else:
            features[f'PRA_lag{lag}'] = 0

    # Rolling averages
    for window in windows:
        if len(history) >= window:
            features[f'PRA_L{window}_mean'] = history.iloc[:window]['PRA'].mean()
            features[f'PRA_L{window}_std'] = history.iloc[:window]['PRA'].std()
        else:
            features[f'PRA_L{window}_mean'] = 0
            features[f'PRA_L{window}_std'] = 0

    # EWMA
    for span in [5, 10, 15]:
        if len(history) >= span:
            features[f'PRA_ewma{span}'] = history['PRA'].ewm(span=span).mean().iloc[0]
        else:
            features[f'PRA_ewma{span}'] = 0

    return features

# Walk forward through each date
for i, pred_date in enumerate(tqdm(unique_dates, desc="Walk-forward validation")):
    # Get games on this date (to predict)
    games_today = raw_gamelogs[raw_gamelogs['GAME_DATE'] == pred_date].copy()

    if len(games_today) == 0:
        continue

    # Get all games BEFORE this date (for feature calculation)
    past_games = raw_gamelogs[raw_gamelogs['GAME_DATE'] < pred_date].copy()

    # For each game today, calculate features using only past data
    for idx, row in games_today.iterrows():
        player_id = row['PLAYER_ID']

        # Get this player's history BEFORE today
        player_history = past_games[past_games['PLAYER_ID'] == player_id]

        # Skip if player has insufficient history (< 5 games)
        if len(player_history) < 5:
            continue

        # Calculate lag features using only past games
        lag_feats = calculate_lag_features(player_history)

        # Core stats (from current game's context, not the outcome)
        # Use last game's stats as proxy for expected usage
        if len(player_history) > 0:
            last_game = player_history.iloc[-1]  # Most recent game
            core_feats = {stat: last_game.get(stat, 0) for stat in core_stats if stat in last_game}
        else:
            core_feats = {stat: 0 for stat in core_stats}

        # Combine features
        features = {**core_feats, **lag_feats}

        # Create feature vector (match training feature order)
        feature_vector = []
        for col in feature_cols:
            feature_vector.append(features.get(col, 0))

        # Make prediction
        pred_pra = model.predict([feature_vector])[0]

        # Store prediction
        all_predictions.append({
            'PLAYER_NAME': row['PLAYER_NAME'],
            'PLAYER_ID': player_id,
            'GAME_ID': row.get('GAME_ID', ''),
            'GAME_DATE': pred_date,
            'PRA': row['PRA'],
            'predicted_PRA': pred_pra,
            'error': pred_pra - row['PRA'],
            'abs_error': abs(pred_pra - row['PRA']),
            'games_in_history': len(player_history)
        })

# Create predictions dataframe
predictions_df = pd.DataFrame(all_predictions)

print(f"\n✅ Walk-forward validation complete!")
print(f"   Total predictions: {len(predictions_df):,}")
print(f"   Date range: {predictions_df['GAME_DATE'].min()} to {predictions_df['GAME_DATE'].max()}")

# Calculate prediction accuracy
mae = mean_absolute_error(predictions_df['PRA'], predictions_df['predicted_PRA'])
print(f"\n5. Prediction Accuracy (NO TEMPORAL LEAKAGE):")
print(f"   MAE: {mae:.2f} points")
print(f"   Within ±3 pts: {(predictions_df['abs_error'] <= 3).mean()*100:.1f}%")
print(f"   Within ±5 pts: {(predictions_df['abs_error'] <= 5).mean()*100:.1f}%")
print(f"   Within ±10 pts: {(predictions_df['abs_error'] <= 10).mean()*100:.1f}%")

# Save predictions
output_file = 'data/results/walkforward_predictions_2023-24.csv'
predictions_df.to_csv(output_file, index=False)
print(f"\n✅ Saved walk-forward predictions to {output_file}")

print("\n" + "="*80)
print("WALK-FORWARD VALIDATION SUMMARY")
print("="*80)

print(f"""
Validation Approach:
  ✅ Temporal isolation: Each prediction uses ONLY past games
  ✅ No future information in lag features
  ✅ Proper time-series validation
  ✅ Trained on 2003-2023 ONLY (excluding 2023-24)

Results:
  Total predictions: {len(predictions_df):,}
  MAE: {mae:.2f} points
  Prediction accuracy: {(predictions_df['abs_error'] <= 5).mean()*100:.1f}% within ±5 pts

Comparison:
  Original 2023-24 (val.parquet):  4.82 MAE, 79.66% win rate
  Walk-forward 2023-24 (this run): {mae:.2f} MAE, TBD win rate
  Walk-forward 2024-25:            8.83 MAE, 51.98% win rate

Next Steps:
  1. Match these predictions to betting odds (data/historical_odds/2023-24/pra_odds.csv)
  2. Calculate edge using new edge_calculator.py
  3. Run backtest to get TRUE out-of-sample performance
  4. Expected result: ~52% win rate (confirming temporal leakage in val.parquet)
""")

print("="*80)
