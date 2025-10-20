#!/usr/bin/env python3
"""
Generate Oct 22, 2025 Predictions - V2 Model (Pre-Game Features Only)

Uses only pre-game features:
- Lag features (PRA_lag1, PRA_L5_mean, etc.)
- CTG season stats (USG%, PSA, etc.)
- Contextual features (Minutes_Projected, Days_Rest)

NO in-game features (FGA, MIN, FG_PCT).
"""

import pandas as pd
import numpy as np
import pickle
from datetime import datetime
from pathlib import Path

# ======================================================================
# CONFIGURATION
# ======================================================================

PREDICTION_DATE = "2025-10-22"
MODEL_PATH = "models/production_model_FIXED_V2_latest.pkl"
GAME_LOGS_PATH = "data/game_logs/all_game_logs_through_2025.csv"
GAMES_PATH = "data/upcoming/games_2025_10_22.csv"
ODDS_PATH = "data/upcoming/odds_2025_10_22.csv"
OUTPUT_PATH = "data/results/predictions_2025_10_22_FIXED_V2.csv"

print("=" * 80)
print("OCT 22, 2025 PREDICTIONS - V2 MODEL (PRE-GAME FEATURES ONLY)")
print("=" * 80)
print()

# ======================================================================
# LOAD MODEL
# ======================================================================

print("1. Loading V2 model...")
with open(MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
feature_cols = model_dict['feature_cols']

print(f"   ✅ Model loaded")
print(f"   Features: {len(feature_cols)}")
print(f"   Training MAE: {model_dict.get('train_mae', 'N/A'):.2f} pts")
print(f"   Top feature: {model_dict['feature_importance'].iloc[0]['feature']} ({model_dict['feature_importance'].iloc[0]['importance']*100:.1f}%)")
print()

# ======================================================================
# LOAD DATA
# ======================================================================

print("2. Loading historical game logs...")
df_all = pd.read_csv(GAME_LOGS_PATH)
df_all['GAME_DATE'] = pd.to_datetime(df_all['GAME_DATE'])

# Historical data (before Oct 22, 2025)
df_historical = df_all[df_all['GAME_DATE'] < PREDICTION_DATE].copy()
print(f"   ✅ Loaded {len(df_historical):,} historical games")
print(f"   Date range: {df_historical['GAME_DATE'].min()} to {df_historical['GAME_DATE'].max()}")
print()

# Add CTG features to historical data
print("2b. Loading and merging CTG features...")
import sys
sys.path.insert(0, str(Path(__file__).parent.parent.parent))
from utils.ctg_feature_builder import CTGFeatureBuilder

ctg_builder = CTGFeatureBuilder()
print(f"   ✅ CTG data loaded")

# Merge CTG features
df_historical['CTG_SEASON'] = df_historical['SEASON'].apply(lambda x: x if '-' in str(x) else f"{x[:4]}-{x[4:6]}")

# Get unique player-seasons
unique_player_seasons = df_historical[['PLAYER_NAME', 'CTG_SEASON']].drop_duplicates()
print(f"   Unique player-seasons: {len(unique_player_seasons):,}")

ctg_data = []
for i, (idx, row) in enumerate(unique_player_seasons.iterrows(), 1):
    player_name = row['PLAYER_NAME']
    season = row['CTG_SEASON']

    # Get CTG features for this player-season
    ctg_features = ctg_builder.get_player_ctg_features(player_name, season)
    if ctg_features:
        ctg_features['PLAYER_NAME'] = player_name
        ctg_features['CTG_SEASON'] = season
        ctg_data.append(ctg_features)

    if i % 500 == 0:
        print(f"      Processed {i} / {len(unique_player_seasons)} player-seasons...")

# Deduplicate CTG data BEFORE merge
ctg_df = pd.DataFrame(ctg_data)
if len(ctg_df) > 0:
    ctg_df = ctg_df.drop_duplicates(subset=['PLAYER_NAME', 'CTG_SEASON'])

print(f"   CTG records: {len(ctg_df):,}")

# Merge CTG to historical data
df_historical = df_historical.merge(
    ctg_df,
    on=['PLAYER_NAME', 'CTG_SEASON'],
    how='left'
)

# Fill missing CTG values
ctg_cols = ['CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', 'CTG_TOV_PCT', 'CTG_eFG', 'CTG_REB_PCT']
for col in ctg_cols:
    if col in df_historical.columns:
        df_historical[col] = df_historical[col].fillna(df_historical[col].median())

print(f"   ✅ CTG features merged")
print()

print("3. Loading Oct 22 games and odds...")
df_games = pd.read_csv(GAMES_PATH)
df_odds = pd.read_csv(ODDS_PATH)

print(f"   ✅ {len(df_games)} games on Oct 22")
print(f"   ✅ {len(df_odds)} prop lines available")
print()

# ======================================================================
# GENERATE PREDICTIONS
# ======================================================================

print("4. Generating predictions (pre-game features only)...")
print()

# Get unique players from odds
players_with_odds = df_odds['player_name'].unique()
print(f"   Players with odds: {len(players_with_odds)}")

predictions = []

for player_name in players_with_odds:
    # Find player in historical data
    player_past = df_historical[df_historical['PLAYER_NAME'] == player_name].copy()

    if len(player_past) == 0:
        print(f"   ⚠️  No historical data for {player_name}, skipping")
        continue

    if len(player_past) < 5:
        print(f"   ⚠️  Only {len(player_past)} games for {player_name}, skipping")
        continue

    # Sort by date
    player_past = player_past.sort_values('GAME_DATE')

    # Get player info
    player_id = player_past['PLAYER_ID'].iloc[-1]
    # Extract team from MATCHUP (e.g., "BOS vs. LAL" or "BOS @ LAL")
    matchup = player_past['MATCHUP'].iloc[-1] if 'MATCHUP' in player_past.columns else ''
    team = matchup.split()[0] if matchup else 'UNK'

    # Calculate pre-game features
    pra_values = player_past['PRA'].values
    min_values = player_past['MIN'].values if 'MIN' in player_past.columns else np.array([])

    features = {}

    # ========== LAG FEATURES ==========
    features['PRA_lag1'] = pra_values[-1] if len(pra_values) >= 1 else np.nan
    features['PRA_lag3'] = pra_values[-3] if len(pra_values) >= 3 else np.nan
    features['PRA_lag5'] = pra_values[-5] if len(pra_values) >= 5 else np.nan
    features['PRA_lag7'] = pra_values[-7] if len(pra_values) >= 7 else np.nan
    features['PRA_lag10'] = pra_values[-10] if len(pra_values) >= 10 else np.nan

    # ========== ROLLING MEANS ==========
    features['PRA_L3_mean'] = np.mean(pra_values[-3:]) if len(pra_values) >= 3 else np.nan
    features['PRA_L5_mean'] = np.mean(pra_values[-5:]) if len(pra_values) >= 5 else np.nan
    features['PRA_L10_mean'] = np.mean(pra_values[-10:]) if len(pra_values) >= 10 else np.nan
    features['PRA_L20_mean'] = np.mean(pra_values[-20:]) if len(pra_values) >= 20 else np.nan

    # ========== ROLLING STD ==========
    features['PRA_L3_std'] = np.std(pra_values[-3:]) if len(pra_values) >= 3 else np.nan
    features['PRA_L5_std'] = np.std(pra_values[-5:]) if len(pra_values) >= 5 else np.nan
    features['PRA_L10_std'] = np.std(pra_values[-10:]) if len(pra_values) >= 10 else np.nan
    features['PRA_L20_std'] = np.std(pra_values[-20:]) if len(pra_values) >= 20 else np.nan

    # ========== EWMA ==========
    if len(pra_values) >= 5:
        features['PRA_ewma5'] = pd.Series(pra_values).ewm(span=5).mean().iloc[-1]
    else:
        features['PRA_ewma5'] = np.nan

    if len(pra_values) >= 10:
        features['PRA_ewma10'] = pd.Series(pra_values).ewm(span=10).mean().iloc[-1]
    else:
        features['PRA_ewma10'] = np.nan

    if len(pra_values) >= 15:
        features['PRA_ewma15'] = pd.Series(pra_values).ewm(span=15).mean().iloc[-1]
    else:
        features['PRA_ewma15'] = np.nan

    # ========== TREND ==========
    if len(pra_values) >= 20:
        features['PRA_trend'] = np.mean(pra_values[-5:]) - np.mean(pra_values[-20:])
    else:
        features['PRA_trend'] = np.nan

    # ========== CONTEXTUAL FEATURES ==========

    # Minutes projected (L5 average)
    if len(min_values) >= 5:
        features['Minutes_Projected'] = np.mean(min_values[-5:])
    else:
        features['Minutes_Projected'] = np.nan

    # Days rest
    last_game_date = player_past['GAME_DATE'].iloc[-1]
    pred_date = pd.to_datetime(PREDICTION_DATE)
    days_rest = (pred_date - last_game_date).days
    features['Days_Rest'] = min(days_rest, 7)  # Cap at 7
    features['Is_BackToBack'] = 1 if days_rest <= 1 else 0

    # Games in last 7 days
    last_7_days = player_past[player_past['GAME_DATE'] >= pred_date - pd.Timedelta(days=7)]
    features['Games_Last7'] = len(last_7_days)

    # ========== CTG FEATURES ==========
    # Use most recent CTG stats
    ctg_cols = ['CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', 'CTG_TOV_PCT', 'CTG_eFG', 'CTG_REB_PCT']
    latest_ctg = player_past.iloc[-1]

    for col in ctg_cols:
        if col in latest_ctg and pd.notna(latest_ctg[col]):
            features[col] = latest_ctg[col]
        else:
            features[col] = np.nan

    # Create feature vector
    X_pred = pd.DataFrame([features])

    # Ensure all required features exist
    for col in feature_cols:
        if col not in X_pred.columns:
            X_pred[col] = np.nan

    X_pred = X_pred[feature_cols]

    # Fill missing values (same as training)
    X_pred = X_pred.fillna(0)

    # Make prediction
    pred_pra = model.predict(X_pred)[0]

    # Store result
    result = {
        'PLAYER_ID': player_id,
        'PLAYER_NAME': player_name,
        'TEAM': team,
        'predicted_PRA': pred_pra,
        'last_game_PRA': features['PRA_lag1'],
        'L5_mean_PRA': features['PRA_L5_mean'],
        'L10_mean_PRA': features['PRA_L10_mean'],
        'ewma10_PRA': features['PRA_ewma10'],
        'Minutes_Projected': features['Minutes_Projected'],
        'Days_Rest': features['Days_Rest'],
        'Is_BackToBack': features['Is_BackToBack'],
        'CTG_USG': features.get('CTG_USG', np.nan),
        'CTG_PSA': features.get('CTG_PSA', np.nan)
    }

    predictions.append(result)

print(f"   ✅ Generated {len(predictions)} predictions")
print()

# ======================================================================
# SAVE PREDICTIONS
# ======================================================================

print("5. Saving predictions...")

df_pred = pd.DataFrame(predictions)

# Sort by predicted PRA
df_pred = df_pred.sort_values('predicted_PRA', ascending=False)

# Save
df_pred.to_csv(OUTPUT_PATH, index=False)

print(f"   ✅ Predictions saved to {OUTPUT_PATH}")
print()

# ======================================================================
# SUMMARY
# ======================================================================

print("=" * 80)
print("PREDICTION SUMMARY")
print("=" * 80)
print()

print(f"📊 Predictions:")
print(f"   Total: {len(df_pred)}")
print(f"   Range: {df_pred['predicted_PRA'].min():.1f} to {df_pred['predicted_PRA'].max():.1f}")
print(f"   Mean: {df_pred['predicted_PRA'].mean():.1f}")
print(f"   Median: {df_pred['predicted_PRA'].median():.1f}")
print()

print(f"🔍 Quality Checks:")
print(f"   Negative predictions: {(df_pred['predicted_PRA'] < 0).sum()} ✅")
print(f"   Missing CTG_USG: {df_pred['CTG_USG'].isna().sum()}")
print(f"   Missing Minutes_Projected: {df_pred['Minutes_Projected'].isna().sum()}")
print()

print(f"🏀 Top 10 Predictions:")
print(df_pred[['PLAYER_NAME', 'TEAM', 'predicted_PRA', 'L5_mean_PRA', 'Minutes_Projected']].head(10).to_string(index=False))
print()

print("=" * 80)
print("Next step: Create betting recommendations")
print("=" * 80)
