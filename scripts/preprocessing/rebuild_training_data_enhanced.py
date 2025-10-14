"""
Rebuild Training Data with Enhanced Features

This adds CTG stats, L3 features, and rest/schedule to the training data.
Then we can retrain the model and see TRUE impact on MAE and win rate.

Expected Impact: MAE 8.8 → 5.5-6.5, Win Rate 51.6% → 54-56%
"""

import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import timedelta
import sys
sys.path.append('utils')
from ctg_feature_builder import CTGFeatureBuilder

print("="*80)
print("REBUILD TRAINING DATA WITH ENHANCED FEATURES")
print("="*80)

# Load CTG builder
print("\n1. Loading CTG feature builder...")
ctg_builder = CTGFeatureBuilder()

# Load original training data
print("\n2. Loading original training data...")
train_df = pd.read_parquet('data/processed/train.parquet')
print(f"✅ Loaded {len(train_df):,} training rows")
print(f"   Original features: {len(train_df.columns)}")

# Ensure GAME_DATE is datetime
train_df['GAME_DATE'] = pd.to_datetime(train_df['GAME_DATE'])

# Sort by date for temporal features
train_df = train_df.sort_values(['PLAYER_ID', 'GAME_DATE'])

print("\n3. Adding enhanced features...")
print("   This will take several minutes...")

# Initialize new feature columns
train_df['CTG_USG'] = 0.0
train_df['CTG_PSA'] = 0.0
train_df['CTG_AST_PCT'] = 0.0
train_df['CTG_TOV_PCT'] = 0.0
train_df['CTG_eFG'] = 0.0
train_df['CTG_REB_PCT'] = 0.0
train_df['CTG_Available'] = 0

train_df['PRA_L3_mean'] = 0.0
train_df['PRA_L3_std'] = 0.0
train_df['PRA_L3_trend'] = 0.0
train_df['MIN_L3_mean'] = 0.0

train_df['Days_Rest'] = 7
train_df['Is_BackToBack'] = 0
train_df['Games_Last7'] = 0

# Process in chunks by player
unique_players = train_df['PLAYER_ID'].unique()
print(f"   Processing {len(unique_players):,} unique players...")

for player_id in tqdm(unique_players, desc="Adding enhanced features"):
    # Get this player's games
    player_mask = train_df['PLAYER_ID'] == player_id
    player_games = train_df[player_mask].copy()

    if len(player_games) == 0:
        continue

    # Get player name and determine seasons
    player_name = player_games.iloc[0]['PLAYER_NAME']
    seasons = player_games['SEASON'].unique()

    # Add CTG features (same for all games in a season)
    for season in seasons:
        season_mask = (train_df['PLAYER_ID'] == player_id) & (train_df['SEASON'] == season)
        ctg_feats = ctg_builder.get_player_ctg_features(player_name, season)

        train_df.loc[season_mask, 'CTG_USG'] = ctg_feats['CTG_USG']
        train_df.loc[season_mask, 'CTG_PSA'] = ctg_feats['CTG_PSA']
        train_df.loc[season_mask, 'CTG_AST_PCT'] = ctg_feats['CTG_AST_PCT']
        train_df.loc[season_mask, 'CTG_TOV_PCT'] = ctg_feats['CTG_TOV_PCT']
        train_df.loc[season_mask, 'CTG_eFG'] = ctg_feats['CTG_eFG']
        train_df.loc[season_mask, 'CTG_REB_PCT'] = ctg_feats['CTG_REB_PCT']
        train_df.loc[season_mask, 'CTG_Available'] = ctg_feats['CTG_Available']

    # Add L3 and rest features (temporal - different for each game)
    for i, (idx, row) in enumerate(player_games.iterrows()):
        # Get games BEFORE this one
        past_games = player_games.iloc[:i]

        if len(past_games) >= 3:
            # L3 features
            last_3 = past_games.iloc[-3:]
            train_df.loc[idx, 'PRA_L3_mean'] = last_3['PRA'].mean()
            train_df.loc[idx, 'PRA_L3_std'] = last_3['PRA'].std() if len(last_3) > 1 else 0
            train_df.loc[idx, 'PRA_L3_trend'] = (last_3.iloc[-1]['PRA'] - last_3.iloc[0]['PRA']) / 2
            train_df.loc[idx, 'MIN_L3_mean'] = last_3['MIN'].mean() if 'MIN' in last_3.columns else 0

        if len(past_games) > 0:
            # Rest features
            last_game_date = past_games.iloc[-1]['GAME_DATE']
            current_date = row['GAME_DATE']
            days_rest = (current_date - last_game_date).days

            train_df.loc[idx, 'Days_Rest'] = min(days_rest, 7)
            train_df.loc[idx, 'Is_BackToBack'] = 1 if days_rest == 0 else 0

            # Games in last 7 days
            week_ago = current_date - timedelta(days=7)
            games_last_7 = len(past_games[past_games['GAME_DATE'] >= week_ago])
            train_df.loc[idx, 'Games_Last7'] = games_last_7

print(f"\n✅ Enhanced features added!")
print(f"   New feature count: {len(train_df.columns)}")
print(f"   CTG coverage: {(train_df['CTG_Available'] == 1).mean()*100:.1f}%")

# Save enhanced training data
print("\n4. Saving enhanced training data...")
output_file = 'data/processed/train_enhanced.parquet'
train_df.to_parquet(output_file, index=False)
print(f"✅ Saved to {output_file}")

# Summary
print("\n" + "="*80)
print("ENHANCED TRAINING DATA SUMMARY")
print("="*80)

print(f"""
Original Training Data:
  Rows: {len(train_df):,}
  Features: {len([c for c in train_df.columns if c not in ['PRA', 'PLAYER_NAME', 'PLAYER_ID', 'GAME_ID', 'GAME_DATE', 'SEASON']])}

Enhanced Features Added:
  ✅ CTG Stats: USG%, PSA, AST%, TOV%, eFG%, REB% (7 features)
  ✅ L3 Recent Form: Mean, Std, Trend, MIN (4 features)
  ✅ Rest/Schedule: Days rest, B2B, Games last 7 (3 features)

Total New Features: 14
CTG Coverage: {(train_df['CTG_Available'] == 1).mean()*100:.1f}%

Next Steps:
  1. Train new model on train_enhanced.parquet
  2. Run walk-forward validation with new model
  3. Expected: MAE 8.8 → 5.5-6.5 pts
  4. Expected: Win Rate 51.6% → 54-56%

Saved to: {output_file}
""")

print("="*80)
