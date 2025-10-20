#!/usr/bin/env python3
"""
Quick Phase 1 Validation - Uses Test Set from Training

Since we already filtered 2024-25 data during Phase 1 training (test_df),
we can directly use those predictions for validation instead of recomputing features.

This is MUCH faster than walk-forward (seconds vs hours).
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error

print("=" * 80)
print("QUICK PHASE 1 VALIDATION - 2024-25 TEST SET")
print("=" * 80)

# Load Phase 1 model
print("\n1. Loading Phase 1 model...")
with open('models/production_model_PHASE1_latest.pkl', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
feature_cols = model_dict['feature_cols']

print(f"   ✅ Model loaded ({len(feature_cols)} features)")
print(f"   Train MAE: {model_dict['train_mae']:.2f}")
print(f"   Val MAE: {model_dict['val_mae']:.2f}")
print(f"   Test MAE: {model_dict['test_mae']:.2f}")

# Load all processed data (includes 2024-25)
print("\n2. Loading processed game data...")
df = pd.read_csv('data/game_logs/all_game_logs_through_2025.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

if 'PRA' not in df.columns:
    df['PRA'] = df['PTS'] + df['REB'] + df['AST']

print(f"   ✅ Loaded {len(df):,} games")

# Filter to 2024-25 season (test period)
test_df = df[df['GAME_DATE'] > '2024-06-30'].copy()

print(f"\n3. 2024-25 test data:")
print(f"   Games: {len(test_df):,}")
print(f"   Date range: {test_df['GAME_DATE'].min()} to {test_df['GAME_DATE'].max()}")
print(f"   Players: {test_df['PLAYER_ID'].nunique()}")

# Calculate features (simplified - just what we need for test)
print("\n4. Calculating features for test set...")

# Note: This uses the SAME feature calculation as training,
# but only on 2024-25 data. Features are calculated using .shift(1)
# so they only use past games (no leakage).

# Import feature calculation from training script
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

from src.features.advanced_stats import AdvancedStatsFeatures
from src.features.consistency_features import ConsistencyFeatures
from src.features.recent_form_features import RecentFormFeatures
from utils.ctg_feature_builder import CTGFeatureBuilder

# Sort by player and date
df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

# Add SEASON if needed
if 'SEASON' not in df.columns:
    df['SEASON'] = df['GAME_DATE'].apply(lambda x: f"{x.year}-{str(x.year+1)[-2:]}")

# Calculate baseline lag features
print("   Calculating baseline lag features...")
player_groups = df.groupby('PLAYER_ID')

for lag in [1, 3, 5, 7, 10]:
    df[f'PRA_lag{lag}'] = player_groups['PRA'].shift(lag)

for window in [3, 5, 10, 20]:
    df[f'PRA_L{window}_mean'] = player_groups['PRA'].shift(1).rolling(window=window, min_periods=1).mean()
    df[f'PRA_L{window}_std'] = player_groups['PRA'].shift(1).rolling(window=window, min_periods=2).std()

for span in [5, 10, 15]:
    df[f'PRA_ewma{span}'] = player_groups['PRA'].shift(1).ewm(span=span, min_periods=1).mean()

df['PRA_trend_L5_L20'] = df['PRA_L5_mean'] - df['PRA_L20_mean']

# Contextual features
df['Minutes_Projected'] = player_groups['MIN'].shift(1).rolling(5, min_periods=1).mean()
df['Days_Since_Last_Game'] = df.groupby('PLAYER_ID')['GAME_DATE'].diff().dt.days
df['Days_Rest'] = df['Days_Since_Last_Game'].fillna(7).clip(upper=7)
df['Is_BackToBack'] = (df['Days_Rest'] <= 1).astype(int)

def calculate_games_last_7(group):
    result = []
    for i, current_date in enumerate(group['GAME_DATE']):
        past_7_days = (group['GAME_DATE'] < current_date) & (group['GAME_DATE'] >= current_date - pd.Timedelta(days=7))
        result.append(past_7_days.sum())
    return pd.Series(result, index=group.index)

df['Games_Last7'] = df.groupby('PLAYER_ID').apply(calculate_games_last_7).reset_index(level=0, drop=True).fillna(0).astype(int)

# CTG features
print("   Adding CTG features...")
ctg_builder = CTGFeatureBuilder()
df['CTG_SEASON'] = df['SEASON'].apply(lambda x: x if '-' in str(x) else f"{x[:4]}-{x[4:6]}")

unique_player_seasons = df[['PLAYER_NAME', 'CTG_SEASON']].drop_duplicates()
ctg_data = []
for _, row in unique_player_seasons.iterrows():
    ctg_feats = ctg_builder.get_player_ctg_features(row['PLAYER_NAME'], row['CTG_SEASON'])
    ctg_feats['PLAYER_NAME'] = row['PLAYER_NAME']
    ctg_feats['CTG_SEASON'] = row['CTG_SEASON']
    ctg_data.append(ctg_feats)

ctg_df = pd.DataFrame(ctg_data)
df = df.merge(ctg_df, on=['PLAYER_NAME', 'CTG_SEASON'], how='left')

ctg_defaults = {'CTG_USG': 0.20, 'CTG_PSA': 1.10, 'CTG_AST_PCT': 0.12, 'CTG_TOV_PCT': 0.12, 'CTG_eFG': 0.53, 'CTG_REB_PCT': 0.10}
for col, default_val in ctg_defaults.items():
    df[col] = df[col].fillna(default_val)

# Phase 1 features
print("   Calculating Phase 1 features (this may take a few minutes)...")
advanced_stats = AdvancedStatsFeatures()
consistency_features = ConsistencyFeatures()
recent_form = RecentFormFeatures()

df = advanced_stats.add_all_features(df)
df = consistency_features.add_all_features(df)
df = recent_form.add_all_features(df)

print("   ✅ All features calculated")

# Extract 2024-25 test set with features
test_df_with_features = df[df['GAME_DATE'] > '2024-06-30'].copy()

# Fill missing features
for col in feature_cols:
    if col not in test_df_with_features.columns:
        test_df_with_features[col] = 0
    else:
        test_df_with_features[col] = test_df_with_features[col].fillna(0)

print(f"\n5. Making predictions on test set...")

X_test = test_df_with_features[feature_cols]
y_test = test_df_with_features['PRA']

# Make predictions
predictions = model.predict(X_test)

# Store results
results_df = pd.DataFrame({
    'PLAYER_NAME': test_df_with_features['PLAYER_NAME'],
    'PLAYER_ID': test_df_with_features['PLAYER_ID'],
    'GAME_DATE': test_df_with_features['GAME_DATE'],
    'actual_pra': y_test,
    'predicted_pra': predictions,
    'error': np.abs(predictions - y_test),
    'residual': predictions - y_test
})

# Calculate metrics
mae = mean_absolute_error(y_test, predictions)

print(f"   ✅ Predictions complete")
print(f"\n6. Phase 1 Model Performance on 2024-25:")
print(f"   Games: {len(results_df):,}")
print(f"   MAE: {mae:.2f} points")
print(f"   Within ±3 pts: {(results_df['error'] <= 3).mean()*100:.1f}%")
print(f"   Within ±5 pts: {(results_df['error'] <= 5).mean()*100:.1f}%")
print(f"   Within ±10 pts: {(results_df['error'] <= 10).mean()*100:.1f}%")
print(f"\n   Mean residual (bias): {results_df['residual'].mean():+.2f} pts")
print(f"   Std residual: {results_df['residual'].std():.2f} pts")

# Save results
output_file = 'data/results/phase1_test_predictions_2024_25.csv'
results_df.to_csv(output_file, index=False)

print(f"\n✅ Saved predictions to {output_file}")

# Comparison
print("\n" + "=" * 80)
print("COMPARISON TO BASELINE")
print("=" * 80)

baseline_mae = 8.83
improvement = ((baseline_mae - mae) / baseline_mae) * 100

print(f"""
Baseline (FIXED_V2):
  - MAE: {baseline_mae:.2f} points
  - Win Rate: 52.94%

Phase 1:
  - MAE: {mae:.2f} points ({improvement:+.1f}% improvement)
  - Win Rate: TBD (needs betting odds)

Next Step: Match to betting odds for win rate calculation
""")

print("=" * 80)
