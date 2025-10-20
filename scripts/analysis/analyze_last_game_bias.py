"""
Last Game Bias Analysis

Investigates if the model is overfitting to the last game PRA value,
which would explain negative predictions when players have DNP or low-minute games.
"""

import pandas as pd
import numpy as np
import pickle

print("=" * 80)
print("LAST GAME BIAS ANALYSIS")
print("=" * 80)

# Load the model
print("\nLoading production model...")
with open('models/production_model_latest.pkl', 'rb') as f:
    model_dict = pickle.load(f)

model = model_dict['model']
feature_cols = model_dict['feature_cols']

print(f"Model features: {len(feature_cols)}")

# Get feature importance
importance_df = model_dict['feature_importance']
print("\nTop 20 most important features:")
print(importance_df.head(20).to_string(index=False))

# Check if lag1 is the dominant feature
lag1_importance = importance_df[importance_df['feature'] == 'PRA_lag1']
if not lag1_importance.empty:
    lag1_rank = importance_df[importance_df['feature'] == 'PRA_lag1'].index[0] + 1
    lag1_score = lag1_importance['importance'].values[0]
    print(f"\nPRA_lag1 rank: #{lag1_rank}")
    print(f"PRA_lag1 importance: {lag1_score:.4f}")
    print(f"Total importance sum: {importance_df['importance'].sum():.4f}")
    print(f"PRA_lag1 % of total importance: {(lag1_score / importance_df['importance'].sum()) * 100:.2f}%")

# Load training data to analyze predictions on problematic cases
print("\n" + "=" * 80)
print("ANALYZING PREDICTIONS FOR LAST-GAME DNP SCENARIOS")
print("=" * 80)

print("\nLoading training data...")
df = pd.read_csv('data/game_logs/all_game_logs_through_2025.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

if 'PRA' not in df.columns:
    df['PRA'] = df['PTS'] + df['REB'] + df['AST']

# Calculate lag features
player_groups = df.groupby('PLAYER_ID')
for lag in [1, 3, 5, 7, 10]:
    df[f'PRA_lag{lag}'] = player_groups['PRA'].shift(lag)

for window in [3, 5, 10, 20]:
    df[f'PRA_L{window}_mean'] = (
        player_groups['PRA'].shift(1).rolling(window=window, min_periods=1).mean()
    )
    df[f'PRA_L{window}_std'] = (
        player_groups['PRA'].shift(1).rolling(window=window, min_periods=2).std()
    )

for span in [5, 10, 15]:
    df[f'PRA_ewma{span}'] = (
        player_groups['PRA'].shift(1).ewm(span=span, min_periods=1).mean()
    )

df['PRA_trend_L5_L20'] = df['PRA_L5_mean'] - df['PRA_L20_mean']

if 'MIN' in df.columns:
    df['MIN_L5_mean'] = player_groups['MIN'].shift(1).rolling(window=5, min_periods=1).mean()

# Find games where player had DNP (0 MIN) or very low minutes (< 5)
# as their LAST game before a normal game
print("\nFinding cases where last game was DNP/low-minutes...")

df['prev_MIN'] = player_groups['MIN'].shift(1)
df['prev_PRA'] = player_groups['PRA'].shift(1)

# Games where:
# 1. Current game had normal minutes (>= 15)
# 2. Previous game was DNP or very low minutes (<= 5)
# 3. Player has good history (L10 mean > 15)
problematic_cases = df[
    (df['MIN'] >= 15) &
    (df['prev_MIN'] <= 5) &
    (df['PRA_L10_mean'] > 15) &
    (~df['PRA_lag1'].isna())
]

print(f"\nFound {len(problematic_cases):,} games matching criteria:")
print("  - Current game: >= 15 MIN")
print("  - Previous game: <= 5 MIN (DNP or garbage time)")
print("  - Historical average (L10): > 15 PRA")

if len(problematic_cases) > 0:
    print("\nSample cases:")
    sample = problematic_cases[['PLAYER_NAME', 'GAME_DATE', 'MIN', 'PRA',
                                'prev_MIN', 'prev_PRA', 'PRA_lag1',
                                'PRA_L5_mean', 'PRA_L10_mean']].head(10)
    print(sample.to_string(index=False))

    # Analyze average actual PRA in these cases
    print(f"\nActual PRA in these cases:")
    print(f"  Mean: {problematic_cases['PRA'].mean():.2f}")
    print(f"  Median: {problematic_cases['PRA'].median():.2f}")
    print(f"  Std: {problematic_cases['PRA'].std():.2f}")

# Check the specific problematic predictions
print("\n" + "=" * 80)
print("SIMULATING PREDICTIONS FOR JARRETT ALLEN & MIKAL BRIDGES")
print("=" * 80)

problematic_players = [
    ('Jarrett Allen', '2025-04-13'),
    ('Mikal Bridges', '2025-04-13')
]

for player_name, last_game_date in problematic_players:
    print(f"\n{'=' * 60}")
    print(f"Player: {player_name}")
    print(f"Last game: {last_game_date}")
    print('=' * 60)

    # Get player's history
    player_data = df[df['PLAYER_NAME'] == player_name].sort_values('GAME_DATE')
    last_game = player_data[player_data['GAME_DATE'] == last_game_date]

    if len(last_game) > 0:
        last_game = last_game.iloc[0]

        print(f"\nLast game details:")
        print(f"  Date: {last_game['GAME_DATE']}")
        print(f"  MIN: {last_game['MIN']}")
        print(f"  PRA: {last_game['PRA']}")

        print(f"\nLag features:")
        print(f"  PRA_lag1: {last_game['PRA_lag1']:.2f}")
        print(f"  PRA_lag3: {last_game.get('PRA_lag3', 'N/A')}")
        print(f"  PRA_lag5: {last_game.get('PRA_lag5', 'N/A')}")
        print(f"  PRA_L5_mean: {last_game['PRA_L5_mean']:.2f}")
        print(f"  PRA_L10_mean: {last_game['PRA_L10_mean']:.2f}")
        print(f"  PRA_L20_mean: {last_game.get('PRA_L20_mean', 'N/A')}")

        # Check if we have all features needed
        available_features = [col for col in feature_cols if col in last_game.index]
        missing_features = [col for col in feature_cols if col not in last_game.index]

        print(f"\nFeatures available: {len(available_features)} / {len(feature_cols)}")
        if missing_features:
            print(f"Missing features: {missing_features[:10]}")  # Show first 10

# Check distribution of predictions when PRA_lag1 is very low
print("\n" + "=" * 80)
print("CORRELATION: PRA_lag1 vs Model Predictions")
print("=" * 80)

print("\nAnalyzing model behavior on training data...")

# Get a sample of training data with all features
valid_mask = ~df[feature_cols].isna().any(axis=1)
train_sample = df[valid_mask].sample(n=min(50000, valid_mask.sum()), random_state=42)

print(f"Sample size: {len(train_sample):,}")

# Make predictions
X_sample = train_sample[feature_cols].fillna(0)
y_pred = model.predict(X_sample)
train_sample['predicted_PRA'] = y_pred

# Analyze by PRA_lag1 bins
print("\nPredictions grouped by PRA_lag1:")

lag1_bins = [0, 1, 5, 10, 15, 20, 25, 30, 40, 100]
train_sample['lag1_bin'] = pd.cut(train_sample['PRA_lag1'], bins=lag1_bins)

bin_analysis = train_sample.groupby('lag1_bin', observed=True).agg({
    'predicted_PRA': ['mean', 'std', 'min', 'max'],
    'PRA': ['mean', 'std'],
    'PRA_lag1': 'count'
}).round(2)

print(bin_analysis)

# Count negative predictions
negative_preds = train_sample[train_sample['predicted_PRA'] < 0]
print(f"\nNegative predictions in sample: {len(negative_preds)}")
if len(negative_preds) > 0:
    print(f"  Mean PRA_lag1: {negative_preds['PRA_lag1'].mean():.2f}")
    print(f"  Mean PRA_L5: {negative_preds['PRA_L5_mean'].mean():.2f}")
    print(f"  Actual PRA mean: {negative_preds['PRA'].mean():.2f}")

print("\n" + "=" * 80)
print("ANALYSIS COMPLETE")
print("=" * 80)
