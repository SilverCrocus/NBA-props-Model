"""
Data Quality Investigation - Negative PRA Predictions

Analyzes:
1. Missing/invalid data in training set
2. Players with negative predictions
3. Feature distributions and outliers
4. Data merge quality (historical + 2024-25)
"""

import pandas as pd
import numpy as np
from pathlib import Path
import sys

print("=" * 80)
print("DATA QUALITY INVESTIGATION - NEGATIVE PRA PREDICTIONS")
print("=" * 80)

# ============================================================================
# PART 1: Load and inspect training data
# ============================================================================
print("\n" + "=" * 80)
print("PART 1: TRAINING DATA INSPECTION")
print("=" * 80)

print("\nLoading training data...")
df = pd.read_csv('data/game_logs/all_game_logs_through_2025.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])
df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

print(f"\n✅ Loaded {len(df):,} games")
print(f"   Date range: {df['GAME_DATE'].min()} to {df['GAME_DATE'].max()}")
print(f"   Players: {df['PLAYER_NAME'].nunique():,}")

# Add PRA
if 'PRA' not in df.columns:
    df['PRA'] = df['PTS'] + df['REB'] + df['AST']

# ============================================================================
# Check 1: Missing values in key columns
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 1: Missing Values in Key Columns")
print("-" * 80)

key_cols = ['PLAYER_ID', 'PLAYER_NAME', 'GAME_DATE', 'PTS', 'REB', 'AST',
            'PRA', 'MIN', 'FGA', 'FG_PCT', 'FG3A', 'FTA']

missing_analysis = pd.DataFrame({
    'Column': key_cols,
    'Missing_Count': [df[col].isna().sum() for col in key_cols],
    'Missing_Pct': [df[col].isna().mean() * 100 for col in key_cols]
})

print(missing_analysis.to_string(index=False))

# ============================================================================
# Check 2: Invalid PRA values in training data
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 2: Invalid PRA Values (Negative or Zero)")
print("-" * 80)

negative_pra = df[df['PRA'] < 0]
zero_pra = df[df['PRA'] == 0]

print(f"\nGames with NEGATIVE PRA: {len(negative_pra):,}")
if len(negative_pra) > 0:
    print("\nSample of negative PRA games:")
    print(negative_pra[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'REB', 'AST', 'PRA']].head(10).to_string(index=False))

print(f"\nGames with ZERO PRA: {len(zero_pra):,}")
if len(zero_pra) > 0:
    print(f"   {(len(zero_pra) / len(df)) * 100:.2f}% of all games")
    print("\nSample of zero PRA games:")
    print(zero_pra[['PLAYER_NAME', 'GAME_DATE', 'MIN', 'PTS', 'REB', 'AST', 'PRA']].head(10).to_string(index=False))

# ============================================================================
# Check 3: PRA distribution
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 3: PRA Distribution")
print("-" * 80)

print("\nPRA Statistics:")
print(df['PRA'].describe())

print("\nPRA Percentiles:")
percentiles = [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]
for p in percentiles:
    val = np.percentile(df['PRA'].dropna(), p)
    print(f"   {p:3d}th percentile: {val:6.1f}")

# ============================================================================
# Check 4: Minutes distribution (DNP games)
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 4: Minutes Distribution (DNP Games)")
print("-" * 80)

if 'MIN' in df.columns:
    print("\nMinutes Statistics:")
    print(df['MIN'].describe())

    dnp_games = df[df['MIN'] == 0]
    print(f"\nGames with 0 minutes (DNP): {len(dnp_games):,}")
    print(f"   {(len(dnp_games) / len(df)) * 100:.2f}% of all games")

    # DNP games with non-zero PRA (data error)
    dnp_with_stats = dnp_games[dnp_games['PRA'] > 0]
    if len(dnp_with_stats) > 0:
        print(f"\n⚠️  WARNING: {len(dnp_with_stats):,} DNP games have non-zero PRA (DATA ERROR)")
        print("\nSample:")
        print(dnp_with_stats[['PLAYER_NAME', 'GAME_DATE', 'MIN', 'PTS', 'REB', 'AST', 'PRA']].head(10).to_string(index=False))

# ============================================================================
# PART 2: Investigate players with negative predictions
# ============================================================================
print("\n" + "=" * 80)
print("PART 2: PLAYERS WITH NEGATIVE/LOW PREDICTIONS")
print("=" * 80)

print("\nLoading predictions...")
preds = pd.read_csv('data/results/predictions_2025_10_22.csv')

print(f"\n✅ Loaded {len(preds)} predictions")

# Find problematic predictions
negative_preds = preds[preds['predicted_PRA'] < 0]
low_preds = preds[preds['predicted_PRA'] < 5]

print(f"\nNegative predictions: {len(negative_preds)}")
print(f"Very low predictions (<5): {len(low_preds)}")

print("\n" + "-" * 80)
print("Players with Negative or Near-Zero Predictions:")
print("-" * 80)
print(low_preds.to_string(index=False))

# ============================================================================
# Check 5: Historical data for problematic players
# ============================================================================
print("\n" + "-" * 80)
print("CHECK 5: Historical Data for Problematic Players")
print("-" * 80)

problematic_players = ['Jarrett Allen', 'Mikal Bridges', 'Jalen Johnson']

for player in problematic_players:
    print(f"\n{'=' * 60}")
    print(f"Player: {player}")
    print('=' * 60)

    player_games = df[df['PLAYER_NAME'] == player].sort_values('GAME_DATE')

    print(f"\nTotal games in history: {len(player_games):,}")
    print(f"Date range: {player_games['GAME_DATE'].min()} to {player_games['GAME_DATE'].max()}")

    print(f"\nPRA Statistics:")
    print(player_games['PRA'].describe())

    # Recent games (last 20)
    recent_games = player_games.tail(20)
    print(f"\nLast 20 games:")
    print(recent_games[['GAME_DATE', 'MIN', 'PTS', 'REB', 'AST', 'PRA']].to_string(index=False))

    # Games in 2024-25 season
    games_2024_25 = player_games[player_games['GAME_DATE'] >= '2024-10-01']
    print(f"\n2024-25 Season games: {len(games_2024_25)}")
    if len(games_2024_25) > 0:
        print(games_2024_25[['GAME_DATE', 'MIN', 'PTS', 'REB', 'AST', 'PRA']].to_string(index=False))

    # Check for gaps in timeline
    if len(player_games) > 1:
        player_games['days_since_last'] = player_games['GAME_DATE'].diff().dt.days
        large_gaps = player_games[player_games['days_since_last'] > 180]  # 6 months
        if len(large_gaps) > 0:
            print(f"\n⚠️  Large gaps (>6 months) in game history: {len(large_gaps)}")
            print(large_gaps[['GAME_DATE', 'days_since_last']].to_string(index=False))

# ============================================================================
# PART 3: Lag feature analysis
# ============================================================================
print("\n" + "=" * 80)
print("PART 3: LAG FEATURE ANALYSIS")
print("=" * 80)

print("\nCalculating lag features for analysis...")
player_groups = df.groupby('PLAYER_ID')

# Calculate key lag features
df['PRA_lag1'] = player_groups['PRA'].shift(1)
df['PRA_L5_mean'] = player_groups['PRA'].shift(1).rolling(window=5, min_periods=1).mean()
df['PRA_L10_mean'] = player_groups['PRA'].shift(1).rolling(window=10, min_periods=1).mean()

lag_features = ['PRA_lag1', 'PRA_L5_mean', 'PRA_L10_mean']

print("\n" + "-" * 80)
print("Lag Feature Missing Values:")
print("-" * 80)

lag_missing = pd.DataFrame({
    'Feature': lag_features,
    'Missing_Count': [df[col].isna().sum() for col in lag_features],
    'Missing_Pct': [df[col].isna().mean() * 100 for col in lag_features]
})
print(lag_missing.to_string(index=False))

print("\n" + "-" * 80)
print("Lag Feature Distributions:")
print("-" * 80)

for feature in lag_features:
    print(f"\n{feature}:")
    print(df[feature].describe())

    # Check for extreme outliers
    q99 = df[feature].quantile(0.99)
    q01 = df[feature].quantile(0.01)
    outliers = df[(df[feature] > q99) | (df[feature] < q01)][feature].dropna()
    print(f"   Outliers (>99th or <1st percentile): {len(outliers):,} ({len(outliers)/len(df)*100:.2f}%)")

# ============================================================================
# PART 4: Data merge quality check
# ============================================================================
print("\n" + "=" * 80)
print("PART 4: DATA MERGE QUALITY")
print("=" * 80)

# Check for duplicate games
print("\n" + "-" * 80)
print("Duplicate Games Check:")
print("-" * 80)

df['game_key'] = df['PLAYER_ID'].astype(str) + '_' + df['GAME_DATE'].astype(str)
duplicates = df[df.duplicated(subset=['game_key'], keep=False)]

print(f"\nDuplicate player-game combinations: {len(duplicates):,}")
if len(duplicates) > 0:
    print(f"   ⚠️  WARNING: {len(duplicates):,} duplicate rows found!")
    print("\nSample duplicates:")
    print(duplicates[['PLAYER_NAME', 'GAME_DATE', 'PTS', 'REB', 'AST', 'PRA']].head(20).to_string(index=False))

# Check for timeline gaps
print("\n" + "-" * 80)
print("Timeline Continuity Check:")
print("-" * 80)

# Games by year
df['YEAR'] = df['GAME_DATE'].dt.year
games_by_year = df.groupby('YEAR').size().sort_index()

print("\nGames by year:")
for year, count in games_by_year.items():
    print(f"   {year}: {count:,}")

# Check for missing months
df['YEAR_MONTH'] = df['GAME_DATE'].dt.to_period('M')
all_months = pd.period_range(start=df['GAME_DATE'].min(), end=df['GAME_DATE'].max(), freq='M')
actual_months = df['YEAR_MONTH'].unique()

missing_months = set(all_months) - set(actual_months)
if missing_months:
    print(f"\n⚠️  WARNING: {len(missing_months)} months with no games:")
    for month in sorted(missing_months)[:10]:  # Show first 10
        print(f"   {month}")

# ============================================================================
# PART 5: Feature values for problematic predictions
# ============================================================================
print("\n" + "=" * 80)
print("PART 5: FEATURE VALUES FOR PROBLEMATIC PREDICTIONS")
print("=" * 80)

# For each problematic player, show their most recent feature values
print("\nRecalculating features for October 2025 predictions...")

# Get most recent game for each problematic player (before Oct 22, 2025)
cutoff_date = pd.to_datetime('2025-10-22')

for player in problematic_players:
    print(f"\n{'=' * 60}")
    print(f"Player: {player}")
    print('=' * 60)

    player_data = df[df['PLAYER_NAME'] == player].copy()

    # Most recent game before cutoff
    before_cutoff = player_data[player_data['GAME_DATE'] < cutoff_date].sort_values('GAME_DATE')

    if len(before_cutoff) > 0:
        latest_game = before_cutoff.iloc[-1]
        print(f"\nMost recent game: {latest_game['GAME_DATE']}")
        print(f"PRA: {latest_game['PRA']}")
        print(f"MIN: {latest_game['MIN']}")

        print(f"\nLag features:")
        for col in ['PRA_lag1', 'PRA_L5_mean', 'PRA_L10_mean']:
            if col in latest_game.index:
                print(f"   {col}: {latest_game[col]:.2f}")

        # Show last 5 games
        last_5 = before_cutoff.tail(5)
        print(f"\nLast 5 games before prediction date:")
        print(last_5[['GAME_DATE', 'MIN', 'PTS', 'REB', 'AST', 'PRA']].to_string(index=False))

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "=" * 80)
print("SUMMARY OF FINDINGS")
print("=" * 80)

print("\n1. DATA QUALITY ISSUES:")
print(f"   - Games with negative PRA: {len(negative_pra):,}")
print(f"   - Games with zero PRA: {len(zero_pra):,} ({len(zero_pra)/len(df)*100:.2f}%)")
print(f"   - Duplicate games: {len(duplicates):,}")
print(f"   - DNP games (0 minutes): {len(dnp_games):,} ({len(dnp_games)/len(df)*100:.2f}%)")

print("\n2. PREDICTIONS:")
print(f"   - Negative predictions: {len(negative_preds)}")
print(f"   - Very low predictions (<5): {len(low_preds)}")

print("\n3. LAG FEATURES:")
for feature in lag_features:
    missing_pct = df[feature].isna().mean() * 100
    print(f"   - {feature} missing: {missing_pct:.2f}%")

print("\n" + "=" * 80)
print("INVESTIGATION COMPLETE")
print("=" * 80)
