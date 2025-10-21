#!/usr/bin/env python3
"""
Quick Backtest: Lineup Features on 2024-25 Season

Purpose: Validate if lineup features improve MAE before full implementation

Process:
1. Build lineup features for 5 key teams (LAL, BOS, DEN, MIL, PHX)
2. Merge to existing training data
3. Retrain model with lineup features
4. Walk-forward backtest on 2024-25
5. Compare MAE: Baseline vs With Lineup Features

Expected Impact:
- Current MAE: ~8.83 points
- Target MAE: 7.5-8.0 points (0.8-1.3 improvement)

Runtime: ~10-15 minutes
"""

import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent.parent.parent))

import pickle
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import mean_absolute_error

from scripts.utils.fast_feature_builder import FastFeatureBuilder
from scripts.utils.lineup_feature_builder import LineupFeatureBuilder

print("=" * 80)
print("LINEUP FEATURES BACKTEST - 2024-25 Season")
print("=" * 80)

# Configuration
TEST_TEAMS = {
    "LAL": ["LeBron James", "Anthony Davis"],
    "BOS": ["Jayson Tatum", "Jaylen Brown"],
    "DEN": ["Nikola Jokic", "Jamal Murray"],
    "MIL": ["Giannis Antetokounmpo", "Damian Lillard"],
    "PHX": ["Devin Booker", "Kevin Durant"],
}

# ============================================================================
# STEP 1: Load Data
# ============================================================================

print("\n📂 Loading game logs...")
game_logs = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
game_logs["GAME_DATE"] = pd.to_datetime(game_logs["GAME_DATE"])

if "PRA" not in game_logs.columns:
    game_logs["PRA"] = game_logs["PTS"] + game_logs["REB"] + game_logs["AST"]

print(f"   ✓ Loaded {len(game_logs):,} player-game records")


# Extract team from MATCHUP column
def extract_team(matchup):
    if " @ " in matchup:
        return matchup.split(" @ ")[0].strip()
    elif " vs. " in matchup:
        return matchup.split(" vs. ")[0].strip()
    return "UNK"


game_logs["TEAM_ABBREVIATION"] = game_logs["MATCHUP"].apply(extract_team)

# ============================================================================
# STEP 2: Build Lineup Features for Test Teams
# ============================================================================

print("\n🏀 Building lineup features for 5 key teams...")
lineup_builder = LineupFeatureBuilder(game_logs)

all_lineup_features = []

for team, stars in TEST_TEAMS.items():
    print(f"\n   {team}: {', '.join(stars)}")

    # Get all players for this team (excluding the stars themselves)
    team_players = game_logs[
        (game_logs["TEAM_ABBREVIATION"] == team) & (~game_logs["PLAYER_NAME"].isin(stars))
    ]["PLAYER_NAME"].unique()

    # Filter to players with enough games
    player_game_counts = (
        game_logs[game_logs["PLAYER_NAME"].isin(team_players)].groupby("PLAYER_NAME").size()
    )

    qualified_players = player_game_counts[player_game_counts >= 20].index.tolist()

    print(f"      Players with 20+ games: {len(qualified_players)}")

    # Build features for top 5 players per team (for speed)
    for i, player in enumerate(qualified_players[:5], 1):
        try:
            player_features = lineup_builder.build_teammate_features(
                player_name=player, teammates=stars, lookback_window=20
            )

            if len(player_features) > 0:
                all_lineup_features.append(player_features)
                print(f"         [{i}/5] {player}: {len(player_features)} games ✓")
        except Exception as e:
            print(f"         [{i}/5] {player}: Error - {e}")
            continue

if len(all_lineup_features) == 0:
    print("\n❌ No lineup features generated. Exiting.")
    sys.exit(1)

lineup_features_df = pd.concat(all_lineup_features, ignore_index=True)
print(f"\n✓ Total lineup features: {len(lineup_features_df):,} games")

# CRITICAL: Deduplicate by GAME_ID (multiple players can have same GAME_ID)
# Keep first occurrence (lineup features are the same for all players in a game)
print(f"   Deduplicating by GAME_ID...")
print(
    f"   Before: {len(lineup_features_df):,} rows, {lineup_features_df['GAME_ID'].nunique():,} unique games"
)

lineup_features_df = lineup_features_df.drop_duplicates(subset=["GAME_ID"], keep="first")

print(
    f"   After: {len(lineup_features_df):,} rows, {lineup_features_df['GAME_ID'].nunique():,} unique games"
)

# Save for inspection
lineup_features_df.to_csv("data/results/lineup_features_test_teams.csv", index=False)
print(f"   Saved to: data/results/lineup_features_test_teams.csv")

# ============================================================================
# STEP 3: Prepare Training Data
# ============================================================================

print("\n📊 Preparing training data...")

# Split data
train_cutoff = pd.to_datetime("2023-06-30")
test_cutoff = pd.to_datetime("2024-06-30")

historical_df = game_logs[game_logs["GAME_DATE"] <= train_cutoff].copy()
test_df = game_logs[
    (game_logs["GAME_DATE"] > test_cutoff) & (game_logs["GAME_DATE"] <= "2025-04-30")
].copy()

print(f"   Historical games (pre-2023): {len(historical_df):,}")
print(f"   Test games (2024-25): {len(test_df):,}")

# Build standard features (existing pipeline)
print("\n🔧 Building standard features...")
feature_builder = FastFeatureBuilder()
full_df = feature_builder.build_features(historical_df, test_df, verbose=True)

# Filter to test set
test_with_features = full_df[full_df["GAME_DATE"] > test_cutoff].copy()
print(f"   Test set with features: {len(test_with_features):,} games")

# ============================================================================
# STEP 4: Two Training Scenarios
# ============================================================================

# Load existing model for feature list
try:
    with open("models/backtest_model.pkl", "rb") as f:
        existing_model = pickle.load(f)
    base_feature_cols = existing_model["feature_cols"]
    print(f"\n✓ Loaded existing model with {len(base_feature_cols)} features")
except:
    print("\n⚠️  Could not load existing model, will use all numeric columns")
    base_feature_cols = test_with_features.select_dtypes(include=[np.number]).columns.tolist()
    base_feature_cols = [
        col for col in base_feature_cols if col not in ["PRA", "PTS", "REB", "AST"]
    ]

# Prepare train/val/test splits
train_df = full_df[full_df["GAME_DATE"] <= train_cutoff].copy()
val_df = full_df[
    (full_df["GAME_DATE"] > train_cutoff) & (full_df["GAME_DATE"] <= test_cutoff)
].copy()

print(f"\nDataset sizes:")
print(f"   Train: {len(train_df):,} games")
print(f"   Val: {len(val_df):,} games")
print(f"   Test: {len(test_with_features):,} games")

# ============================================================================
# SCENARIO 1: Baseline (No Lineup Features)
# ============================================================================

print("\n" + "=" * 80)
print("SCENARIO 1: Baseline Model (No Lineup Features)")
print("=" * 80)

# Prepare data
X_train_base = train_df[base_feature_cols].fillna(0)
y_train = train_df["PRA"]

X_val_base = val_df[base_feature_cols].fillna(0)
y_val = val_df["PRA"]

X_test_base = test_with_features[base_feature_cols].fillna(0)
y_test = test_with_features["PRA"]

print(f"\nFeatures: {len(base_feature_cols)}")
print(f"Train samples: {len(X_train_base):,}")

# Train model
print("\n🔄 Training baseline model...")
model_base = xgb.XGBRegressor(
    n_estimators=500,  # Reduced for faster testing
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model_base.fit(X_train_base, y_train, verbose=False)

# Evaluate
y_pred_base = model_base.predict(X_test_base)
mae_base = mean_absolute_error(y_test, y_pred_base)

print(f"\n📊 Baseline Results:")
print(f"   Test MAE: {mae_base:.3f} points")

# ============================================================================
# SCENARIO 2: With Lineup Features
# ============================================================================

print("\n" + "=" * 80)
print("SCENARIO 2: Model WITH Lineup Features")
print("=" * 80)

# Merge lineup features
print("\n🔀 Merging lineup features...")

train_with_lineup = train_df.merge(
    lineup_features_df, on="GAME_ID", how="left", suffixes=("", "_lineup")
)

val_with_lineup = val_df.merge(
    lineup_features_df, on="GAME_ID", how="left", suffixes=("", "_lineup")
)

test_with_lineup = test_with_features.merge(
    lineup_features_df, on="GAME_ID", how="left", suffixes=("", "_lineup")
)

# Get lineup feature columns
lineup_feature_cols = [
    col
    for col in lineup_features_df.columns
    if col
    not in ["GAME_ID", "GAME_DATE", "PLAYER_ID", "PLAYER_NAME", "TEAM_ID", "TEAM_ABBREVIATION"]
]

print(f"   Lineup features added: {len(lineup_feature_cols)}")
print(f"   Examples: {lineup_feature_cols[:5]}")

# Combined feature list
enhanced_feature_cols = base_feature_cols + lineup_feature_cols

# Prepare data
X_train_enhanced = train_with_lineup[enhanced_feature_cols].fillna(0)
X_val_enhanced = val_with_lineup[enhanced_feature_cols].fillna(0)
X_test_enhanced = test_with_lineup[enhanced_feature_cols].fillna(0)

print(f"\nEnhanced features: {len(enhanced_feature_cols)}")
print(f"Train samples: {len(X_train_enhanced):,}")

# Train model
print("\n🔄 Training enhanced model...")
model_enhanced = xgb.XGBRegressor(
    n_estimators=500,  # Reduced for faster testing
    learning_rate=0.01,
    max_depth=5,
    subsample=0.8,
    colsample_bytree=0.8,
    random_state=42,
    n_jobs=-1,
)

model_enhanced.fit(X_train_enhanced, y_train, verbose=False)

# Evaluate
y_pred_enhanced = model_enhanced.predict(X_test_enhanced)
mae_enhanced = mean_absolute_error(y_test, y_pred_enhanced)

print(f"\n📊 Enhanced Results:")
print(f"   Test MAE: {mae_enhanced:.3f} points")

# ============================================================================
# COMPARISON
# ============================================================================

print("\n" + "=" * 80)
print("FINAL COMPARISON")
print("=" * 80)

improvement = mae_base - mae_enhanced
pct_improvement = (improvement / mae_base) * 100

print(f"\n📈 Results:")
print(f"   Baseline MAE:  {mae_base:.3f} points")
print(f"   Enhanced MAE:  {mae_enhanced:.3f} points")
print(f"   Improvement:   {improvement:+.3f} points ({pct_improvement:+.1f}%)")

if improvement > 0:
    print(f"\n✅ LINEUP FEATURES IMPROVED MAE!")
    print(f"   Worth pursuing full implementation.")
else:
    print(f"\n⚠️  Lineup features did NOT improve MAE")
    print(f"   Consider other features (opponent defense, TS%, etc.)")

# Save results
results = {
    "baseline_mae": mae_base,
    "enhanced_mae": mae_enhanced,
    "improvement": improvement,
    "pct_improvement": pct_improvement,
    "test_teams": list(TEST_TEAMS.keys()),
    "lineup_features_count": len(lineup_feature_cols),
    "test_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
}

results_df = pd.DataFrame([results])
results_df.to_csv("data/results/lineup_features_backtest_results.csv", index=False)
print(f"\n💾 Results saved to: data/results/lineup_features_backtest_results.csv")

# Feature importance comparison
print("\n🔍 Top Lineup Features by Importance:")
feature_importance = pd.DataFrame(
    {"feature": enhanced_feature_cols, "importance": model_enhanced.feature_importances_}
).sort_values("importance", ascending=False)

lineup_importance = feature_importance[
    feature_importance["feature"].str.contains("lebron|tatum|jokic|giannis|booker", case=False)
].head(10)

if len(lineup_importance) > 0:
    print(lineup_importance.to_string(index=False))
else:
    print("   No lineup features in top features")

print("\n" + "=" * 80)
print("✅ Backtest Complete!")
print("=" * 80)
