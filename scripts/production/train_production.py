#!/usr/bin/env python3
"""
Train Production Model - Simple Single Ensemble
================================================

Trains a single 3-model ensemble on ALL available data.
No validation, no testing - just maximum training data.

Uses the exact same architecture that achieved 64.3% win rate in backtest.

Output:
- models/production.pkl (contains all 3 models + metadata)

Usage: uv run python scripts/production/train_production.py
"""

import pickle
import sys
from datetime import datetime

import numpy as np
import pandas as pd
import xgboost as xgb

# Add scripts/utils to path
sys.path.append("scripts/utils")
from fast_feature_builder import FastFeatureBuilder

print("=" * 80)
print("TRAIN PRODUCTION MODEL (ALL DATA)")
print("=" * 80)
print()

# ======================================================================
# CONFIGURATION
# ======================================================================

# XGBoost hyperparameters (validated from backtest)
XGBOOST_PARAMS = {
    "objective": "reg:squarederror",
    "max_depth": 4,
    "learning_rate": 0.05,
    "n_estimators": 200,
    "subsample": 0.7,
    "colsample_bytree": 0.7,
    "min_child_weight": 5,
    "gamma": 1.0,
    "reg_alpha": 0.5,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

# Train 3 models with different random seeds for ensemble diversity
ENSEMBLE_SEEDS = [42, 123, 456]

# ======================================================================
# 1. LOAD DATA
# ======================================================================

print("STEP 1: Loading ALL historical game logs...")
print()

# Load all historical game logs
historical_df = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
historical_df["GAME_DATE"] = pd.to_datetime(historical_df["GAME_DATE"])

# Sort by date for temporal integrity
historical_df = historical_df.sort_values(["PLAYER_ID", "GAME_DATE"])

print(f"✅ Loaded: {len(historical_df):,} games")
print(
    f"   Date range: {historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}"
)
print()

# Add PRA if not present
if "PRA" not in historical_df.columns:
    historical_df["PRA"] = historical_df["PTS"] + historical_df["REB"] + historical_df["AST"]

# Filter to MIN >= 25 (match betting population)
historical_df = historical_df[historical_df["MIN"] >= 25].copy()
print(f"After MIN >= 25: {len(historical_df):,} games")
print()

# ======================================================================
# 2. BUILD FEATURES
# ======================================================================

print("STEP 2: Building features...")
print()

builder = FastFeatureBuilder()
full_df = builder.build_features(historical_df, pd.DataFrame(), verbose=True)

print(f"✅ Features built: {full_df.shape}")
print()

# Get feature columns (exclude metadata and target)
exclude_cols = [
    "PLAYER_ID",
    "GAME_DATE",
    "PRA",
    "PLAYER_NAME",
    "TEAM",
    "SEASON",
    "SEASON_TYPE",
    "MATCHUP",
    "WL",
]
feature_cols = [c for c in full_df.columns if c not in exclude_cols]

# Filter out non-numeric columns
numeric_feature_cols = []
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(full_df[col]):
        numeric_feature_cols.append(col)

# Remove same-game box score stats (DATA LEAKAGE)
leaking_features = [
    "PTS",
    "REB",
    "AST",
    "DREB",
    "OREB",
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
    "MIN",
]

feature_cols = [c for c in numeric_feature_cols if c not in leaking_features]

print(f"✅ Clean features: {len(feature_cols)}")
print()

# ======================================================================
# 3. TRAIN ENSEMBLE (3 MODELS)
# ======================================================================

print("=" * 80)
print("STEP 3: TRAINING ENSEMBLE (3 MODELS)")
print("=" * 80)
print()

# Prepare training data (ALL data)
X_train = full_df[feature_cols].fillna(0)
y_train = full_df["PRA"]

print(f"Training on {len(X_train):,} games")
print()

models = []

for i, seed in enumerate(ENSEMBLE_SEEDS, 1):
    print(f"Training Model {i}/3 (seed={seed})...")

    # Update params with new seed
    params = XGBOOST_PARAMS.copy()
    params["random_state"] = seed

    # Train model
    model = xgb.XGBRegressor(**params)
    model.fit(X_train, y_train, verbose=False)

    models.append(model)
    print(f"   ✅ Model {i} trained")

print()
print("✅ All 3 models trained!")
print()

# ======================================================================
# 4. SAVE PRODUCTION MODEL
# ======================================================================

print("STEP 4: Saving production model...")
print()

production_model = {
    "models": models,
    "feature_cols": feature_cols,
    "xgboost_params": XGBOOST_PARAMS,
    "ensemble_seeds": ENSEMBLE_SEEDS,
    "training_games": len(X_train),
    "training_date_range": f"{historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}",
    "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "version": "v1",
    "description": "Production ensemble (3 models) trained on ALL data for live predictions",
}

with open("models/production.pkl", "wb") as f:
    pickle.dump(production_model, f)

print("✅ Saved to models/production.pkl")
print()

# ======================================================================
# USAGE INSTRUCTIONS
# ======================================================================

print("=" * 80)
print("✅ PRODUCTION MODEL READY!")
print("=" * 80)
print()
print("Model Details:")
print(f"   - 3-model ensemble")
print(f"   - {len(feature_cols)} features")
print(f"   - Trained on {len(X_train):,} games")
print(
    f"   - Date range: {historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}"
)
print()
print("Usage:")
print("   1. Load model: production_model = pickle.load(open('models/production.pkl', 'rb'))")
print("   2. Get predictions from each model")
print("   3. Average the 3 predictions")
print()
