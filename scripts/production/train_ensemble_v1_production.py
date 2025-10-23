#!/usr/bin/env python3
"""
Train Production Ensemble Model v1
===================================

Trains the FINAL production model on ALL available data (2003-2025).

This model will be used for live predictions going forward.

Training Strategy:
- Train on ALL data from 2003 through current season
- Use same 3-fold time-series CV structure (but all dates move forward)
- Apply isotonic regression calibration
- Same hyperparameters as validated backtest model

Folds (ALL data included):
- Fold 1: Train 2003-2022, Val 2023-24
- Fold 2: Train 2003-2023, Val 2024-25 (first half)
- Fold 3: Train 2003-2024, Val 2024-25 (second half)

Output:
- models/ensemble_v1_production_fold_1.pkl
- models/ensemble_v1_production_fold_2.pkl
- models/ensemble_v1_production_fold_3.pkl
- models/ensemble_v1_production_meta.pkl

Usage: uv run python scripts/production/train_ensemble_v1_production.py
"""

import pickle
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

# Add scripts/utils to path
sys.path.append("scripts/utils")
from fast_feature_builder import FastFeatureBuilder

print("=" * 80)
print("TRAIN PRODUCTION ENSEMBLE MODEL V1 (ALL DATA)")
print("=" * 80)
print()

# ======================================================================
# CONFIGURATION
# ======================================================================

# Time-series CV folds - using ALL available data
FOLDS = [
    {
        "name": "Fold 1",
        "train_start": "2003-01-01",
        "train_end": "2022-06-30",
        "val_start": "2022-07-01",
        "val_end": "2023-06-30",
        "model_path": "models/production_fold_1.pkl",
    },
    {
        "name": "Fold 2",
        "train_start": "2003-01-01",
        "train_end": "2023-06-30",
        "val_start": "2023-07-01",
        "val_end": "2024-06-30",
        "model_path": "models/production_fold_2.pkl",
    },
    {
        "name": "Fold 3",
        "train_start": "2003-01-01",
        "train_end": "2024-06-30",
        "val_start": "2024-07-01",
        "val_end": "2025-06-30",
        "model_path": "models/production_fold_3.pkl",
    },
]

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

print(f"✅ Historical: {len(historical_df):,} games")
print(
    f"   Date range: {historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}"
)
print()

# Add PRA if not present
if "PRA" not in historical_df.columns:
    historical_df["PRA"] = historical_df["PTS"] + historical_df["REB"] + historical_df["AST"]

# Filter to MIN >= 25 (match betting population)
print(
    f"Before MIN filter: {len(historical_df):,} games, PRA avg = {historical_df['PRA'].mean():.2f}"
)
historical_df = historical_df[historical_df["MIN"] >= 25].copy()
print(
    f"After MIN >= 25:   {len(historical_df):,} games, PRA avg = {historical_df['PRA'].mean():.2f}"
)
print()

# ======================================================================
# 2. BUILD FEATURES (ONCE FOR ALL DATA)
# ======================================================================

print("STEP 2: Building features for ALL data...")
print("   (This takes ~60 seconds but only done once)")
print()

builder = FastFeatureBuilder()
full_df = builder.build_features(historical_df, pd.DataFrame(), verbose=True)

print(f"✅ Features built: {full_df.shape}")
print(
    f"   Total features: {len([c for c in full_df.columns if c not in ['PLAYER_ID', 'GAME_DATE', 'PRA']])}"
)
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

# Filter out non-numeric columns (XGBoost requirement)
numeric_feature_cols = []
for col in feature_cols:
    if pd.api.types.is_numeric_dtype(full_df[col]):
        numeric_feature_cols.append(col)
    else:
        print(f"   Dropping non-numeric column: {col} (dtype: {full_df[col].dtype})")

# CRITICAL: Remove same-game box score stats (DATA LEAKAGE)
leaking_features = [
    # Direct PRA components
    "PTS",
    "REB",
    "AST",
    "DREB",
    "OREB",
    # Components used to calculate PTS
    "FGM",
    "FGA",
    "FG3M",
    "FG3A",
    "FTM",
    "FTA",
    # Other same-game stats
    "FG_PCT",
    "FG3_PCT",
    "FT_PCT",
    "STL",
    "BLK",
    "TOV",
    "PF",
    "PLUS_MINUS",
    "MIN",  # Can leak through correlations
]

print("\n⚠️  REMOVING LEAKING FEATURES (same-game box score stats):")
non_leaking_cols = []
for col in numeric_feature_cols:
    if col in leaking_features:
        print(f"   ✗ Removing: {col} (same-game stat)")
    else:
        non_leaking_cols.append(col)

feature_cols = non_leaking_cols

print(f"\n✅ Features after leakage removal: {len(feature_cols)}")
print()

# ======================================================================
# 3. TIME-SERIES CROSS-VALIDATION TRAINING
# ======================================================================

print("=" * 80)
print("STEP 3: TIME-SERIES CROSS-VALIDATION (3 FOLDS - ALL DATA)")
print("=" * 80)
print()

fold_results = []
fold_models = []
fold_calibrators = []

for i, fold in enumerate(FOLDS, 1):
    print(f"\n{'=' * 80}")
    print(f"FOLD {i}: {fold['name']}")
    print(f"{'=' * 80}")
    print(f"Train: {fold['train_start']} to {fold['train_end']}")
    print(f"Val:   {fold['val_start']} to {fold['val_end']}")
    print()

    # Split data for this fold
    train_mask = (full_df["GAME_DATE"] >= fold["train_start"]) & (
        full_df["GAME_DATE"] <= fold["train_end"]
    )
    val_mask = (full_df["GAME_DATE"] >= fold["val_start"]) & (
        full_df["GAME_DATE"] <= fold["val_end"]
    )

    train_df = full_df[train_mask].copy()
    val_df = full_df[val_mask].copy()

    print(f"Train: {len(train_df):,} games")
    print(f"Val:   {len(val_df):,} games")
    print()

    # Prepare train data
    X_train = train_df[feature_cols].fillna(0)
    y_train = train_df["PRA"]

    # Prepare validation data
    X_val = val_df[feature_cols].fillna(0)
    y_val = val_df["PRA"]

    # Train XGBoost model
    print(f"Training XGBoost model (Fold {i})...")
    model = xgb.XGBRegressor(**XGBOOST_PARAMS)
    model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)

    # Make predictions on validation set
    y_pred_raw = model.predict(X_val)
    mae_raw = mean_absolute_error(y_val, y_pred_raw)

    print(f"   ✅ Model trained")
    print(f"   Raw MAE: {mae_raw:.2f} pts")
    print()

    # ======================================================================
    # CALIBRATION (on validation set)
    # ======================================================================

    print(f"Training calibrator (Fold {i})...")

    # Split validation set: 80% for calibration, 20% for testing calibration
    val_split_idx = int(len(val_df) * 0.8)

    X_val_calib = X_val.iloc[:val_split_idx]
    y_val_calib = y_val.iloc[:val_split_idx]
    X_val_test = X_val.iloc[val_split_idx:]
    y_val_test = y_val.iloc[val_split_idx:]

    # Get raw predictions for calibration
    y_pred_calib = model.predict(X_val_calib)
    y_pred_test = model.predict(X_val_test)

    # Fit isotonic regression calibrator
    calibrator = IsotonicRegression(out_of_bounds="clip")
    calibrator.fit(y_pred_calib, y_val_calib)

    # Apply calibration to test portion
    y_pred_calibrated = calibrator.transform(y_pred_test)
    mae_calibrated = mean_absolute_error(y_val_test, y_pred_calibrated)

    improvement = mae_raw - mae_calibrated

    print(f"   ✅ Calibrator trained")
    print(f"   Calibrated MAE: {mae_calibrated:.2f} pts")
    print(f"   Improvement: {improvement:.2f} pts ({improvement/mae_raw*100:.1f}%)")
    print()

    # Save model and calibrator
    print(f"Saving model and calibrator...")
    fold_model_dict = {
        "model": model,
        "calibrator": calibrator,
        "feature_cols": feature_cols,
        "fold_name": fold["name"],
        "train_period": f"{fold['train_start']} to {fold['train_end']}",
        "val_period": f"{fold['val_start']} to {fold['val_end']}",
        "mae_raw": mae_raw,
        "mae_calibrated": mae_calibrated,
        "improvement": improvement,
        "xgboost_params": XGBOOST_PARAMS,
        "version": "v1",
        "type": "production",
    }

    with open(fold["model_path"], "wb") as f:
        pickle.dump(fold_model_dict, f)

    print(f"   ✅ Saved to {fold['model_path']}")
    print()

    # Store results
    fold_results.append(
        {
            "fold": fold["name"],
            "train_games": len(train_df),
            "val_games": len(val_df),
            "mae_raw": mae_raw,
            "mae_calibrated": mae_calibrated,
            "improvement": improvement,
        }
    )

    fold_models.append(model)
    fold_calibrators.append(calibrator)

# ======================================================================
# 4. CROSS-VALIDATION SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("CROSS-VALIDATION SUMMARY (ALL DATA)")
print("=" * 80)
print()

results_df = pd.DataFrame(fold_results)
print(results_df.to_string(index=False))
print()

print(
    f"Average MAE (raw):        {results_df['mae_raw'].mean():.2f} ± {results_df['mae_raw'].std():.2f} pts"
)
print(
    f"Average MAE (calibrated): {results_df['mae_calibrated'].mean():.2f} ± {results_df['mae_calibrated'].std():.2f} pts"
)
print(f"Average improvement:      {results_df['improvement'].mean():.2f} pts")
print()

# ======================================================================
# 5. SAVE ENSEMBLE METADATA
# ======================================================================

print("Saving ensemble metadata...")

ensemble_meta = {
    "version": "v1",
    "type": "production",
    "folds": FOLDS,
    "fold_results": fold_results,
    "feature_cols": feature_cols,
    "xgboost_params": XGBOOST_PARAMS,
    "created_at": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
    "training_data_range": f"{historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}",
    "total_training_games": len(historical_df),
    "description": "Production ensemble model v1 - trained on ALL data (2003-2025) for live predictions",
}

with open("models/production_meta.pkl", "wb") as f:
    pickle.dump(ensemble_meta, f)

print(f"   ✅ Saved to models/production_meta.pkl")
print()

# ======================================================================
# 6. FEATURE IMPORTANCE (AVERAGED ACROSS FOLDS)
# ======================================================================

print("=" * 80)
print("FEATURE IMPORTANCE (Averaged Across Folds)")
print("=" * 80)
print()

# Get feature importances from each fold
importances = []
for model in fold_models:
    importances.append(model.feature_importances_)

# Average importance across folds
avg_importance = np.mean(importances, axis=0)
std_importance = np.std(importances, axis=0)

# Create dataframe
importance_df = pd.DataFrame(
    {"feature": feature_cols, "importance_mean": avg_importance, "importance_std": std_importance}
).sort_values("importance_mean", ascending=False)

print(importance_df.head(20).to_string(index=False))
print()

# Save importance
importance_df.to_csv("data/results/ensemble_v1_production_feature_importance.csv", index=False)
print("   ✅ Saved to data/results/ensemble_v1_production_feature_importance.csv")
print()

# ======================================================================
# FINAL SUMMARY
# ======================================================================

print("\n" + "=" * 80)
print("✅ PRODUCTION MODEL TRAINING COMPLETE!")
print("=" * 80)
print()
print("Models saved:")
print(f"   - models/production_fold_1.pkl")
print(f"   - models/production_fold_2.pkl")
print(f"   - models/production_fold_3.pkl")
print(f"   - models/production_meta.pkl")
print()
print("Performance Summary:")
print(
    f"   - CV MAE (averaged): {results_df['mae_calibrated'].mean():.2f} ± {results_df['mae_calibrated'].std():.2f} pts"
)
print(f"   - Training games: {len(historical_df):,}")
print(
    f"   - Date range: {historical_df['GAME_DATE'].min().date()} to {historical_df['GAME_DATE'].max().date()}"
)
print()
print("✅ READY FOR PRODUCTION USE!")
print()
print("Usage:")
print("   Load all 3 fold models, make predictions with each, average results")
print("   Apply averaged calibration for final prediction")
print()
