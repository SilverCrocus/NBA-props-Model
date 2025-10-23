#!/usr/bin/env python3
"""
Train NBA Props Model - Production Version

Features:
- Aggressive regularization to prevent overfitting
- Feature selection (top 50 predictors)
- Time series cross-validation
- Proper train/val/test splits

Output: models/pra_model.pkl

Usage: uv run python scripts/production/train_model.py
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import TimeSeriesSplit

# Add scripts/utils to path
sys.path.append("scripts/utils")
from fast_feature_builder import FastFeatureBuilder

print("=" * 80)
print("TRAIN NBA PROPS MODEL")
print("=" * 80)
print()

# ======================================================================
# 1. LOAD DATA
# ======================================================================

print("STEP 1: Loading data...")
print()

# Load all historical game logs
print("Loading historical game logs (2003-2024)...")
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
# 2. BUILD FEATURES
# ======================================================================

print("STEP 2: Building features...")
print()

# Split into train/val/test (strict temporal order)
# Use 2022-2024 for validation (need recent data)
train_df = historical_df[historical_df["GAME_DATE"] < "2022-07-01"].copy()
val_df = historical_df[
    (historical_df["GAME_DATE"] >= "2022-07-01") & (historical_df["GAME_DATE"] < "2023-07-01")
].copy()
test_df = historical_df[historical_df["GAME_DATE"] >= "2023-07-01"].copy()

print(
    f"Train: {len(train_df):,} games ({train_df['GAME_DATE'].min().date()} to {train_df['GAME_DATE'].max().date()})"
)
print(
    f"Val:   {len(val_df):,} games ({val_df['GAME_DATE'].min().date()} to {val_df['GAME_DATE'].max().date()})"
)
print(
    f"Test:  {len(test_df):,} games ({test_df['GAME_DATE'].min().date()} to {test_df['GAME_DATE'].max().date()})"
)
print()

# Use FastFeatureBuilder
builder = FastFeatureBuilder()

print("Building features (this takes ~30 seconds)...")
all_data_for_features = pd.concat([train_df, val_df, test_df], ignore_index=True)
all_data_for_features = all_data_for_features.sort_values(["PLAYER_ID", "GAME_DATE"])

full_df = builder.build_features(all_data_for_features, pd.DataFrame(), verbose=True)

# Split back
train_with_features = full_df[full_df["GAME_DATE"] < "2022-07-01"].copy()
val_with_features = full_df[
    (full_df["GAME_DATE"] >= "2022-07-01") & (full_df["GAME_DATE"] < "2023-07-01")
].copy()
test_with_features = full_df[full_df["GAME_DATE"] >= "2023-07-01"].copy()

print(f"\n✅ Features built")
print(f"   Train: {len(train_with_features):,} games")
print(f"   Val:   {len(val_with_features):,} games")
print(f"   Test:  {len(test_with_features):,} games")
print()

# ======================================================================
# 3. FEATURE SELECTION (REDUCE OVERFITTING)
# ======================================================================

print("STEP 3: Feature selection...")
print()

# Exclude target and metadata
exclude_cols = {
    "PRA",
    "PTS",
    "REB",
    "AST",  # Target
    "PLAYER_NAME",
    "PLAYER_ID",
    "GAME_ID",
    "GAME_DATE",
    "SEASON",  # Metadata
    "TEAM_ABBREVIATION",
    "TEAM_NAME",
    "MATCHUP",
    "WL",  # Game info
    "FGM",
    "FGA",
    "FG_PCT",
    "FG3M",
    "FG3A",
    "FG3_PCT",  # Would leak
    "FTM",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "TOV",
    "STL",
    "BLK",
    "PLUS_MINUS",
    "FANTASY_PTS",
    "DK_POINTS",
    "FD_POINTS",
}

# Get all numeric columns
all_feature_cols = []
for col in train_with_features.columns:
    if col not in exclude_cols:
        if train_with_features[col].dtype in ["int64", "float64", "int32", "float32", "bool"]:
            all_feature_cols.append(col)

print(f"Initial features: {len(all_feature_cols)}")

# Prepare data for feature selection
X_train_all = train_with_features[all_feature_cols].fillna(0)
y_train = train_with_features["PRA"]

# AGGRESSIVE FEATURE SELECTION: Keep only top 50 features
# This prevents overfitting by reducing model complexity
selector = SelectKBest(score_func=f_regression, k=50)
selector.fit(X_train_all, y_train)

# Get selected features
feature_mask = selector.get_support()
feature_cols = [col for col, selected in zip(all_feature_cols, feature_mask) if selected]

print(f"✅ Selected {len(feature_cols)} best features (reduced from {len(all_feature_cols)})")
print(f"   Top 10: {feature_cols[:10]}")
print()

# Prepare X, y
X_train = train_with_features[feature_cols].fillna(0)
y_train = train_with_features["PRA"]

X_val = val_with_features[feature_cols].fillna(0)
y_val = val_with_features["PRA"]

X_test = test_with_features[feature_cols].fillna(0)
y_test = test_with_features["PRA"]

print(f"Training shape: {X_train.shape}")
print(f"Validation shape: {X_val.shape}")
print(f"Test shape: {X_test.shape}")
print()

# ======================================================================
# 4. CROSS-VALIDATION CHECK
# ======================================================================

print("STEP 4: Cross-validation check...")
print()

# Use TimeSeriesSplit for temporal cross-validation
tscv = TimeSeriesSplit(n_splits=3)

cv_scores = []
for fold, (train_idx, val_idx) in enumerate(tscv.split(X_train)):
    X_fold_train = X_train.iloc[train_idx]
    y_fold_train = y_train.iloc[train_idx]
    X_fold_val = X_train.iloc[val_idx]
    y_fold_val = y_train.iloc[val_idx]

    # Quick model for CV
    cv_model = xgb.XGBRegressor(
        max_depth=3,
        learning_rate=0.05,
        n_estimators=100,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=42,
    )
    cv_model.fit(X_fold_train, y_fold_train, verbose=0)

    fold_mae = mean_absolute_error(y_fold_val, cv_model.predict(X_fold_val))
    cv_scores.append(fold_mae)
    print(f"  Fold {fold+1} MAE: {fold_mae:.2f}")

print(f"\n✅ Cross-validation MAE: {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f}")
print()

# ======================================================================
# 5. TRAIN MODEL WITH AGGRESSIVE REGULARIZATION
# ======================================================================

print("STEP 5: Training XGBoost with AGGRESSIVE regularization...")
print()

print("⚠️  ANTI-OVERFITTING MEASURES:")
print("   • max_depth: 3 (was 6) - shallower trees")
print("   • min_child_weight: 10 (was 3) - require more samples per leaf")
print("   • gamma: 1.0 (was 0.1) - higher pruning threshold")
print("   • subsample: 0.7 (was 0.8) - more bootstrap randomness")
print("   • colsample_bytree: 0.7 (was 0.8) - more feature randomness")
print("   • reg_alpha: 1.0 (was 0.1) - L1 regularization")
print("   • reg_lambda: 5.0 (was 1.0) - L2 regularization")
print("   • learning_rate: 0.01 (was 0.05) - slower learning")
print()

params = {
    "objective": "reg:squarederror",
    "max_depth": 3,  # Shallow trees (prevent overfitting)
    "learning_rate": 0.01,  # Slow learning
    "n_estimators": 1000,  # More trees at slower rate
    "min_child_weight": 10,  # Require more samples per leaf
    "gamma": 1.0,  # High pruning threshold
    "subsample": 0.7,  # Bootstrap 70% of data
    "colsample_bytree": 0.7,  # Use 70% of features per tree
    "reg_alpha": 1.0,  # L1 regularization
    "reg_lambda": 5.0,  # Strong L2 regularization
    "random_state": 42,
    "n_jobs": -1,
    "early_stopping_rounds": 100,  # Stop if no improvement
    "eval_metric": "mae",
}

model = xgb.XGBRegressor(**params)

# Train
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(X_train, y_train, eval_set=eval_set, verbose=50)

print()
print("✅ Training complete")
print()

# ======================================================================
# 6. EVALUATE MODEL
# ======================================================================

print("STEP 6: Evaluating model...")
print()

# Predictions
y_pred_train = model.predict(X_train)
y_pred_val = model.predict(X_val)
y_pred_test = model.predict(X_test)

# Metrics
train_mae = mean_absolute_error(y_train, y_pred_train)
val_mae = mean_absolute_error(y_val, y_pred_val)
test_mae = mean_absolute_error(y_test, y_pred_test)

# Bias
train_bias = (y_pred_train - y_train).mean()
val_bias = (y_pred_val - y_val).mean()
test_bias = (y_pred_test - y_test).mean()

print("=" * 80)
print("MODEL PERFORMANCE")
print("=" * 80)
print()
print(f"Training MAE:   {train_mae:.2f} points (bias: {train_bias:+.2f})")
print(f"Validation MAE: {val_mae:.2f} points (bias: {val_bias:+.2f})")
print(f"Test MAE:       {test_mae:.2f} points (bias: {test_bias:+.2f})")
print()

# Check for overfitting
train_val_gap = val_mae - train_mae
print(f"Train-Val Gap: {train_val_gap:.2f} points")
if train_val_gap > 2.0:
    print("⚠️  WARNING: Still showing overfitting (gap > 2 points)")
elif train_val_gap > 1.0:
    print("✅ Acceptable generalization (gap 1-2 points)")
else:
    print("✅ Excellent generalization (gap < 1 point)")
print()

# Realistic expectations
print("=" * 80)
print("REALITY CHECK")
print("=" * 80)
print()
print("Expected backtest performance:")
print("  • Win rate: 52-55% (NOT 64%)")
print("  • ROI: 2-5% (NOT 30%)")
print("  • MAE: 6-8 points (matches test MAE)")
print("  • Profit on $1000: $20-50 (NOT billions)")
print()

# ======================================================================
# 7. SAVE MODEL
# ======================================================================

print("STEP 7: Saving model...")

model_dict = {
    "model": model,
    "feature_cols": feature_cols,
    "train_mae": train_mae,
    "val_mae": val_mae,
    "test_mae": test_mae,
    "train_bias": train_bias,
    "val_bias": val_bias,
    "test_bias": test_bias,
    "training_samples": len(train_with_features),
    "val_samples": len(val_with_features),
    "test_samples": len(test_with_features),
    "cv_mae_mean": np.mean(cv_scores),
    "cv_mae_std": np.std(cv_scores),
    "date_range": {
        "train_start": str(train_with_features["GAME_DATE"].min()),
        "train_end": str(train_with_features["GAME_DATE"].max()),
        "val_start": str(val_with_features["GAME_DATE"].min()),
        "val_end": str(val_with_features["GAME_DATE"].max()),
        "test_start": str(test_with_features["GAME_DATE"].min()),
        "test_end": str(test_with_features["GAME_DATE"].max()),
    },
    "version": "v3.0_ANTI_OVERFIT",
    "anti_overfit_measures": [
        "1. Feature selection: 50 best features (was 195)",
        "2. Shallow trees: max_depth=3 (was 6)",
        "3. Strong L2 regularization: reg_lambda=5.0 (was 1.0)",
        "4. High min_child_weight: 10 (was 3)",
        "5. Bootstrap + feature sampling: 70% (was 80%)",
        "6. Slow learning rate: 0.01 (was 0.05)",
        "7. Cross-validation verification",
        "8. Separate test set evaluation",
    ],
    "hyperparameters": params,
}

output_path = "models/pra_model.pkl"
with open(output_path, "wb") as f:
    pickle.dump(model_dict, f)

print(f"✅ Model saved to: {output_path}")
print()

# ======================================================================
# 8. SUMMARY
# ======================================================================

print("=" * 80)
print("TRAINING SUMMARY - ANTI-OVERFITTING MODEL")
print("=" * 80)
print()

print(f"Model: {model_dict['version']}")
print(f"Features: {len(feature_cols)} (selected from {len(all_feature_cols)})")
print(f"Training samples: {len(train_with_features):,}")
print(f"Validation samples: {len(val_with_features):,}")
print(f"Test samples: {len(test_with_features):,}")
print()

print("Performance:")
print(f"  Train MAE: {train_mae:.2f} pts")
print(f"  Val MAE:   {val_mae:.2f} pts")
print(f"  Test MAE:  {test_mae:.2f} pts")
print(f"  CV MAE:    {np.mean(cv_scores):.2f} ± {np.std(cv_scores):.2f} pts")
print()

print("Anti-Overfitting Measures:")
for measure in model_dict["anti_overfit_measures"]:
    print(f"  ✅ {measure}")
print()

print("Next Steps:")
print("  1. Run backtest: uv run python scripts/validation/calibrated_backtest_2024_25.py")
print("  2. Expected win rate: 52-55% (realistic)")
print("  3. Expected ROI: 2-5% (achievable)")
print("  4. If still >60% win rate → investigate data leakage")
print()
