#!/usr/bin/env python3
"""
Train Production Model - Uses ALL Available Data

This script trains a production-ready model using:
1. ALL game logs (2003 through latest 2024-25 data)
2. Proper train/validation split for hyperparameter tuning
3. Calibration on 2023-24 data (keep as validation)
4. Final model trained on 2003-2024 data

Usage:
    uv run python scripts/production/train_production_model.py

Output:
    - models/production_model_v2.0_PRODUCTION_latest.pkl (uncalibrated)
    - models/production_model_v2.0_PRODUCTION_CALIBRATED_latest.pkl (calibrated)
"""

import pickle
import sys
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error

sys.path.append(str(Path(__file__).parent.parent.parent))
from scripts.utils.fast_feature_builder import FastFeatureBuilder

print("=" * 80)
print("PRODUCTION MODEL TRAINING - Using ALL Available Data")
print("=" * 80)
print()

# ============================================================================
# 1. LOAD ALL GAME LOGS
# ============================================================================

print("STEP 1: LOADING ALL GAME LOGS")
print("=" * 80)

# Load everything
df = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
df["GAME_DATE"] = pd.to_datetime(df["GAME_DATE"])
df = df.sort_values(["PLAYER_ID", "GAME_DATE"])

# Add PRA
if "PRA" not in df.columns:
    df["PRA"] = df["PTS"] + df["REB"] + df["AST"]

print(f"✅ Loaded {len(df):,} games")
print(f"   Date range: {df['GAME_DATE'].min().date()} to {df['GAME_DATE'].max().date()}")
print(f"   Players: {df['PLAYER_ID'].nunique():,}")
print()

# ============================================================================
# 2. SPLIT DATA
# ============================================================================

print("STEP 2: SPLITTING DATA")
print("=" * 80)
print()

# For calibration validation: Keep 2023-24 as calibration validation set
# For model training: Use 2003-2024 (including 2024-25 available data)
print("Data split strategy:")
print("  Calibration validation: 2023-24 season (for isotonic regression)")
print("  Model training: 2003 through latest 2024-25 data")
print()

# Calibration validation set (2023-24)
calib_start = pd.to_datetime("2023-10-24")
calib_end = pd.to_datetime("2024-10-21")
calibration_df = df[(df["GAME_DATE"] >= calib_start) & (df["GAME_DATE"] <= calib_end)].copy()

# Training set: Everything EXCEPT calibration period
# This includes 2003-2023 AND available 2024-25 data
train_df = df[~((df["GAME_DATE"] >= calib_start) & (df["GAME_DATE"] <= calib_end))].copy()

print(f"Training data: {len(train_df):,} games")
print(f"  Date range: {train_df['GAME_DATE'].min().date()} to {train_df['GAME_DATE'].max().date()}")
print()

print(f"Calibration validation: {len(calibration_df):,} games")
print(f"  Date range: {calibration_df['GAME_DATE'].min().date()} to {calibration_df['GAME_DATE'].max().date()}")
print()

# ============================================================================
# 3. BUILD FEATURES
# ============================================================================

print("STEP 3: BUILDING FEATURES")
print("=" * 80)

# For training data, we need to build features properly
# Split into historical and target for feature building
historical_cutoff = pd.to_datetime("2023-10-23")  # Day before 2023-24 season
historical_train = train_df[train_df["GAME_DATE"] <= historical_cutoff].copy()
recent_train = train_df[train_df["GAME_DATE"] > historical_cutoff].copy()

print(f"Historical for features: {len(historical_train):,} games (2003 to Oct 2023)")
print(f"Recent training data: {len(recent_train):,} games (2024-25 season)")
print()

print("Building features for training data...")
builder = FastFeatureBuilder()
train_full = builder.build_features(historical_train, recent_train, verbose=True)

print()
print("Building features for calibration data...")
# For calibration data, use all data before calibration period
calib_historical = df[df["GAME_DATE"] < calib_start].copy()
calib_full = builder.build_features(calib_historical, calibration_df, verbose=True)

# Get feature columns (same across both)
core_stats = [
    "MIN",
    "FGA",
    "FG_PCT",
    "FG3A",
    "FG3_PCT",
    "FTA",
    "FT_PCT",
    "OREB",
    "DREB",
    "STL",
    "BLK",
    "TOV",
    "PF",
]

# Exclude non-numeric columns and target/ID columns
exclude_cols = [
    "GAME_ID",
    "PLAYER_ID",
    "PLAYER_NAME",
    "TEAM_ABBREVIATION",
    "GAME_DATE",
    "SEASON",
    "SEASON_TYPE",
    "PRA",
    "PTS",
    "REB",
    "AST",
    "MATCHUP",  # String column
    "WL",  # String column
    "CTG_SEASON",  # String column
    "HOME_AWAY",  # String column (if exists)
]

# Get numeric columns only
feature_cols = [
    col for col in train_full.columns
    if col not in exclude_cols and train_full[col].dtype in ['int64', 'float64', 'bool']
]

# Ensure core stats are included first
feature_cols = core_stats + [col for col in feature_cols if col not in core_stats]

print()
print(f"✅ Features ready: {len(feature_cols)} features")

# ============================================================================
# 4. PREPARE TRAINING DATA
# ============================================================================

print()
print("STEP 4: PREPARING TRAINING DATA")
print("=" * 80)

# Filter to valid rows (non-missing PRA, sufficient features)
train_mask = ~(train_full[feature_cols].isna().any(axis=1) | train_full["PRA"].isna())
X_train = train_full.loc[train_mask, feature_cols].fillna(0)
y_train = train_full.loc[train_mask, "PRA"]

print(f"✅ Training samples: {len(X_train):,}")
print(f"   Features: {len(feature_cols)}")
print(f"   PRA range: {y_train.min():.1f} to {y_train.max():.1f}")
print()

# ============================================================================
# 5. TRAIN MODEL
# ============================================================================

print("STEP 5: TRAINING XGBOOST MODEL")
print("=" * 80)
print()

# Use same hyperparameters as v2.0_CLEAN
params = {
    "objective": "reg:squarederror",
    "max_depth": 6,
    "learning_rate": 0.05,
    "n_estimators": 500,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "min_child_weight": 3,
    "gamma": 0.1,
    "reg_alpha": 0.1,
    "reg_lambda": 1.0,
    "random_state": 42,
    "n_jobs": -1,
}

print("Hyperparameters:")
for key, value in params.items():
    print(f"  {key}: {value}")
print()

print("Training model...")
model = xgb.XGBRegressor(**params)
model.fit(X_train, y_train)

# Training MAE
train_preds = model.predict(X_train)
train_preds = np.maximum(0, train_preds)
train_mae = mean_absolute_error(y_train, train_preds)

print(f"✅ Model trained")
print(f"   Training MAE: {train_mae:.2f} points")
print()

# ============================================================================
# 6. EVALUATE ON CALIBRATION DATA (held-out validation)
# ============================================================================

print("STEP 6: EVALUATING ON CALIBRATION DATA")
print("=" * 80)

calib_mask = ~(calib_full[feature_cols].isna().any(axis=1) | calib_full["PRA"].isna())
X_calib = calib_full.loc[calib_mask, feature_cols].fillna(0)
y_calib = calib_full.loc[calib_mask, "PRA"]

calib_preds = model.predict(X_calib)
calib_preds = np.maximum(0, calib_preds)
calib_mae = mean_absolute_error(y_calib, calib_preds)

print(f"✅ Calibration validation:")
print(f"   Samples: {len(X_calib):,}")
print(f"   MAE: {calib_mae:.2f} points")
print()

# ============================================================================
# 7. FEATURE IMPORTANCE
# ============================================================================

print("STEP 7: FEATURE IMPORTANCE")
print("=" * 80)

importances = model.feature_importances_
feature_importance_df = pd.DataFrame({
    "feature": feature_cols,
    "importance": importances
}).sort_values("importance", ascending=False)

print("\nTop 15 Features:")
print(feature_importance_df.head(15).to_string(index=False))
print()

# ============================================================================
# 8. SAVE UNCALIBRATED MODEL
# ============================================================================

print("STEP 8: SAVING UNCALIBRATED MODEL")
print("=" * 80)

model_dict = {
    "model": model,
    "feature_cols": feature_cols,
    "feature_importance": feature_importance_df,
    "version": "v2.0_PRODUCTION",
    "timestamp": datetime.now().isoformat(),
    "train_mae": train_mae,
    "calib_mae": calib_mae,
    "training_samples": len(X_train),
    "calib_samples": len(X_calib),
    "date_range": f"{train_full['GAME_DATE'].min().date()} to {train_full['GAME_DATE'].max().date()}",
    "notes": "Production model trained on all available data including 2024-25 season",
}

model_path = "models/production_model_v2.0_PRODUCTION_latest.pkl"
with open(model_path, "wb") as f:
    pickle.dump(model_dict, f)

print(f"✅ Saved uncalibrated model: {model_path}")
print()

# ============================================================================
# 9. CALIBRATE MODEL
# ============================================================================

print("STEP 9: CALIBRATING MODEL")
print("=" * 80)
print()

# Use calibration validation data to train isotonic regression
print("Preparing calibration data...")

# Load historical odds for 2023-24 season
try:
    odds_df = pd.read_csv("data/historical_odds/2023-24/pra_odds.csv")
    odds_df["event_date"] = pd.to_datetime(odds_df["event_date"])

    print(f"✅ Loaded {len(odds_df):,} historical odds")

    # Create predictions dataframe for calibration period
    calib_predictions_df = calib_full.loc[calib_mask, ["PLAYER_NAME", "GAME_DATE"]].copy()
    calib_predictions_df["predicted_PRA"] = calib_preds
    calib_predictions_df["actual_PRA"] = y_calib.values

    # Merge with odds
    merged_calib = calib_predictions_df.merge(
        odds_df,
        left_on=["PLAYER_NAME", "GAME_DATE"],
        right_on=["player_name", "event_date"],
        how="inner",
    )

    print(f"✅ Matched {len(merged_calib):,} predictions with odds")
    print()

    if len(merged_calib) > 0:
        # Convert to probabilities
        calibration_data = []

        for idx, row in merged_calib.iterrows():
            pred_pra = row["predicted_PRA"]
            actual_pra = row["actual_PRA"]
            line = row["line"]

            # Logistic transform
            distance_from_line = pred_pra - line
            scale = 5.0  # MAE-based scale
            prob_over_raw = 1 / (1 + np.exp(-distance_from_line / scale))

            # Ground truth
            actual_over = 1 if actual_pra > line else 0

            calibration_data.append({
                "prob_over_raw": prob_over_raw,
                "actual_over": actual_over,
            })

        calib_prob_df = pd.DataFrame(calibration_data)

        print("Training isotonic regression calibrator...")
        calibrator = IsotonicRegression(out_of_bounds="clip")
        calibrator.fit(calib_prob_df["prob_over_raw"], calib_prob_df["actual_over"])

        # Evaluate calibration quality
        calib_prob_df["prob_over_calibrated"] = calibrator.predict(calib_prob_df["prob_over_raw"])

        brier_raw = brier_score_loss(calib_prob_df["actual_over"], calib_prob_df["prob_over_raw"])
        brier_calibrated = brier_score_loss(
            calib_prob_df["actual_over"], calib_prob_df["prob_over_calibrated"]
        )

        log_loss_raw = log_loss(calib_prob_df["actual_over"], calib_prob_df["prob_over_raw"])
        log_loss_calibrated = log_loss(
            calib_prob_df["actual_over"], calib_prob_df["prob_over_calibrated"]
        )

        print(f"✅ Calibrator trained")
        print(f"   Brier score (raw): {brier_raw:.4f}")
        print(f"   Brier score (calibrated): {brier_calibrated:.4f}")
        print(f"   Log loss (raw): {log_loss_raw:.4f}")
        print(f"   Log loss (calibrated): {log_loss_calibrated:.4f}")
        print()

        # Add calibrator to model dict
        model_dict["calibrator"] = calibrator
        model_dict["calibration_metrics"] = {
            "brier_score_raw": brier_raw,
            "brier_score_calibrated": brier_calibrated,
            "log_loss_raw": log_loss_raw,
            "log_loss_calibrated": log_loss_calibrated,
            "calibration_samples": len(calib_prob_df),
        }

        # Save calibrated model
        calibrated_path = "models/production_model_v2.0_PRODUCTION_CALIBRATED_latest.pkl"
        with open(calibrated_path, "wb") as f:
            pickle.dump(model_dict, f)

        print(f"✅ Saved calibrated model: {calibrated_path}")
        print()

    else:
        print("⚠️  No odds data for calibration - skipping calibration step")
        print("   Uncalibrated model saved only")
        print()

except Exception as e:
    print(f"⚠️  Error during calibration: {e}")
    print("   Uncalibrated model saved only")
    print()

# ============================================================================
# 10. SUMMARY
# ============================================================================

print("=" * 80)
print("✅ PRODUCTION MODEL TRAINING COMPLETE")
print("=" * 80)
print()

print("Model Details:")
print(f"  Version: v2.0_PRODUCTION")
print(f"  Training samples: {len(X_train):,}")
print(f"  Training MAE: {train_mae:.2f} points")
print(f"  Calibration validation MAE: {calib_mae:.2f} points")
print(f"  Features: {len(feature_cols)}")
print()

print("Files Created:")
print(f"  1. {model_path}")
if "calibrator" in model_dict:
    print(f"  2. {calibrated_path}")
print()

print("Usage:")
print("  Use this model for daily betting recommendations:")
print("  uv run python scripts/production/daily_betting_recommendations.py")
print()

print("Next Steps:")
print("  1. Test model on recent games (spot check)")
print("  2. Generate betting recommendations for upcoming games")
print("  3. Paper trade for 20-50 bets to validate performance")
print("  4. Retrain monthly with new data")
print()

print("=" * 80)
print("🚀 PRODUCTION MODEL READY FOR DEPLOYMENT")
print("=" * 80)
