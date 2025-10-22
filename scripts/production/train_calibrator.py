#!/usr/bin/env python3
"""
Train Isotonic Regression Calibrator

This calibrates the model's predicted probabilities to actual win rates.
Isotonic regression is ideal for calibration because:
1. Non-parametric (doesn't assume linear relationship)
2. Preserves ranking (monotonic transformation)
3. Reduces overconfidence in extreme predictions
"""

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import brier_score_loss, log_loss

print("=" * 80)
print("TRAIN ISOTONIC REGRESSION CALIBRATOR")
print("=" * 80)
print()

# ======================================================================
# 1. LOAD MODEL
# ======================================================================

print("STEP 1: Loading trained model...")

with open("models/pra_model.pkl", "rb") as f:
    model_dict = pickle.load(f)

model = model_dict["model"]
feature_cols = model_dict["feature_cols"]

print(f"✅ Model loaded: {model_dict['version']}")
print(f"   Features: {len(feature_cols)}")
print(f"   Val MAE: {model_dict['val_mae']:.2f}")
print(f"   Val Bias: {model_dict['val_bias']:.2f}")
print()

# ======================================================================
# 2. LOAD VALIDATION DATA
# ======================================================================

print("STEP 2: Loading validation data...")

# Load historical game logs
historical_df = pd.read_csv("data/game_logs/all_game_logs_through_2025.csv")
historical_df["GAME_DATE"] = pd.to_datetime(historical_df["GAME_DATE"])

# Filter to validation period (2023-24 season)
val_df = historical_df[
    (historical_df["GAME_DATE"] >= "2023-06-30") & (historical_df["GAME_DATE"] < "2024-10-22")
].copy()

# Filter to MIN >= 25 (match betting population)
val_df = val_df[val_df["MIN"] >= 25].copy()

# Add PRA if not present
if "PRA" not in val_df.columns:
    val_df["PRA"] = val_df["PTS"] + val_df["REB"] + val_df["AST"]

print(f"✅ Validation data: {len(val_df):,} games")
print(f"   Date range: {val_df['GAME_DATE'].min().date()} to {val_df['GAME_DATE'].max().date()}")
print(f"   PRA avg: {val_df['PRA'].mean():.2f}")
print()

# ======================================================================
# 3. BUILD FEATURES FOR VALIDATION SET
# ======================================================================

print("STEP 3: Building features for validation set...")
print("(Using FastFeatureBuilder)")
print()

# Add scripts/utils to path for FastFeatureBuilder
sys.path.append("scripts/utils")
from fast_feature_builder import FastFeatureBuilder

# Need historical data before validation for feature building
train_historical = historical_df[historical_df["GAME_DATE"] < "2023-06-30"].copy()
train_historical = train_historical[train_historical["MIN"] >= 25].copy()

# Combine for feature building
combined = pd.concat([train_historical, val_df], ignore_index=True)
combined = combined.sort_values(["PLAYER_ID", "GAME_DATE"])

# Build features
builder = FastFeatureBuilder()
combined_with_features = builder.build_features(combined, pd.DataFrame(), verbose=False)

# Extract validation set
val_with_features = combined_with_features[
    combined_with_features["GAME_DATE"] >= "2023-06-30"
].copy()

print(f"✅ Features built: {len(val_with_features.columns)} columns")
print()

# ======================================================================
# 4. GENERATE PREDICTIONS
# ======================================================================

print("STEP 4: Generating predictions on validation set...")

X_val = val_with_features[feature_cols].fillna(0)
y_val = val_with_features["PRA"]

# Get predicted PRA values
y_pred = model.predict(X_val)

print(f"✅ Predictions generated: {len(y_pred):,}")
print(f"   MAE: {np.abs(y_pred - y_val).mean():.2f}")
print(f"   Bias: {(y_pred - y_val).mean():.2f}")
print()

# ======================================================================
# 5. CREATE CALIBRATION DATASET
# ======================================================================

print("STEP 5: Creating calibration dataset...")
print("(Converting predictions to binary outcomes)")
print()

# For calibration, we need to convert to binary classification
# We'll use a sliding window of PRA thresholds (similar to betting lines)

calibration_data = []

# Create synthetic "betting lines" at various thresholds
# This simulates what we see in real betting markets
thresholds = np.arange(10, 50, 0.5)  # PRA lines from 10 to 50

for threshold in thresholds:
    # For each game, check if actual PRA went OVER the threshold
    went_over = (y_val >= threshold).astype(int)

    # Predicted probability of OVER
    # Assume normal distribution around prediction with std ~7 points (empirical)
    from scipy.stats import norm

    std_dev = 7.0  # Average prediction error
    prob_over = 1 - norm.cdf(threshold, loc=y_pred, scale=std_dev)

    # Add to calibration dataset
    for actual, pred_prob in zip(went_over, prob_over):
        if 0.01 < pred_prob < 0.99:  # Only use non-extreme predictions
            calibration_data.append({"predicted_prob": pred_prob, "actual_outcome": actual})

calib_df = pd.DataFrame(calibration_data)

print(f"✅ Calibration dataset created: {len(calib_df):,} samples")
print(f"   Mean predicted prob: {calib_df['predicted_prob'].mean():.3f}")
print(f"   Actual win rate: {calib_df['actual_outcome'].mean():.3f}")
print()

# ======================================================================
# 6. TRAIN ISOTONIC REGRESSION
# ======================================================================

print("STEP 6: Training isotonic regression calibrator...")

# Split calibration data into train/test
n_train = int(len(calib_df) * 0.8)
calib_train = calib_df.iloc[:n_train]
calib_test = calib_df.iloc[n_train:]

X_calib_train = calib_train["predicted_prob"].values.reshape(-1, 1)
y_calib_train = calib_train["actual_outcome"].values

X_calib_test = calib_test["predicted_prob"].values.reshape(-1, 1)
y_calib_test = calib_test["actual_outcome"].values

# Train isotonic regression
calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)
calibrator.fit(X_calib_train.ravel(), y_calib_train)

print(f"✅ Calibrator trained")
print()

# ======================================================================
# 7. EVALUATE CALIBRATION
# ======================================================================

print("STEP 7: Evaluating calibration...")
print()

# Predictions before calibration
y_pred_prob_uncalibrated = X_calib_test.ravel()

# Predictions after calibration
y_pred_prob_calibrated = calibrator.predict(X_calib_test.ravel())

# Brier score (lower is better)
brier_before = brier_score_loss(y_calib_test, y_pred_prob_uncalibrated)
brier_after = brier_score_loss(y_calib_test, y_pred_prob_calibrated)

# Log loss (lower is better)
logloss_before = log_loss(y_calib_test, y_pred_prob_uncalibrated)
logloss_after = log_loss(y_calib_test, y_pred_prob_calibrated)

print("Calibration Quality Metrics:")
print(f"  Brier Score:")
print(f"    Before: {brier_before:.4f}")
print(f"    After:  {brier_after:.4f}")
print(f"    Improvement: {(brier_before - brier_after)/brier_before*100:.1f}%")
print()
print(f"  Log Loss:")
print(f"    Before: {logloss_before:.4f}")
print(f"    After:  {logloss_after:.4f}")
print(f"    Improvement: {(logloss_before - logloss_after)/logloss_before*100:.1f}%")
print()

# Show calibration curve samples
print("Calibration Curve (sample points):")
print(f"{'Predicted Prob':<20} {'Calibrated Prob':<20} {'Adjustment':<15}")
print("-" * 60)
sample_probs = [0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
for p in sample_probs:
    p_calibrated = calibrator.predict([p])[0]
    adjustment = p_calibrated - p
    print(f"{p:<20.3f} {p_calibrated:<20.3f} {adjustment:+.3f}")
print()

# ======================================================================
# 8. SAVE CALIBRATOR
# ======================================================================

print("STEP 8: Saving calibrator to model...")

# Add calibrator to model dict
model_dict["calibrator"] = calibrator
model_dict["calibration_metrics"] = {
    "brier_score_before": brier_before,
    "brier_score_after": brier_after,
    "log_loss_before": logloss_before,
    "log_loss_after": logloss_after,
    "calibration_samples": len(calib_df),
}

# Save updated model
with open("models/pra_model.pkl", "wb") as f:
    pickle.dump(model_dict, f)

# Also save to BIAS_FIXED version
with open("models/pra_model.pkl", "wb") as f:
    pickle.dump(model_dict, f)

print(f"✅ Calibrator saved to model")
print(f"   Calibration samples: {len(calib_df):,}")
print(f"   Brier score improvement: {(brier_before - brier_after)/brier_before*100:.1f}%")
print()

# ======================================================================
# 9. SUMMARY
# ======================================================================

print("=" * 80)
print("CALIBRATION SUMMARY")
print("=" * 80)
print()

print(f"Model: {model_dict['version']}")
print(f"Calibration method: Isotonic Regression")
print(f"Calibration samples: {len(calib_df):,}")
print()

print("Performance:")
print(
    f"  Brier Score: {brier_before:.4f} → {brier_after:.4f} ({(brier_before - brier_after)/brier_before*100:.1f}% improvement)"
)
print(
    f"  Log Loss: {logloss_before:.4f} → {logloss_after:.4f} ({(logloss_before - logloss_after)/logloss_before*100:.1f}% improvement)"
)
print()

print("What this means:")
print("  • Win probabilities are now better calibrated")
print("  • Extreme predictions (very high/low) are adjusted")
print("  • Edges will be more realistic (likely smaller)")
print("  • Betting strategy will be more conservative")
print()

print("Next Steps:")
print("  1. Run backtest with calibrated model")
print("  2. Compare edges: uncalibrated vs calibrated")
print("  3. Check if profit is still strong with realistic edges")
print()
