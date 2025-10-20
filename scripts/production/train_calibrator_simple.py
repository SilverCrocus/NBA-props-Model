#!/usr/bin/env python3
"""
Train Isotonic Calibrator - Simple Version (No Data Leakage)

Train on 2023-24 walk-forward predictions (out-of-sample)
Apply to 2024-25 predictions (truly future data)
"""

import pickle

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.metrics import mean_absolute_error

print("=" * 80)
print("TRAIN ISOTONIC CALIBRATOR - LEAK-FREE VERSION")
print("=" * 80)
print()

# Load 2023-24 walk-forward predictions (OUT-OF-SAMPLE)
print("1. Loading 2023-24 validation predictions...")
df_2023 = pd.read_csv("data/results/backtest_walkforward_2023_24.csv")
df_2023 = df_2023.dropna(subset=["predicted_pra", "actual_pra"])

print(f"   ✅ Loaded {len(df_2023):,} predictions from 2023-24")
print(f"   Date range: {df_2023['game_date'].min()} to {df_2023['game_date'].max()}")
print()

# Extract predictions and actuals
y_pred_2023 = df_2023["predicted_pra"].values
y_actual_2023 = df_2023["actual_pra"].values

# Calculate baseline MAE
mae_before = mean_absolute_error(y_actual_2023, y_pred_2023)
mean_residual_before = (y_pred_2023 - y_actual_2023).mean()

print("2. Baseline performance (before calibration):")
print(f"   MAE: {mae_before:.2f} pts")
print(f"   Mean residual: {mean_residual_before:+.2f} pts")
print()

# Train isotonic calibrator
print("3. Training isotonic calibrator on 2023-24 data...")
calibrator = IsotonicRegression(out_of_bounds="clip")
calibrator.fit(y_pred_2023, y_actual_2023)
print("   ✅ Calibrator trained")
print()

# Apply calibration to 2023-24 (sanity check)
y_pred_2023_calibrated = calibrator.predict(y_pred_2023)
mae_after = mean_absolute_error(y_actual_2023, y_pred_2023_calibrated)
mean_residual_after = (y_pred_2023_calibrated - y_actual_2023).mean()

print("4. Calibration performance on training data (2023-24):")
print(f"   MAE: {mae_after:.2f} pts (before: {mae_before:.2f})")
print(f"   Mean residual: {mean_residual_after:+.2f} pts (before: {mean_residual_before:+.2f})")
print(f"   Improvement: {mae_before - mae_after:.2f} pts")
print()

# Save calibrator
print("5. Saving calibrator...")
calibrator_dict = {
    "calibrator": calibrator,
    "training_date": "2023-24 season",
    "training_samples": len(df_2023),
    "mae_before": mae_before,
    "mae_after": mae_after,
    "mean_residual_before": mean_residual_before,
    "mean_residual_after": mean_residual_after,
}

with open("models/isotonic_calibrator_FIXED.pkl", "wb") as f:
    pickle.dump(calibrator_dict, f)

print("   ✅ Saved: models/isotonic_calibrator_FIXED.pkl")
print()

# Now apply to 2024-25 (OUT-OF-SAMPLE TEST)
print("6. Applying calibrator to 2024-25 predictions (OUT-OF-SAMPLE)...")
df_2024 = pd.read_csv("data/results/backtest_walkforward_2024_25.csv")
df_2024 = df_2024.dropna(subset=["predicted_pra", "actual_pra"])

print(f"   ✅ Loaded {len(df_2024):,} predictions from 2024-25")
print()

y_pred_2024_raw = df_2024["predicted_pra"].values
y_actual_2024 = df_2024["actual_pra"].values

# Apply calibration
y_pred_2024_calibrated = calibrator.predict(y_pred_2024_raw)

# Save calibrated predictions
df_2024["predicted_PRA_calibrated"] = y_pred_2024_calibrated
df_2024.to_csv("data/results/backtest_2024_25_CALIBRATED_FIXED.csv", index=False)

print("   ✅ Saved: data/results/backtest_2024_25_CALIBRATED_FIXED.csv")
print()

# Calculate performance
mae_2024_before = mean_absolute_error(y_actual_2024, y_pred_2024_raw)
mae_2024_after = mean_absolute_error(y_actual_2024, y_pred_2024_calibrated)
mean_residual_2024_before = (y_pred_2024_raw - y_actual_2024).mean()
mean_residual_2024_after = (y_pred_2024_calibrated - y_actual_2024).mean()

print("7. TRUE OUT-OF-SAMPLE PERFORMANCE (2024-25):")
print(f"   MAE before calibration: {mae_2024_before:.2f} pts")
print(f"   MAE after calibration:  {mae_2024_after:.2f} pts")
print(f"   Mean residual before: {mean_residual_2024_before:+.2f} pts")
print(f"   Mean residual after:  {mean_residual_2024_after:+.2f} pts")
print()

if mae_2024_after < mae_2024_before:
    print(f"   ✅ Calibration improved MAE by {mae_2024_before - mae_2024_after:.2f} pts")
else:
    print(f"   ⚠️  Calibration did not improve MAE (this is normal for out-of-sample)")

if abs(mean_residual_2024_after) < abs(mean_residual_2024_before):
    print(
        f"   ✅ Calibration reduced bias by {abs(mean_residual_2024_before) - abs(mean_residual_2024_after):.2f} pts"
    )
else:
    print(f"   ⚠️  Calibration did not reduce bias")

print()
print("=" * 80)
print("✅ CALIBRATION COMPLETE - NO DATA LEAKAGE")
print("=" * 80)
print()
print("Next steps:")
print("1. Use backtest_2024_25_CALIBRATED_FIXED.csv for betting strategy")
print("2. Apply ultra-selective filters to get realistic win rate")
print("3. Verify win rate is 54-58% (not 63%)")
