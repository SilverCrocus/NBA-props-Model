#!/usr/bin/env python3
"""
Apply Calibration to Full 2024-25 Backtest

Takes the uncalibrated backtest predictions and applies isotonic calibration
to validate that calibration fixes the systematic -7 pt bias.
"""

import pandas as pd
import numpy as np
import pickle
from sklearn.metrics import mean_absolute_error

# ======================================================================
# CONFIGURATION
# ======================================================================

CALIBRATED_MODEL_PATH = "models/production_model_calibrated.pkl"
UNCALIBRATED_BACKTEST_PATH = "data/results/backtest_2024_25_FIXED_V2.csv"
OUTPUT_PATH = "data/results/backtest_2024_25_CALIBRATED.csv"

print("=" * 80)
print("APPLY CALIBRATION TO FULL 2024-25 BACKTEST")
print("=" * 80)
print()

# ======================================================================
# LOAD CALIBRATOR
# ======================================================================

print("1. Loading calibrated model...")
with open(CALIBRATED_MODEL_PATH, 'rb') as f:
    model_dict = pickle.load(f)

calibrator = model_dict['calibrator']
print(f"   ✅ Calibrator loaded")
print()

# ======================================================================
# LOAD UNCALIBRATED BACKTEST
# ======================================================================

print("2. Loading uncalibrated backtest...")
df = pd.read_csv(UNCALIBRATED_BACKTEST_PATH)

print(f"   ✅ Loaded {len(df):,} predictions")
print()

# ======================================================================
# APPLY CALIBRATION
# ======================================================================

print("3. Applying calibration...")

# Get raw predictions
raw_predictions = df['predicted_PRA'].values
actual_values = df['actual_PRA'].values

# Apply calibrator
calibrated_predictions = calibrator.predict(raw_predictions)

# Calculate adjustments
adjustments = calibrated_predictions - raw_predictions

print(f"   ✅ Calibration applied")
print(f"   Mean adjustment: {adjustments.mean():.2f} pts")
print(f"   Median adjustment: {np.median(adjustments):.2f} pts")
print()

# Add to dataframe
df['predicted_PRA_raw'] = raw_predictions
df['predicted_PRA_calibrated'] = calibrated_predictions
df['calibration_adjustment'] = adjustments

# Calculate errors
df['error_calibrated'] = actual_values - calibrated_predictions

# ======================================================================
# VALIDATION METRICS
# ======================================================================

print("4. Validation metrics...")
print()

print("   BEFORE CALIBRATION:")
mae_before = df['error'].abs().mean()
residual_before = (actual_values - raw_predictions).mean()
print(f"     Mean predicted PRA: {raw_predictions.mean():.2f}")
print(f"     Mean actual PRA: {actual_values.mean():.2f}")
print(f"     Mean residual: {residual_before:+.2f} pts")
print(f"     MAE: {mae_before:.2f} pts")
print()

print("   AFTER CALIBRATION:")
mae_after = df['error_calibrated'].abs().mean()
residual_after = (actual_values - calibrated_predictions).mean()
print(f"     Mean predicted PRA: {calibrated_predictions.mean():.2f}")
print(f"     Mean actual PRA: {actual_values.mean():.2f}")
print(f"     Mean residual: {residual_after:+.2f} pts")
print(f"     MAE: {mae_after:.2f} pts")
print()

print("   IMPROVEMENT:")
print(f"     MAE reduction: {mae_before - mae_after:.2f} pts ({(mae_before - mae_after)/mae_before*100:.1f}%)")
print(f"     Bias reduction: {abs(residual_before) - abs(residual_after):.2f} pts")
print()

# ======================================================================
# SUCCESS CRITERIA CHECK
# ======================================================================

print("=" * 80)
print("SUCCESS CRITERIA VALIDATION")
print("=" * 80)
print()

success = True

# Criterion 1: Mean residual near 0
if abs(residual_after) <= 1.0:
    print("✅ Mean residual: {:.2f} pts (target: -1 to +1)".format(residual_after))
else:
    print("❌ Mean residual: {:.2f} pts (target: -1 to +1)".format(residual_after))
    success = False

# Criterion 2: MAE improvement
if mae_after < mae_before:
    print("✅ MAE improved: {:.2f} → {:.2f} pts".format(mae_before, mae_after))
else:
    print("❌ MAE did not improve: {:.2f} → {:.2f} pts".format(mae_before, mae_after))
    success = False

# Criterion 3: MAE < 7 pts
if mae_after < 7.0:
    print("✅ MAE < 7 pts: {:.2f} pts".format(mae_after))
else:
    print("⚠️  MAE ≥ 7 pts: {:.2f} pts (acceptable but not ideal)".format(mae_after))

print()

if success:
    print("🎉 CALIBRATION VALIDATION: PASSED")
else:
    print("⚠️  CALIBRATION VALIDATION: NEEDS REVIEW")

print()

# ======================================================================
# SAVE CALIBRATED BACKTEST
# ======================================================================

print("5. Saving calibrated backtest...")
df.to_csv(OUTPUT_PATH, index=False)
print(f"   ✅ Saved to {OUTPUT_PATH}")
print()

print("=" * 80)
print("CALIBRATION APPLIED SUCCESSFULLY")
print("=" * 80)
print()
print("Next step: Apply ultra-selective filters to achieve 56-58% win rate")
print("=" * 80)
