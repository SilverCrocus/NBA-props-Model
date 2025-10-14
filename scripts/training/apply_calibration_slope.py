"""
Apply Calibration Slope Adjustment to Fix Over-Confident Predictions

The model is over-predicting extremes:
- High PRA predictions are too high
- Low PRA predictions are too low

Calibration slope adjustment:
1. Fit linear regression: actual = slope * predicted + intercept
2. Apply calibration: calibrated = intercept + slope * raw_pred
3. If slope < 1, this "shrinks" predictions toward mean (fixes over-confidence)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt

print("="*80)
print("CALIBRATION SLOPE ADJUSTMENT")
print("="*80)

# Load walk-forward predictions for 2024-25 (for calibration analysis)
print("\n1. Loading 2024-25 walk-forward predictions...")
df = pd.read_csv('data/results/walkforward_predictions_2024-25.csv')

print(f"✅ Loaded {len(df):,} predictions")
print(f"   MAE (uncalibrated): {mean_absolute_error(df['PRA'], df['predicted_PRA']):.2f}")

# STEP 1: Analyze calibration
print("\n2. Analyzing calibration...")

X = df['predicted_PRA'].values.reshape(-1, 1)
y = df['PRA'].values

lr = LinearRegression()
lr.fit(X, y)

slope = lr.coef_[0]
intercept = lr.intercept_

print(f"   Calibration slope: {slope:.4f}")
print(f"   Calibration intercept: {intercept:.4f}")
print(f"   Perfect calibration: slope=1.0, intercept=0.0")

if slope < 1.0:
    print(f"   ⚠️  Slope < 1.0 indicates OVER-CONFIDENCE (predictions too extreme)")
    print(f"   Model over-predicts high values and under-predicts low values")
elif slope > 1.0:
    print(f"   ⚠️  Slope > 1.0 indicates UNDER-CONFIDENCE (predictions too conservative)")
else:
    print(f"   ✅ Slope ≈ 1.0 indicates good calibration")

# STEP 2: Apply calibration
print("\n3. Applying calibration slope adjustment...")

df['calibrated_PRA'] = intercept + slope * df['predicted_PRA']

# Calculate improved MAE
mae_before = mean_absolute_error(df['PRA'], df['predicted_PRA'])
mae_after = mean_absolute_error(df['PRA'], df['calibrated_PRA'])

print(f"   MAE before calibration: {mae_before:.2f}")
print(f"   MAE after calibration:  {mae_after:.2f}")
print(f"   Improvement: {mae_before - mae_after:+.2f} points ({(mae_after/mae_before - 1)*100:+.1f}%)")

# STEP 3: Analyze calibration by prediction range
print("\n4. Calibration analysis by prediction range...")

bins = [0, 15, 20, 25, 30, 35, 40, 100]
labels = ['<15', '15-20', '20-25', '25-30', '30-35', '35-40', '40+']

df['pred_bin'] = pd.cut(df['predicted_PRA'], bins=bins, labels=labels)

print("\nPrediction Range | Count | Avg Pred | Avg Actual | Bias | MAE Before | MAE After")
print("-"*80)

for bin_label in labels:
    subset = df[df['pred_bin'] == bin_label]
    if len(subset) == 0:
        continue

    avg_pred = subset['predicted_PRA'].mean()
    avg_actual = subset['PRA'].mean()
    bias = avg_pred - avg_actual
    mae_before_bin = mean_absolute_error(subset['PRA'], subset['predicted_PRA'])
    mae_after_bin = mean_absolute_error(subset['PRA'], subset['calibrated_PRA'])

    print(f"{bin_label:>16} | {len(subset):>5,} | {avg_pred:>8.2f} | {avg_actual:>10.2f} | "
          f"{bias:>+4.2f} | {mae_before_bin:>10.2f} | {mae_after_bin:>9.2f}")

# STEP 4: Save calibrated predictions
print("\n5. Saving calibrated predictions...")

output_file = 'data/results/walkforward_predictions_2024-25_calibrated.csv'
df.to_csv(output_file, index=False)
print(f"✅ Saved to {output_file}")

# STEP 5: Create calibration plot
print("\n6. Creating calibration plot...")

fig, axes = plt.subplots(1, 2, figsize=(14, 6))

# Plot 1: Calibration curve (before)
axes[0].scatter(df['predicted_PRA'], df['PRA'], alpha=0.1, s=10)
axes[0].plot([0, 60], [0, 60], 'r--', label='Perfect calibration (y=x)', linewidth=2)
axes[0].plot(df['predicted_PRA'].sort_values(),
             intercept + slope * df['predicted_PRA'].sort_values(),
             'g-', label=f'Fitted (y={slope:.3f}x + {intercept:.2f})', linewidth=2)
axes[0].set_xlabel('Predicted PRA')
axes[0].set_ylabel('Actual PRA')
axes[0].set_title(f'Calibration Curve (Before)\nSlope={slope:.4f}, MAE={mae_before:.2f}')
axes[0].legend()
axes[0].grid(alpha=0.3)

# Plot 2: Calibration curve (after)
axes[1].scatter(df['calibrated_PRA'], df['PRA'], alpha=0.1, s=10)
axes[1].plot([0, 60], [0, 60], 'r--', label='Perfect calibration (y=x)', linewidth=2)
axes[1].set_xlabel('Calibrated PRA')
axes[1].set_ylabel('Actual PRA')
axes[1].set_title(f'Calibration Curve (After)\nMAE={mae_after:.2f} (improved {mae_before - mae_after:.2f} pts)')
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plot_file = 'data/results/calibration_plot_2024_25.png'
plt.savefig(plot_file, dpi=150, bbox_inches='tight')
print(f"✅ Saved calibration plot to {plot_file}")

# STEP 6: Apply same calibration to 2023-24
print("\n7. Applying calibration to 2023-24 predictions...")

df_2023 = pd.read_csv('data/results/walkforward_predictions_2023-24.csv')
df_2023['calibrated_PRA'] = intercept + slope * df_2023['predicted_PRA']

mae_2023_before = mean_absolute_error(df_2023['PRA'], df_2023['predicted_PRA'])
mae_2023_after = mean_absolute_error(df_2023['PRA'], df_2023['calibrated_PRA'])

print(f"   2023-24 MAE before: {mae_2023_before:.2f}")
print(f"   2023-24 MAE after:  {mae_2023_after:.2f}")
print(f"   Improvement: {mae_2023_before - mae_2023_after:+.2f} points")

output_file_2023 = 'data/results/walkforward_predictions_2023-24_calibrated.csv'
df_2023.to_csv(output_file_2023, index=False)
print(f"✅ Saved to {output_file_2023}")

# Summary
print("\n" + "="*80)
print("CALIBRATION SLOPE ADJUSTMENT SUMMARY")
print("="*80)

print(f"""
Calibration Parameters:
  Slope: {slope:.4f} {'⚠️ (over-confident)' if slope < 1.0 else '✅ (good)'}
  Intercept: {intercept:.4f}

2024-25 Results:
  MAE before: {mae_before:.2f} points
  MAE after:  {mae_after:.2f} points
  Improvement: {mae_before - mae_after:+.2f} points ({(mae_after/mae_before - 1)*100:+.1f}%)

2023-24 Results:
  MAE before: {mae_2023_before:.2f} points
  MAE after:  {mae_2023_after:.2f} points
  Improvement: {mae_2023_before - mae_2023_after:+.2f} points ({(mae_2023_after/mae_2023_before - 1)*100:+.1f}%)

Calibration Formula:
  calibrated_PRA = {intercept:.4f} + {slope:.4f} × raw_predicted_PRA

Next Steps:
  1. Re-run backtests with calibrated predictions
  2. Check if win rate improves (expect ~53-54% from ~52%)
  3. Verify large edges (7-10 pts) now perform better
  4. Continue with Phase 2: Feature Engineering
""")

print("="*80)
