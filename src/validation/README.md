# Statistical Validation Module

Production-ready statistical validation for sports betting models.

## Overview

This module provides three critical validation methods:

1. **Block Bootstrap** - Confidence intervals that respect temporal correlation
2. **Isotonic Calibration** - Fix non-monotonic edge buckets
3. **Edge Bucket Analysis** - Diagnostic tools for model quality

## Quick Start

```bash
# Run complete validation pipeline
uv run scripts/validation/run_statistical_validation.py

# Run detailed edge bucket analysis
uv run scripts/validation/analyze_edge_buckets.py
```

## Files

```
src/validation/
├── statistical_validation.py    # Core validation module
└── README.md                     # This file

scripts/validation/
├── run_statistical_validation.py  # Full validation pipeline
└── analyze_edge_buckets.py        # Edge bucket diagnostics

docs/
└── STATISTICAL_VALIDATION_GUIDE.md  # Comprehensive guide

data/validation_results/
├── bootstrap_win_rate.png         # Win rate distribution
├── bootstrap_roi.png              # ROI distribution
├── calibration_curve.png          # Before/after calibration
├── edge_bucket_analysis.png       # Edge diagnostics
├── edge_bucket_win_rate.png       # Win rate by edge
├── threshold_optimization.png     # Optimal threshold
└── statistical_validation_summary.txt  # Text report
```

## Usage Examples

### Example 1: Block Bootstrap Win Rate

```python
from src.validation.statistical_validation import BlockBootstrap

# Load betting results
bets_df = pd.read_csv('data/results/backtest_2024_25.csv')

# Create bootstrap validator
bootstrap = BlockBootstrap(
    df=bets_df,
    date_col='GAME_DATE',
    block_size=1,           # 1 day per block
    n_bootstrap=5000,       # 5000 iterations
    random_seed=42
)

# Validate win rate
result, passes = bootstrap.validate_win_rate(
    win_col='win',
    breakeven_wr=0.5238,    # -110 odds
    confidence_level=0.95
)

print(result)
# Output:
# Win Rate: 0.5450 [0.5289, 0.5611] (SE: 0.0082)
# ✅ PASS: Lower bound (0.5289) > breakeven (0.5238)
```

### Example 2: Isotonic Calibration

```python
from src.validation.statistical_validation import IsotonicCalibration

# Split data
train = df[df['GAME_DATE'] < '2024-11-01']
test = df[df['GAME_DATE'] >= '2024-11-01']

# Fit calibrator
calibrator = IsotonicCalibration()
calibrator.fit(
    y_pred=train['predicted_PRA'].values,
    y_true=train['PRA'].values
)

# Apply to test set
test['calibrated_PRA'] = calibrator.predict(test['predicted_PRA'].values)

# Evaluate
mae_before = mean_absolute_error(test['PRA'], test['predicted_PRA'])
mae_after = mean_absolute_error(test['PRA'], test['calibrated_PRA'])

print(f"MAE improvement: {mae_before - mae_after:.2f} pts")
```

### Example 3: Edge Bucket Analysis

```python
from src.validation.statistical_validation import IsotonicCalibration

calibrator = IsotonicCalibration()

# Analyze edge buckets
bucket_stats = calibrator.analyze_edge_buckets(
    df=test_df,
    pred_col='predicted_PRA',
    actual_col='PRA',
    line_col='line',
    n_bins=10
)

# Check monotonicity
for i in range(len(bucket_stats) - 1):
    if bucket_stats.iloc[i+1]['win_rate'] < bucket_stats.iloc[i]['win_rate']:
        print(f"⚠️ Non-monotonic at edge {bucket_stats.iloc[i]['edge_mean']:.1f}")
```

### Example 4: Custom Metric Bootstrap

```python
# Define custom metric (e.g., Sharpe ratio)
def sharpe_ratio(df):
    returns = df['profit'] / df['bet_size']
    return returns.mean() / returns.std() if returns.std() > 0 else 0

# Bootstrap custom metric
sharpe_result = bootstrap.bootstrap_metric(
    sharpe_ratio,
    "Sharpe Ratio",
    confidence_level=0.95
)

print(sharpe_result)
# Sharpe Ratio: 0.3245 [0.2891, 0.3599] (SE: 0.0181)
```

## Key Parameters

### Block Bootstrap

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `block_size` | 1 day | Conservative for daily correlation |
| `n_bootstrap` | 5000 | Stable CI estimates |
| `confidence_level` | 0.95 | Industry standard |

### Isotonic Calibration

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `increasing` | True | Monotonic constraint |
| `out_of_bounds` | 'clip' | Extrapolation strategy |
| Train/test split | 70/30 | Temporal split only |

### Edge Buckets

| Parameter | Recommended | Notes |
|-----------|-------------|-------|
| `n_bins` | 10 | Balance granularity vs samples |
| `min_samples` | 30 | Statistical validity threshold |

## Acceptance Criteria

### Win Rate Bootstrap

✅ **PASS:** `ci_lower > breakeven_wr`
❌ **FAIL:** `ci_lower ≤ breakeven_wr`

### ROI Bootstrap

✅ **PASS:** `ci_lower > 0`
❌ **FAIL:** `ci_lower ≤ 0`

### Isotonic Calibration

✅ **PASS:** `mae_after < mae_before`
⚠️ **MARGINAL:** `mae_after ≈ mae_before`
❌ **FAIL:** `mae_after > mae_before` (overfitting)

### Edge Monotonicity

✅ **PASS:** All edge buckets show increasing win rate
⚠️ **MINOR:** 1-2 non-monotonic buckets (small samples OK)
❌ **FAIL:** Consistent non-monotonicity

## Common Use Cases

### Production Deployment

```python
# 1. Fit calibrator on validation set (one-time)
calibrator = IsotonicCalibration()
calibrator.fit(val_pred, val_actual)

import pickle
with open('models/isotonic_calibrator.pkl', 'wb') as f:
    pickle.dump(calibrator, f)

# 2. Apply daily in production
with open('models/isotonic_calibrator.pkl', 'rb') as f:
    calibrator = pickle.load(f)

today['calibrated_PRA'] = calibrator.predict(today['predicted_PRA'])
```

### Weekly Monitoring

```python
# Run bootstrap on last 30 days
recent_bets = df[df['GAME_DATE'] >= pd.Timestamp.now() - pd.Timedelta(days=30)]

bootstrap = BlockBootstrap(recent_bets, block_size=1, n_bootstrap=2000)
wr_result, passes = bootstrap.validate_win_rate()

if not passes:
    send_alert("Model degraded - investigate!")
```

### Model Comparison

```python
# Compare two models
bootstrap_v1 = BlockBootstrap(v1_bets, block_size=1, n_bootstrap=5000)
bootstrap_v2 = BlockBootstrap(v2_bets, block_size=1, n_bootstrap=5000)

wr_v1 = bootstrap_v1.validate_win_rate()
wr_v2 = bootstrap_v2.validate_win_rate()

if wr_v2.ci_lower > wr_v1.ci_upper:
    print("✅ V2 is significantly better")
```

## Troubleshooting

### Issue: Wide Confidence Intervals

**Symptom:** `[0.45, 0.65]` - too wide to be useful
**Cause:** Insufficient data or high variance
**Fix:**
- Collect more data
- Reduce variance (smaller bets, tighter thresholds)
- Consider increasing block size

### Issue: Calibration Doesn't Help

**Symptom:** `mae_after ≈ mae_before`
**Cause:** Model already well-calibrated
**Fix:**
- Check calibration curve (points on diagonal = already good)
- Don't force calibration if not needed

### Issue: Non-Monotonic Edge Buckets

**Symptom:** Large edges underperform small edges
**Cause:** Model miscalibration at extremes
**Fix:**
- Apply isotonic calibration
- Improve features for extreme predictions
- Use conservative Kelly fraction (0.1-0.25)

## Best Practices

1. **Always validate out-of-sample** - Never fit and test on same data
2. **Use temporal splits** - Respect time series nature
3. **Monitor regularly** - Re-validate monthly on rolling window
4. **Conservative parameters** - Block size = 1 day, Kelly fraction = 0.25
5. **Document assumptions** - Track validation dates and parameters

## References

- See `docs/STATISTICAL_VALIDATION_GUIDE.md` for comprehensive guide
- Academic references for methods and best practices
- Industry standards for sports betting validation

## Support

For issues or questions:
1. Check `docs/STATISTICAL_VALIDATION_GUIDE.md`
2. Review example scripts in `scripts/validation/`
3. Examine diagnostic plots in `data/validation_results/`
