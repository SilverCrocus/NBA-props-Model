# Quick Reference: EDA Findings
**Date:** October 7, 2025 | **Status:** 🔴 MODEL NOT PROFITABLE - DO NOT USE FOR LIVE BETTING

---

## TL;DR - Top 5 Issues

1. **Edge calculation is broken** - Using simple difference instead of probability-based approach → -34.6% calibration error
2. **High push rate (26.3%)** - 3× higher than normal, indicates weak signal
3. **Inverted calibration** - "Good bets" lose (18% win rate), "bad bets" win (53% win rate)
4. **Systematic bias** - Under-predicts by 1.33 points on average
5. **Missing contextual features** - No opponent defense, injury context, or recent trends

---

## Key Numbers

| Metric | Value | Status |
|--------|-------|--------|
| **MAE** | 7.97 points | ✅ Acceptable |
| **Win Rate** | 38.3% | 🔴 Poor (need 52.4%) |
| **Push Rate** | 26.3% | 🔴 Too high (expect <10%) |
| **Edge Calibration** | -34.6% error | 🔴 Critical |
| **Systematic Bias** | -1.33 points | 🟡 Moderate |

---

## Error Patterns

**By Prediction Range:**
- 0-10 PRA: -5.3 bias (severe under-prediction)
- 10-20 PRA: -1.4 bias (moderate)
- 20-30 PRA: +0.8 bias (slight over)
- 30-40 PRA: +4.8 bias (severe over-prediction)
- 40-50 PRA: +8.1 bias (critical over-prediction)
- 50+ PRA: +15.9 bias (catastrophic over-prediction)

**Worst Players (MAE > 18):**
- Over-predicted: Lauri Markkanen (+23), Anthony Davis (+21), Keegan Murray (+20)
- Under-predicted: Brandon Miller (-18), Mark Williams (-17), Quentin Grimes (-17)

---

## What to Fix First

### Priority 1 (This Week):
1. Rebuild edge calculation (probability-based)
2. Apply isotonic regression calibration
3. Impute missing lag features (17% missing)
4. Set minimum edge threshold (>5%)

### Priority 2 (Next 2 Weeks):
5. Add opponent defense features
6. Add injury/lineup context
7. Apply player-specific bias corrections

### Priority 3 (Next Month):
8. Implement prediction intervals
9. Build separate models by player type
10. Add ensemble methods

---

## Betting Guidance (Until Fixed)

### ❌ DO NOT BET:
- Positive edge bets (0-10%) - only 18-31% win rate
- High PRA overs (>40 predicted) - model over-predicts by 8-16 points
- Low minutes players (<15 MPG) - too volatile
- Close calls (pred ≈ line ±2) - high push rate

### ⚠️ EXPERIMENTAL:
- Negative edge bets (-10 to 0%) - paradoxically win at 51-54%
- Mid-range unders (20-30 PRA) - model slightly over-predicts
- Star unders (if pred >50) - model misses elite performances

**WARNING:** Even "good" patterns are from a miscalibrated model. Re-evaluate after fixes.

---

## Data Quality Issues

- **Missing lag features:** 17% (lag10), 12% (lag7)
- **Missing CTG features:** 12% (shooting stats)
- **Zero variance:** SEASON_ID (remove)
- **Feature count:** 167 features (likely redundant lags)

---

## Files Generated

**Reports:**
- 📄 [EXECUTIVE_SUMMARY.md](./EXECUTIVE_SUMMARY.md) - Full findings (11 KB)
- 📄 [EDA_FINDINGS_REPORT.md](./EDA_FINDINGS_REPORT.md) - Detailed analysis (16 KB)
- 📄 [QUICK_REFERENCE.md](./QUICK_REFERENCE.md) - This file

**Visualizations:**
- 📊 [1_prediction_quality.png](./eda_plots/1_prediction_quality.png) - Scatter, distributions, errors
- 📊 [2_error_by_history.png](./eda_plots/2_error_by_history.png) - Training data impact
- 📊 [3_player_level_errors.png](./eda_plots/3_player_level_errors.png) - Best/worst players
- 📊 [4_edge_calibration.png](./eda_plots/4_edge_calibration.png) - Edge analysis
- 📊 [5_feature_quality.png](./eda_plots/5_feature_quality.png) - Missing data, correlations

**Raw Output:**
- 📝 eda_output.txt - Console output (11 KB)
- 📝 edge_audit_output.txt - Edge calculation analysis (6.5 KB)

**Scripts:**
- 🐍 eda_analysis.py - Main analysis script
- 🐍 eda_visualizations.py - Visualization generator
- 🐍 edge_calculation_audit.py - Edge audit script

---

## Expected Timeline to Profitability

| Phase | Timeline | Win Rate Target | Key Actions |
|-------|----------|----------------|-------------|
| **Phase 1** | 1 week | 48-52% | Fix edge calc, calibration, imputation |
| **Phase 2** | 2 weeks | 53-55% | Add features, bias corrections |
| **Phase 3** | 1 month | 55-57% | Ensemble, prediction intervals, monitoring |

**Target ROI:** 3-5% (after Phase 3)

---

## Contact & Next Steps

1. Review EXECUTIVE_SUMMARY.md for full findings
2. Review visualizations in eda_plots/
3. Prioritize Phase 1 fixes (edge calculation is critical)
4. Re-run walk-forward validation after fixes
5. Compare new results to this baseline

**Last Updated:** October 7, 2025
