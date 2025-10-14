# Exploratory Data Analysis - Walk-Forward Validation Results
## NBA Props Model - 2024-25 Season

**Analysis Date:** October 7, 2025
**Analyst:** Data Analysis Team

---

## Executive Summary

This report presents findings from a comprehensive exploratory data analysis of the walk-forward validation results for the NBA props model (2024-25 season). The analysis identified several critical data quality issues and systematic patterns that are impacting model performance and edge calibration.

**Key Findings:**
- Model shows systematic under-prediction bias (-1.33 points average)
- Severe edge miscalibration, especially in positive edge buckets (-35% error)
- Strong player-specific biases suggest missing contextual features
- High missing data rates in lag features (17% for 10-game lags)
- Model performance degrades significantly for high-scoring players

---

## 1. Dataset Overview

### Data Dimensions
- **Predictions Dataset:** 23,204 game predictions
- **Backtest Dataset:** 3,386 betting opportunities
- **Features Dataset:** 21,869 observations × 167 features
- **Coverage:** Full 2024-25 season walk-forward validation

### Data Quality Summary
- **Predictions:** No missing values, no invalid ranges
- **Features:** 60 features with >5% missing data
- **Outliers:** 0% using 3×IQR method (predictions well-bounded)

---

## 2. Prediction Distribution Analysis

### Overall Prediction Statistics

| Metric | Predicted PRA | Actual PRA | Interpretation |
|--------|--------------|------------|----------------|
| Mean | 16.39 | 17.72 | **-1.33 systematic bias** |
| Median | 13.83 | 16.00 | Model skews toward lower predictions |
| Std Dev | 11.82 | 12.24 | Similar variance |
| Min | 0.69 | 0.00 | Reasonable bounds |
| Max | 63.24 | 81.00 | **Missing top-end performance** |
| 25th %ile | 7.02 | 8.00 | Conservative on bench players |
| 75th %ile | 23.87 | 26.00 | Conservative on starters |

### Key Issues Identified

#### Issue 1: Systematic Under-Prediction
- **Bias:** -1.33 points (model under-predicts)
- **Magnitude:** 54.2% of predictions are under-predictions vs 45.8% over-predictions
- **Impact:** Leads to negative edge opportunities being missed

#### Issue 2: Non-Normal Distribution
- **Shapiro-Wilk Test:** p-value < 0.0001
- **Interpretation:** Predictions NOT normally distributed
- **Implication:** Residuals may violate model assumptions

#### Issue 3: Range Compression
- **Max Prediction:** 63.24 (Actual max: 81.00)
- **Gap:** Model fails to predict extreme performances (>65 PRA)
- **Consequence:** Underbetting on elite player performances

---

## 3. Error Analysis

### Overall Error Metrics

| Metric | Value | Benchmark | Status |
|--------|-------|-----------|--------|
| **MAE** | 7.97 | < 8.0 | ✓ ACCEPTABLE |
| **RMSE** | 10.40 | < 11.0 | ✓ ACCEPTABLE |
| **MAPE** | inf | N/A | ⚠️ ISSUE (division by zero) |
| **Bias** | -1.33 | ±0.5 | ✗ POOR |

### Error by Prediction Range

| Prediction Range | MAE | Bias | Count | Severity |
|-----------------|-----|------|-------|----------|
| 0-10 PRA | 7.09 | -5.32 | 8,561 | **SEVERE under-prediction** |
| 10-20 PRA | 7.29 | -1.43 | 6,680 | Moderate under-prediction |
| 20-30 PRA | 8.74 | +0.78 | 4,931 | Slight over-prediction |
| 30-40 PRA | 9.79 | +4.84 | 1,846 | **SEVERE over-prediction** |
| 40-50 PRA | 10.97 | +8.10 | 950 | **CRITICAL over-prediction** |
| 50+ PRA | 16.53 | +15.87 | 236 | **CATASTROPHIC over-prediction** |

### Critical Finding: Bimodal Error Pattern

The model exhibits a **U-shaped error pattern**:
- **Low PRA (0-10):** Under-predicts by 5.3 points (bench/limited-minutes players)
- **Mid PRA (10-30):** Relatively accurate (±1.5 points)
- **High PRA (30+):** Severely over-predicts (+4.8 to +15.9 points)

**Root Causes:**
1. **Low PRA:** Insufficient training samples for low-minute players (min 5 games required)
2. **High PRA:** Model trained on historical averages, unable to capture elite game performances
3. **Missing Features:** Lack of opponent-specific defensive matchups, injury contexts, usage spikes

---

## 4. Player-Level Error Patterns

### Most Problematic Players (Highest MAE, min 10 games)

| Player | MAE | Bias | Avg Pred | Avg Actual | Issue |
|--------|-----|------|----------|------------|-------|
| Lauri Markkanen | 23.28 | +23.28 | 49.78 | 26.50 | Massive over-prediction |
| Anthony Davis | 20.92 | +20.61 | 59.80 | 39.20 | Severe over-prediction |
| Keegan Murray | 20.22 | +20.22 | 40.50 | 20.28 | Dramatic over-prediction |
| Chet Holmgren | 19.14 | +19.14 | 42.99 | 23.85 | Severe over-prediction |
| Micah Potter | 18.69 | +18.69 | 27.75 | 9.06 | Over-prediction (low minutes) |
| Brandon Miller | 17.93 | -17.93 | 12.98 | 30.91 | **Severe under-prediction** |
| Quentin Grimes | 17.50 | -17.10 | 6.05 | 23.14 | **Severe under-prediction** |
| Mark Williams | 17.12 | -17.12 | 12.37 | 29.49 | **Severe under-prediction** |
| Nikola Jokić | 16.58 | -15.42 | 44.20 | 59.62 | Under-predicts MVP performance |
| Karl-Anthony Towns | 15.50 | -14.80 | 32.50 | 47.30 | Under-predicts star performance |

### Player-Specific Issues

#### Over-Predicted Players (10 worst)
**Pattern:** High usage rate players, recent role changes, injury comebacks
- **Lauri Markkanen:** Likely trained on All-Star season, but regressed
- **Anthony Davis:** Injury-prone, inconsistent availability not captured
- **Keegan Murray:** Sophomore slump not captured by features
- **Chet Holmgren:** Rookie season, limited training data

**Recommended Features:**
- Season-over-season usage rate changes
- Injury history/load management indicators
- Team roster changes (new player additions affecting usage)

#### Under-Predicted Players (10 worst)
**Pattern:** Breakout seasons, mid-season role expansions, trades
- **Brandon Miller:** Rookie breakout not predicted
- **Quentin Grimes:** Role expansion mid-season
- **Mark Williams:** Increased minutes due to injury/trades
- **Nikola Jokić:** Outlier performances (MVP-level games)
- **Karl-Anthony Towns:** Trade to Knicks, usage spike

**Recommended Features:**
- Recent minutes trend (last 5 games)
- Team injury context (who's out → usage boost)
- Trade indicators (new team environment)

---

## 5. Edge Calibration Analysis

### Overall Betting Performance

| Metric | Value | Expected | Gap |
|--------|-------|----------|-----|
| Total Bets | 3,386 | - | - |
| Wins | 1,297 (38.3%) | 52.4% | **-14.1%** |
| Losses | 1,198 (35.4%) | 47.6% | +12.2% |
| Pushes | 891 (26.3%) | ~5% | **+21.3%** |

**Critical Issue:** 26.3% push rate is abnormally high
- **Expected:** 5-8% for standard betting lines
- **Actual:** 26.3% (3.3× higher than expected)
- **Root Cause:** Model predictions very close to betting lines (no clear edge)

### Edge Calibration by Bucket

| Edge Bucket | Bet Count | Win Rate | Expected Win Rate | Calibration Error |
|-------------|-----------|----------|-------------------|-------------------|
| < -10% | 576 | 53.5% | 35.8% | **+17.7%** ✓ Good |
| -10 to -5% | 674 | 50.7% | 42.7% | **+8.0%** ✓ Good |
| -5 to 0% | 886 | 23.6% | 47.4% | **-23.8%** ✗ POOR |
| 0 to 5% | 591 | 17.8% | 52.4% | **-34.6%** ✗ CRITICAL |
| 5 to 10% | 397 | 53.1% | 57.2% | **-4.1%** ~ Acceptable |
| > 10% | 262 | 46.6% | 63.8% | **-17.2%** ✗ POOR |

### Critical Findings

#### Finding 1: Inverted Calibration in Positive Edge Buckets
- **0 to 5% Edge:** 17.8% win rate (should be ~52%)
  - **Loss:** -34.6 percentage points
  - **Severity:** CRITICAL
- **Implication:** Model thinks it has slight positive edge, but actually has large negative edge

#### Finding 2: Negative Edge Buckets Perform Well
- **-10% Edge:** 53.5% win rate (expected 36%)
  - **Gain:** +17.7 percentage points
- **-5 to -10% Edge:** 50.7% win rate (expected 43%)
  - **Gain:** +8.0 percentage points
- **Implication:** Model's "bad bets" are actually good bets (edge calculation is inverted)

#### Finding 3: High Edge Buckets Fail
- **>10% Edge:** 46.6% win rate (expected 64%)
  - **Loss:** -17.2 percentage points
- **Implication:** Model is overconfident on high-edge opportunities

### Root Cause Analysis

The edge calibration issues stem from:

1. **Probability Calibration Error**
   - Model predictions are not well-calibrated probabilities
   - Need isotonic regression or Platt scaling post-processing

2. **Line Integration Issue**
   - Edge calculated as: `predicted_PRA - betting_line`
   - Does not account for variance/uncertainty in predictions
   - Should use: `P(PRA > line) - implied_probability_of_line`

3. **Vig Not Properly Handled**
   - Standard -110 odds require 52.4% win rate to break even
   - Edge calculation may not properly account for vig

4. **Small Sample Sizes**
   - High edge buckets (>10%) have only 262 bets
   - Insufficient data to validate extreme edges

---

## 6. Feature Quality Issues

### Missing Data Analysis

**Features with >10% Missing Data:**

| Feature | Missing Rate | Impact | Recommendation |
|---------|-------------|--------|----------------|
| PRA_lag10, PTS_lag10, REB_lag10, AST_lag10, MIN_lag10 | 17.11% | HIGH | Increase minimum games requirement OR use forward-fill |
| PRA_lag7, PTS_lag7, REB_lag7, AST_lag7, MIN_lag7 | 11.88% | MEDIUM | Forward-fill from shorter lags |
| CTG shooting features (All Three %, FT%, etc.) | 11.74% | HIGH | Impute with player career averages |

### Zero Variance Features

- **SEASON_ID:** Remove (all values are 2024-25)

### Feature Correlation Issues

**Top Correlated Features (sample):**
- Should check for multicollinearity (VIF > 10)
- May have redundant lag features (lag1, lag3, lag5, lag7, lag10)

---

## 7. Recommended Data Cleaning Steps

### Priority 1: Critical Issues (Address Immediately)

1. **Fix Edge Calculation**
   - Implement proper probability-based edge: `edge = P(win) - breakeven_probability`
   - Account for vig in all calculations
   - Use prediction intervals, not point predictions

2. **Impute Missing Lag Features**
   - For lag10: Use lag7 → lag5 → lag3 → season average cascade
   - For lag7: Use lag5 → lag3 → lag1 cascade
   - Document imputation flags for model

3. **Remove Zero-Variance Features**
   - Drop SEASON_ID from feature set

4. **Calibrate Predictions**
   - Apply isotonic regression on validation set
   - Ensure predicted probabilities match actual frequencies

### Priority 2: Model Improvements (Next Sprint)

5. **Add Missing Contextual Features**
   - **Opponent defensive rating** (last 10 games)
   - **Team injury report** (number of starters out)
   - **Minutes trend** (last 5 games vs season average)
   - **Usage rate trend** (recent vs season average)
   - **Trade/roster change indicator** (binary)

6. **Handle Player-Specific Biases**
   - Add player random effects (mixed-effects model)
   - Or: Train separate models for player archetypes (star, starter, bench)

7. **Improve High-PRA Predictions**
   - Add non-linear terms (squared, cubic)
   - Increase weight on recent games for high-usage players
   - Add ceiling indicators (max PRA last 10 games)

### Priority 3: Data Quality Monitoring (Ongoing)

8. **Implement Data Validation Pipeline**
   - Check missing rates before training
   - Alert if >20% missing in critical features
   - Validate prediction ranges (0-100)

9. **Feature Engineering Pipeline**
   - Automate lag feature creation with proper backfill
   - Add unit tests for feature calculations
   - Version control feature definitions

10. **Prediction Post-Processing**
    - Clip predictions to reasonable ranges (0-80)
    - Apply player-specific bias corrections
    - Smooth predictions using ensemble methods

---

## 8. Specific Player Types Where Model Fails

### Player Archetype Analysis

#### Type 1: Limited-Minutes Bench Players (0-15 MPG)
- **Issue:** Severe under-prediction (MAE: 7.09, Bias: -5.32)
- **Volume:** 8,561 predictions (36.9%)
- **Cause:** Insufficient training samples (min 5 games), high variance
- **Fix:** Either exclude from betting pool OR use separate model

#### Type 2: Elite High-Usage Stars (40+ PRA)
- **Issue:** Severe over-prediction (MAE: 10.97-16.53, Bias: +8 to +16)
- **Volume:** 1,186 predictions (5.1%)
- **Cause:** Model regression to mean, missing peak performance indicators
- **Fix:** Add ceiling features, recent trend momentum

#### Type 3: Breakout/Role-Change Players
- **Examples:** Brandon Miller, Quentin Grimes, Mark Williams
- **Issue:** Dramatic under-prediction (MAE: 15-18, Bias: -15 to -17)
- **Cause:** Training data reflects old role, features don't capture usage spikes
- **Fix:** Add recent minutes/usage trends, injury context

#### Type 4: Injury-Prone Stars
- **Examples:** Anthony Davis, Lauri Markkanen
- **Issue:** Over-prediction (MAE: 20-23, Bias: +20)
- **Cause:** Model uses full-season averages, doesn't account for load management
- **Fix:** Add injury history, games-since-injury, minutes restriction indicators

#### Type 5: Rookies/Sophomores
- **Examples:** Chet Holmgren, Keegan Murray
- **Issue:** High error variance (MAE: 19-20)
- **Cause:** Limited historical data, performance inconsistency
- **Fix:** Use draft pick as prior, college stats integration

---

## 9. Betting Strategy Recommendations

### Avoid These Bet Types (Until Model Fixed)

1. **Positive Edge Bets (0-10%)**
   - Actual performance: 18-53% win rate
   - These are losing bets despite positive calculated edge

2. **High-PRA Player Overs (>40 PRA predicted)**
   - Model over-predicts by 8-16 points
   - Avoid "Over" bets on stars

3. **Low-Minutes Player Bets (<15 MPG)**
   - Too much variance, insufficient data
   - 36% of bet pool, but poor calibration

### Consider These Bet Types (Promising)

1. **Negative Edge Bets (-10 to 0%)**
   - Actual performance: 51-54% win rate
   - Model's "bad bets" are actually +EV (edge calculation is inverted)

2. **Mid-Range PRA Unders (20-30 PRA)**
   - Model slightly over-predicts (+0.78 bias)
   - "Under" bets may have hidden edge

3. **Star Player Unders (if model predicts >50)**
   - Model over-predicts elite performances by 15+ points
   - "Under" bets likely +EV

---

## 10. Conclusion

### Summary of Critical Issues

| Issue | Severity | Impact on Betting | Recommended Fix |
|-------|----------|-------------------|-----------------|
| Edge miscalibration (0-10%) | CRITICAL | -35% win rate error | Rebuild edge calculation |
| High push rate (26%) | CRITICAL | Capital inefficiency | Improve line integration |
| Systematic under-prediction | HIGH | Missing negative edge bets | Recalibrate predictions |
| Player-specific biases | HIGH | Large errors on 20+ players | Add contextual features |
| Missing lag features (17%) | MEDIUM | Reduced model accuracy | Implement imputation |
| High-PRA over-prediction | MEDIUM | Losing "over" bets | Add ceiling features |

### Next Steps

1. **Immediate (This Week):**
   - Fix edge calculation formula
   - Apply isotonic regression calibration
   - Impute missing lag features

2. **Short-Term (Next 2 Weeks):**
   - Add opponent defense, injury context features
   - Implement player-specific bias corrections
   - Retrain model with improved feature set

3. **Long-Term (Next Month):**
   - Build separate models for player archetypes
   - Implement ensemble methods
   - Deploy real-time calibration monitoring

### Expected Improvements

With these fixes, we expect:
- **Edge calibration error:** <5% (currently -35%)
- **Push rate:** <10% (currently 26%)
- **MAE:** <7.0 (currently 7.97)
- **Overall win rate:** >52.4% (currently 38.3%)

---

## Appendix: Visualizations

All analysis visualizations saved to:
`/Users/diyagamah/Documents/nba_props_model/data/results/eda_plots/`

1. `1_prediction_quality.png` - Prediction vs Actual scatter, error distributions
2. `2_error_by_history.png` - Error patterns by training data size
3. `3_player_level_errors.png` - Top/bottom performing players
4. `4_edge_calibration.png` - Edge calibration curves and errors
5. `5_feature_quality.png` - Missing data and correlation analysis

---

**Report Generated:** October 7, 2025
**Analysis Scripts:**
- `/Users/diyagamah/Documents/nba_props_model/eda_analysis.py`
- `/Users/diyagamah/Documents/nba_props_model/eda_visualizations.py`
**Output:** `/Users/diyagamah/Documents/nba_props_model/eda_output.txt`
