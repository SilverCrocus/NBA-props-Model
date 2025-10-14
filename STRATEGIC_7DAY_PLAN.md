# NBA Props Model - 7-Day Strategic Improvement Plan
**Date:** October 14, 2025
**Prepared By:** Data Science Team
**Objective:** Transform 51.98% win rate model to 55%+ production-ready system in 7 days

---

## Executive Summary

### Current State (Brutal Reality)
```
Metric               Current    Target     Gap        Status
Win Rate             51.98%     55.00%     -3.02 pp   BELOW TARGET
ROI                  +0.91%     +5.00%     -4.09 pp   BARELY PROFITABLE
MAE                  9.92 pts   5.00 pts   +4.92 pts  NEARLY 2X WORSE
CLV Rate             73.7%      40-50%     +23.7 pp   ELITE (only bright spot)
```

### The Paradox
**Elite at finding edges (73.7% CLV), terrible at winning bets (51.98% win rate)**

This disconnect reveals the core issue: **The model identifies market inefficiencies but predictions are too inaccurate to capitalize on them.**

---

## Root Cause Analysis

### Issue #1: Catastrophic Feature Gap (Primary Problem)
**Impact on MAE:** 5.10 → 9.92 points (+94% increase)

**Missing Critical Features:**
```
CATEGORY                    STATUS      MAE IMPACT
L3 Recent Form              MISSING     -1.5 pts
Opponent Defense            MISSING     -1.2 pts
Rest/Back-to-Back           MISSING     -0.8 pts
Minutes Projection Model    MISSING     -0.7 pts
CTG Advanced Stats (full)   PARTIAL     -0.6 pts
Home/Away Splits            MISSING     -0.4 pts
```

**Evidence from walk_forward_validation_enhanced.py:**
- Current model trained WITHOUT enhanced features (lines 268-272)
- Comment states: "Model was trained on old features, so we just use what overlaps"
- Enhanced features calculated but NOT used in training

**The Smoking Gun:**
```python
# Current approach (WRONG):
old_feature_vector = [all_features.get(col, 0) for col in feature_cols]  # OLD FEATURES
pred_pra = model.predict([old_feature_vector])[0]  # PREDICTING WITH INCOMPLETE DATA

# What we need (CORRECT):
enhanced_feature_vector = [all_features.get(col, 0) for col in enhanced_feature_cols]
retrained_model.predict([enhanced_feature_vector])[0]  # FULL FEATURES
```

---

### Issue #2: Edge Calculation Fundamentally Broken
**Impact on Win Rate:** -14.1 percentage points (38.3% → expected 52.4%)

**From Executive Summary (EDA Findings):**
```
Edge Range        Win Rate    Expected    Error
0-5% (positive)   17.8%       52.4%       -34.6 pp  ❌ CRITICAL
-5 to 0%          23.6%       47.4%       -23.8 pp  ❌ POOR
> 10%             46.6%       63.8%       -17.2 pp  ❌ POOR
```

**Current Formula (WRONG):**
```python
edge = predicted_PRA - betting_line  # Treats prediction as certainty
```

**Correct Formula (RESEARCH-VALIDATED):**
```python
# Account for prediction uncertainty
pred_std = model_std_dev  # ~8 points based on MAE
prob_win = norm.cdf((predicted_PRA - betting_line) / pred_std)
implied_prob = convert_odds_to_probability(odds)
edge = (prob_win - implied_prob) * 100
```

---

### Issue #3: Model Trained on Wrong Features
**Impact:** Cannot assess until retraining

**From REALITY_CHECK.md:**
- Training MAE: 4.82 points (with full features)
- Walk-forward MAE: 9.92 points (with limited features)
- **106% increase in error** due to missing features

**The Problem:**
```
Training Data (2003-2024):
- Has all CTG stats ✓
- Has all historical lags ✓
- Model learns patterns from rich features

Prediction Data (2024-25):
- Missing L3 features ❌
- Missing opponent features ❌
- Missing rest features ❌
- Model applies rich-feature patterns to poor-feature data = FAILURE
```

---

## Prioritized Feature Impact Analysis

### Tier 1: CRITICAL (Must-Have for Production)

#### 1. L3 Recent Form Features (HIGHEST IMPACT)
**Expected MAE Reduction:** 9.92 → 8.4 (-1.5 pts)
**Expected Win Rate Improvement:** 51.98% → 53.2% (+1.2 pp)
**Implementation Effort:** 1 day
**Research Evidence:** "Most recent 3-5 games have 3x predictive power of season average"

**Features:**
```python
# Last 3 games rolling stats
PRA_L3_mean         # Average PRA last 3 games
PRA_L3_std          # Volatility last 3 games
PRA_L3_trend        # Linear trend (improving/declining)
MIN_L3_mean         # Minutes stability indicator
PTS_L3_mean         # Scoring form
REB_L3_mean         # Rebounding form
AST_L3_mean         # Assists form
```

**Why This Matters:**
- Research shows L3 > L5 > L10 > Season for NBA props
- Captures hot/cold streaks better than season averages
- Low correlation with existing lag features (orthogonal signal)

---

#### 2. Opponent Defense Features (HIGH IMPACT)
**Expected MAE Reduction:** 8.4 → 7.2 (-1.2 pts)
**Expected Win Rate Improvement:** 53.2% → 54.3% (+1.1 pp)
**Implementation Effort:** 2 days
**Research Evidence:** "Opponent defense creates edge opportunities, markets slow to adjust"

**Features:**
```python
# Team-level defensive metrics
OPP_DRtg                    # Defensive rating (last 10 games)
OPP_Pace                    # Pace factor (possessions/48 min)
OPP_PRA_Allowed_per100      # PRA allowed per 100 possessions

# Position-specific (HIGHEST VALUE)
OPP_DRtg_vs_Guards          # Defense vs guards
OPP_DRtg_vs_Forwards        # Defense vs forwards
OPP_DRtg_vs_Centers         # Defense vs centers
OPP_PTS_Allowed_to_Pos      # Points allowed to position
OPP_REB_Allowed_to_Pos      # Rebounds allowed to position
OPP_AST_Allowed_to_Pos      # Assists allowed to position
```

**Data Sources:**
- NBA API: `leaguedashptdefend` endpoint
- CTG Team Data: Already collected (270 files, 100% complete)
- Can extract from existing `data/ctg_team_data/`

---

#### 3. Rest & Schedule Features (MEDIUM-HIGH IMPACT)
**Expected MAE Reduction:** 7.2 → 6.4 (-0.8 pts)
**Expected Win Rate Improvement:** 54.3% → 54.9% (+0.6 pp)
**Implementation Effort:** 0.5 days (already partially implemented)
**Research Evidence:** "+37.6% win likelihood with 1+ day rest"

**Features (ALREADY IN CODE):**
```python
# From walk_forward_validation_enhanced.py lines 150-176
Days_Rest               # 0, 1, 2, 3, 4+
Is_BackToBack          # Binary indicator
Games_Last7            # Fatigue indicator
```

**Just Need:** Extract from existing implementation and verify

---

#### 4. Probability-Based Edge Calculation (CRITICAL)
**Expected Win Rate Improvement:** 54.9% → 55.8% (+0.9 pp)
**Expected ROI Improvement:** +0.91% → +3.5% (+2.59 pp)
**Implementation Effort:** 1 day
**Research Evidence:** "+34.69% ROI vs -35.17% ROI (calibration vs accuracy)"

**Current vs Correct:**
```python
# CURRENT (BROKEN)
edge = predicted_PRA - betting_line
bet_decision = edge >= 3  # Arbitrary threshold

# CORRECT (PROBABILITY-BASED)
from scipy.stats import norm

def calculate_true_edge(predicted_pra, betting_line, odds, model_uncertainty):
    """
    Calculate true expected value accounting for prediction uncertainty
    """
    # Model uncertainty from historical MAE
    pred_std = model_uncertainty  # Will be ~6.5 after fixes

    # Probability of beating the line
    z_score = (predicted_pra - betting_line) / pred_std
    prob_over = norm.cdf(z_score)
    prob_under = 1 - prob_over

    # Convert American odds to probabilities
    if odds > 0:
        implied_prob = 100 / (odds + 100)
    else:
        implied_prob = -odds / (-odds + 100)

    # Calculate EV
    if prob_over > 0.5:  # Bet OVER
        ev = (prob_over * (odds/100 if odds > 0 else 1)) - (prob_under * 1)
    else:  # Bet UNDER
        ev = (prob_under * (odds/100 if odds > 0 else 1)) - (prob_over * 1)

    edge_pct = ev * 100
    return edge_pct, prob_over

# Only bet if edge > 5% AND confidence > 60%
```

---

### Tier 2: HIGH VALUE (Improve Beyond 55%)

#### 5. Minutes Projection Model (HIGH VALUE)
**Expected MAE Reduction:** 6.4 → 5.8 (-0.6 pts)
**Expected Win Rate Improvement:** 55.8% → 56.4% (+0.6 pp)
**Implementation Effort:** 3 days
**Research Priority:** "Most critical opportunity stat"

**Why Separate Model:**
- Minutes volatility is high (std ~8 minutes)
- Injuries, blowouts, coach decisions affect minutes
- Minutes * Usage = Opportunity (multiplicative impact on PRA)

**Approach:**
```python
# Simple XGBoost regression for minutes prediction
Features:
- Last 5 games minutes average
- Last 10 games minutes std
- Starter vs bench indicator
- Team pace (more possessions = more minutes)
- Opponent strength (blowout probability)
- Home vs away (rotation patterns differ)
- Back-to-back indicator (rest starters)

Target: Actual minutes played
Validation: Walk-forward MAE < 5 minutes
```

---

#### 6. Home/Away Splits (MEDIUM VALUE)
**Expected MAE Reduction:** 5.8 → 5.5 (-0.3 pts)
**Implementation Effort:** 1 day
**Research Evidence:** "Player-specific edge opportunities"

**Features:**
```python
PRA_Home_L10        # Last 10 home games
PRA_Away_L10        # Last 10 away games
PRA_Home_vs_Away    # Differential
MIN_Home_L10        # Minutes at home
MIN_Away_L10        # Minutes away
```

---

#### 7. CTG Advanced Stats (Full Integration)
**Expected MAE Reduction:** 5.5 → 5.2 (-0.3 pts)
**Implementation Effort:** 1 day
**Current Status:** Partially implemented

**Missing CTG Features:**
```python
# From CTGFeatureBuilder but not in training
CTG_USG             # Usage rate percentile
CTG_PSA             # Points per shot attempt
CTG_AST_PCT         # Assist percentage
CTG_TOV_PCT         # Turnover percentage
CTG_eFG             # Effective FG%
CTG_REB_PCT         # Total rebound percentage
CTG_Available       # Data availability flag
```

---

### Tier 3: CALIBRATION & VALIDATION (Production Polish)

#### 8. Isotonic Regression Calibration
**Expected Win Rate Improvement:** 56.4% → 57.0% (+0.6 pp)
**Implementation Effort:** 0.5 days
**Research Evidence:** "Calibration > Accuracy for betting"

```python
from sklearn.isotonic import IsotonicRegression

# After model training, calibrate probabilities
iso_reg = IsotonicRegression(out_of_bounds='clip')
calibrated_probs = iso_reg.fit_transform(y_val, raw_predictions)

# Use calibrated predictions for betting decisions
```

---

#### 9. Confidence Intervals & Uncertainty Quantification
**Expected Win Rate Improvement:** Minor, but CRITICAL for risk management
**Implementation Effort:** 1 day

```python
# Quantile regression for prediction intervals
import xgboost as xgb

# Train 3 models: 10th, 50th, 90th percentiles
model_10 = xgb.XGBRegressor(objective='quantile', quantile_alpha=0.1)
model_50 = xgb.XGBRegressor(objective='reg:squarederror')
model_90 = xgb.XGBRegressor(objective='quantile', quantile_alpha=0.9)

# Prediction intervals
lower = model_10.predict(X)
median = model_50.predict(X)
upper = model_90.predict(X)

interval_width = upper - lower  # Uncertainty metric
only_bet_if_interval_width < 12  # High confidence only
```

---

## 7-Day Implementation Roadmap

### Day 1: Foundation Fixes
**Goal:** Retrain model with existing enhanced features
**Expected Outcome:** MAE 9.92 → 8.5 pts, Win Rate 51.98% → 52.8%

**Tasks:**
1. **Extract L3 Features from walk_forward_validation_enhanced.py** [2 hours]
   - Lines 127-147 already have L3 calculation
   - Add to training pipeline

2. **Extract Rest Features** [1 hour]
   - Lines 150-176 already have rest calculation
   - Integrate into feature builder

3. **Retrain Model with Enhanced Features** [4 hours]
   ```bash
   # Create new training script
   uv run scripts/train_model_enhanced_v1.py

   # Steps:
   # 1. Load train.parquet (2003-2024 data)
   # 2. Calculate L3 + Rest features for ALL training data
   # 3. Train XGBoost with enhanced features
   # 4. Save model as models/xgboost_enhanced_v1.pkl
   ```

4. **Validate on 2024-25** [1 hour]
   ```bash
   uv run scripts/validate_enhanced_v1.py

   # Calculate MAE, Win Rate, ROI
   # Compare to baseline
   ```

**Success Criteria:**
- MAE < 8.5 points
- Win Rate > 52.5%
- Training completes without errors

---

### Day 2: Opponent Defense Features
**Goal:** Add opponent-adjusted features
**Expected Outcome:** MAE 8.5 → 7.5 pts, Win Rate 52.8% → 53.8%

**Tasks:**
1. **Build Opponent Defense Data Pipeline** [4 hours]
   ```python
   # scripts/build_opponent_features.py

   # Extract from CTG team data (already collected)
   team_data_path = 'data/ctg_team_data/'

   # Create opponent defensive metrics:
   # - DRtg by team, by date
   # - Pace by team, by date
   # - PRA allowed per 100 possessions
   # - Position-specific defensive ratings

   # Save to: data/processed/opponent_defense_ratings.parquet
   ```

2. **Integrate Opponent Features into Training** [3 hours]
   - Merge opponent data with training set
   - Handle missing opponent data (use league average)
   - Create interaction features (Usage * OPP_DRtg)

3. **Retrain Model v2** [1 hour]
   ```bash
   uv run scripts/train_model_enhanced_v2.py
   ```

**Success Criteria:**
- Opponent features available for 90%+ of games
- MAE < 7.8 points
- Win Rate > 53.5%

---

### Day 3: Probability-Based Edge Calculation
**Goal:** Fix edge calculation and improve win rate
**Expected Outcome:** Win Rate 53.8% → 55.2%, ROI +0.91% → +3.0%

**Tasks:**
1. **Implement True Edge Calculator** [3 hours]
   ```python
   # utils/edge_calculator.py

   class EdgeCalculator:
       def __init__(self, model_mae=7.5):
           self.pred_std = model_mae  # Use model MAE as std proxy

       def calculate_ev(self, pred_pra, line, odds):
           # Full implementation from Tier 1 #4 above
           pass
   ```

2. **Recalculate 2024-25 Betting Results** [2 hours]
   ```bash
   # Use existing predictions but NEW edge calculation
   uv run scripts/recalculate_betting_results.py

   # Compare:
   # - Old edge (difference) vs New edge (probability)
   # - Win rate improvement
   # - ROI improvement
   ```

3. **Set Optimal Thresholds** [2 hours]
   ```python
   # Optimize edge threshold via grid search
   for min_edge in [3, 4, 5, 6, 7]:
       for min_prob in [0.55, 0.60, 0.65]:
           results = backtest(min_edge, min_prob)
           # Track: Win Rate, ROI, # Bets

   # Select threshold that maximizes ROI with >52.4% win rate
   ```

4. **Generate Updated Performance Report** [1 hour]

**Success Criteria:**
- Win Rate > 54.5%
- ROI > +2.5%
- Clear edge thresholds defined

---

### Day 4: Minutes Projection Model
**Goal:** Build separate minutes predictor
**Expected Outcome:** MAE 7.5 → 6.8 pts, Win Rate 55.2% → 55.8%

**Tasks:**
1. **Build Minutes Dataset** [2 hours]
   ```python
   # Features for minutes prediction:
   # - Last 5/10 games minutes average
   # - Starter indicator
   # - Team pace
   # - Back-to-back indicator
   # - Opponent strength (blowout proxy)
   # - Home/away

   # Target: Actual minutes played
   ```

2. **Train Minutes Model** [2 hours]
   ```python
   # Simple XGBoost regressor
   model = xgb.XGBRegressor(
       n_estimators=100,
       max_depth=4,
       learning_rate=0.1
   )

   # Validate: MAE < 5 minutes
   ```

3. **Integrate Minutes Projections** [3 hours]
   ```python
   # Add projected_minutes as feature to PRA model
   # Interaction features:
   # - projected_minutes * usage_rate
   # - projected_minutes * pace
   ```

4. **Retrain PRA Model v3** [1 hour]

**Success Criteria:**
- Minutes MAE < 5 minutes
- PRA MAE < 7.0 points
- Win Rate > 55.5%

---

### Day 5: Home/Away + Full CTG Integration
**Goal:** Add remaining high-value features
**Expected Outcome:** MAE 6.8 → 6.2 pts, Win Rate 55.8% → 56.5%

**Tasks:**
1. **Home/Away Split Features** [3 hours]
   ```python
   # Calculate from historical data:
   # - PRA_Home_L10 (last 10 home games)
   # - PRA_Away_L10 (last 10 away games)
   # - MIN_Home_L10
   # - MIN_Away_L10
   # - Home_Away_Diff (differential)
   ```

2. **Full CTG Integration** [3 hours]
   ```python
   # Ensure ALL CTG features used:
   # - Usage percentile
   # - PSA percentile
   # - Assist %
   # - Turnover %
   # - Rebound %
   # - eFG%
   ```

3. **Retrain Model v4** [1 hour]

4. **Feature Importance Analysis** [1 hour]
   ```python
   import shap

   explainer = shap.TreeExplainer(model)
   shap_values = explainer.shap_values(X_val)

   # Identify top 20 features
   # Remove features with importance < 0.01
   ```

**Success Criteria:**
- MAE < 6.5 points
- Win Rate > 56.0%
- All high-value features integrated

---

### Day 6: Calibration & Validation
**Goal:** Polish model for production
**Expected Outcome:** Win Rate 56.5% → 57.0%, ROI +3.0% → +4.0%

**Tasks:**
1. **Isotonic Regression Calibration** [2 hours]
   ```python
   from sklearn.isotonic import IsotonicRegression

   iso_reg = IsotonicRegression(out_of_bounds='clip')
   calibrated_preds = iso_reg.fit_transform(y_val, raw_preds)
   ```

2. **Confidence Intervals** [3 hours]
   ```python
   # Train quantile models (10th, 50th, 90th percentiles)
   # Calculate prediction intervals
   # Only bet when interval width < threshold
   ```

3. **Multi-Season Validation** [2 hours]
   ```bash
   # Test on 2023-24 season (as second validation)
   uv run scripts/validate_2023_24.py

   # Ensure consistent performance across seasons
   ```

4. **Edge Case Testing** [1 hour]
   ```python
   # Test model on:
   # - Low minutes players (<10 MPG)
   # - Back-to-back games
   # - Blowout scenarios
   # - Injury returns
   ```

**Success Criteria:**
- Calibration error < 3%
- Consistent performance across 2023-24 and 2024-25
- Prediction intervals validated

---

### Day 7: Final Testing & Documentation
**Goal:** Production readiness assessment
**Expected Outcome:** Go/No-Go decision with confidence

**Tasks:**
1. **Final Performance Report** [2 hours]
   ```markdown
   # NBA Props Model - Production Readiness Report

   ## Performance Metrics (2024-25)
   - Win Rate: X.XX%
   - ROI: +X.XX%
   - MAE: X.XX points
   - CLV Rate: XX.X%

   ## Validation Results (2023-24)
   - Win Rate: X.XX%
   - ROI: +X.XX%
   - MAE: X.XX points

   ## Risk Assessment
   - Confidence intervals
   - Edge case performance
   - Drawdown analysis

   ## Production Criteria
   - [✓/✗] Win Rate > 55%
   - [✓/✗] ROI > 3%
   - [✓/✗] MAE < 6.5 points
   - [✓/✗] Consistent across seasons
   ```

2. **Feature Documentation** [2 hours]
   ```python
   # Document all features:
   # - Description
   # - Calculation method
   # - Data source
   # - SHAP importance
   # - Missing data handling
   ```

3. **Production Deployment Plan** [2 hours]
   ```markdown
   # Deployment Checklist
   - [ ] Model serialization (pickle/joblib)
   - [ ] Feature pipeline automation
   - [ ] Real-time data ingestion
   - [ ] Bet logging system
   - [ ] Performance monitoring dashboard
   - [ ] Alerting for model degradation
   ```

4. **Go/No-Go Decision** [2 hours]
   ```
   IF all criteria met:
       → Proceed to live testing (paper money)
       → Start with $10-25 bets
       → Track CLV daily
       → Scale gradually

   IF criteria not met:
       → Identify bottleneck
       → Extend development 1-2 weeks
       → Focus on weakest metric
   ```

**Success Criteria:**
- All production criteria documented
- Clear deployment plan
- Risk mitigation strategies defined

---

## Expected Performance Trajectory

### Daily Progress (Conservative Estimates)

```
Day   Feature Addition          MAE      Win Rate   ROI      Status
0     Baseline                  9.92     51.98%     +0.91%   CURRENT
1     L3 + Rest                 8.50     52.80%     +1.50%   Improving
2     Opponent Defense          7.50     53.80%     +2.20%   Promising
3     Probability Edge          7.50     55.20%     +3.00%   Good
4     Minutes Model             6.80     55.80%     +3.40%   Very Good
5     Home/Away + Full CTG      6.20     56.50%     +3.80%   Excellent
6     Calibration              6.00     57.00%     +4.20%   Elite
7     Final Testing             5.80     57.00%     +4.50%   PRODUCTION READY
```

### Confidence Levels

```
Target Achievement Probability:

Win Rate > 55%:   85% confidence  ✓ LIKELY
MAE < 6.5 pts:    75% confidence  ✓ PROBABLE
ROI > 3%:         80% confidence  ✓ LIKELY

Overall Success: 75% probability of meeting ALL criteria
```

---

## Risk Assessment & Mitigation

### Critical Risks

#### Risk #1: Features Don't Improve MAE as Expected
**Probability:** 30%
**Impact:** High (miss 55% win rate target)

**Mitigation:**
- Day 3 checkpoint: If MAE > 8.0, pivot to ensemble methods
- Add LightGBM model and average predictions
- Increase focus on feature selection (remove noise)

**Fallback Plan:**
```python
# If single model underperforms
ensemble_pred = (
    0.5 * xgboost_pred +
    0.3 * lightgbm_pred +
    0.2 * ridge_regression_pred
)
# Often reduces MAE by 0.3-0.5 points
```

---

#### Risk #2: Opponent Data Quality Issues
**Probability:** 40%
**Impact:** Medium (lose 0.5-1.0 pts MAE improvement)

**Mitigation:**
- Use multiple data sources (CTG + NBA API)
- Implement league average fallback
- Test with/without opponent features

**Contingency:**
```python
if opponent_data_missing_rate > 20%:
    # Use simpler opponent features
    opp_features = {
        'opp_win_pct': team_record / games,  # Simple proxy
        'opp_strength': league_avg_rating  # Fallback
    }
```

---

#### Risk #3: Overfitting During Rapid Development
**Probability:** 50%
**Impact:** High (model fails in production)

**Mitigation:**
- Strict walk-forward validation (no data leakage)
- Test on BOTH 2023-24 and 2024-25 seasons
- Use regularization (max_depth=6, learning_rate=0.05)
- Never tune on test set

**Detection:**
```python
if val_mae < train_mae:  # Red flag
    print("WARNING: Validation better than training = DATA LEAKAGE")

if abs(mae_2023_24 - mae_2024_25) > 1.5:  # Red flag
    print("WARNING: Inconsistent across seasons = OVERFITTING")
```

---

#### Risk #4: Time Constraints (7 days is aggressive)
**Probability:** 60%
**Impact:** Medium (need to prioritize)

**Mitigation:**
- Focus on Tier 1 features only (Days 1-3)
- Skip Tier 2 if behind schedule
- Accept 55% win rate vs 57% as minimum viable

**Minimum Viable Product (MVP):**
```
Must-Have Features:
1. L3 Recent Form         ✓ (Day 1)
2. Probability Edge       ✓ (Day 3)
3. Opponent Defense       ✓ (Day 2)

Nice-to-Have Features:
4. Minutes Model          ⚠ (Skip if needed)
5. Home/Away              ⚠ (Skip if needed)
6. Calibration           ⚠ (Skip if needed)

Target with MVP: 55.0% win rate, +3.0% ROI
```

---

## Production Readiness Criteria

### Performance Metrics (ALL must pass)

```
CRITERIA                          THRESHOLD       CURRENT    TARGET    STATUS
1. Win Rate (2024-25)             > 55.0%         51.98%     57.0%     ❌
2. Win Rate (2023-24)             > 54.0%         51.19%     56.0%     ❌
3. MAE (2024-25)                  < 6.5 pts       9.92       5.8       ❌
4. ROI (2024-25)                  > 3.0%          +0.91%     +4.5%     ❌
5. CLV Rate                       > 40%           73.7%      73.7%     ✓
6. Minimum Bets                   > 1000          2,495      2,495     ✓
7. Consistency (2023 vs 2024)     < 2 pp diff     0.79 pp    <2 pp     ✓
8. Calibration Error              < 5%            N/A        <3%       ❌
9. Max Drawdown                   < 20%           N/A        <15%      ❌
10. Sharpe Ratio                  > 1.0           N/A        >1.5      ❌
```

**Current Status:** 3/10 criteria met (30%)
**Target Status:** 10/10 criteria met (100%)

---

### Technical Criteria

```
CRITERIA                              STATUS
1. Walk-forward validation             ✓ Implemented
2. No temporal leakage                 ✓ Verified
3. Feature documentation              ❌ Incomplete
4. Model serialization                ❌ Not implemented
5. Automated feature pipeline         ❌ Manual
6. Real-time prediction API           ❌ Not built
7. Bet logging system                 ❌ Not implemented
8. Performance monitoring             ❌ Not implemented
9. Alerting for degradation           ❌ Not implemented
10. Reproducible training pipeline    ⚠️  Partial
```

**Current Status:** 2/10 criteria met (20%)
**Target for Day 7:** 8/10 criteria met (80%)

---

## Success Definition

### Minimum Viable Production Model

**Performance Thresholds:**
- Win Rate: 55.0%+ (vs 52.38% breakeven)
- ROI: 3.0%+ (vs 0.91% current)
- MAE: 6.5 points or less (vs 9.92 current)
- CLV Rate: Maintain 70%+ (vs 73.7% current)

**Validation Requirements:**
- Consistent across 2023-24 and 2024-25 seasons (<2 pp difference)
- Walk-forward validated (no temporal leakage)
- Edge calculation probability-based (not difference-based)

**Risk Management:**
- Confidence intervals implemented
- Maximum bet size: 2-3% of bankroll
- Stop-loss: Pause betting if win rate < 53% over 100 bets

**Business Criteria:**
- Minimum 1000 bets for statistical significance
- Positive CLV in 70%+ of bets
- Expected value > 3% per bet placed

---

### Stretch Goals (If Time Permits)

**Performance:**
- Win Rate: 57.0%+
- ROI: 5.0%+
- MAE: 5.5 points or less

**Technical:**
- Full automation (data → features → predictions → bets)
- Real-time monitoring dashboard
- Multi-model ensemble (XGBoost + LightGBM)

---

## Implementation Commands

### Day 1: Foundation
```bash
# Create L3 + Rest feature pipeline
uv run scripts/create_l3_features.py

# Retrain model
uv run scripts/train_enhanced_v1.py

# Validate
uv run scripts/validate_enhanced_v1.py
```

### Day 2: Opponent Features
```bash
# Build opponent defense data
uv run scripts/build_opponent_defense.py

# Retrain with opponent features
uv run scripts/train_enhanced_v2.py

# Validate
uv run scripts/validate_enhanced_v2.py
```

### Day 3: Edge Calculation
```bash
# Implement probability-based edge
uv run scripts/implement_true_edge.py

# Recalculate betting results
uv run scripts/recalculate_betting_results.py

# Optimize thresholds
uv run scripts/optimize_edge_thresholds.py
```

### Day 4: Minutes Model
```bash
# Train minutes predictor
uv run scripts/train_minutes_model.py

# Integrate into PRA model
uv run scripts/train_enhanced_v3.py

# Validate
uv run scripts/validate_enhanced_v3.py
```

### Day 5: Final Features
```bash
# Add home/away and full CTG
uv run scripts/add_final_features.py

# Retrain final model
uv run scripts/train_enhanced_v4.py

# Feature importance
uv run scripts/analyze_feature_importance.py
```

### Day 6: Calibration
```bash
# Calibrate model
uv run scripts/calibrate_model.py

# Multi-season validation
uv run scripts/validate_multi_season.py

# Test edge cases
uv run scripts/test_edge_cases.py
```

### Day 7: Final Testing
```bash
# Generate final report
uv run scripts/generate_final_report.py

# Create deployment plan
uv run scripts/create_deployment_plan.py

# Go/No-Go decision
uv run scripts/production_readiness_check.py
```

---

## Appendix: Quick Reference

### Key Files to Modify
```
Priority 1 (Days 1-3):
- scripts/train_enhanced_v1.py          (NEW - retrain with L3+Rest)
- scripts/build_opponent_defense.py     (NEW - opponent features)
- utils/edge_calculator.py              (NEW - probability edge)
- scripts/recalculate_betting_results.py (NEW - apply new edge calc)

Priority 2 (Days 4-5):
- scripts/train_minutes_model.py        (NEW - minutes predictor)
- scripts/add_final_features.py         (NEW - home/away + CTG)
- scripts/train_enhanced_v4.py          (NEW - final model)

Priority 3 (Days 6-7):
- scripts/calibrate_model.py            (NEW - isotonic regression)
- scripts/validate_multi_season.py      (NEW - 2023-24 + 2024-25)
- scripts/production_readiness_check.py (NEW - go/no-go)
```

### Key Data Files
```
Training Data:
- data/processed/train.parquet          (2003-2024, 227M)
- data/processed/train_enhanced.parquet (WITH new features)

Validation Data:
- data/game_logs/game_logs_2024_25_preprocessed.csv
- data/results/backtest_walkforward_2024_25.csv

Odds Data:
- data/historical_odds/*.csv            (For backtesting)
```

### Performance Tracking
```python
# Track daily progress in this format:
daily_metrics = {
    'day': 1,
    'features_added': ['L3', 'Rest'],
    'mae': 8.50,
    'win_rate': 52.80,
    'roi': 1.50,
    'num_features': 85,
    'training_time_min': 12,
    'notes': 'L3 features strongest signal'
}
```

---

## Contact & Support

**Questions During Implementation:**
- Check research/RESEARCH_SUMMARY.md for academic backing
- Review docs/validation/REALITY_CHECK.md for current state
- See walk_forward_validation_enhanced.py for L3/Rest implementation

**Red Flags to Watch:**
1. Validation MAE < Training MAE (data leakage)
2. Win rate decreases after adding features (overfitting)
3. Huge performance difference between seasons (instability)
4. Edge calculation produces >80% win rate (broken logic)

---

**Bottom Line:** This is aggressive but achievable. The 73.7% CLV rate proves the model finds real edges. The 9.92 MAE proves we just need better features. If we execute Days 1-3 perfectly (L3 + Opponent + Edge), we hit 55% win rate. Everything else is gravy.

**Let's build a production-ready model in 7 days.**
