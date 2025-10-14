# Week 1 Day 4: Advanced Features Results

**Date**: October 14, 2025
**Task**: Add opponent, efficiency, and normalization features
**MLflow Run ID**: b429e7ec622644a78ff2b263119b5b1d

---

## Executive Summary

Added 13 advanced features (opponent defense, efficiency metrics, per-36 stats) but achieved **minimal improvement**: MAE 6.11 → 6.10 points (only 0.01 improvement). This reveals that current bottleneck is NOT missing features, but rather model optimization and/or feature quality.

### Key Finding
⚠️ **Feature quantity ≠ Model quality: 13 new features → 0.01 MAE improvement (0.16%)**

---

## Comparison: Day 3 (Basic) vs Day 4 (Advanced)

| Metric | Day 3 (34 features) | Day 4 (47 features) | Change |
|--------|---------------------|---------------------|--------|
| **Features** | 34 | **47** | +13 (+38.2%) |
| **Train MAE** | 4.99 points | **4.89 points** | -0.10 pts (-2.0%) |
| **Val MAE** | 6.11 points | **6.10 points** | **-0.01 pts (-0.16%)** |
| **Val RMSE** | 7.86 points | **7.84 points** | -0.02 pts (-0.25%) |
| **R²** | 0.589 | **0.591** | +0.002 (+0.34%) |
| **Within ±3 pts** | 30.6% | **30.7%** | +0.1 pp |
| **Within ±5 pts** | 50.3% | **50.0%** | -0.3 pp |
| **Within ±10 pts** | 81.6% | **82.1%** | +0.5 pp |
| **Training Time** | ~1.5 minutes | **~4.5 minutes** | +3 minutes (+200%) |

---

## New Features Added (13 total)

### Opponent Features (3)
1. **opp_DRtg**: Opponent defensive rating (proxy: PRA allowed per game)
2. **opp_pace**: Opponent pace factor (proxy: avg PRA / 30 * 100)
3. **opp_PRA_allowed**: Opponent average PRA allowed (last 10 games)

### Efficiency Features (5)
4. **TS_pct**: True Shooting % = PTS / (2 * (FGA + 0.44 * FTA))
5. **PER**: Player Efficiency Rating (simplified version)
6. **USG_per_36**: Usage rate per 36 minutes
7. **PTS_per_shot**: Points per shot attempt
8. **eFG_pct**: Effective FG% = (FGM + 0.5 * FG3M) / FGA

### Normalization Features (5)
9. **PRA_per_36**: PRA normalized to 36 minutes
10. **PTS_per_36**: Points per 36 minutes
11. **REB_per_36**: Rebounds per 36 minutes
12. **AST_per_36**: Assists per 36 minutes
13. **MIN_avg**: Average minutes played (last 10 games)

---

## Results: 2024-25 Season Validation

### Overall Performance
```
Total Predictions:  25,431 (same as Day 3)
MAE:                6.10 points (Day 3: 6.11)
RMSE:               7.84 points (Day 3: 7.86)
R²:                 0.591 (Day 3: 0.589)

Accuracy Breakdown:
  Within ±3 pts:    30.7%  (7,807 predictions)
  Within ±5 pts:    50.0%  (12,716 predictions)
  Within ±10 pts:   82.1%  (20,874 predictions)

Training Performance:
  Train MAE:        4.89 points (Day 3: 4.99)
  Validation gap:   1.21 points (Day 3: 1.12)
```

---

## Analysis: Why Only 0.01 Improvement?

### Expected vs Actual
- **Expected**: MAE 6.11 → 5.2-5.5 points (Day 3 roadmap)
- **Actual**: MAE 6.11 → 6.10 points
- **Miss**: 0.60-0.91 points below target

### Hypothesis 1: Opponent Features Are Weak ⚠️

**Issue**: Opponent features may not be calculated correctly or lack predictive power.

**Evidence**:
- Opponent defensive rating calculated as simple proxy (PRA allowed)
- Does NOT account for:
  - Actual team defensive schemes
  - Position-specific defense matchups
  - Home/away differences
  - Recent vs season-long defensive performance
  - Strength of schedule adjustments

**Example of weak opponent feature**:
```python
# Current (oversimplified):
opp_DRtg = 100.0 + (opp_PRA_allowed - 30.0)

# Better (not implemented):
opp_DRtg_by_position[player_position]
opp_DRtg_last_10_games
opp_DRtg_home_vs_away
```

### Hypothesis 2: Efficiency Features Redundant 📊

**Issue**: Efficiency features may be highly correlated with existing features.

**Correlation Analysis Needed**:
- TS% likely correlated with FG%, FG3%, FT%
- PER likely correlated with PTS, REB, AST (which make up PRA)
- PRA_per_36 likely correlated with MIN and PRA_lag features

**Result**: XGBoost may be ignoring redundant features.

### Hypothesis 3: Model Saturated 🔒

**Issue**: XGBoost with current hyperparameters may be at capacity.

**Evidence**:
- Train MAE improved (4.99 → 4.89) but Val MAE barely changed
- Suggests model can fit training data better but not generalizing
- May need:
  - More trees (currently 300)
  - Different learning rate
  - Feature selection to remove noise

### Hypothesis 4: Feature Quality > Feature Quantity ✓

**Key Insight**: Adding more features doesn't help if they're not informative.

**Learning**:
- Day 3: Doubled training data → 1% improvement
- Day 4: Added 13 features → 0.16% improvement
- **Conclusion**: Neither data quantity nor feature quantity is the answer

---

## What Worked

### ✅ Training Improved
- Train MAE: 4.99 → 4.89 (-2.0%)
- Model can learn patterns from new features in training data

### ✅ Code Architecture Scales
- Successfully calculated 13 additional features on-the-fly
- No significant performance degradation
- Feature calculation functions are modular and maintainable

### ✅ Leak-Free Validation Maintained
- Opponent features calculated using ONLY past games
- No temporal leakage introduced
- Validation methodology remains sound

---

## What Didn't Work

### ❌ Validation Improvement Negligible
- Val MAE: 6.11 → 6.10 (only 0.16% improvement)
- Accuracy within ±5 pts: 50.3% → 50.0% (actually slightly worse!)
- Features not generalizing to validation set

### ❌ Training Time Increased 200%
- Day 3: ~1.5 minutes
- Day 4: ~4.5 minutes
- Cost/benefit ratio is poor (3 extra minutes for 0.01 MAE)

### ❌ Opponent Features Ineffective
- opp_DRtg, opp_pace, opp_PRA_allowed added minimal value
- Suggests oversimplified opponent modeling

---

## Root Cause Analysis

### Why Opponent Features Failed

1. **Too Simple**: Using team-level PRA allowed doesn't capture:
   - Position-specific matchups (guards vs bigs)
   - Individual defender assignments
   - Defensive schemes (zone vs man)
   - Contextual factors (home/away, injuries)

2. **Wrong Granularity**: Should track opponent defense by:
   - Player position
   - Recent form (last 5-10 games)
   - Home vs away splits
   - Specific defensive metrics (opponent FG% allowed, etc.)

3. **Implementation Issue**: Opponent team lookup may be failing
   - Need to verify OPP_TEAM matching correctly
   - Check if opponent stats being calculated for enough games

### Why Efficiency Features Failed

1. **High Correlation**: TS%, eFG%, PTS_per_shot all measure shooting efficiency
   - XGBoost can use FG%, FG3%, FT% directly
   - Derived metrics add noise without new information

2. **Already Captured**: PRA_per_36 is mathematically related to:
   - PRA (target variable)
   - MIN (already a feature)
   - XGBoost can learn this relationship implicitly

### Why Normalization Features Failed

1. **Leakage Risk**: Per-36 stats using recent games may create subtle leakage
2. **Redundancy**: Model can learn minutes-based adjustments from MIN + PRA features
3. **Overfitting**: More features = more risk of fitting noise in training data

---

## Revised Strategy for Week 1 Days 4-5

### Immediate Actions (Rest of Day 4)

1. **Feature Importance Analysis** (30 minutes)
   - Examine which of the 47 features are actually used by XGBoost
   - Identify and remove low-importance features
   - Expected: Many new features have near-zero importance

2. **Feature Correlation Analysis** (30 minutes)
   - Calculate correlation matrix for all 47 features
   - Remove highly correlated features (r > 0.9)
   - Expected: TS% and eFG% highly correlated with existing features

3. **Opponent Feature Validation** (30 minutes)
   - Verify OPP_TEAM data quality
   - Check if opponent stats being calculated correctly
   - Add position-aware opponent features if possible

### Day 5: Model Optimization (Revised)

**Original Plan** was hyperparameter tuning, but Day 4 results suggest we should:

1. **Feature Selection** (2 hours)
   - Remove features with importance < 0.01
   - Target: Reduce from 47 → ~25-30 most important features
   - Expected Impact: MAE 6.10 → 5.8-6.0 (removing noise helps)

2. **Hyperparameter Optimization** (4 hours)
   - Grid search on:
     - n_estimators: [300, 500, 1000]
     - max_depth: [4, 6, 8]
     - learning_rate: [0.01, 0.05, 0.1]
     - subsample: [0.7, 0.8, 0.9]
   - Expected Impact: MAE 5.8-6.0 → 5.5-5.8

3. **Ensemble Methods** (2 hours)
   - Train LightGBM alongside XGBoost
   - Simple averaging ensemble
   - Expected Impact: MAE 5.5-5.8 → 5.3-5.6

---

## Technical Details

### Data Pipeline
```
Raw Game Logs (54,012 games)
    ↓
Extract TEAM_NAME and OPP_TEAM from MATCHUP
    ↓
Walk-Forward Training (204 dates)
    ↓
Calculate 47 features per prediction:
  - 34 basic features (Day 3)
  - 13 advanced features (Day 4)
    ↓
Train XGBoost (22,717 samples)
    ↓
Walk-Forward Validation (163 dates, 25,431 predictions)
    ↓
Results: MAE 6.10, RMSE 7.84, R² 0.591
```

### Feature Calculation Time Breakdown
- **Day 3 (34 features)**: ~2.5 it/sec → 1.5 minutes
- **Day 4 (47 features)**: ~0.8 it/sec → 4.5 minutes
- **Bottleneck**: Opponent feature calculation (requires filtering all_games_df)

### Code Changes
- Added `calculate_efficiency_features()` function
- Added `calculate_normalization_features()` function
- Added `calculate_opponent_features()` function
- Modified `calculate_all_features()` to include new features
- Updated data loading to use `all_game_logs_with_opponent.csv`

---

## Industry Context

### Current Performance
- **MAE: 6.10 points**
  - Good: 4-5 points
  - Elite: 3.5-4 points
  - **Status**: Still 22% above "Good" threshold ❌

- **Betting Performance (Estimated)**:
  - Win Rate: 52-53%
  - ROI: 2-4%
  - **Status**: NOT profitable after transaction costs ❌

### Comparison to Week 1 Goals
- **Day 1**: Infrastructure ✅
- **Day 2**: Eliminate leakage (6.17 MAE) ✅
- **Day 3**: Full training data (6.11 MAE) ✅
- **Day 4**: Advanced features (6.10 MAE) ⚠️ **Disappointing**
- **Day 5**: Target 5.5 MAE - **Seems difficult now**

---

## Key Learnings

### 1. Feature Engineering is an Art, Not Science
- Adding "logical" features doesn't guarantee improvement
- Need empirical validation and feature importance analysis
- Quality > Quantity (13 features → 0.01 improvement)

### 2. Model Diagnostics Are Critical
- Should have checked feature importance BEFORE adding more features
- Need to understand what model is actually learning
- Correlation analysis should precede feature engineering

### 3. Simple Proxies Often Fail
- Opponent "defensive rating" based on PRA allowed is too crude
- Need actual defensive metrics or more sophisticated opponent modeling
- Domain expertise needed for good feature engineering

### 4. Diminishing Returns Everywhere
- Day 2: Removed leakage → Big jump (0.28 → 6.17)
- Day 3: 2x training data → 1% improvement
- Day 4: 13 new features → 0.16% improvement
- **Pattern**: Each optimization yields smaller gains

---

## Honest Assessment

### What This Means
The Day 4 results are **sobering but valuable**. We learned that:
1. Feature quantity doesn't drive improvement
2. Current approach is near its limit without major changes
3. Need to shift from "add more features" to "optimize what we have"

### Probability of Reaching 5.5 MAE
With current approach:
- **Through feature engineering alone**: 10% (proved ineffective)
- **Through hyperparameter tuning**: 40% (best remaining option)
- **Through feature selection + tuning**: 60% (remove noise, optimize)
- **Through ensemble methods**: 70% (combine multiple models)

### Revised Week 1 Conclusion
- **Reaching 5.5 MAE by end of Week 1**: 50% probability (down from 75%)
- **Reaching 6.0 MAE by end of Week 1**: 80% probability
- **Profitable betting (55%+ win rate)**: 40% probability (needs calibration work)

---

## Next Steps

### Immediate (Complete Day 4)
1. ✅ Document Day 4 results (this file)
2. ⏳ Run feature importance analysis
3. ⏳ Run correlation analysis
4. ⏳ Identify features to remove
5. ⏳ Create pruned feature set for Day 5

### Day 5 Strategy (Revised)
**Focus**: Optimize existing model, not add more features

1. **Feature Selection** (Remove low-importance features)
2. **Hyperparameter Grid Search** (Find optimal XGBoost params)
3. **Ensemble with LightGBM** (Diversity improves robustness)

**Expected Final Result**: MAE 5.5-6.0 points (realistic target)

---

## Files Created/Modified

### New Files
- `scripts/training/walk_forward_training_advanced_features.py` (787 lines)
- `data/game_logs/all_game_logs_with_opponent.csv` (54,012 rows, ~15 MB)
- `data/results/walk_forward_advanced_features_2024_25.csv` (25,431 predictions)
- `docs/validation/WEEK1_DAY4_ADVANCED_FEATURES_RESULTS.md` (this file)

### MLflow Artifacts
- Run ID: `b429e7ec622644a78ff2b263119b5b1d`
- Experiment: Phase1_Foundation
- Model: XGBoost (300 trees)
- Features: 47 (logged with importance scores)

---

## Conclusion

✅ **Mission Partially Complete**: Successfully added 13 advanced features with leak-free validation.

⚠️ **Reality Check**: Features added minimal value (0.01 MAE improvement), revealing that bottleneck is model optimization, not missing features.

🎯 **Pivot Required**: Days 5 must focus on feature selection, hyperparameter tuning, and ensembling rather than adding more features.

📊 **Honest Status**: Currently at 6.10 MAE, need 0.60-1.10 points improvement to reach profitable range (5.0-5.5 MAE). This is achievable but requires different approach than originally planned.

🚀 **Next Milestone**: Feature selection and hyperparameter optimization to reach 5.8-6.0 MAE by end of Day 5.

---

**Status**: Week 1 Day 4 Complete ✅ (but results disappointing)
**Confidence**: Medium (learned what doesn't work, but target seems harder)
**Timeline**: Behind schedule (hoped for 5.2-5.5 MAE, got 6.10 MAE)
**Recommendation**: Shift from feature engineering to model optimization for Day 5

---

*Analysis Date: October 14, 2025*
*Training: 2023-24 Season (204 dates, 22,717 samples)*
*Validation: 2024-25 Season (163 dates, 25,431 predictions)*
*Method: Walk-Forward with On-the-Fly Advanced Feature Calculation*
