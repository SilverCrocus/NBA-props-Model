# Data Quality Investigation Report
## Negative PRA Predictions - Root Cause Analysis

**Date:** October 20, 2025
**Investigator:** Data Quality Analysis
**Model:** Production Model (trained on 587K games, 2003-2025)

---

## Executive Summary

### Critical Finding: LAST GAME DNP BUG

The model is producing **negative predictions** (-0.16 for Jarrett Allen, -0.02 for Mikal Bridges) because:

1. **Both players had DNP (Did Not Play) games as their most recent game**
   - Jarrett Allen: Last game on 2025-04-13 with **1 minute, 1 PRA**
   - Mikal Bridges: Last game on 2025-04-13 with **0 minutes, 0 PRA**

2. **The model was trained WITH these DNP games but is predicting AFTER them**
   - Training data includes 25,093 zero-PRA games (4.27% of all games)
   - Training data includes 2,634 DNP games (0.45% of all games)
   - Model learned that zero/low PRA values are valid outputs

3. **Missing CTG features during prediction time**
   - Model expects 37 features including 6 CTG features
   - Prediction script only has 31/37 features available
   - CTG features (USG%, PSA, etc.) are missing from the CSV

4. **Training MAE of 1.40 is IMPOSSIBLE and indicates severe overfitting**
   - Realistic MAE should be 7-9 points
   - Model memorized training data rather than learning patterns
   - This explains negative predictions on edge cases

---

## Detailed Findings

### 1. Data Quality Issues in Training Set

#### Missing Values
| Column | Missing Count | Missing % |
|--------|--------------|-----------|
| FG_PCT | 29,546 | 5.03% |
| All other core stats | 0 | 0.00% |

#### Invalid PRA Values
- **Negative PRA games:** 0 ✅
- **Zero PRA games:** 25,093 (4.27% of dataset)
- **DNP games (0 minutes):** 2,634 (0.45%)
- **⚠️ DATA ERROR:** 175 DNP games have non-zero PRA (e.g., 0 MIN but 1 REB recorded)

#### Duplicate Games
- **2 duplicate player-game combinations found:**
  - Shawn Marion on 2007-12-19 (two conflicting stat lines)
  - One shows 23 PTS, 10 REB (35 PRA)
  - Another shows 0 PTS, 1 REB (1 PRA)

---

### 2. Analysis of Players with Negative Predictions

#### Jarrett Allen (-0.16 PRA predicted)

**Historical Stats:**
- Total games: 586
- Career PRA average: 23.7 (excellent)
- Career PRA range: 1 to 54

**Last Game (2025-04-13):**
- Minutes: **1** (clear DNP/garbage time)
- PRA: **1** (0 PTS, 0 REB, 1 AST)
- Context: Last game of season, likely rested

**Recent Form (Last 5 games before DNP):**
- 2025-04-11: 26 PRA (33 MIN)
- 2025-04-10: 14 PRA (16 MIN)
- 2025-04-08: 24 PRA (29 MIN)
- 2025-04-06: 26 PRA (31 MIN)
- **Average: 22.5 PRA** - completely normal

**Lag Features at Prediction Time:**
- PRA_lag1: 26.00 (good)
- PRA_L5_mean: 20.20 (good)
- PRA_L10_mean: 25.00 (excellent)

**⚠️ The model saw:** MIN=1, PRA=1 as the most recent game
**Expected prediction:** ~24 PRA based on historical average
**Actual prediction:** -0.16 PRA (IMPOSSIBLE)

---

#### Mikal Bridges (-0.02 PRA predicted)

**Historical Stats:**
- Total games: 595
- Career PRA average: 21.4 (excellent)
- Career PRA range: 0 to 59

**Last Game (2025-04-13):**
- Minutes: **0** (DNP)
- PRA: **0**
- Context: Last game of season, likely rested

**Recent Form (Last 5 games before DNP):**
- 2025-04-11: 27 PRA (36 MIN)
- 2025-04-10: 26 PRA (33 MIN)
- 2025-04-08: 22 PRA (38 MIN)
- 2025-04-06: 32 PRA (39 MIN)
- **Average: 26.75 PRA** - excellent form

**Lag Features at Prediction Time:**
- PRA_lag1: 27.00 (excellent)
- PRA_L5_mean: 27.00 (excellent)
- PRA_L10_mean: 26.90 (excellent)

**⚠️ The model saw:** MIN=0, PRA=0 as the most recent game
**Expected prediction:** ~27 PRA based on recent form
**Actual prediction:** -0.02 PRA (IMPOSSIBLE)

---

#### Jalen Johnson (5.14 PRA predicted - also suspiciously low)

**Historical Stats:**
- Total games: 192
- Career PRA average: 19.3
- Recent form (2024-25): Averaging 35+ PRA in many games

**Last Game (2025-01-23):**
- Minutes: **11** (left game early?)
- PRA: **5**
- Context: Injury or foul trouble?

**Recent Form (Last 5 games before injury):**
- 2025-01-22: 30 PRA
- 2025-01-20: 27 PRA
- 2025-01-18: 30 PRA
- 2024-12-29: 22 PRA
- **Average: 27.3 PRA** - excellent form

**⚠️ Another case of low-minute game dragging prediction down**

---

### 3. Feature Analysis

#### Lag Feature Coverage
| Feature | Missing Count | Missing % |
|---------|--------------|-----------|
| PRA_lag1 | 2,280 | 0.39% |
| PRA_L5_mean | 1 | 0.00% |
| PRA_L10_mean | 1 | 0.00% |

✅ Lag features have excellent coverage

#### Feature Importance (from model)
| Rank | Feature | Importance | % of Total |
|------|---------|-----------|------------|
| 1 | FGA | 0.4555 | 45.55% |
| 2 | PRA_ewma15 | 0.1434 | 14.34% |
| 3 | MIN | 0.1226 | 12.26% |
| 4 | FG_PCT | 0.0780 | 7.80% |
| 5 | DREB | 0.0609 | 6.09% |
| ... | ... | ... | ... |
| **14** | **PRA_lag1** | **0.0003** | **0.03%** ⚠️ |

**⚠️ Critical Finding:** PRA_lag1 has almost ZERO importance (0.03%)

This is suspicious because:
- Recent performance should be highly predictive
- The model may be overfitting to in-game stats (FGA, MIN, etc.)
- This explains why DNP games (MIN=0, FGA=0) produce negative predictions

---

### 4. Model Training Issues

#### Training Performance
- **Training MAE: 1.40 points** ⚠️ IMPOSSIBLE
- Realistic MAE should be 7-9 points
- This level of accuracy indicates **severe overfitting**

#### Evidence of Overfitting
1. **Training MAE too low:** Model memorized rather than learned
2. **Negative predictions exist:** Model extrapolates beyond valid range
3. **FGA dominates (45.55%):** Model relies on in-game stats, not pre-game predictors
4. **Lag features weak (0.03%):** Model ignores recent performance trends

#### Root Cause
The training script uses **all available features including in-game stats**:
- FGA (field goal attempts)
- MIN (minutes played)
- FG_PCT (shooting percentage)

**These are NOT known before the game!**

The model is essentially learning: "If FGA is high, PRA is high" (circular logic)

For a **prediction model**, we should ONLY use:
- Pre-game features (CTG stats, historical averages)
- Lag features (recent performance)
- Contextual features (opponent, rest days)

---

### 5. Missing CTG Features During Prediction

#### Expected Features (37 total)
Model was trained with:
- Core stats: MIN, FGA, FG_PCT, etc.
- Lag features: PRA_lag1, PRA_L5_mean, etc.
- **CTG features: CTG_USG, CTG_PSA, CTG_AST_PCT, CTG_TOV_PCT, CTG_eFG, CTG_REB_PCT**

#### Available Features at Prediction Time (31 total)
The CSV `all_game_logs_through_2025.csv` does NOT contain CTG features.

**Missing:** 6 CTG features

#### Impact
- Model receives incomplete feature set
- Missing features are filled with 0 (via `.fillna(0)`)
- This creates invalid feature combinations
- Model extrapolates into untrained territory → negative predictions

---

### 6. DNP Game Analysis

#### Training Set Contains 529 "Last Game DNP" Scenarios
Games where:
- Current game had normal minutes (≥15 MIN)
- Previous game was DNP/garbage time (≤5 MIN)
- Player has good history (L10 avg >15 PRA)

**Actual PRA in these cases:**
- Mean: **19.34 PRA** (players bounce back)
- Median: **17.00 PRA**
- Std: 9.44

**Model Learned:** After a DNP, players return to normal performance

**Problem:** When the DNP game is the LAST game in history (like Allen & Bridges), the model has no "bounce back" game to learn from. Combined with missing CTG features, it predicts negative values.

---

## Root Cause Summary

### Why Negative Predictions Occur

1. **Data Leakage in Training**
   - Model uses in-game stats (FGA, MIN, FG_PCT) which aren't known pre-game
   - FGA alone accounts for 45.55% of predictions
   - Model overfits to training data (MAE = 1.40, unrealistic)

2. **Missing Features at Prediction Time**
   - CTG features (6 columns) missing from CSV
   - Model fills with 0, creating invalid input
   - Model extrapolates beyond trained range

3. **Last Game DNP Effect**
   - Both problematic players had DNP as last game
   - Model sees: MIN=0, FGA=0, PRA=0/1
   - With missing CTG features, model predicts continuation: negative PRA

4. **Weak Temporal Features**
   - PRA_lag1 has only 0.03% importance
   - Model ignores recent form (which was excellent for both players)
   - Model over-relies on current game stats (which are zeros for DNP)

---

## Recommended Solutions

### Immediate Fixes (Required)

1. **Remove in-game stats from features**
   - Remove: FGA, FGM, FG_PCT, FG3A, FG3M, FG3_PCT, FTA, FTM, FT_PCT
   - Remove: PTS, REB, AST, OREB, DREB, STL, BLK, TOV, PF
   - Remove: MIN (this is a POST-game stat)
   - Keep ONLY: Lag features, CTG features, contextual features

2. **Add CTG features to the CSV**
   - Merge CTG stats (USG%, PSA, etc.) into `all_game_logs_through_2025.csv`
   - Follow the merge logic from `game_log_builder.py:180-205`
   - Verify no duplicates after merge

3. **Filter out DNP games from training**
   - Remove games where MIN < 5 (DNP/garbage time)
   - These games have unreliable stats and distort the model
   - Alternative: Add a "DNP flag" feature

4. **Add prediction bounds**
   - Clip predictions to [0, 100] range
   - PRA cannot be negative (it's Points + Rebounds + Assists)
   - Add post-processing: `predictions = np.clip(predictions, 0, 100)`

5. **Retrain model with proper features**
   - Expected MAE: 7-9 points (realistic)
   - Verify lag features gain importance (should be top 5)
   - Cross-validate on walk-forward basis

### Medium-Term Improvements

6. **Add minutes projection feature**
   - Use L5/L10 average minutes as a feature
   - This helps model understand playing time context

7. **Add opponent defensive features**
   - Opponent DRtg (defensive rating)
   - Opponent pace
   - Position-specific matchup data

8. **Improve feature engineering**
   - True Shooting % (game-level calculation)
   - Rest days, back-to-back flags
   - Home/away splits
   - See: `FEATURE_ENGINEERING_RECOMMENDATIONS.md`

---

## Data Cleaning Steps (Before Retraining)

### Step 1: Remove Duplicate Games
```python
# Remove Shawn Marion duplicate (2007-12-19)
df = df.drop_duplicates(subset=['PLAYER_ID', 'GAME_DATE'], keep='first')
```

### Step 2: Filter DNP Games
```python
# Remove games with < 5 minutes (DNP/garbage time)
df = df[df['MIN'] >= 5]
```

### Step 3: Add CTG Features
```python
# Use CTGFeatureBuilder to merge season stats
from utils.ctg_feature_builder import CTGFeatureBuilder
ctg_builder = CTGFeatureBuilder()

# Map season format
df['CTG_SEASON'] = df['SEASON'].apply(lambda x: f"{x[:4]}-{x[4:6]}")

# Get CTG features for each player-season
for idx, row in df.iterrows():
    ctg_feats = ctg_builder.get_player_ctg_features(
        row['PLAYER_NAME'], row['CTG_SEASON']
    )
    df.loc[idx, 'CTG_USG'] = ctg_feats['CTG_USG']
    df.loc[idx, 'CTG_PSA'] = ctg_feats['CTG_PSA']
    # ... etc
```

### Step 4: Recalculate Lag Features
```python
# After filtering, recalculate lag features
player_groups = df.groupby('PLAYER_ID')

df['PRA_lag1'] = player_groups['PRA'].shift(1)
df['PRA_L5_mean'] = player_groups['PRA'].shift(1).rolling(5, min_periods=1).mean()
# ... etc
```

### Step 5: Use Only Pre-Game Features
```python
feature_cols = [
    # Lag features (recent performance)
    'PRA_lag1', 'PRA_lag3', 'PRA_lag5',
    'PRA_L5_mean', 'PRA_L10_mean', 'PRA_L20_mean',
    'PRA_ewma5', 'PRA_ewma10', 'PRA_ewma15',

    # CTG season stats (player baseline)
    'CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', 'CTG_TOV_PCT',
    'CTG_eFG', 'CTG_REB_PCT',

    # Contextual features (add these)
    'MIN_L5_mean',  # Minutes projection
    # 'opponent_DRtg',  # TODO: Add opponent defense
    # 'rest_days',      # TODO: Add rest/schedule
]
```

---

## Validation Standards

After retraining, verify:

1. **No negative predictions:** `assert (predictions >= 0).all()`
2. **Training MAE is realistic:** 7-9 points (not 1.4)
3. **Lag features have importance:** PRA_lag1 should be top 5 features
4. **Walk-forward validation:** Test on 2024-25 season, expect 52-55% win rate
5. **DNP handling:** Test predictions for players with recent DNP games

---

## Files to Update

1. **Training script:** `scripts/production/train_production_model_2025.py`
   - Remove in-game stats from features
   - Add CTG merge logic
   - Filter MIN >= 5
   - Add prediction clipping

2. **Game logs CSV:** `data/game_logs/all_game_logs_through_2025.csv`
   - Add CTG features columns
   - Remove duplicates
   - Mark DNP games

3. **Prediction script:** (create new) `scripts/production/predict_oct_22_2025.py`
   - Use only pre-game features
   - Add minutes projection
   - Clip predictions to [0, 100]

---

## Expected Outcomes After Fixes

### Training Performance
- **MAE:** 7-9 points (realistic for NBA props)
- **No negative predictions:** All predictions >= 0
- **Feature importance:** Lag features + CTG features dominate

### Prediction Quality
- **Allen's prediction:** ~24 PRA (based on L10 avg of 25)
- **Bridges' prediction:** ~27 PRA (based on L5 avg of 27)
- **Johnson's prediction:** ~30 PRA (based on recent form)

### Win Rate (Betting Simulation)
- **Current (broken):** Unknown (negative predictions invalidate results)
- **Expected (fixed):** 52-55% win rate, 2-5% ROI

---

## Conclusion

The negative predictions are caused by a **combination of issues**:

1. ✅ **Data quality:** Minor issues (2 duplicates, 175 DNP errors)
2. ❌ **Model overfitting:** Training MAE of 1.40 is impossible
3. ❌ **Data leakage:** Using in-game stats (FGA, MIN) which aren't pre-game
4. ❌ **Missing features:** CTG features not in CSV
5. ❌ **Last game DNP:** Both players had 0-minute games as last entry

**Priority 1:** Retrain model with ONLY pre-game features
**Priority 2:** Add CTG features to CSV
**Priority 3:** Filter DNP games from training
**Priority 4:** Add prediction bounds (clip to [0, 100])

After these fixes, the model should produce realistic predictions in the 20-30 PRA range for these players, based on their excellent recent form.
