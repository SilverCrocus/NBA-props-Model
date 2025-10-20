# Immediate Action Items - Fix Negative Predictions

## Critical Issues Identified

### 🚨 Root Cause: DATA LEAKAGE + MISSING FEATURES

The model is using **in-game stats** (FGA, MIN, etc.) which creates:
1. Unrealistic training MAE of 1.40 (overfitting)
2. Negative predictions when MIN=0 or FGA=0
3. Weak temporal features (PRA_lag1 only 0.03% importance)

---

## Quick Fix Checklist (Do These Now)

### ✅ Step 1: Add Prediction Bounds (5 minutes)

**File:** `scripts/production/predict_oct_22_2025.py` (or wherever predictions are made)

```python
# After getting predictions
predictions = model.predict(X)

# Add bounds - PRA cannot be negative
predictions = np.clip(predictions, 0, 100)

# Save predictions
```

**Expected Result:** No more negative predictions (temporary band-aid)

---

### ✅ Step 2: Filter DNP Games from Training (10 minutes)

**File:** `scripts/production/train_production_model_2025.py` (line 46)

```python
# After loading data
df = pd.read_csv('data/game_logs/all_game_logs_through_2025.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

# ADD THIS LINE - Filter out DNP/garbage time games
print(f"Before DNP filter: {len(df):,} games")
df = df[df['MIN'] >= 5].copy()  # Remove games with < 5 minutes
print(f"After DNP filter: {len(df):,} games")

df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])
```

**Expected Result:** Remove 2,634 DNP games that distort model

---

### ✅ Step 3: Remove Duplicate Games (5 minutes)

**File:** `scripts/production/train_production_model_2025.py` (line 48)

```python
# After loading data and sorting
df = df.sort_values(['PLAYER_ID', 'GAME_DATE'])

# ADD THIS LINE - Remove duplicates
df = df.drop_duplicates(subset=['PLAYER_ID', 'GAME_DATE'], keep='first')
print(f"Removed {len_before - len(df)} duplicate games")
```

**Expected Result:** Remove 2 duplicate Shawn Marion games

---

## Major Refactor Required (Do These Next)

### ⚠️ Step 4: Remove In-Game Stats from Features (30 minutes)

**Problem:** Model uses FGA (45.55% importance), MIN (12.26%), etc. - these aren't known before the game!

**File:** `scripts/production/train_production_model_2025.py` (line 127-136)

**Current code:**
```python
# Define feature columns
core_stats = ['MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT',
              'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF']

lag_features = [col for col in df.columns if any(x in col for x in
                ['_lag', '_L3_', '_L5_', '_L10_', '_L20_', '_ewma', '_trend'])]

ctg_features = ['CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', 'CTG_TOV_PCT', 'CTG_eFG', 'CTG_REB_PCT']

feature_cols = core_stats + lag_features + ctg_features
```

**Replace with:**
```python
# Define feature columns - ONLY PRE-GAME FEATURES
# Remove core_stats entirely - those are in-game stats!

lag_features = [col for col in df.columns if any(x in col for x in
                ['_lag', '_L3_', '_L5_', '_L10_', '_L20_', '_ewma', '_trend'])]

ctg_features = ['CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', 'CTG_TOV_PCT', 'CTG_eFG', 'CTG_REB_PCT']

contextual_features = ['MIN_L5_mean']  # Minutes projection (historical average)

feature_cols = lag_features + ctg_features + contextual_features
feature_cols = [col for col in feature_cols if col in df.columns]

print(f"\n⚠️  USING ONLY PRE-GAME FEATURES")
print(f"   Lag features: {len([c for c in feature_cols if 'lag' in c or 'L' in c or 'ewma' in c])}")
print(f"   CTG features: {len([c for c in feature_cols if 'CTG' in c])}")
print(f"   Contextual: {len([c for c in feature_cols if c in contextual_features])}")
```

**Expected Result:**
- Training MAE will increase to 7-9 points (realistic!)
- Feature importance will shift to lag features + CTG
- Model will use actual predictive features

---

### ⚠️ Step 5: Fix CTG Feature Merge (1 hour)

**Problem:** CTG features are calculated during training but NOT saved to CSV, so they're missing at prediction time.

**Solution A: Add CTG to CSV (Recommended)**

Create new script: `scripts/data_processing/add_ctg_to_game_logs.py`

```python
"""
Add CTG features to game logs CSV
"""
import pandas as pd
import sys
sys.path.append('utils')
from ctg_feature_builder import CTGFeatureBuilder

print("Loading game logs...")
df = pd.read_csv('data/game_logs/all_game_logs_through_2025.csv')
df['GAME_DATE'] = pd.to_datetime(df['GAME_DATE'])

print("Initializing CTG builder...")
ctg_builder = CTGFeatureBuilder()

# Map season format
df['CTG_SEASON'] = df['SEASON'].apply(lambda x: x if '-' in str(x) else f"{x[:4]}-{x[4:6]}")

# Initialize CTG columns
df['CTG_USG'] = 0.0
df['CTG_PSA'] = 0.0
df['CTG_AST_PCT'] = 0.0
df['CTG_TOV_PCT'] = 0.0
df['CTG_eFG'] = 0.0
df['CTG_REB_PCT'] = 0.0
df['CTG_Available'] = 0

print("Adding CTG features...")
ctg_cache = {}
for idx, row in df.iterrows():
    player_name = row['PLAYER_NAME']
    season = row['CTG_SEASON']

    cache_key = f"{player_name}_{season}"

    if cache_key not in ctg_cache:
        ctg_cache[cache_key] = ctg_builder.get_player_ctg_features(player_name, season)

    ctg_feats = ctg_cache[cache_key]

    df.loc[idx, 'CTG_USG'] = ctg_feats['CTG_USG']
    df.loc[idx, 'CTG_PSA'] = ctg_feats['CTG_PSA']
    df.loc[idx, 'CTG_AST_PCT'] = ctg_feats['CTG_AST_PCT']
    df.loc[idx, 'CTG_TOV_PCT'] = ctg_feats['CTG_TOV_PCT']
    df.loc[idx, 'CTG_eFG'] = ctg_feats['CTG_eFG']
    df.loc[idx, 'CTG_REB_PCT'] = ctg_feats['CTG_REB_PCT']
    df.loc[idx, 'CTG_Available'] = ctg_feats['CTG_Available']

    if idx % 10000 == 0:
        print(f"   Processed {idx:,} / {len(df):,} games...")

print(f"\nCTG coverage: {(df['CTG_Available'] == 1).mean() * 100:.1f}%")

# Save updated CSV
output_path = 'data/game_logs/all_game_logs_with_ctg.csv'
df.to_csv(output_path, index=False)
print(f"\n✅ Saved to {output_path}")
```

Run: `uv run scripts/data_processing/add_ctg_to_game_logs.py`

**Solution B: Recalculate CTG at Prediction Time**

Update prediction script to run CTG builder before predictions (slower but works)

---

## Expected Outcomes After Fixes

### Quick Fixes (Steps 1-3)
- ✅ No negative predictions (clipped to 0)
- ✅ No duplicate games
- ✅ No DNP games distorting model

### Major Refactor (Steps 4-5)
- ✅ **Training MAE: 7-9 points** (realistic)
- ✅ **Feature importance:** Lag features become top features (not FGA)
- ✅ **Predictions for Allen/Bridges:** 24-27 PRA (not negative)
- ✅ **Model uses only pre-game data:** No data leakage

---

## Testing After Changes

### Test 1: No Negative Predictions
```python
# After training
predictions = model.predict(X_test)
assert (predictions >= 0).all(), "Found negative predictions!"
print("✅ No negative predictions")
```

### Test 2: Realistic MAE
```python
# After training
train_mae = mean_absolute_error(y_train, model.predict(X_train))
assert 6 <= train_mae <= 10, f"MAE {train_mae} is unrealistic"
print(f"✅ Training MAE: {train_mae:.2f} (realistic)")
```

### Test 3: Lag Features Have Importance
```python
# After training
importance_df = pd.DataFrame({
    'feature': feature_cols,
    'importance': model.feature_importances_
}).sort_values('importance', ascending=False)

lag_features = importance_df[importance_df['feature'].str.contains('lag|L5|L10|ewma')]
top_10_has_lag = any(feat in lag_features['feature'].values for feat in importance_df.head(10)['feature'])

assert top_10_has_lag, "Lag features should be in top 10!"
print("✅ Lag features are important")
```

### Test 4: Predict Allen & Bridges
```python
# Load their latest stats
allen_features = get_player_features('Jarrett Allen', '2025-10-22')
bridges_features = get_player_features('Mikal Bridges', '2025-10-22')

allen_pred = model.predict([allen_features])[0]
bridges_pred = model.predict([bridges_features])[0]

print(f"Jarrett Allen prediction: {allen_pred:.1f} PRA")
print(f"Mikal Bridges prediction: {bridges_pred:.1f} PRA")

# Should be in reasonable ranges
assert 15 <= allen_pred <= 35, f"Allen prediction {allen_pred} seems off"
assert 18 <= bridges_pred <= 35, f"Bridges prediction {bridges_pred} seems off"
print("✅ Predictions are reasonable")
```

---

## Priority Order

### 🔥 DO FIRST (Today)
1. **Step 2:** Filter DNP games (removes 0.45% of distorted data)
2. **Step 3:** Remove duplicates (fixes data integrity)
3. **Step 1:** Add prediction bounds (prevents negative outputs)

### 🚀 DO NEXT (This Week)
4. **Step 4:** Remove in-game features (fixes data leakage)
5. **Step 5:** Add CTG to CSV (fixes missing features)
6. **Retrain model** with new feature set
7. **Re-run predictions** for Oct 22, 2025

### 📊 DO LATER (Improvements)
8. Add opponent defensive features
9. Add rest/schedule features
10. Add minutes projection logic
11. Implement walk-forward validation
12. Backtest on 2024-25 season

---

## Questions?

If you encounter issues:

1. **"Training MAE is still 1.40"**
   - Check: Did you remove in-game stats (FGA, MIN, etc.)?
   - Check: Are you using ONLY lag + CTG features?

2. **"Still getting negative predictions"**
   - Check: Did you add `np.clip(predictions, 0, 100)`?
   - Check: Are CTG features available in the data?

3. **"CTG features all zero"**
   - Check: Did you run the CTG merge script?
   - Check: Does `all_game_logs_with_ctg.csv` exist?

4. **"Model won't train"**
   - Check: After filtering DNP games, do you still have enough data?
   - Check: Are there NaN values in the feature columns?

---

## Success Criteria

✅ Model trained successfully
✅ Training MAE between 7-9 points
✅ No negative predictions
✅ Lag features in top 10 importance
✅ Allen prediction: 20-28 PRA
✅ Bridges prediction: 22-30 PRA
✅ Johnson prediction: 25-35 PRA

When all criteria pass, the model is ready for production use.
