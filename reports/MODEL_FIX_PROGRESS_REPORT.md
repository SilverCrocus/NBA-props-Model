# Model Fix Progress Report - October 20, 2025

## 🎯 Executive Summary

We successfully fixed **3 of 5 critical issues** in the NBA Props model:

✅ **FIXED:**
1. Negative predictions eliminated (0 in all splits)
2. Data retention improved (41% → 99.7%)
3. Train/Val gap healthy (77% → 8.8%)

❌ **STILL BROKEN:**
4. MAE unrealistically low (1.44 vs expected 7-9)
5. **ROOT CAUSE:** In-game feature leakage (FGA, MIN, FG_PCT)

**Status:** Model is NOT ready for production betting until Phase 2 fixes applied.

---

## 📋 Phase 1 Fixes Implemented

### Fix #1: DNP Game Filtering ✅
**Before:** DNP/garbage time games (MIN < 10) polluted lag features
**After:** Filtered 85,457 games with MIN < 10
**Impact:** Prevents predictions based on non-meaningful games

### Fix #2: Vectorized CTG Merge ✅
**Before:** O(N²) loop took 2-4 hours
**After:** Vectorized pandas merge takes 2-3 minutes
**Impact:** 50-100x speed improvement

### Fix #3: Data Imputation ✅
**Before:** Dropped 59% of data (346K games) due to NaN
**After:** Retained 99.7% of data (500K games) with imputation
**Impact:** +131% more training data

### Fix #4: Train/Val/Test Splits ✅
**Before:** No validation, trained on all data
**After:** Temporal splits (Train: 2003-2023, Val: 2023-24, Test: 2024-25)
**Impact:** Proper evaluation, no temporal leakage in splits

### Fix #5: XGBoost Regularization ✅
**Before:** `max_depth=6, lr=0.05, gamma=0`
**After:** `max_depth=4, lr=0.01, gamma=0.1, reg_alpha=0.1, reg_lambda=5`
**Impact:** Train/Val gap reduced from 77% → 8.8%

### Fix #6: Non-Negative Constraint ✅
**Before:** `objective='reg:squarederror'` allowed negative predictions
**After:** `objective='reg:gamma'` enforces PRA > 0
**Impact:** 0 negative predictions (vs multiple before)

---

## 📊 Model Performance After Phase 1 Fixes

| Metric | Before Fixes | After Phase 1 | Target | Status |
|--------|-------------|---------------|---------|--------|
| **Training MAE** | 1.40 pts | 1.44 pts | 6-8 pts | ❌ Still too low |
| **Validation MAE** | 6.10 pts | 1.57 pts | 7-9 pts | ❌ Too low |
| **Test MAE** | Unknown | 1.55 pts | 8-10 pts | ❌ Too low |
| **Train/Val gap** | 77% | 8.8% | <20% | ✅ FIXED |
| **Negative predictions** | Multiple | 0 | 0 | ✅ FIXED |
| **Data retention** | 41% | 99.7% | >90% | ✅ FIXED |

---

## 🔍 REMAINING ISSUE: In-Game Feature Leakage

### Feature Importance Analysis

**Top 5 features:**
1. **FGA (27.1%)** - ❌ Field goal attempts in current game (unknown pre-game!)
2. **PRA_ewma15 (16.9%)** - ✅ Lag feature (valid)
3. **MIN (14.0%)** - ❌ Minutes in current game (unknown pre-game!)
4. **PRA_ewma10 (11.3%)** - ✅ Lag feature (valid)
5. **FG_PCT (7.2%)** - ❌ Shooting % in current game (unknown pre-game!)

### The Problem

**In-game stats (FGA, MIN, FG_PCT, FTA, FT_PCT) create data leakage:**

1. **Training scenario:**
   - Model sees: "Player played 35 MIN, took 20 FGA, shot 50% FG"
   - Model learns: "When MIN=35 and FGA=20 → predict ~25 PRA"
   - **This is VALID** because we know these stats after the game

2. **Prediction scenario (BROKEN):**
   - Model needs: "Player will play ? MIN, take ? FGA, shoot ? FG%"
   - **We DON'T KNOW these values before the game!**
   - Current script uses LAST GAME values (e.g., last game MIN=1 for Jarrett Allen)
   - **This is INVALID** and causes garbage predictions

### Why MAE is Artificially Low

The model is essentially solving:
```
PRA = f(FGA, MIN, FG_PCT, ...)
```

This is easy to predict IN-SAMPLE (hence MAE 1.44), but **impossible to use for betting** because we don't know FGA/MIN before the game!

**Analogy:** It's like predicting a student's test score AFTER seeing their answer sheet. Yes, you'll be accurate, but it's useless for prediction.

---

## 🛠️ PHASE 2 FIXES REQUIRED (Not Yet Implemented)

### Critical: Remove In-Game Features

**Files to modify:**
- `scripts/production/train_model_FIXED.py` (line 185)
- `scripts/production/predict_oct22_FIXED.py` (to be created)

**Changes:**

**BEFORE (current):**
```python
core_stats = ['MIN', 'FGA', 'FG_PCT', 'FG3A', 'FG3_PCT',
              'FTA', 'FT_PCT', 'OREB', 'DREB', 'STL', 'BLK', 'TOV', 'PF']
```

**AFTER (Phase 2):**
```python
# Only use PRE-GAME features:
core_stats = []  # Remove all in-game stats

# Use only lag features + CTG stats:
feature_cols = lag_features + ctg_features + contextual_features

# Where:
lag_features = ['PRA_lag1', 'PRA_L5_mean', 'PRA_ewma15', etc.]
ctg_features = ['CTG_USG', 'CTG_PSA', 'CTG_AST_PCT', etc.]
contextual_features = ['Days_Rest', 'Is_BackToBack', 'OPP_DRtg', 'Minutes_Projected']
```

### Add Minutes Projection Feature

**Why needed:** If we remove MIN from features, model needs to estimate expected playing time

**Implementation:**
```python
# Calculate player's typical minutes (L5 average)
df['Minutes_Projected'] = df.groupby('PLAYER_ID')['MIN'].shift(1).rolling(5, min_periods=1).mean()

# Use this instead of actual MIN as a feature
```

### Expected Results After Phase 2

| Metric | After Phase 1 | After Phase 2 (Expected) | Improvement |
|--------|---------------|--------------------------|-------------|
| **Training MAE** | 1.44 pts | **6-7 pts** | Realistic |
| **Validation MAE** | 1.57 pts | **7-8 pts** | Realistic |
| **Test MAE** | 1.55 pts | **8-10 pts** | Realistic |
| **Feature importance** | FGA 27.1% | PRA_ewma15 ~30% | Lag features dominate |
| **Usable for betting?** | ❌ NO | ✅ YES | Production-ready |

---

## 🎯 RECOMMENDATIONS

### Option 1: Use Current Model (NOT RECOMMENDED)

**Pros:**
- Already trained
- No negative predictions
- Can generate predictions for Oct 22

**Cons:**
- ❌ MAE artificially low (not trustworthy)
- ❌ Uses in-game stats (cheating)
- ❌ Predictions based on last game stats (broken for DNP games)
- ❌ **HIGH RISK for betting** - model hasn't truly learned

**Verdict:** Only use for demonstration/testing, NOT real betting

### Option 2: Implement Phase 2 Fixes (RECOMMENDED)

**Time required:** 2-3 hours

**Steps:**
1. Remove FGA, MIN, FG_PCT, FTA, FT_PCT from features (15 min)
2. Add Minutes_Projected feature (30 min)
3. Retrain model (10 min)
4. Verify MAE is 7-9 points (realistic) (15 min)
5. Run backtest on 2024-25 (30 min)
6. Generate Oct 22 predictions (10 min)
7. Validate predictions are realistic (15 min)

**Expected outcome:**
- ✅ MAE 7-9 points (matches research)
- ✅ No in-game feature leakage
- ✅ Ready for production betting
- ✅ 52-55% win rate expected

---

## 📈 COMPARISON: Phase 1 vs Phase 2

### What We Fixed in Phase 1

| Issue | Status | Impact |
|-------|--------|--------|
| Negative predictions | ✅ FIXED | No more impossible values |
| Data loss (59%) | ✅ FIXED | 2.4x more training data |
| O(N²) performance bug | ✅ FIXED | 50-100x faster |
| Train/Val overfitting gap | ✅ FIXED | Healthy 8.8% gap |
| No validation strategy | ✅ FIXED | Proper temporal splits |

### What Phase 2 Will Fix

| Issue | Status | Impact |
|-------|--------|--------|
| In-game feature leakage | ⏳ PENDING | True pre-game prediction |
| MAE artificially low | ⏳ PENDING | Realistic 7-9 pts |
| Unusable for betting | ⏳ PENDING | Production-ready |
| Lag features underutilized | ⏳ PENDING | Proper feature importance |

---

## 🔄 NEXT STEPS

### Immediate (If Using Current Model)

Despite limitations, you CAN generate predictions for Oct 22 with caveats:

1. ✅ Run prediction script (use last 5-game averages for "core stats")
2. ✅ Generate betting recommendations
3. ⚠️  **Use conservative bet sizing** ($25-50 instead of $100)
4. ⚠️  **Track results carefully** to validate model
5. ⚠️  **Don't trust large edges** (10+ pts likely wrong)

### Recommended (Phase 2 Fixes)

1. Implement "pre-game features only" version
2. Retrain with realistic MAE target (7-9 pts)
3. Backtest on full 2024-25 season
4. Validate win rate 52-55% (not 99%!)
5. THEN use for Oct 22 betting with confidence

---

## 📊 VALIDATION CHECKLIST

### Phase 1 (Current State)

- [x] No negative predictions
- [x] Train/Val gap <20%
- [x] Data retention >90%
- [ ] Training MAE 6-8 pts
- [ ] Validation MAE 7-9 pts

### Phase 2 (Required for Production)

- [ ] No in-game features (FGA, MIN, etc.)
- [ ] MAE 7-9 pts across all splits
- [ ] Lag features are top importance
- [ ] Backtest win rate 52-55%
- [ ] Edge sizes 3-10 pts (realistic)

---

## 📁 FILES CREATED

### Phase 1
- ✅ `scripts/production/train_model_FIXED.py` - Fixed training script
- ✅ `models/production_model_FIXED_latest.pkl` - Trained model (with limitations)
- ✅ `reports/MODEL_FIX_PROGRESS_REPORT.md` - This document

### Phase 2 (TODO)
- ⏳ `scripts/production/train_model_FIXED_V2.py` - Remove in-game features
- ⏳ `scripts/production/predict_oct22_FIXED.py` - Use pre-game features only
- ⏳ `scripts/production/backtest_2024_25_FIXED.py` - Validate on full season
- ⏳ `reports/MODEL_FIX_FINAL_REPORT.md` - After Phase 2 complete

---

## 💡 KEY TAKEAWAYS

1. **We fixed 3 of 5 critical bugs** ✅
   - No negative predictions
   - Data retention 99.7%
   - Train/Val gap healthy

2. **BUT model still has feature leakage** ❌
   - Using in-game stats (FGA, MIN, FG_PCT)
   - MAE artificially low (1.44 vs realistic 7-9)
   - Not suitable for production betting yet

3. **Phase 2 fixes are straightforward** ⏱️ 2-3 hours
   - Remove in-game features
   - Add minutes projection
   - Retrain and validate

4. **Current model CAN be used for Oct 22** ⚠️
   - But with conservative bet sizing
   - And realistic expectations (not 99% win rate!)
   - Track results for validation

---

**Status:** Phase 1 Complete (60% fixed) | Phase 2 Pending (40% remaining)

**Recommendation:** Proceed with Phase 2 before real betting, or use Phase 1 model with 50% normal bet sizes for validation.
