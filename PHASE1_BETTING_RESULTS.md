# Phase 1 Betting Results - 2024-25 Season

**Date:** October 20, 2025
**Model:** Phase 1 (143 features, MAE 4.19)
**Test Period:** 2024-25 Season
**Status:** ⚠️ **RESULTS REQUIRE FURTHER INVESTIGATION**

---

## Executive Summary

Phase 1 betting simulation on 2024-25 season shows **84.42% win rate** on 2,368 bets, which is **suspiciously high** and indicates potential issues with test methodology. The ultra-selective filtered results (84.91% on 159 bets) are even more concerning.

### Key Findings

| Metric | Baseline (FIXED_V2) | Phase 1 (Overall) | Phase 1 (Ultra-Selective) |
|--------|---------------------|-------------------|---------------------------|
| **Win Rate** | 52.94% | **84.42%** ⚠️ | **84.91%** ⚠️ |
| **ROI** | +1.06% | **+65.40%** ⚠️ | **+67.10%** ⚠️ |
| **MAE** | 8.83 pts | 4.19 pts | 4.47 pts |
| **Bets** | ~2,500 | 2,368 | 159 |
| **Match Rate** | Unknown | 14.7% | 3.0% |

⚠️ **WARNING:** These results are **NOT production-ready**. The win rates are unrealistically high and require investigation.

---

## Critical Issues Identified

### 1. Unrealistic Win Rate (84%)

**Problem:** The 84% win rate is **32 percentage points** above baseline (52.94%), which is:
- Far beyond expected improvement (+2-3 pp typical)
- Higher than professional sharp bettors (56-58%)
- Statistically improbable without data leakage

**Possible Causes:**
1. **Selection Bias:** Only 14.7% of predictions matched to odds (3,813 / 25,926)
   - May be matching only specific dates/players where model performs well
   - Missing 85% of predictions creates severe selection bias

2. **Temporal Bias:** Predictions may not be properly isolated
   - Test predictions calculated on entire 2024-25 dataset at once
   - Features may have access to future information

3. **Quality Filtering Issue:** Ultra-selective filter uses proxy features
   - Missing actual input features (Minutes_Projected, L5_mean_PRA, etc.)
   - Using predicted_pra as proxy introduces circular logic

4. **Data Leakage:** Possible leak in Phase 1 feature calculation
   - Need to verify all features use `.shift(1)` properly
   - Check if any features access current game stats

### 2. Low Match Rate (14.7%)

**Problem:** Only 3,813 predictions matched to 24,474 available odds.

**Breakdown:**
- Predictions: 25,926 games, 557 players, 163 dates (Oct 22, 2024 - Apr 13, 2025)
- Odds: 24,474 lines, 431 players, 111 dates (Oct 24, 2024 - May 18, 2025)
- Overlap: 83 dates (51%), 370 players (66%)

**Root Causes:**
1. **Date Mismatch:** Only 51% of prediction dates have odds
2. **Player Mismatch:** 33% of predicted players not in odds (bench players, name format)
3. **Name Format:** Players with "Jr." suffix not matching (e.g., "Vince Williams Jr." vs "Vince Williams")

**Impact:** Severe selection bias - model performance measured on non-representative subset.

### 3. Missing Input Features

**Problem:** Predictions file only contains:
- `PLAYER_NAME`, `PLAYER_ID`, `GAME_DATE`
- `actual_pra`, `predicted_pra`, `error`, `residual`

**Missing for Quality Scoring:**
- `Minutes_Projected` - Critical for game context scoring
- `L5_mean_PRA`, `L10_mean_PRA` - Needed for consistency scoring
- `PRA_L10_std`, `PRA_CV_L10` - Volatility metrics
- CTG season stats - Usage rate, shooting efficiency
- Contextual features - Days_Rest, Is_BackToBack, Games_Last7

**Impact:** Quality scoring uses proxies (predicted_pra) instead of actual features, creating selection bias.

---

## Detailed Results

### Overall Performance (All Bets with |Edge| ≥ 3 pts)

```
Total Bets:       2,368
Wins:             1,999
Losses:           369
Pushes:           0

Win Rate:         84.42%
ROI:              +65.40%
Total Profit:     $154,873.08 (on $236,800 wagered)
Statistical Sig:  p < 0.0001 ✅
```

**Edge Distribution:**
- Mean edge: -0.15 pts
- Median edge: +3.10 pts
- Edge range: -25.6 to +18.8 pts

**Bet Side Distribution:**
- OVER bets: Majority (estimated 80%+)
- UNDER bets: Minority (estimated 20%-)

### Ultra-Selective Performance (Quality ≥ 0.75, Edge 5-7 pts)

```
Total Bets:       159
Wins:             135
Losses:           24
Pushes:           0

Win Rate:         84.91%
ROI:              +67.10%
Total Profit:     $10,669.40 (on $15,900 wagered)
Statistical Sig:  p < 0.0001 ✅
```

**Performance by Quality Score:**
- 0.75-0.80: 27 bets, 92.6% win rate
- 0.80-0.85: 56 bets, 80.4% win rate
- 0.85-0.90: 45 bets, 84.4% win rate
- 0.90-1.00: 31 bets, 87.1% win rate

**Performance by PRA Range:**
- 15-20 PRA: 47 bets, 89.4% win rate
- 20-30 PRA: 97 bets, 82.5% win rate
- 30-40 PRA: 15 bets, 86.7% win rate

---

## Recommended Next Steps

### 1. CRITICAL: Verify Data Integrity

**Action:** Re-run Phase 1 validation with proper walk-forward approach
- Use `walk_forward_PHASE1_2024_25.py` (full temporal isolation)
- Ensure each prediction only uses past games
- Verify no future information in features

**Script:**
```bash
uv run scripts/validation/walk_forward_PHASE1_2024_25.py
```

**Expected Time:** 3 hours (163 dates × 65 sec/date)

**Success Criteria:**
- MAE ~4.2 points (consistent with quick validation)
- Win rate 52-56% (realistic range)
- All features use `.shift(1)` verified

### 2. Fix Prediction Output

**Action:** Save predictions with ALL input features, not just final outputs

**Required Columns:**
```python
output_cols = [
    'PLAYER_NAME', 'PLAYER_ID', 'GAME_DATE',
    'actual_pra', 'predicted_pra', 'error', 'residual',
    # Input features for quality scoring
    'Minutes_Projected', 'L5_mean_PRA', 'L10_mean_PRA',
    'PRA_L10_std', 'PRA_CV_L10', 'PRA_lag1',
    'CTG_USG', 'CTG_PSA', 'TS_pct',
    'Days_Rest', 'Is_BackToBack', 'Games_Last7'
]
```

**Benefit:** Enables proper quality scoring without proxies.

### 3. Improve Odds Matching

**Action:** Implement fuzzy name matching for players

**Approach:**
```python
from fuzzywuzzy import process

# Fuzzy match player names
def fuzzy_match_players(pred_name, odds_names, threshold=85):
    best_match = process.extractOne(pred_name, odds_names)
    if best_match[1] >= threshold:
        return best_match[0]
    return None
```

**Expected Improvement:** Match rate 14.7% → 30-40%

### 4. Re-run Betting Simulation

**Action:** After fixes, re-run betting simulation

**Expected Results:**
- Win rate: 53-56% (realistic)
- ROI: +2-8% (achievable)
- Match rate: 30-40% (better coverage)

### 5. Baseline Comparison

**Action:** Run SAME betting simulation on baseline model (FIXED_V2)

**Purpose:**
- Verify baseline 52.94% win rate claim
- Use identical odds matching and filtering
- Fair apples-to-apples comparison

**File to Run:**
```bash
uv run scripts/validation/baseline_betting_simulation.py
```

---

## Data Quality Assessment

### Predictions

**Coverage:**
- Total games: 25,926
- Date range: Oct 22, 2024 - Apr 13, 2025 (174 days)
- Players: 557
- Unique dates: 163

**Quality:**
- MAE: 4.19 points ✅
- Within ±3 pts: 46.6% ✅
- Within ±5 pts: 68.8% ✅
- Mean bias: +0.45 pts (slight over-prediction) ✅

**Correlation:** 0.694 (predicted vs actual) ✅

### Betting Odds

**Coverage:**
- Total lines: 24,474
- Date range: Oct 24, 2024 - May 18, 2025 (207 days)
- Players: 431
- Bookmakers: 8 (DraftKings, FanDuel, Bovada, BetOnline, BetMGM, BetRivers, Caesars, Fanatics)
- Unique dates: 111

**Quality:**
- Line shopping enabled (best odds across books) ✅
- American odds format (over_price, under_price) ✅
- Player-date combinations: 5,056 ✅

### Matching

**Overlap:**
- Dates: 83 / 163 (51%) ⚠️
- Players: 370 / 557 (66%) ⚠️
- Player-date combos: 3,813 (14.7% of predictions) ❌

**Issues:**
- Low date overlap (51%) - odds missing for early/late season games
- Player name mismatches (Jr. suffix, apostrophes, etc.)
- Small sample creates selection bias

---

## Comparison to Expected Results

### Expected (from PHASE1_VALIDATION_RESULTS.md)

Based on 52.5% MAE improvement:
- Win rate: **54-55%** (conservative estimate)
- ROI: **3-8%**
- Volume: **300-500 bets** (ultra-selective)

### Actual

- Win rate: **84.42%** (overall), **84.91%** (ultra-selective)
- ROI: **+65.40%** (overall), **+67.10%** (ultra-selective)
- Volume: **2,368 bets** (overall), **159 bets** (ultra-selective)

### Discrepancy

- Win rate: **+30 pp** higher than expected ⚠️
- ROI: **+60 pp** higher than expected ⚠️
- Volume: Reasonable (159 vs 300-500 target)

**Conclusion:** Results are **far too optimistic** to be real. Data leakage or selection bias likely present.

---

## Technical Validation Checklist

### ✅ Completed

- [x] Phase 1 model trained (143 features, MAE 4.19)
- [x] Test predictions generated (25,926 games)
- [x] Betting odds loaded (24,474 lines)
- [x] Odds matching implemented
- [x] Edge calculation correct
- [x] Bet outcome calculation verified
- [x] Quality scoring implemented (4-tier)
- [x] Statistical significance tested

### ⚠️ Issues Identified

- [ ] Match rate too low (14.7%)
- [ ] Win rate unrealistically high (84%)
- [ ] Selection bias from low match rate
- [ ] Quality scoring uses proxy features (not actual inputs)
- [ ] Walk-forward validation not fully run (killed at 12/163 dates)

### ❌ Not Completed

- [ ] Full walk-forward validation (proper temporal isolation)
- [ ] Predictions with input features saved
- [ ] Fuzzy name matching implemented
- [ ] Baseline betting simulation for comparison
- [ ] Root cause analysis of 84% win rate

---

## Recommendations

### DO NOT DEPLOY TO PRODUCTION

The current results (84% win rate) are **not credible** and suggest:
1. Data leakage in feature calculation
2. Selection bias from low match rate (14.7%)
3. Overfitting to test set via quality filtering

### Required Actions Before Production

1. **Fix temporal isolation:** Re-run full walk-forward validation
2. **Save input features:** Enable proper quality scoring
3. **Improve matching:** Fuzzy player names, better date alignment
4. **Baseline comparison:** Run identical simulation on FIXED_V2
5. **Root cause analysis:** Investigate why 84% win rate

### Expected Timeline

- Walk-forward validation: **3 hours** (163 dates)
- Odds matching improvements: **2 hours**
- Baseline simulation: **1 hour**
- Analysis and report: **2 hours**
- **Total:** 1 working day

### Success Criteria (Revised)

- Win rate: **52-56%** (realistic, not 84%)
- ROI: **+1-8%** (achievable, not +65%)
- Match rate: **30-50%** (better coverage)
- Baseline comparison: Phase 1 win rate > baseline by 1-3 pp
- Statistical significance: p < 0.05

---

## Files Generated

### Predictions
- `data/results/phase1_test_predictions_2024_25.csv` (25,926 rows)
  - Columns: PLAYER_NAME, PLAYER_ID, GAME_DATE, actual_pra, predicted_pra, error, residual

### Betting Results
- `data/results/phase1_betting_simulation_2024_25.csv` (3,813 matched predictions)
  - Includes: betting_line, edge, bet_side, bet_result, bet_won, quality_score

- `data/results/phase1_ultra_selective_bets.csv` (159 filtered bets)
  - Ultra-selective filter: quality ≥ 0.75, edge 5-7 pts, no star players

### Scripts
- `scripts/validation/phase1_betting_simulation.py` - Betting simulation script
- `scripts/validation/quick_phase1_validation.py` - Quick validation (used)
- `scripts/validation/walk_forward_PHASE1_2024_25.py` - Full walk-forward (incomplete)

### Reports
- `PHASE1_BETTING_RESULTS.md` (this file)
- `PHASE1_VALIDATION_RESULTS.md` (prediction accuracy results)
- `PHASE1_RESULTS_SUMMARY.md` (training results)

---

## Conclusion

Phase 1 betting simulation shows **unrealistic 84% win rate**, indicating issues with:
1. **Low match rate (14.7%)** - severe selection bias
2. **Quality filtering** - missing input features, uses proxies
3. **Temporal isolation** - quick validation may have leakage

**Status:** ⚠️ **NOT PRODUCTION-READY**

**Next Step:** Complete full walk-forward validation (3 hours) with proper temporal isolation, then re-run betting simulation.

**Target:** Win rate 52-56%, ROI +1-8%, match rate 30-50%

---

**Last Updated:** October 20, 2025
**Model Version:** PHASE1
**Status:** Results Require Investigation ⚠️
**Next Milestone:** Full walk-forward validation → Credible betting results
