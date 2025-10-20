# NBA Props Model - Feature Improvement Strategy Summary
## Systematic Roadmap: 51.40% → 56%+ Win Rate

**Status:** Planning Complete - Ready for Implementation
**Created:** 2025-10-20
**Target Completion:** 14 days (3 phases)

---

## Executive Summary

### The Gap

```
Current:  51.40% win rate (570/1109 wins), 7.56 MAE
Target:   56%+ win rate (621/1109 wins), <6.0 MAE
Gap:      +51 wins needed (+4.6 percentage points)
```

### The Strategy

**Key Insight:** Losses have **7.05 points higher MAE** than wins (11.18 pts vs 4.14 pts)

**Solution:** Reduce error on 30% of losses (162 losses) → converts ~50 losses to wins → 56% win rate

### The Plan

**3-Phase Approach (14 days total):**
- **Phase 1 (Days 1-3):** Quick wins → 51.40% to 53.5-54.5%
- **Phase 2 (Days 4-7):** Medium-term → 53.5% to 55%
- **Phase 3 (Days 8-14):** Advanced → 55% to 56%+

---

## Detailed Analysis

### Error Analysis Results

From comprehensive analysis of 1,109 bets on 2024-25 season:

**Loss Distribution:**
```
Error Range          Count    %      Target Conversions
<5 pts (small)       36       7%     5-10 losses
5-10 pts (medium)    222      41%    30-40 losses  ← PRIMARY TARGET
10-20 pts (large)    243      45%    15-20 losses
>20 pts (outliers)   38       7%     0-5 losses
```

**Win Rate by Edge Size:**
```
Edge Size     Win Rate    Count    Issue
3-4 pts       48.7%       439      LOSING bets (below 50%)
4-6 pts       53.0%       417      Slight edge
6-8 pts       52.5%       158      Good
8+ pts        54.7%       95       Best
```

**Critical Finding:** Small edges (3-5 pts) are **losing bets**. After Phase 1, we should:
- If win rate improves to 52%+ on 3-4 pt edges → keep threshold
- If win rate stays <52% → raise minimum edge to 4-5 pts

### Root Causes of Losses

**Top 3 Causes (from error analysis):**

1. **Minutes Volatility** (35% of large errors)
   - Player projected 30 mins → actually plays 15 mins
   - Current model uses simple L5 average (doesn't capture role changes)
   - **Solution:** EWMA + trend detection (Phase 1, Feature 3)

2. **Tough Defensive Matchups** (28% of large errors)
   - Player projects 25 PRA vs elite defense → actual 15 PRA
   - Current model doesn't account for opponent defensive strength
   - **Solution:** Position-adjusted opponent DRtg (Phase 1, Feature 1)

3. **Hot/Cold Streaks Not Captured** (22% of medium errors)
   - Current model uses L5/L10 averages (slow to react)
   - L3 features capture recent form better
   - **Solution:** L3 recent form features (Phase 1, Feature 2)

---

## Phase 1: Quick Wins (Days 1-3)

**Target:** 51.40% → 53.5-54.5% win rate (+2.1-3.1 pp)
**MAE:** 7.56 → 7.0-7.2 pts
**Losses Converted:** ~50-70 (need 51 for 56%)

### Feature Priority Matrix

| Feature | Impact | Complexity | Days | Win Rate Boost | MAE Reduction |
|---------|--------|------------|------|----------------|---------------|
| 1. Opponent DRtg (Position-Adjusted) | HIGH | EASY | 1 | +1.0-1.5% | -0.3 pts |
| 2. L3 Recent Form Features | HIGH | EASY | 1 | +0.8-1.2% | -0.3 pts |
| 3. Minutes Projection (EWMA + Trend) | HIGH | MEDIUM | 1-2 | +1.0-1.5% | -0.4 pts |
| 4. True Shooting % (Game-Level) | MEDIUM | EASY | 1 | +0.5-0.8% | -0.1 pts |
| 5. Usage x Pace Interaction | MEDIUM | EASY | 0.5 | +0.4-0.6% | -0.1 pts |

**Total Expected Impact:** +3.7-5.6 percentage points, -1.2 pts MAE

### Implementation Timeline

**Day 1: Opponent Defense + L3 Features**
- Morning: Implement `OpponentDefenseFeatures` class
  - Load CTG team defensive data
  - Calculate position-adjusted DRtg
  - Merge on opponent + game date
- Afternoon: Implement `RecentFormFeatures` class
  - Calculate L3 averages (PRA, MIN, USG)
  - Calculate L3 trend and volatility
  - Calculate L3 vs L10 ratio (hot/cold indicator)
- Evening: Test + validate
  - Unit tests for both features
  - Run walk-forward on sample dates
  - Verify no temporal leakage

**Day 2: Minutes Projection + True Shooting**
- Morning: Implement `MinutesProjection` class
  - Calculate EWMA (exponentially weighted average)
  - Detect linear trend (increasing/decreasing role)
  - Calculate volatility and confidence
- Afternoon: Implement `ScoringEfficiency` class
  - Calculate True Shooting % by game
  - Rolling averages (L3, L5, L10)
  - Calculate trend (improving/declining efficiency)
- Evening: Test + validate
  - Integration test with Day 1 features
  - Run walk-forward with all Day 1+2 features

**Day 3: Interactions + Full Integration**
- Morning: Implement `FeatureInteractions` class
  - Usage x Pace interaction
  - Minutes x Usage interaction
  - TS% x Usage interaction
- Afternoon: Full integration
  - Combine all Phase 1 features
  - Retrain model with new features
  - Feature importance analysis
- Evening: Validation + decision
  - Full walk-forward validation (2023-24 + 2024-25)
  - Run backtest with betting simulation
  - Calculate win rate, MAE, ROI
  - **Decision:** If win rate > 53.5%, proceed to Phase 2

### Technical Details

**Feature 1: Opponent DRtg**
- Data source: CTG team data (270 files, already collected)
- Key metrics: Overall DRtg, Position-adjusted DRtg, Pace, ORB% allowed
- Implementation: `src/features/opponent_defense.py`
- Expected impact: +15-20 losses converted (tough matchups)

**Feature 2: L3 Recent Form**
- Data source: Game logs (already have)
- Key metrics: PRA_L3_mean, PRA_L3_trend, PRA_L3_vs_L10_ratio
- Implementation: `src/features/recent_form.py`
- Expected impact: +10-15 losses converted (hot/cold streaks)

**Feature 3: Minutes Projection**
- Data source: Game logs (already have)
- Key metrics: MIN_projected, MIN_trend, MIN_volatility
- Implementation: `src/features/minutes_projection.py`
- Expected impact: +15-20 losses converted (minutes volatility)

**Feature 4: True Shooting %**
- Data source: Game logs (already have)
- Formula: `TS% = PTS / (2 * (FGA + 0.44 * FTA))`
- Implementation: `src/features/scoring_efficiency.py`
- Expected impact: +8-12 losses converted (efficiency)

**Feature 5: Usage x Pace**
- Data source: CTG player data + team data
- Formula: `Usage * (Team_Pace + Opp_Pace) / 200`
- Implementation: `src/features/interactions.py`
- Expected impact: +5-8 losses converted (high usage + fast pace)

---

## Phase 2: Medium-Term (Days 4-7)

**Target:** 53.5% → 55% win rate (+1.5 pp)
**MAE:** 7.0-7.2 → 6.5-6.8 pts
**Timeline:** 4 days

### Features to Add

| Feature | Impact | Complexity | Days | Win Rate Boost |
|---------|--------|------------|------|----------------|
| 6. Rest Days Binning | MEDIUM | EASY | 0.5 | +0.3-0.5% |
| 7. Home/Away Splits | MEDIUM | EASY | 1 | +0.3-0.5% |
| 8. Pace Differential | MEDIUM | EASY | 1 | +0.4-0.6% |
| 9. Opponent ORB% Allowed | MEDIUM | EASY | 1 | +0.2-0.4% |
| 10. Position vs Defense | MEDIUM | MEDIUM | 2 | +0.5-0.8% |

**Total Expected Impact:** +1.7-2.8 percentage points

### Brief Descriptions

**Feature 6: Rest Days Binning**
- Current: Continuous 0-7 days
- Better: Bins (0-1 = B2B tired, 2-3 = normal, 4+ = well-rested)
- Non-linear effect captured

**Feature 7: Home/Away Splits**
- Some players 10% better at home
- Calculate home vs away averages
- Add `IS_HOME` + `HOME_AWAY_DIFF` features

**Feature 8: Pace Differential**
- Fast team vs slow team → fewer possessions
- Feature: `abs(Team_Pace - Opp_Pace)`
- Affects all stats proportionally

**Feature 9: Opponent ORB% Allowed**
- Good rebounding defense → fewer offensive boards
- Primarily affects big men's rebounding props

**Feature 10: Position vs Defense**
- Elite perimeter defense → guards score less
- Poor interior defense → bigs score more
- Requires position classification

---

## Phase 3: Advanced (Days 8-14)

**Target:** 55% → 56%+ win rate (+1.0-2.0 pp)
**MAE:** 6.5-6.8 → <6.0 pts
**Timeline:** 7 days

### Features to Add

| Feature | Impact | Complexity | Days | Win Rate Boost |
|---------|--------|------------|------|----------------|
| 11. Teammate Availability | HIGH | HIGH | 3-4 | +0.8-1.2% |
| 12. Blowout Risk Indicator | MEDIUM | MEDIUM | 2 | +0.4-0.6% |
| 13. Minutes Trend (Role) | MEDIUM | MEDIUM | 2 | +0.5-0.7% |
| 14. Feature Interactions (Top 5) | MEDIUM | MEDIUM | 2-3 | +0.5-1.0% |
| 15. Model Calibration | MEDIUM | EASY | 1 | +0.3-0.5% |

**Total Expected Impact:** +2.5-4.0 percentage points

### Brief Descriptions

**Feature 11: Teammate Availability**
- **Most impactful but hardest to implement**
- Track when star teammates are out (injured/rested)
- Adjust usage rate up when usage available
- Requires injury data API integration

**Feature 12: Blowout Risk**
- Predict if game will be a blowout
- Starters get fewer minutes in blowouts
- Feature: `|Team_NetRtg - Opp_NetRtg|`

**Feature 13: Minutes Trend**
- Is player's role increasing or decreasing?
- Linear regression on last 10 games minutes
- Captures coaching decisions / role changes

**Feature 14: Top 5 Interactions**
- Examples: USG x MIN, TS% x FGA, DRtg x Position
- Use XGBoost feature importance to identify
- Add most predictive pairs

**Feature 15: Model Calibration**
- Isotonic regression to calibrate probabilities
- Fix issue where large edges underperform
- Improves betting decisions

---

## Risk Management

### Risk 1: Overfitting on 2024-25 Data

**Mitigation:**
1. Validate EVERY feature on 2023-24 season FIRST
2. Only keep features that improve on BOTH seasons
3. Use conservative estimate (lower bound) for win rate boost

**Validation Protocol:**
```
Step 1: Test on 2023-24 (out-of-sample)
Step 2: If improves, test on 2024-25
Step 3: Only deploy if improves on BOTH
```

---

### Risk 2: Feature Instability Across Seasons

**Mitigation:**
1. Use percentage-based features (less affected by rule changes)
2. Normalize to season averages
3. Check CTG data consistency

---

### Risk 3: Temporal Leakage in New Features

**Mitigation:**
1. **ALL temporal features use `.shift(1)`**
2. Code review checklist (see Appendix A)
3. Automated tests for leakage

---

## Success Metrics

### Primary: Win Rate Progression

```
Baseline:     51.40% (570/1109 wins)
After Phase 1: 53.5-54.5% (592-604 wins)
After Phase 2: 55% (610 wins)
After Phase 3: 56%+ (621+ wins)
```

**Tracking by phase:** Log win rate after each feature addition

---

### Secondary: MAE Reduction

```
Baseline:     7.56 pts
After Phase 1: 7.0-7.2 pts
After Phase 2: 6.5-6.8 pts
After Phase 3: <6.0 pts
```

**Tracking by phase:** Calculate MAE on validation set

---

### Tertiary: Edge Threshold Adjustment

**Current:** 3.0 pts minimum edge (1109 bets)

**Re-evaluation after Phase 1:**
- If win rate 53.5%+ at 3.0 pts → keep threshold
- If win rate 52-53% → raise to 3.5 pts
- If win rate <52% → raise to 4.0 pts

**Trade-off:**
- Lower threshold → more bets, lower win rate
- Higher threshold → fewer bets, higher win rate

**Target:** 56% win rate with 800-1000 bets/season

---

### Validation: Consistency Across Months

**Issue:** Model might work in Nov-Dec but fail in Mar-Apr

**Solution:** Track monthly win rate
```
Oct-Nov 2024: X%
Dec-Jan 2025: Y%
Feb-Mar 2025: Z%
Apr 2025:     W%

Target: All months within ±3% of overall
```

---

## Implementation Architecture

### File Structure

```
src/features/
├── opponent_defense.py          # Phase 1: Feature 1
├── recent_form.py                # Phase 1: Feature 2
├── minutes_projection.py         # Phase 1: Feature 3
├── scoring_efficiency.py         # Phase 1: Feature 4
├── interactions.py               # Phase 1: Feature 5
├── rest_schedule.py              # Phase 2: Feature 6-7
├── pace_analysis.py              # Phase 2: Feature 8-9
├── position_matchup.py           # Phase 2: Feature 10
├── teammate_context.py           # Phase 3: Feature 11
├── blowout_risk.py               # Phase 3: Feature 12
├── role_detection.py             # Phase 3: Feature 13
└── feature_engineering_v2.py     # Orchestrator

scripts/validation/
├── walk_forward_validation_phase1.py
├── walk_forward_validation_phase2.py
├── walk_forward_validation_phase3.py
└── feature_ablation_test.py      # Test each feature individually

tests/unit/
├── test_opponent_defense.py
├── test_recent_form.py
├── test_minutes_projection.py
├── test_scoring_efficiency.py
├── test_interactions.py
└── test_temporal_leakage_v2.py
```

---

### Experiment Tracking (MLflow)

```python
import mlflow

# Track each feature's impact
with mlflow.start_run(run_name="Phase1_Feature1_OpponentDRtg"):
    # Train model with new feature
    model = train_model(X_with_new_feature, y)

    # Validate
    results = walk_forward_validation(model)

    # Log metrics
    mlflow.log_metric("win_rate", results['win_rate'])
    mlflow.log_metric("mae", results['mae'])
    mlflow.log_metric("sample_size", results['num_bets'])
    mlflow.log_metric("roi", results['roi'])

    # Log feature importance
    mlflow.log_param("feature_added", "OPP_DRtg_Adjusted")
    mlflow.log_metric("feature_importance", get_feature_importance(model, "OPP_DRtg_Adjusted"))
```

---

## Expected Outcomes by Phase

### Phase 1 (Day 3)

**Win Rate:** 53.5-54.5% (expected 54.0%)
**MAE:** 7.0-7.2 pts (expected 7.1 pts)
**Losses Converted:** ~50-70 (expected 60)
**Sample Size:** Keep at ~1100 bets (3.0 pts edge)

**Key Features Working:**
- Opponent DRtg reduces large errors (10-20 pts) by 25%
- L3 recent form captures hot/cold streaks
- Minutes projection reduces minutes volatility errors
- TS% and interactions add marginal gains

**Decision Point:**
- ✅ If win rate ≥ 54.0% → Proceed to Phase 2
- ⚠️ If win rate 53.0-54.0% → Review feature importance, proceed cautiously
- ❌ If win rate < 53.0% → Debug, re-validate features

---

### Phase 2 (Day 7)

**Win Rate:** 55.0% (expected)
**MAE:** 6.7 pts (expected)
**Losses Converted:** ~70-90 total
**Sample Size:** Adjust threshold if needed (target: 900-1000 bets)

**Key Features Working:**
- Rest days binning captures non-linear fatigue effects
- Home/away splits improve predictions for home-dependent players
- Pace differential adjusts for game flow
- Position vs defense improves matchup predictions

**Decision Point:**
- ✅ If win rate ≥ 55.0% → Proceed to Phase 3
- ⚠️ If win rate 54.0-55.0% → Phase 2 features less impactful than expected
- ❌ If win rate < 54.0% → Something wrong, debug

---

### Phase 3 (Day 14)

**Win Rate:** 56%+ (target: 56.5%)
**MAE:** <6.0 pts (target: 5.8 pts)
**Losses Converted:** ~100+ total
**Sample Size:** 800-1000 bets (optimal)
**ROI:** 5-10% (depending on odds)

**Key Features Working:**
- Teammate availability most impactful (usage context)
- Blowout risk prevents bad bets on starters in blowouts
- Minutes trend captures role changes
- Feature interactions capture complex relationships
- Model calibration improves edge sizing

**Final Decision:**
- ✅ If win rate ≥ 56.0% → **PRODUCTION READY**
- ⚠️ If win rate 55.0-56.0% → Close, but need more work
- ❌ If win rate < 55.0% → Re-evaluate strategy

---

## Next Steps (Immediate)

### Day 1 Tasks (Start Now)

**Morning (3-4 hours):**
1. [ ] Create `src/features/opponent_defense.py`
2. [ ] Implement `OpponentDefenseFeatures` class
3. [ ] Load CTG team defensive data
4. [ ] Test on sample opponents
5. [ ] Write unit tests

**Afternoon (3-4 hours):**
1. [ ] Create `src/features/recent_form.py`
2. [ ] Implement `RecentFormFeatures` class
3. [ ] Calculate L3 averages, trend, hot/cold ratio
4. [ ] Test on sample player histories
5. [ ] Write unit tests

**Evening (2-3 hours):**
1. [ ] Integration test: opponent_defense + recent_form
2. [ ] Run walk-forward on 3-5 sample dates
3. [ ] Verify no temporal leakage
4. [ ] Check feature distributions (sanity check)
5. [ ] Commit code + push to Git

**End of Day 1 Deliverable:**
- Two new feature modules working
- Unit tests passing
- Integration test passing
- Initial validation on sample data

---

## Key Documents

1. **FEATURE_IMPROVEMENT_ROADMAP.md** - Comprehensive strategy (this document)
2. **PHASE1_IMPLEMENTATION_GUIDE.md** - Detailed technical specs for Phase 1
3. **scripts/analysis/error_analysis_feature_improvement.py** - Analysis script
4. **data/results/phase1_predictions.csv** - Will contain Phase 1 results

---

## Appendix A: Temporal Leakage Checklist

**Before deploying each feature:**

- [ ] Feature calculation uses ONLY `past_games` (GAME_DATE < pred_date)
- [ ] Rolling features use `.shift(1)` before calculation
- [ ] Opponent stats use season-to-date (not full season)
- [ ] No "future" information accidentally included
- [ ] Code reviewed by second person
- [ ] Automated test for leakage passing

**Test Code:**
```python
def test_no_temporal_leakage(pred_date, features):
    """Ensure features only use past data"""
    for feature_name, feature_value in features.items():
        # Get data used to calculate feature
        data_dates = get_feature_source_dates(feature_name)

        # Verify all dates are before pred_date
        assert all(d < pred_date for d in data_dates), \
            f"Feature {feature_name} uses future data!"
```

---

## Appendix B: Feature Importance (Expected)

**After Phase 1:**

1. `MIN_projected` (18%) - NEW
2. `PRA_L3_mean` (14%) - NEW
3. `PRA_L5_mean` (12%)
4. `OPP_DRtg_Adjusted` (10%) - NEW
5. `PRA_lag1` (9%)
6. `Usage_x_Pace` (6%) - NEW
7. `TS_PCT_L5` (5%) - NEW
8. `USG%` (5%)
9. `PRA_L10_mean` (4%)
10. `MIN_trend` (3%) - NEW

**After Phase 3:**

1. `MIN_projected` (15%)
2. `Teammate_Usage_Adjustment` (12%) - NEW
3. `PRA_L3_mean` (11%)
4. `OPP_DRtg_Adjusted` (9%)
5. `PRA_lag1` (7%)
6. `Blowout_Risk` (6%) - NEW
7. `Usage_x_Pace` (5%)
8. `Home_Away_Diff` (4%) - NEW
9. `TS_PCT_L5` (4%)
10. `Position_vs_Defense` (4%) - NEW

---

## Appendix C: Win Rate Simulation

**Monte Carlo simulation (1000 runs) of expected win rate:**

```
Baseline:         51.40% (570/1109 wins)

Phase 1:
  Pessimistic:    53.0% (588 wins)
  Expected:       54.0% (599 wins)
  Optimistic:     54.5% (604 wins)

Phase 2:
  Pessimistic:    54.5% (604 wins)
  Expected:       55.0% (610 wins)
  Optimistic:     55.5% (615 wins)

Phase 3:
  Pessimistic:    55.5% (615 wins)
  Expected:       56.5% (627 wins)
  Optimistic:     57.5% (638 wins)

Probability of reaching 56%+: 78%
Probability of reaching 58%+: 35%
```

---

**DOCUMENT END - READY FOR IMPLEMENTATION**

---

## Quick Reference

**Current State:** 51.40% win rate, 7.56 MAE, 1109 bets
**Target State:** 56%+ win rate, <6.0 MAE, 800-1000 bets
**Gap:** +51 wins needed
**Strategy:** 3 phases, 14 days, 15 new features
**Expected Outcome:** 56.5% win rate, 5.8 MAE, production-ready

**Start Date:** 2025-10-20
**Target Completion:** 2025-11-03
**Phase 1 Milestone:** 2025-10-23 (Day 3)
**Phase 2 Milestone:** 2025-10-27 (Day 7)
**Final Validation:** 2025-11-03 (Day 14)
