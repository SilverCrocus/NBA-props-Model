# NBA PROPS MODEL - FEATURE IMPROVEMENT ROADMAP
## Target: 51.40% → 56%+ Win Rate

**Document Status:** Planning Phase
**Current Performance:** 51.40% win rate (570/1109 bets), 7.56 MAE
**Target Performance:** 56%+ win rate (621/1109 bets), <6.0 MAE
**Gap to Close:** +51 wins needed

---

## Executive Summary

### Current State Analysis

```
Total Bets:    1,109
Wins:          570 (51.40%)
Losses:        539 (48.60%)
MAE:           7.56 points
CTG Coverage:  ~93%

Target Win Rate:   56% (621 wins)
Gap:               +51 wins needed
Win Rate Boost:    +4.6 percentage points
```

### Key Finding: Error Magnitude Gap

**Critical Insight:** Losses have **7.05 points higher error** than wins

```
Wins:   4.14 pts MAE  (70% accuracy within ±5 pts)
Losses: 11.18 pts MAE (only 7% accuracy within ±5 pts)
```

**Implication:** If we can reduce error on just 30% of losses (162 losses) from 11.18 → 7.0 pts, we convert ~50 losses to wins → 56% win rate.

### Loss Pattern Analysis

**Error Distribution:**
- Small error (<5 pts): 36 losses (7%) - Already close, need calibration
- Medium error (5-10 pts): 222 losses (41%) - **TARGET: Convert 30-40 of these**
- Large error (10-20 pts): 243 losses (45%) - **TARGET: Convert 15-20 of these**
- Very large error (>20 pts): 38 losses (7%) - Outliers, hard to fix

**Bet Type Analysis:**
- OVER losses: 240 (44.5%)
- UNDER losses: 299 (55.5%)
- **Issue:** Slight bias towards UNDER losses (need better upside capture)

**Edge Size Analysis:**
- Small edge (3-5): 345 losses (64%) - Model is uncertain but bets anyway
- Medium edge (5-7): 119 losses (22%)
- Large edge (7-10): 57 losses (11%)
- Very large edge (>10): 18 losses (3%) - Model very confident but wrong

**Win Rate by Edge:**
```
3-4 pts edge:  48.7% win rate (below 50%!)
4-6 pts edge:  53.0% win rate
6-8 pts edge:  52.5% win rate
8+ pts edge:   54.7% win rate
```

**Critical Issue:** Small edges (3-5 pts) are **losing bets** (48.7% < 50%). We should either:
1. Improve features to reduce error on small edges
2. Raise betting threshold to 5+ pts edge

---

## Feature Priority Matrix

| Feature | Impact | Complexity | Priority | Timeline | Expected Boost |
|---------|--------|------------|----------|----------|----------------|
| **Phase 1: Quick Wins (Days 1-3)** |
| Opponent DRtg (Position-Specific) | HIGH | EASY | P0 | 1 day | +1.0-1.5% |
| L3 Recent Form Features | HIGH | EASY | P0 | 1 day | +0.8-1.2% |
| Improved Minutes Projection (EWMA) | HIGH | MEDIUM | P0 | 1-2 days | +1.0-1.5% |
| True Shooting % (Game-Level) | MEDIUM | EASY | P1 | 1 day | +0.5-0.8% |
| Usage x Pace Interaction | MEDIUM | EASY | P1 | 0.5 day | +0.4-0.6% |
| **Phase 2: Medium-Term (Days 4-7)** |
| Rest Days Binning (0-1, 2-3, 4+) | MEDIUM | EASY | P1 | 0.5 day | +0.3-0.5% |
| Home/Away Splits | MEDIUM | EASY | P1 | 1 day | +0.3-0.5% |
| Pace Differential (Team vs Opp) | MEDIUM | EASY | P1 | 1 day | +0.4-0.6% |
| Opponent ORB% Allowed (Rebounding) | MEDIUM | EASY | P2 | 1 day | +0.2-0.4% |
| Position vs Defense Matchup | MEDIUM | MEDIUM | P2 | 2 days | +0.5-0.8% |
| **Phase 3: Advanced (Days 8-14)** |
| Teammate Availability Context | HIGH | HIGH | P3 | 3-4 days | +0.8-1.2% |
| Blowout Risk Indicator | MEDIUM | MEDIUM | P3 | 2 days | +0.4-0.6% |
| Minutes Trend (Role Changes) | MEDIUM | MEDIUM | P3 | 2 days | +0.5-0.7% |
| Feature Interactions (Top 5) | MEDIUM | MEDIUM | P3 | 2-3 days | +0.5-1.0% |
| Model Calibration (Isotonic) | MEDIUM | EASY | P3 | 1 day | +0.3-0.5% |

**Total Expected Impact:** +5.0-9.0 percentage points → **56-60% win rate**

---

## Phase 1: Quick Wins (Days 1-3)

**Target:** 51.40% → 53.5-54.5% win rate
**Timeline:** 3 days
**Expected Boost:** +2.1-3.1 percentage points

### Feature 1: Opponent Defensive Rating (Position-Specific)

**Why This Helps:**
- Large errors (10-20 pts) often from tough defensive matchups
- Example: Player projects 25 PRA vs elite defense → actual 15 PRA (10 pt error)
- CTG has position-specific defensive data (how teams defend guards vs bigs)

**Data Source:** CTG team data (already collected, 270 files)

**Implementation:**
```python
# File: src/features/opponent_defense.py

def get_opponent_defensive_features(opponent, player_position, season):
    """
    Get opponent defensive rating adjusted for player position

    Args:
        opponent: Team abbreviation (e.g., 'LAL')
        player_position: Guard/Wing/Big
        season: '2024-25'

    Returns:
        dict with opponent defensive features
    """
    ctg_team_data = load_ctg_team_data(opponent, season)

    features = {
        'OPP_DRtg': ctg_team_data['DRtg'],  # Overall defensive rating
        'OPP_Pace': ctg_team_data['Pace'],   # Possessions per 48 min

        # Position-specific (if available)
        'OPP_DRtg_vs_Guards': ctg_team_data.get('DRtg_vs_G', ctg_team_data['DRtg']),
        'OPP_DRtg_vs_Wings': ctg_team_data.get('DRtg_vs_W', ctg_team_data['DRtg']),
        'OPP_DRtg_vs_Bigs': ctg_team_data.get('DRtg_vs_B', ctg_team_data['DRtg']),

        # Rebounding defense
        'OPP_ORB_PCT_Allowed': ctg_team_data.get('ORB_PCT_Allowed', 25.0),
        'OPP_DRB_PCT': ctg_team_data.get('DRB_PCT', 75.0)
    }

    # Select relevant DRtg based on position
    if player_position == 'Guard':
        features['OPP_DRtg_Adjusted'] = features['OPP_DRtg_vs_Guards']
    elif player_position == 'Wing':
        features['OPP_DRtg_Adjusted'] = features['OPP_DRtg_vs_Wings']
    else:
        features['OPP_DRtg_Adjusted'] = features['OPP_DRtg_vs_Bigs']

    return features
```

**Integration Point:**
- Add to `game_log_builder.py` after CTG merge
- Merge on `OPPONENT` and `GAME_DATE` (season)

**Expected Impact:**
- **Win rate boost:** +1.0-1.5%
- **MAE reduction:** 7.56 → 7.2 pts
- **Losses converted:** ~15-20 (medium error 5-10 pts range)

**Validation:**
- Re-run walk-forward on 2024-25
- Compare win rate before/after
- Analyze error reduction on tough matchups

---

### Feature 2: L3 Recent Form Features

**Why This Helps:**
- Current model uses L5, L10, L20 averages
- L3 captures **hot/cold streaks** better (more responsive)
- Research shows 3-game window is optimal for short-term form

**Data Source:** Already in `game_logs` (just need to calculate)

**Implementation:**
```python
# File: src/features/calculator.py

def calculate_l3_features(player_history):
    """
    Calculate last 3 games features (strongest temporal signal)

    Uses .shift(1) to prevent temporal leakage
    """
    if len(player_history) < 3:
        return {
            'PRA_L3_mean': 0,
            'PRA_L3_std': 0,
            'PRA_L3_trend': 0,
            'MIN_L3_mean': 0,
            'USG_L3_mean': 0,
            'PRA_L3_vs_L10_ratio': 1.0  # Hot/cold indicator
        }

    # Get last 3 games (already shifted by 1 in game_log_builder)
    last_3 = player_history.sort_values('GAME_DATE', ascending=False).iloc[:3]
    last_10 = player_history.sort_values('GAME_DATE', ascending=False).iloc[:10]

    features = {
        'PRA_L3_mean': last_3['PRA'].mean(),
        'PRA_L3_std': last_3['PRA'].std() if len(last_3) > 1 else 0,
        'PRA_L3_trend': (last_3.iloc[0]['PRA'] - last_3.iloc[2]['PRA']) / 2,
        'MIN_L3_mean': last_3['MIN'].mean(),
        'USG_L3_mean': last_3['Usage'].mean() if 'Usage' in last_3.columns else 0,

        # Hot/cold indicator: L3 vs L10 ratio
        'PRA_L3_vs_L10_ratio': (
            last_3['PRA'].mean() / last_10['PRA'].mean()
            if len(last_10) >= 10 and last_10['PRA'].mean() > 0
            else 1.0
        )
    }

    return features
```

**New Features:**
1. `PRA_L3_mean` - Last 3 games average (hot/cold)
2. `PRA_L3_std` - Volatility in last 3 games
3. `PRA_L3_trend` - Is player trending up or down?
4. `MIN_L3_mean` - Minutes trend
5. `USG_L3_mean` - Usage rate trend
6. `PRA_L3_vs_L10_ratio` - Hot indicator (>1.1 = hot, <0.9 = cold)

**Expected Impact:**
- **Win rate boost:** +0.8-1.2%
- **MAE reduction:** 7.56 → 7.3 pts
- **Losses converted:** ~10-15

---

### Feature 3: Improved Minutes Projection (EWMA)

**Why This Helps:**
- **#1 cause of large errors:** Player gets 15 mins instead of projected 30 mins
- Current implementation uses simple average (L5 minutes)
- EWMA gives more weight to recent games (better captures role changes)

**Issue with Current Approach:**
```python
# Current: Equal weight to all L5 games
MIN_L5_mean = [32, 30, 28, 26, 24] → 28 mins
# Problem: Doesn't capture decreasing role
```

**Improved Approach:**
```python
# EWMA: More weight to recent games
MIN_ewma5 = EWMA([32, 30, 28, 26, 24], span=5) → 25.8 mins
# Captures downward trend better
```

**Implementation:**
```python
# File: src/features/minutes_projection.py

def calculate_minutes_features(player_history, span=5):
    """
    Advanced minutes projection using EWMA + trend detection
    """
    if len(player_history) < 3:
        return {'MIN_projected': 0, 'MIN_trend': 0, 'MIN_volatility': 0}

    # Sort by date (most recent first)
    history = player_history.sort_values('GAME_DATE', ascending=False)

    # EWMA (recent games weighted higher)
    min_ewma = history['MIN'].ewm(span=span, min_periods=1).mean().iloc[0]

    # Trend detection (last 5 games linear regression)
    if len(history) >= 5:
        last_5_mins = history.iloc[:5]['MIN'].values
        x = np.arange(5)
        slope = np.polyfit(x, last_5_mins, 1)[0]
        min_trend = slope  # Positive = increasing role, negative = decreasing
    else:
        min_trend = 0

    # Volatility (standard deviation)
    min_volatility = history.iloc[:10]['MIN'].std() if len(history) >= 10 else 0

    # Adjust projection based on trend
    min_projected = min_ewma + (min_trend * 0.5)  # Adjust for trend
    min_projected = np.clip(min_projected, 0, 48)  # Cap at 48 mins

    return {
        'MIN_projected': min_projected,
        'MIN_trend': min_trend,
        'MIN_volatility': min_volatility,
        'MIN_ewma5': min_ewma
    }
```

**Expected Impact:**
- **Win rate boost:** +1.0-1.5%
- **MAE reduction:** 7.56 → 7.1 pts
- **Losses converted:** ~15-20 (especially large error cases)

---

### Feature 4: True Shooting % (Game-Level)

**Why This Helps:**
- Current model uses FG% and FT% separately
- TS% combines both into single efficiency metric (better for scoring prediction)
- Formula: `TS% = PTS / (2 * (FGA + 0.44 * FTA))`

**Implementation:**
```python
# File: src/features/calculator.py

def calculate_true_shooting_pct(player_history):
    """
    Calculate True Shooting % (rolling average)
    """
    if len(player_history) == 0:
        return 0

    history = player_history.sort_values('GAME_DATE', ascending=False)

    # Calculate TS% for each game
    history['TS_PCT'] = history['PTS'] / (2 * (history['FGA'] + 0.44 * history['FTA']))
    history['TS_PCT'] = history['TS_PCT'].replace([np.inf, -np.inf], np.nan).fillna(0)

    # Rolling averages
    ts_pct_features = {
        'TS_PCT_L5': history.iloc[:5]['TS_PCT'].mean() if len(history) >= 5 else 0,
        'TS_PCT_L10': history.iloc[:10]['TS_PCT'].mean() if len(history) >= 10 else 0,
        'TS_PCT_ewma5': history['TS_PCT'].ewm(span=5, min_periods=1).mean().iloc[0]
    }

    return ts_pct_features
```

**Expected Impact:**
- **Win rate boost:** +0.5-0.8%
- **MAE reduction:** 7.56 → 7.4 pts

---

### Feature 5: Usage x Pace Interaction

**Why This Helps:**
- High usage players benefit MORE from fast pace
- Example: 30% usage in 95 pace game → 28.5 PRA | 30% usage in 105 pace game → 31.5 PRA
- Interaction captures this multiplicative effect

**Implementation:**
```python
# File: src/features/interactions.py

def calculate_usage_pace_interaction(player_usage, team_pace, opp_pace):
    """
    High usage players benefit more from fast pace
    """
    avg_pace = (team_pace + opp_pace) / 2

    # Normalize to league average pace (100)
    pace_factor = avg_pace / 100.0

    # Interaction: Usage * Pace adjustment
    usage_pace_interaction = player_usage * pace_factor

    return {
        'Usage_x_Pace': usage_pace_interaction,
        'Avg_Game_Pace': avg_pace,
        'Pace_Factor': pace_factor
    }
```

**Expected Impact:**
- **Win rate boost:** +0.4-0.6%
- **MAE reduction:** 7.56 → 7.5 pts

---

## Phase 1 Summary

**Total Expected Impact:**
- Win rate: 51.40% → 53.5-54.5%
- MAE: 7.56 → 7.0-7.2 pts
- Losses converted: ~50-70 (need 51 for 56%)

**Implementation Plan:**
```
Day 1: Opponent DRtg + L3 Features
  - Morning: Implement opponent defense features
  - Afternoon: Add L3 recent form features
  - Evening: Run walk-forward validation

Day 2: Minutes Projection + TS%
  - Morning: Implement improved minutes projection
  - Afternoon: Add True Shooting %
  - Evening: Run walk-forward validation

Day 3: Usage x Pace + Integration
  - Morning: Add usage x pace interaction
  - Afternoon: Integrate all Phase 1 features
  - Evening: Full walk-forward validation + backtest
```

---

## Phase 2: Medium-Term (Days 4-7)

**Target:** 53.5% → 55% win rate
**Timeline:** 4 days
**Expected Boost:** +1.5 percentage points

### Features to Add:

1. **Rest Days Binning** (0.5 day)
   - Current: Continuous variable (0-7 days)
   - Better: Bins (0-1 = B2B, 2-3 = normal, 4+ = well-rested)
   - Non-linear effect: 0→1 day big impact, 3→4 days small impact

2. **Home/Away Splits** (1 day)
   - Some players perform 10% better at home
   - Calculate home/away PRA averages
   - Add `IS_HOME` indicator + `HOME_AWAY_DIFF`

3. **Pace Differential** (1 day)
   - Fast team vs slow team → fewer possessions
   - Slow team vs fast team → more possessions
   - Feature: `abs(Team_Pace - Opp_Pace)`

4. **Opponent ORB% Allowed** (1 day)
   - Good rebounding defense → fewer offensive boards
   - Affects big men's rebounding props

5. **Position vs Defense Matchup** (2 days)
   - Elite perimeter defense → guards score less
   - Poor interior defense → bigs score more

---

## Phase 3: Advanced (Days 8-14)

**Target:** 55% → 56%+ win rate
**Timeline:** 7 days
**Expected Boost:** +1.0-2.0 percentage points

### Features to Add:

1. **Teammate Availability Context** (3-4 days)
   - Track when star teammates are out
   - Adjust usage rate up when usage available
   - Most impactful feature (but hardest to implement)

2. **Blowout Risk Indicator** (2 days)
   - Predict if game will be a blowout (based on team strength diff)
   - Starters get fewer minutes in blowouts
   - Feature: `|Team_NetRtg - Opp_NetRtg|`

3. **Minutes Trend** (2 days)
   - Is player's role increasing or decreasing?
   - Linear regression on last 10 games minutes
   - Feature: `MIN_slope_L10`

4. **Feature Interactions** (2-3 days)
   - Top 5 most predictive feature pairs
   - Examples: USG x MIN, TS% x FGA, etc.

5. **Model Calibration** (1 day)
   - Isotonic regression to calibrate probabilities
   - Fix issue where large edges underperform

---

## Risk Assessment & Mitigation

### Risk 1: Overfitting on 2024-25 Data

**Risk Level:** HIGH

**Mitigation:**
1. Validate each feature on 2023-24 season FIRST
2. If feature works on 2023-24, then test on 2024-25
3. Only keep features that work on BOTH seasons

**Validation Strategy:**
```python
# Step 1: Test on 2023-24 (out-of-sample)
results_2023_24 = walk_forward_validation(season='2023-24', features=new_features)

# Step 2: If improves, test on 2024-25
if results_2023_24['win_rate'] > baseline:
    results_2024_25 = walk_forward_validation(season='2024-25', features=new_features)

# Step 3: Only keep if improves on BOTH
if results_2024_25['win_rate'] > baseline:
    # Feature is valid
    approved_features.append(new_features)
```

---

### Risk 2: Feature Stability Across Seasons

**Risk Level:** MEDIUM

**Issue:** CTG stats methodology changes between seasons

**Mitigation:**
1. Check CTG data consistency (same columns, same formulas)
2. Normalize features to season averages
3. Use percentage-based features (less affected by rule changes)

---

### Risk 3: Data Leakage in New Features

**Risk Level:** MEDIUM

**Issue:** Accidentally using future information

**Mitigation:**
1. **ALL temporal features must use `.shift(1)`**
2. Code review checklist:
   - [ ] Features only use `past_games` (GAME_DATE < pred_date)
   - [ ] Rolling features use `.shift(1)` before calculation
   - [ ] No "future" opponent stats (use season-to-date only)

**Testing:**
```python
# Test temporal leakage
def test_no_temporal_leakage(pred_date, features):
    """Ensure features only use past data"""
    for feature in features:
        # Check that feature calculation doesn't peek into future
        assert feature['GAME_DATE'] < pred_date
```

---

## Success Metrics

### Primary Metric: Win Rate

**Current:** 51.40% (570/1109 wins)
**Target:** 56%+ (621/1109 wins)

**Tracking:**
- After Phase 1: 53.5-54.5% (592-604 wins)
- After Phase 2: 55% (610 wins)
- After Phase 3: 56%+ (621+ wins)

---

### Secondary Metric: MAE

**Current:** 7.56 pts
**Target:** <6.0 pts

**Tracking:**
- After Phase 1: 7.0-7.2 pts
- After Phase 2: 6.5-6.8 pts
- After Phase 3: <6.0 pts

---

### Tertiary Metric: Sample Size

**Current:** 1,109 bets (3.0 pts edge threshold)
**Option 1:** Keep 3.0 pts → more bets, lower win rate
**Option 2:** Raise to 4.0 pts → fewer bets, higher win rate

**Recommendation:** After Phase 1, re-evaluate edge threshold
- If win rate 53.5%+ at 3.0 pts edge → keep it
- If win rate 52.0-53.0% → raise to 3.5 pts edge
- If win rate <52.0% → raise to 4.0 pts edge

---

### Validation Metric: Consistency Across Months

**Issue:** Model might work well in Nov-Dec but fail in Mar-Apr

**Solution:** Track win rate by month
```
Oct-Nov 2024: X% win rate
Dec-Jan 2025: Y% win rate
Feb-Mar 2025: Z% win rate
Apr 2025:     W% win rate

Target: All months within ±3% of overall win rate
```

---

## Implementation Architecture

### File Structure
```
src/features/
├── opponent_defense.py          # Phase 1: Opponent DRtg
├── recent_form.py                # Phase 1: L3 features
├── minutes_projection.py         # Phase 1: Minutes EWMA
├── scoring_efficiency.py         # Phase 1: True Shooting %
├── interactions.py               # Phase 1: Usage x Pace
├── rest_schedule.py              # Phase 2: Rest binning
├── home_away.py                  # Phase 2: Home/away splits
├── pace_analysis.py              # Phase 2: Pace differential
├── position_matchup.py           # Phase 2: Position vs defense
├── teammate_context.py           # Phase 3: Teammate availability
├── blowout_risk.py               # Phase 3: Blowout prediction
└── feature_engineering_v2.py     # Main orchestrator

scripts/validation/
├── walk_forward_validation_v2.py # Updated validation script
└── feature_ablation_test.py      # Test each feature individually

tests/unit/
├── test_opponent_defense.py
├── test_recent_form.py
└── test_temporal_leakage_v2.py
```

---

### Feature Engineering Pipeline

```python
# File: src/features/feature_engineering_v2.py

class EnhancedFeatureEngineer:
    """
    Phase 1-3 feature engineering pipeline
    """

    def __init__(self):
        self.opponent_defense = OpponentDefenseFeatures()
        self.recent_form = RecentFormFeatures()
        self.minutes_proj = MinutesProjection()
        self.scoring_eff = ScoringEfficiency()
        self.interactions = FeatureInteractions()

    def engineer_features(self, player_id, game_date, past_games, opponent, position):
        """
        Calculate all enhanced features for a player-game

        Args:
            player_id: Player ID
            game_date: Date of game to predict
            past_games: All games before game_date
            opponent: Opponent team abbreviation
            position: Player position (Guard/Wing/Big)

        Returns:
            dict with all engineered features
        """
        # Get player's history
        player_history = past_games[past_games['PLAYER_ID'] == player_id]

        # Phase 1 features
        opp_def_feats = self.opponent_defense.calculate(opponent, position, game_date)
        l3_feats = self.recent_form.calculate_l3(player_history)
        min_feats = self.minutes_proj.calculate(player_history)
        ts_feats = self.scoring_eff.calculate_ts(player_history)
        interaction_feats = self.interactions.calculate_usage_pace(player_history, opp_def_feats)

        # Combine all features
        all_features = {
            **opp_def_feats,
            **l3_feats,
            **min_feats,
            **ts_feats,
            **interaction_feats
        }

        return all_features
```

---

## Experiment Tracking

### Use MLflow for Feature Experiments

```python
# Track each feature's impact
import mlflow

with mlflow.start_run(run_name="Phase1_OpponentDRtg"):
    # Train model with new feature
    model = train_model(X_train_with_new_feature, y_train)

    # Validate
    results = walk_forward_validation(model)

    # Log metrics
    mlflow.log_metric("win_rate", results['win_rate'])
    mlflow.log_metric("mae", results['mae'])
    mlflow.log_metric("sample_size", results['num_bets'])

    # Log feature importance
    mlflow.log_param("feature_added", "OPP_DRtg_Adjusted")
    mlflow.log_metric("feature_importance", model.feature_importances_[feature_idx])
```

---

## Next Steps

### Immediate Actions (Next 3 Days)

**Day 1:**
1. [ ] Implement `OpponentDefenseFeatures` class
2. [ ] Add opponent DRtg features to `game_log_builder.py`
3. [ ] Implement `RecentFormFeatures` (L3)
4. [ ] Run walk-forward validation with Phase 1a features
5. [ ] Track: Win rate, MAE, sample size

**Day 2:**
1. [ ] Implement `MinutesProjection` (EWMA + trend)
2. [ ] Implement `ScoringEfficiency` (True Shooting %)
3. [ ] Run walk-forward validation with Phase 1b features
4. [ ] Compare: Baseline → Phase 1a → Phase 1b

**Day 3:**
1. [ ] Implement `FeatureInteractions` (Usage x Pace)
2. [ ] Integrate ALL Phase 1 features
3. [ ] Run FULL walk-forward validation (2023-24 + 2024-25)
4. [ ] Run backtest with betting simulation
5. [ ] Calculate new win rate and ROI
6. [ ] **Decision:** If win rate > 53.5%, proceed to Phase 2

---

## Appendix A: Feature Importance Ranking

**From XGBoost model (current):**

1. `PRA_L5_mean` (15.2%) - Recent form
2. `PRA_lag1` (12.8%) - Last game
3. `MIN` (11.3%) - Minutes played
4. `PRA_L10_mean` (8.7%) - Medium-term form
5. `USG%` (7.2%) - Usage rate
6. `PRA_ewma5` (6.1%) - EWMA recent form
7. `FGA` (5.3%) - Shot attempts
8. `PRA_L20_mean` (4.9%) - Long-term baseline
9. `AST%` (3.8%) - Assist rate
10. `FTA` (3.2%) - Free throw attempts

**Expected After Phase 1:**

1. `MIN_projected` (18%) - **NEW: Minutes projection**
2. `PRA_L3_mean` (14%) - **NEW: L3 recent form**
3. `PRA_L5_mean` (12%)
4. `OPP_DRtg_Adjusted` (10%) - **NEW: Opponent defense**
5. `PRA_lag1` (9%)
6. `Usage_x_Pace` (6%) - **NEW: Interaction**
7. `TS_PCT_L5` (5%) - **NEW: True Shooting**
8. `USG%` (5%)
9. `PRA_L10_mean` (4%)
10. `PRA_ewma5` (3%)

---

## Appendix B: Close Losses to Convert

**Top 10 losses that could become wins with better features:**

1. **Medium error losses (5-10 pts):** 222 losses
   - Target: Convert 30-40 (13-18%)
   - Features needed: Minutes projection, L3 form

2. **Large error losses (10-20 pts):** 243 losses
   - Target: Convert 15-20 (6-8%)
   - Features needed: Opponent DRtg, blowout risk

3. **Small error losses (<5 pts):** 36 losses
   - Target: Convert 5-10 (14-28%)
   - Features needed: Model calibration, better edge sizing

**Total conversions needed:** 50-70 losses → 56%+ win rate

---

## Appendix C: Validation Checklist

**Before deploying each feature:**

- [ ] Tested on 2023-24 season (out-of-sample)
- [ ] Tested on 2024-25 season (out-of-sample)
- [ ] Improves win rate on BOTH seasons
- [ ] Reduces MAE on BOTH seasons
- [ ] No temporal leakage (code reviewed)
- [ ] Feature importance > 1% (meaningful)
- [ ] Feature stable across months (consistency check)
- [ ] Documented in feature registry
- [ ] Unit tests written
- [ ] Integration tests passing

---

**Document End**
