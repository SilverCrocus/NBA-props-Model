# Two-Stage Predictor Design: Minutes → PRA

**Phase:** Phase 3
**Goal:** Reduce MAE from 6.06 → 5.20-5.60 (8-14% improvement)
**Key Insight:** 40-50% of error variance comes from variable playing time

---

## Problem Statement

### The Minutes Variance Problem

**Current Approach:** Predict PRA directly
```
Input: Player stats, opponent, rest days, etc.
Output: 35 PRA prediction

Problem: Player could play 20 min (→20 PRA) or 40 min (→50 PRA)
Model predicts average (35 PRA) → Wrong either way!
```

**Example Cases:**
1. **Blowout Game:** Star plays 25 min instead of 35 min → 10 min difference = ~8 PRA error
2. **Overtime:** Player plays 45 min instead of 35 min → 10 min difference = ~8 PRA error
3. **Injury/Rest:** DNP-Rest or early exit → Massive PRA error

**Data Analysis:**
- Minutes std dev: ~8 minutes per game
- PRA per minute: ~1.0 for average player
- **Expected error from minutes variance: 8 points** (matches our current ~6 MAE!)

---

## Solution: Two-Stage Prediction

### Architecture Overview

```
┌─────────────────────────────────────────────────────────┐
│                    INPUT FEATURES                        │
│  Player stats, opponent, rest, schedule, recent form     │
└─────────────────────────────────────────────────────────┘
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │      STAGE 1: MINUTES PREDICTOR       │
        ├───────────────────────────────────────┤
        │  Model: CatBoost Regressor            │
        │  Target: MIN (minutes played)         │
        │  Features: 40 temporal + contextual   │
        │  Output: predicted_MIN                │
        └───────────────────────────────────────┘
                            │
                            ▼
                    predicted_MIN = 32
                            │
                            ▼
        ┌───────────────────────────────────────┐
        │      STAGE 2: PRA PREDICTOR           │
        ├───────────────────────────────────────┤
        │  Model: CatBoost Regressor            │
        │  Target: PRA (points+rebounds+assists)│
        │  Features: Stage 1 + predicted_MIN    │
        │  Key: PRA normalized by minutes       │
        │  Output: predicted_PRA                │
        └───────────────────────────────────────┘
                            │
                            ▼
                  Final: predicted_PRA
```

---

## Stage 1: Minutes Predictor

### Objective
Predict how many minutes a player will play in the upcoming game.

### Target Variable
- **MIN:** Minutes played (0-48 range, typically 10-40)

### Features (40 total)

**Temporal Features (Player History):**
```python
# Lag features
- MIN_lag1, MIN_lag3, MIN_lag5, MIN_lag7
- MIN_L5_mean, MIN_L10_mean, MIN_L20_mean
- MIN_ewma5, MIN_ewma10

# Trend features
- MIN_trend (L5 vs L10)
- MIN_std_L10 (volatility)
```

**Contextual Features (Game Situation):**
```python
# Rest and fatigue
- days_rest (0-7)
- is_b2b (back-to-back game)
- games_last_7d (fatigue)

# Opponent strength (affects blowouts)
- opp_net_rating (strong opponent = close game = more minutes)
- opp_pace (fast game = more minutes)

# Schedule position
- home_game (0/1)
- month_of_season (load management increases late)
- games_remaining (playoff push affects minutes)
```

**Player-Specific Features:**
```python
# Role indicators
- is_starter (estimated from L10 MIN > 25)
- season_avg_MIN (baseline role)
- CTG_USG (usage rate - starters have higher usage)
```

**Game Script Indicators:**
```python
# Blowout prediction (affects garbage time)
- team_strength (NET rating)
- opp_strength_diff (team - opponent)
- expected_margin (strong team vs weak → blowout → less star minutes)
```

### Model
- **Algorithm:** CatBoost (best from Phase 2)
- **Hyperparameters:**
  ```python
  {
      'iterations': 300,
      'depth': 6,
      'learning_rate': 0.05,
      'loss_function': 'RMSE',
      'eval_metric': 'MAE'
  }
  ```

### Expected Performance
- **MAE:** 4-6 minutes (reasonable given ~8 min std dev)
- **Key:** Even imperfect minutes prediction helps Stage 2!

---

## Stage 2: PRA Predictor (Given Minutes)

### Objective
Predict PRA given we know (or have predicted) the minutes played.

### Target Variable
- **PRA:** Points + Rebounds + Assists

### Features (50 total)

**From Stage 1:**
```python
- predicted_MIN (or actual_MIN in training)
- All Stage 1 features (rest, opponent, etc.)
```

**Efficiency Features (Normalized by Minutes):**
```python
# Per-minute stats
- PRA_per_MIN_L5 = PRA_L5 / MIN_L5
- PRA_per_MIN_L10 = PRA_L10 / MIN_L10
- PRA_per_36 (from Day 4)

# Efficiency metrics
- TS_pct (True Shooting %)
- USG_per_36 (Usage rate normalized)
- PER (Player Efficiency Rating)
```

**Interaction Features (Minutes-Dependent):**
```python
# How does player perform at different minute loads?
- predicted_MIN * PRA_per_MIN_L5 (expected PRA at predicted minutes)
- predicted_MIN * TS_pct (scoring efficiency scaled by minutes)
- (predicted_MIN / season_avg_MIN) (usage relative to normal)
```

**Temporal PRA Features:**
```python
# Same as before
- PRA_lag1, PRA_lag3, PRA_lag5
- PRA_L5_mean, PRA_L10_mean, PRA_L20_mean
- PRA_ewma5, PRA_ewma10
```

**CTG Season Stats:**
```python
# Same as Phase 2
- CTG_USG, CTG_PSA, CTG_AST_PCT, etc.
```

### Key Innovation: Minutes as a Feature

**Training Strategy:**
```python
# During training: Use ACTUAL minutes
X_train['MIN_feature'] = y_actual_MIN

# During prediction: Use PREDICTED minutes
X_val['MIN_feature'] = predicted_MIN_from_stage1
```

**Why this works:**
- Model learns: "If player plays 40 min + has 30 PRA_L5_mean → predict 45 PRA"
- Model learns: "If player plays 20 min + has 30 PRA_L5_mean → predict 22 PRA"
- Minutes becomes a strong moderator of PRA expectations

### Model
- **Algorithm:** CatBoost (best from Phase 2)
- **Hyperparameters:** Same as Stage 1

### Expected Performance
- **MAE:** 5.20-5.60 (8-14% better than 6.06 baseline)

---

## Training Strategy: Walk-Forward with Two Stages

### Challenge
- Stage 2 depends on Stage 1 predictions
- Must avoid data leakage
- Need to validate each stage independently

### Solution: Sequential Walk-Forward

```python
# For each training date:
for train_date in training_dates:
    # Get games today + historical data
    games_today = train_data[train_data['GAME_DATE'] == train_date]
    past_games = train_data[train_data['GAME_DATE'] < train_date]

    # ============ STAGE 1: TRAIN MINUTES PREDICTOR ============
    X_stage1, y_MIN = build_stage1_features(past_games, games_today)
    minutes_model.fit(X_stage1, y_MIN)

    # ============ STAGE 2: TRAIN PRA PREDICTOR ============
    # Use ACTUAL minutes in training (not predicted)
    X_stage2 = build_stage2_features(past_games, games_today, actual_MIN=y_MIN)
    y_PRA = games_today['PRA']
    pra_model.fit(X_stage2, y_PRA)

# For validation:
for val_date in validation_dates:
    games_today = val_data[val_data['GAME_DATE'] == val_date]
    past_games = all_data[all_data['GAME_DATE'] < val_date]

    # ============ STAGE 1: PREDICT MINUTES ============
    X_stage1 = build_stage1_features(past_games, games_today)
    predicted_MIN = minutes_model.predict(X_stage1)

    # ============ STAGE 2: PREDICT PRA USING PREDICTED MINUTES ============
    X_stage2 = build_stage2_features(past_games, games_today, actual_MIN=predicted_MIN)
    predicted_PRA = pra_model.predict(X_stage2)
```

**Key Principle:**
- **Training Stage 2:** Use actual minutes (what would happen if we knew minutes)
- **Validation Stage 2:** Use predicted minutes (realistic scenario)

This creates a slight train-test mismatch, but:
- Stage 1 is good (MAE ~5 min)
- Stage 2 is robust to small minutes errors
- Net result: Better than single-stage model

---

## Error Propagation Analysis

### Understanding the Error Flow

**Total Error = Stage 1 Error + Stage 2 Error**

**Stage 1 Error:**
```
Minutes MAE: 5 minutes (estimated)
Impact on PRA: 5 min * 1.0 PRA/min = 5 points error
```

**Stage 2 Error (Given Perfect Minutes):**
```
PRA MAE given minutes: ~4 points (estimated)
This is the irreducible error (player variance, game flow, etc.)
```

**Combined Error:**
```
Total MAE = sqrt(5² + 4²) = 6.4 points (if uncorrelated)
OR
Total MAE = 5 + 4 = 9 points (if fully correlated - worst case)

Expected (partially correlated): 5.20-5.60 points
```

**Why Better Than Baseline (6.06)?**

Current single-stage model:
- Predicts PRA directly
- Minutes variance adds noise (±8 min std dev)
- Total MAE: 6.06

Two-stage model:
- Explicitly predicts minutes (5 min MAE)
- PRA model uses minutes as strong feature
- Reduces noise from minutes variance
- Total MAE: 5.20-5.60 (14% better!)

---

## Implementation Classes

### Class 1: `TwoStagePredictor`

```python
class TwoStagePredictor:
    """
    Two-stage predictor: Stage 1 (minutes) → Stage 2 (PRA given minutes)
    """

    def __init__(self, stage1_params=None, stage2_params=None):
        self.minutes_model = CatBoostRegressor(**stage1_params)
        self.pra_model = CatBoostRegressor(**stage2_params)

        self.stage1_features = []  # Filled during training
        self.stage2_features = []  # Filled during training

    def fit(self, X, y_pra, y_minutes):
        """
        Train both stages.

        Args:
            X: Full feature set
            y_pra: PRA targets
            y_minutes: Minutes targets
        """
        # Stage 1: Predict minutes
        X_stage1 = X[self.stage1_features]
        self.minutes_model.fit(X_stage1, y_minutes)

        # Stage 2: Predict PRA (using ACTUAL minutes in training)
        X_stage2 = X[self.stage2_features].copy()
        X_stage2['actual_MIN'] = y_minutes  # Key: use actual minutes
        self.pra_model.fit(X_stage2, y_pra)

    def predict(self, X):
        """
        Predict PRA using two-stage approach.

        Args:
            X: Full feature set

        Returns:
            predicted_PRA, predicted_MIN
        """
        # Stage 1: Predict minutes
        X_stage1 = X[self.stage1_features]
        predicted_MIN = self.minutes_model.predict(X_stage1)

        # Stage 2: Predict PRA using PREDICTED minutes
        X_stage2 = X[self.stage2_features].copy()
        X_stage2['predicted_MIN'] = predicted_MIN  # Key: use predicted minutes
        predicted_PRA = self.pra_model.predict(X_stage2)

        return predicted_PRA, predicted_MIN

    def evaluate_stage1(self, X, y_minutes):
        """Evaluate minutes prediction separately."""
        X_stage1 = X[self.stage1_features]
        pred_MIN = self.minutes_model.predict(X_stage1)
        mae = mean_absolute_error(y_minutes, pred_MIN)
        return {'stage1_mae': mae, 'predictions': pred_MIN}

    def evaluate_stage2_oracle(self, X, y_pra, y_minutes):
        """Evaluate PRA prediction with ACTUAL minutes (oracle)."""
        X_stage2 = X[self.stage2_features].copy()
        X_stage2['actual_MIN'] = y_minutes  # Oracle: perfect minutes
        pred_PRA = self.pra_model.predict(X_stage2)
        mae = mean_absolute_error(y_pra, pred_PRA)
        return {'stage2_oracle_mae': mae}
```

### Class 2: `TwoStageFeatureBuilder`

```python
class TwoStageFeatureBuilder:
    """Build features for two-stage prediction."""

    def build_stage1_features(self, player_history, current_date, opponent):
        """Build features for minutes prediction."""
        features = {}

        # Minutes temporal features
        features.update(self._minutes_lag_features(player_history))
        features.update(self._minutes_rolling_features(player_history))
        features.update(self._minutes_ewma_features(player_history))

        # Contextual features
        features.update(self._rest_features(player_history, current_date))
        features.update(self._schedule_features(current_date))
        features.update(self._opponent_features(opponent))

        # Role features
        features['is_starter'] = 1 if self._is_starter(player_history) else 0
        features['season_avg_MIN'] = player_history['MIN'].mean()

        return features

    def build_stage2_features(self, player_history, current_date, minutes_value):
        """Build features for PRA prediction given minutes."""
        features = {}

        # Stage 1 features (context)
        features.update(self.build_stage1_features(player_history, current_date))

        # KEY: Minutes feature
        features['MIN_feature'] = minutes_value  # Actual or predicted

        # PRA temporal features
        features.update(self._pra_lag_features(player_history))
        features.update(self._pra_rolling_features(player_history))
        features.update(self._pra_ewma_features(player_history))

        # Efficiency features (normalized by minutes)
        features.update(self._efficiency_features(player_history))

        # Interaction features
        features['MIN_x_PRA_per_MIN'] = minutes_value * (
            player_history['PRA'].sum() / player_history['MIN'].sum()
        )

        return features
```

---

## Validation Strategy

### Three-Level Evaluation

**Level 1: Stage 1 Performance (Minutes Prediction)**
```python
metrics = {
    'stage1_mae': 5.2,  # Minutes MAE
    'stage1_within_5min': 0.68,  # % within 5 minutes
    'stage1_within_10min': 0.89  # % within 10 minutes
}
```

**Level 2: Stage 2 Oracle (PRA Given Perfect Minutes)**
```python
# Use actual minutes in Stage 2 (best case)
metrics = {
    'stage2_oracle_mae': 4.5,  # Best possible PRA MAE
}
```

**Level 3: Full Two-Stage (Realistic Performance)**
```python
# Use predicted minutes in Stage 2 (realistic)
metrics = {
    'two_stage_mae': 5.4,  # Actual deployed performance
    'baseline_mae': 6.06,  # CatBoost single-stage
    'improvement': 0.66  # 11% improvement
}
```

### Success Criteria

**Minimum Success:**
- ✅ Stage 1 MAE < 6 minutes
- ✅ Two-stage MAE < 6.00 (any improvement)

**Target Success:**
- ✅ Stage 1 MAE < 5 minutes
- ✅ Two-stage MAE < 5.60 (7% improvement)

**Stretch Goal:**
- ✅ Stage 1 MAE < 4 minutes
- ✅ Two-stage MAE < 5.30 (12% improvement)
- ✅ Win rate > 55% (profitable!)

---

## Risk Mitigation

### Risk 1: Stage 1 Poor Performance
**Mitigation:** If minutes MAE > 7, fall back to single-stage CatBoost (6.06)

### Risk 2: Error Propagation
**Mitigation:** Robust Stage 2 features that work with imperfect minutes

### Risk 3: Overfitting Stage 2
**Mitigation:** Use actual minutes in training, test with predicted minutes in validation

### Risk 4: Complexity
**Mitigation:** Modular design, can revert to single-stage if needed

---

## Expected Results

### Conservative Estimate
- Stage 1 MAE: 5.5 minutes
- Stage 2 Oracle MAE: 4.8
- **Two-Stage MAE: 5.80** (4% improvement)

### Target Estimate
- Stage 1 MAE: 5.0 minutes
- Stage 2 Oracle MAE: 4.5
- **Two-Stage MAE: 5.40** (11% improvement)

### Optimistic Estimate
- Stage 1 MAE: 4.5 minutes
- Stage 2 Oracle MAE: 4.2
- **Two-Stage MAE: 5.20** (14% improvement)

---

## Implementation Timeline

### Day 1: Design ✅ (This Document)
- Architecture specification
- Feature engineering plan
- Validation strategy

### Day 2: Implementation
- `TwoStagePredictor` class
- `TwoStageFeatureBuilder` class
- Walk-forward training script

### Day 3-4: Training & Validation
- Train on 2023-24
- Validate on 2024-25
- Compare to CatBoost baseline (6.06)

### Day 5: Analysis
- Error decomposition (Stage 1 vs Stage 2)
- Feature importance by stage
- Profitability assessment

---

## Next Steps

1. ✅ Design complete (this document)
2. **Check existing `src/models/two_stage_predictor.py`**
3. **Implement missing components**
4. **Create training script**
5. **Run walk-forward validation**
6. **Analyze results**

Ready to implement!
