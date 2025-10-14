# NBA Props Model - Implementation Quick Reference

**Last Updated**: October 7, 2025

---

## Quick Stats & Benchmarks

### Model Performance Targets
- **Brier Score**: < 0.20 (lower is better)
- **Hit Rate**: > 55% (breakeven at -110 is ~52.4%)
- **ROI**: > 5% (after 500+ bets)
- **CLV**: Consistently positive
- **Accuracy Ceiling**: 68-72% (NBA upset rate is 28-32%)

### Sample Size Requirements
- **Statistical Significance**: ~400 minutes playing time per player (~10-13 games)
- **Model Evaluation**: 500+ predictions minimum
- **Backtesting**: Multiple seasons for robustness

---

## Critical Features by Priority

### Tier 1: Must-Have Features (Highest ROI)

```python
# 1. OPPONENT DEFENSE (Edge Creator)
opponent_features = {
    'opp_def_rating_vs_position': float,      # Points allowed per 100 poss
    'opp_pace': float,                         # Possessions per game
    'opp_def_rating_last_10': float,           # Recent defensive form
    'historical_vs_opponent': float,           # Player vs. this team
}

# 2. REST & SCHEDULE (High Impact, Easy Implementation)
rest_features = {
    'days_rest': int,                          # 0, 1, 2, 3, 4+
    'is_back_to_back': bool,                   # -37.6% performance
    'games_in_last_7_days': int,
    'opponent_days_rest': int,
    'rest_advantage': int,                     # Our rest - opponent rest
}

# 3. LAG FEATURES (Research-Proven)
lag_windows = [1, 3, 5, 7, 10]                 # games back
for lag in lag_windows:
    features[f'pra_lag_{lag}'] = float
    features[f'points_lag_{lag}'] = float
    features[f'rebounds_lag_{lag}'] = float
    features[f'assists_lag_{lag}'] = float

# 4. ROLLING AVERAGES (Multiple Windows)
rolling_windows = [5, 10, 20, 30]              # games
for window in rolling_windows:
    features[f'pra_rolling_{window}'] = float
    features[f'pra_rolling_std_{window}'] = float

# 5. MINUTES PREDICTION (Most Critical Opportunity Stat)
minutes_features = {
    'projected_minutes': float,                # Separate model output
    'avg_minutes_last_10': float,
    'minutes_volatility': float,
    'starter_indicator': bool,
    'key_teammate_out': bool,
}
```

### Tier 2: Important Features

```python
# 6. PACE NORMALIZATION (Fundamental)
pace_features = {
    'team_pace': float,                        # Possessions per 48 min
    'opponent_pace': float,
    'projected_game_pace': float,
    'per_100_poss_stats': dict,                # All stats normalized
}

# 7. EWMA (Recency Weighting)
alpha_values = [0.1, 0.2, 0.3]
for alpha in alpha_values:
    features[f'pra_ewma_alpha_{alpha}'] = float

# 8. HOME/AWAY SPLITS
location_features = {
    'is_home': bool,
    'player_home_avg_last_20': float,
    'player_away_avg_last_20': float,
    'home_away_pra_diff': float,
}

# 9. USAGE & OPPORTUNITY
usage_features = {
    'usage_rate': float,
    'usage_rate_last_10': float,
    'team_pace_when_on_court': float,
    'projected_possessions_played': float,
}

# 10. VOLATILITY METRICS
volatility_features = {
    'std_last_10': float,
    'coefficient_of_variation': float,
    'games_above_prop_pct': float,             # Historical hit rate
    'floor_projection': float,                 # 25th percentile
    'ceiling_projection': float,               # 75th percentile
}
```

### Tier 3: Nice-to-Have Features

```python
# 11. TREND INDICATORS
trend_features = {
    'form_vs_baseline': float,                 # Last 5 - season avg
    'performance_slope_last_10': float,        # Linear trend
    'is_improving': bool,
    'games_into_season': int,
}

# 12. INJURY CONTEXT
injury_features = {
    'games_since_injury': int,
    'injury_status': str,                      # Out/Quest/Prob/Healthy
    'minute_restriction': bool,
}

# 13. GAME CONTEXT
game_features = {
    'projected_spread': float,
    'projected_total': float,
    'blowout_probability': float,              # Affects minutes
    'playoff_implications': bool,
}
```

---

## Model Selection: Calibration > Accuracy

### PRIMARY FINDING
**Research shows**: Model selection based on calibration yields ROI of **+34.69%** vs. **-35.17%** when based on accuracy.

### Evaluation Metrics Priority

```python
# 1. CALIBRATION METRICS (Most Important for Betting)
from sklearn.metrics import brier_score_loss, log_loss
from sklearn.calibration import calibration_curve

metrics_priority = [
    ('brier_score', brier_score_loss),         # PRIMARY
    ('log_loss', log_loss),                    # SECONDARY
    ('calibration_curve', plot_calibration),   # VISUAL CHECK
]

# 2. BETTING PERFORMANCE
betting_metrics = [
    'closing_line_value',                      # Beat closing by 1-2%
    'expected_value',                          # Positive EV required
    'roi',                                     # After 500+ bets
]

# 3. STATISTICAL METRICS (Lower Priority)
statistical_metrics = [
    'hit_rate',                                # Accuracy
    'mae',                                     # Mean Absolute Error
    'r2',                                      # Variance Explained
]
```

### Calibration Implementation

```python
from sklearn.calibration import CalibratedClassifierCV

# After training XGBoost
calibrated_model = CalibratedClassifierCV(
    xgb_model,
    method='isotonic',                         # or 'sigmoid' (Platt scaling)
    cv=5
)
calibrated_model.fit(X_train, y_train)

# Verify calibration
prob_true, prob_pred = calibration_curve(
    y_test,
    calibrated_model.predict_proba(X_test)[:, 1],
    n_bins=10
)
# Should follow diagonal if well-calibrated
```

---

## Validation: Time-Based Only

### NEVER Do This
```python
# WRONG - Random split for time series
X_train, X_test = train_test_split(X, y, test_size=0.2)  # NO!
```

### ALWAYS Do This

```python
# CORRECT - Walk-Forward Validation
from sklearn.model_selection import TimeSeriesSplit

# Approach 1: TimeSeriesSplit
tscv = TimeSeriesSplit(n_splits=5)
for train_idx, test_idx in tscv.split(X):
    X_train, X_test = X[train_idx], X[test_idx]
    # Train and evaluate

# Approach 2: Manual Walk-Forward
training_window = 1000  # games
test_window = 200       # games
step_size = 200         # roll forward

for i in range(0, len(data) - training_window - test_window, step_size):
    train = data[i:i+training_window]
    test = data[i+training_window:i+training_window+test_window]
    # Train on train, test on test

# Approach 3: Season-Based Splits
train_seasons = ['2018-19', '2019-20', '2020-21', '2021-22']
validation = '2022-23'
test = '2023-24'
```

---

## Common Pitfalls

### 1. Overfitting Dangers
- Limit hyperparameter search iterations
- Use regularization (L1/L2)
- Test on multiple seasons
- Large gap between in-sample and out-of-sample performance = overfitting

### 2. Sample Size Mistakes
- Wait for 10+ games before trusting player predictions
- Need 500+ bets to evaluate profitability
- Early season = wider confidence intervals

### 3. Look-Ahead Bias
- Only use data available at prediction time
- No future information leakage
- Timestamped data essential

### 4. Recency Bias
- Don't overweight last 1-2 games
- Balance recent with larger samples
- EWMA handles this better than simple last-N

### 5. Lineup Data
- "Widely cited but notoriously unreliable"
- Sample size issues
- Focus on individual metrics instead

---

## XGBoost Configuration

### Recommended Hyperparameters

```python
xgb_params = {
    # Model Complexity (Control Overfitting)
    'max_depth': [3, 4, 5, 6],                 # Start shallow
    'min_child_weight': [1, 3, 5],             # Regularization
    'gamma': [0, 0.1, 0.2],                    # Min loss reduction

    # Regularization
    'reg_alpha': [0, 0.1, 1],                  # L1 regularization
    'reg_lambda': [1, 2, 3],                   # L2 regularization

    # Learning
    'learning_rate': [0.01, 0.05, 0.1],        # Lower = more robust
    'n_estimators': [100, 500, 1000],          # With early stopping

    # Sampling
    'subsample': [0.7, 0.8, 0.9],              # Row sampling
    'colsample_bytree': [0.7, 0.8, 0.9],       # Column sampling

    # Other
    'objective': 'reg:squarederror',           # or binary:logistic
    'eval_metric': 'mae',                      # or logloss for classification
}

# Early stopping to prevent overfitting
eval_set = [(X_train, y_train), (X_val, y_val)]
model.fit(
    X_train, y_train,
    eval_set=eval_set,
    early_stopping_rounds=50,
    verbose=False
)
```

---

## Bankroll Management: Kelly Criterion

### Implementation

```python
def kelly_bet_size(prob_win, odds_decimal, bankroll, fraction=0.25):
    """
    Fractional Kelly for risk management

    prob_win: Model probability (0.55 = 55%)
    odds_decimal: 1.91 for -110, 2.00 for +100
    bankroll: Current bankroll
    fraction: 0.25-0.5 recommended (quarter to half Kelly)
    """
    # Calculate edge
    edge = (prob_win * odds_decimal) - 1

    # Kelly percentage
    kelly_pct = edge / (odds_decimal - 1)

    # Apply fractional Kelly (reduce variance)
    bet_pct = kelly_pct * fraction

    # Safety cap at 5% max
    bet_pct = min(bet_pct, 0.05)

    return bankroll * bet_pct if bet_pct > 0 else 0

# Example
prob = 0.55                    # Model says 55% chance of Over
odds = 1.91                    # -110 odds (52.4% implied)
bankroll = 10000               # $10,000 bankroll

bet_amount = kelly_bet_size(prob, odds, bankroll, fraction=0.25)
# Returns bet size in dollars
```

### Key Points
- Professional bettors use 25-50% of Kelly
- Full Kelly too aggressive (high variance)
- Never bet more than 2.5-5% of bankroll per bet
- Edge detection is critical - only bet when edge exists

---

## Feature Engineering Formulas

### Pace Adjustment
```python
# Normalize to league average pace
pace_adjusted = raw_stat * (league_pace / team_pace)

# Per 100 possessions
per_100_poss = (stat / possessions) * 100
```

### EWMA
```python
import pandas as pd

# Exponentially Weighted Moving Average
alpha = 0.2  # Higher = more weight on recent
ewma = df['pra'].ewm(alpha=alpha, adjust=False).mean()
```

### Rolling Volatility
```python
# Coefficient of Variation (normalized volatility)
cv = df['pra'].rolling(10).std() / df['pra'].rolling(10).mean()
```

### Lag Features
```python
# Create multiple lags
for lag in [1, 3, 5, 7, 10]:
    df[f'pra_lag_{lag}'] = df.groupby('player_id')['pra'].shift(lag)
```

---

## Expected Value Calculation

### Over/Under Props

```python
def calculate_ev(model_prob_over, odds_over_decimal, odds_under_decimal):
    """
    Calculate expected value for both sides

    model_prob_over: Model's probability of Over (0.60 = 60%)
    odds_over_decimal: Bookmaker odds for Over (1.91 for -110)
    odds_under_decimal: Bookmaker odds for Under (1.91 for -110)
    """
    # EV for Over
    ev_over = (model_prob_over * odds_over_decimal) - 1

    # EV for Under
    ev_under = ((1 - model_prob_over) * odds_under_decimal) - 1

    # Return best bet if positive EV exists
    if ev_over > 0:
        return 'OVER', ev_over
    elif ev_under > 0:
        return 'UNDER', ev_under
    else:
        return 'NO BET', max(ev_over, ev_under)

# Example
side, ev = calculate_ev(
    model_prob_over=0.55,
    odds_over_decimal=1.91,
    odds_under_decimal=1.91
)
print(f"Bet {side} with {ev:.2%} edge")
```

---

## Rest Days Impact (Research Findings)

### Key Statistics
- **+1 day rest**: 37.6% increase in win likelihood
- **Each rest day**: 15.96% decrease in injury odds
- **Back-to-back games**: Significant performance decline
- **Load management**: Controversial - no clear injury decrease in NBA study

### Implementation
```python
rest_multipliers = {
    0: 0.85,   # Back-to-back penalty
    1: 1.00,   # Normal
    2: 1.05,   # Slight boost
    3: 1.08,   # Good rest
    4+: 1.10,  # Well rested
}

# Adjust projection
adjusted_projection = base_projection * rest_multipliers[days_rest]
```

---

## Opponent Defense Integration

### Data Sources
- **NBA.com**: Defense vs. Position (FREE)
- **Basketball-Reference**: Opponent stats (FREE)
- **Cleaning the Glass**: Defensive percentiles (YOUR CURRENT SOURCE)

### Usage
```python
# Opponent adjustment factor
def opponent_adjustment(player_avg, opp_def_percentile):
    """
    opp_def_percentile: 0-100 (0 = worst defense, 100 = best)

    Adjust player projection based on opponent defense quality
    """
    # Convert percentile to multiplier
    # Tough defense (90th percentile) = 0.90x
    # Easy defense (10th percentile) = 1.10x
    multiplier = 1 + ((50 - opp_def_percentile) / 500)

    return player_avg * multiplier
```

---

## SHAP Feature Importance

### Implementation
```python
import shap

# Create explainer
explainer = shap.TreeExplainer(xgb_model)

# Calculate SHAP values
shap_values = explainer.shap_values(X_test)

# Summary plot
shap.summary_plot(shap_values, X_test, plot_type="bar")

# Individual prediction explanation
shap.force_plot(
    explainer.expected_value,
    shap_values[0],
    X_test.iloc[0]
)

# Dependence plot for specific feature
shap.dependence_plot("days_rest", shap_values, X_test)
```

### Research Findings - Most Important Features
**First Half**:
- Field goal percentage
- Defensive rebounds
- Turnovers
- Assists

**Second Half**:
- Field goal percentage
- Defensive rebounds
- Turnovers
- Offensive rebounds
- Three-point percentage

---

## Backtesting Checklist

### Before Running Backtest
- [ ] Using only data available at prediction time
- [ ] Time-based splits (no random shuffling)
- [ ] No look-ahead bias in features
- [ ] Proper handling of missing data
- [ ] Odds/lines from time of prediction

### During Backtest
- [ ] Track both in-sample and out-of-sample performance
- [ ] Calculate Brier Score and Log Loss
- [ ] Measure CLV (Closing Line Value)
- [ ] Record confidence intervals
- [ ] Log all predictions for analysis

### After Backtest
- [ ] Verify IS/OOS consistency (no large gap)
- [ ] Analyze feature importance
- [ ] Check calibration curves
- [ ] Evaluate across different contexts (home/away, B2B, etc.)
- [ ] Require 500+ predictions before conclusions

---

## Minutes Prediction Model

### Separate Model Approach

```python
# Build dedicated minutes projection model
minutes_features = [
    'avg_minutes_last_10',
    'avg_minutes_season',
    'starter_indicator',
    'teammate_injuries',
    'opponent_strength',
    'projected_game_spread',
    'coach_rotation_pattern',
    'foul_trouble_tendency',
]

# Train regression model
from sklearn.ensemble import RandomForestRegressor
minutes_model = RandomForestRegressor(n_estimators=100)
minutes_model.fit(X_train[minutes_features], y_train_minutes)

# Use predictions as feature in main model
df['projected_minutes'] = minutes_model.predict(X[minutes_features])
```

---

## Quick Reference: What NOT to Do

### DO NOT
1. Use random train/test splits for time series data
2. Optimize for accuracy instead of calibration
3. Overweight last 1-2 games (recency bias)
4. Bet more than 5% of bankroll on single bet
5. Trust lineup data without large sample sizes
6. Add too many features (overfitting risk)
7. Ignore opponent defense adjustments
8. Skip calibration step
9. Evaluate model on <500 predictions
10. Use complex models before testing simple ones

### DO
1. Use walk-forward or time-based cross-validation
2. Optimize for Brier Score / Log Loss
3. Balance recent form with larger samples (EWMA)
4. Use fractional Kelly (25-50%) for bet sizing
5. Focus on individual on/off splits instead
6. Use regularization and feature selection
7. Include opponent-adjusted statistics
8. Calibrate probabilities with isotonic regression
9. Collect 500+ predictions minimum
10. Start with XGBoost baseline, add complexity gradually

---

## Performance Monitoring

### Real-Time Tracking

```python
performance_log = {
    'predictions': [],
    'outcomes': [],
    'probabilities': [],
    'odds': [],
    'bet_sizes': [],
}

# After each prediction
performance_log['predictions'].append(prediction)
performance_log['outcomes'].append(actual_outcome)
performance_log['probabilities'].append(model_probability)
performance_log['odds'].append(closing_line_odds)

# Calculate rolling metrics
def calculate_rolling_metrics(log, window=100):
    recent = log[-window:]
    return {
        'hit_rate': np.mean(recent['predictions'] == recent['outcomes']),
        'roi': calculate_roi(recent),
        'brier_score': brier_score_loss(recent['outcomes'], recent['probabilities']),
        'clv': calculate_clv(recent),
    }
```

---

## Resources

### Essential Reading
1. Research Document: `/Users/diyagamah/Documents/nba_props_model/research/nba_props_modeling_best_practices.md`
2. Calibration Paper: "Machine learning for sports betting: Should model selection be based on accuracy or calibration?"
3. FiveThirtyEight NBA Methodology: https://fivethirtyeight.com/methodology/how-our-nba-predictions-work/

### Data Sources
- **CleaningTheGlass**: Your current source (excellent choice)
- **NBA.com Stats**: Free defensive stats
- **Basketball-Reference**: Free historical data
- **NBA API**: Supplementary data

### Python Libraries
```bash
uv add xgboost lightgbm scikit-learn shap pandas numpy
```

---

**Remember**: Calibration > Accuracy. Temporal validation is non-negotiable. Track CLV religiously.
