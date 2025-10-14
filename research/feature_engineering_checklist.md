# Feature Engineering Checklist for NBA PRA Props Model

**Based on Research**: October 7, 2025
**Purpose**: Comprehensive feature set for PRA (Points + Rebounds + Assists) prediction

---

## Feature Categories

### CATEGORY 1: Player Performance Features (Core)

#### Basic Statistics
- [ ] Points per game (PPG) - current season
- [ ] Rebounds per game (RPG) - current season
- [ ] Assists per game (APG) - current season
- [ ] PRA combined - current season
- [ ] Minutes per game (MPG) - current season
- [ ] Games played - current season
- [ ] Games started - current season

#### Efficiency Metrics
- [ ] True Shooting Percentage (TS%)
- [ ] Effective Field Goal Percentage (eFG%)
- [ ] Field Goal Percentage (FG%)
- [ ] Three-Point Percentage (3P%)
- [ ] Free Throw Percentage (FT%)
- [ ] Two-Point Percentage (2P%)
- [ ] Free Throw Rate (FTr)
- [ ] Three-Point Attempt Rate (3PAr)

#### Advanced Stats (CleaningTheGlass)
- [ ] Usage Rate (USG%)
- [ ] Offensive Rebound Percentage (ORB%)
- [ ] Defensive Rebound Percentage (DRB%)
- [ ] Total Rebound Percentage (TRB%)
- [ ] Assist Percentage (AST%)
- [ ] Turnover Percentage (TOV%)
- [ ] Steal Percentage (STL%)
- [ ] Block Percentage (BLK%)
- [ ] Box Plus/Minus (BPM)

#### Per-Possession Normalization
- [ ] Points per 100 possessions
- [ ] Rebounds per 100 possessions
- [ ] Assists per 100 possessions
- [ ] PRA per 100 possessions
- [ ] Per-36-minute stats (normalized)

---

### CATEGORY 2: Temporal Features (Lag & Rolling)

#### Lag Features (1, 3, 5, 7, 10 games)
- [ ] PRA lag 1 game
- [ ] PRA lag 3 games
- [ ] PRA lag 5 games
- [ ] PRA lag 7 games
- [ ] PRA lag 10 games
- [ ] Points lag 1, 3, 5, 7, 10
- [ ] Rebounds lag 1, 3, 5, 7, 10
- [ ] Assists lag 1, 3, 5, 7, 10
- [ ] Minutes lag 1, 3, 5, 7, 10
- [ ] Usage rate lag 1, 3, 5, 7, 10

#### Simple Moving Averages (SMA)
- [ ] PRA - Last 5 games average
- [ ] PRA - Last 10 games average
- [ ] PRA - Last 20 games average
- [ ] PRA - Last 30 games average
- [ ] Points - Last 5, 10, 20, 30 games
- [ ] Rebounds - Last 5, 10, 20, 30 games
- [ ] Assists - Last 5, 10, 20, 30 games
- [ ] Minutes - Last 5, 10, 20, 30 games

#### Exponentially Weighted Moving Averages (EWMA)
- [ ] PRA EWMA (alpha = 0.1)
- [ ] PRA EWMA (alpha = 0.2)
- [ ] PRA EWMA (alpha = 0.3)
- [ ] Points EWMA (alpha = 0.1, 0.2, 0.3)
- [ ] Rebounds EWMA (alpha = 0.1, 0.2, 0.3)
- [ ] Assists EWMA (alpha = 0.1, 0.2, 0.3)

#### Rolling Standard Deviations
- [ ] PRA std - Last 10 games
- [ ] PRA std - Last 20 games
- [ ] Points std - Last 10, 20 games
- [ ] Rebounds std - Last 10, 20 games
- [ ] Assists std - Last 10, 20 games
- [ ] Minutes std - Last 10 games

#### Volatility Metrics
- [ ] Coefficient of Variation (CV) - Last 10 games
- [ ] CV - Last 20 games
- [ ] Min PRA - Last 10 games (floor)
- [ ] Max PRA - Last 10 games (ceiling)
- [ ] Interquartile Range (IQR) - Last 20 games
- [ ] 25th percentile (floor projection)
- [ ] 75th percentile (ceiling projection)

---

### CATEGORY 3: Trend & Momentum Features

#### Performance Trends
- [ ] Form vs. Baseline (Last 5 avg - Season avg)
- [ ] Form vs. Baseline (Last 10 avg - Season avg)
- [ ] Linear slope of PRA - Last 10 games
- [ ] Linear slope of PRA - Last 20 games
- [ ] Peak performance vs. recent (Max last 30 - Avg last 5)
- [ ] Momentum indicator (improving/declining/stable)

#### Season Context
- [ ] Games into season
- [ ] Days since season start
- [ ] Season phase (early: <20, mid: 20-60, late: 60+, playoffs)
- [ ] Games remaining in season
- [ ] Days until playoffs (if applicable)

---

### CATEGORY 4: Opponent & Matchup Features

#### Opponent Defense Ratings
- [ ] Opponent Defensive Rating (overall)
- [ ] Opponent Defensive Rating vs. Position (Guard/Forward/Center)
- [ ] Opponent Points Allowed per 100 possessions
- [ ] Opponent Points Allowed to position per 100 poss
- [ ] Opponent Defensive Rating percentile (0-100)
- [ ] Opponent Defensive Rating - Last 10 games

#### Opponent Pace & Style
- [ ] Opponent Pace (possessions per 48 min)
- [ ] Opponent Pace percentile
- [ ] Opponent Offensive Rating
- [ ] Opponent Turnover Rate
- [ ] Opponent Rebound Rate allowed
- [ ] Opponent Assist Rate allowed

#### Historical Matchup Data
- [ ] Player PRA avg vs. this opponent (career)
- [ ] Player PRA avg vs. this opponent (this season)
- [ ] Player PRA avg vs. this opponent (last 3 games)
- [ ] Win/Loss record vs. opponent
- [ ] Player performance in rivalry games (if applicable)

#### Pace Projections
- [ ] Projected game pace (team pace + opponent pace) / 2
- [ ] Projected total possessions
- [ ] Projected possessions for player (pace * projected minutes / 48)

---

### CATEGORY 5: Rest & Schedule Features

#### Days Rest
- [ ] Days rest (0, 1, 2, 3, 4+)
- [ ] Is back-to-back (boolean)
- [ ] Opponent days rest
- [ ] Rest advantage (our rest - opponent rest)
- [ ] Games in last 3 days
- [ ] Games in last 5 days
- [ ] Games in last 7 days
- [ ] Games in next 3 days (fatigue anticipation)

#### Travel & Fatigue
- [ ] Miles traveled since last game (if available)
- [ ] Time zone changes (if available)
- [ ] Road trip game number (1st, 2nd, 3rd, etc.)
- [ ] Days on road trip
- [ ] Games on current road trip

#### Load Metrics
- [ ] Minutes played in last 3 days
- [ ] Minutes played in last 7 days
- [ ] Cumulative minutes (season total)
- [ ] Field Goal Attempts in last 3 days (injury risk)
- [ ] Rebounds in last 3 days (injury risk)

---

### CATEGORY 6: Minutes & Opportunity Features

#### Minutes Prediction
- [ ] Projected minutes (from separate model)
- [ ] Average minutes - Last 5 games
- [ ] Average minutes - Last 10 games
- [ ] Average minutes - Season
- [ ] Minutes volatility (std last 10)
- [ ] Starter indicator (boolean)
- [ ] Bench indicator (boolean)

#### Rotation Context
- [ ] Key teammate out (increases usage)
- [ ] Backup available (decreases minutes)
- [ ] Coach rotation pattern score
- [ ] Foul trouble tendency
- [ ] Typical garbage time minutes
- [ ] Blowout probability (affects minutes)

#### Usage Opportunity
- [ ] Usage rate when teammate X is out
- [ ] Usage rate - Last 10 games
- [ ] Usage rate - Season
- [ ] Shot attempts per minute
- [ ] Touches per minute (if available)
- [ ] Time of possession (if available)

---

### CATEGORY 7: Home/Away & Location Features

#### Location Splits
- [ ] Is home game (boolean)
- [ ] Is away game (boolean)
- [ ] Is neutral site (boolean)
- [ ] Home PRA avg - Last 10 games
- [ ] Away PRA avg - Last 10 games
- [ ] Home PRA avg - Season
- [ ] Away PRA avg - Season
- [ ] Home/Away PRA differential

#### Location Advantages
- [ ] Altitude advantage (Denver, Utah)
- [ ] Travel distance to game city
- [ ] Familiar arena (played college/previous team)
- [ ] National TV game (performance boost for some)

---

### CATEGORY 8: Team Context Features

#### Team Performance
- [ ] Team win percentage
- [ ] Team offensive rating
- [ ] Team defensive rating
- [ ] Team pace
- [ ] Team net rating
- [ ] Team recent form (Last 10 wins)

#### Teammate Effects
- [ ] Star teammate availability (boolean)
- [ ] Backup PG/SG/SF/PF/C availability
- [ ] Number of teammates out injured
- [ ] Total minutes available from injuries
- [ ] Expected usage redistribution

#### Team Situation
- [ ] Games above/below .500
- [ ] Playoff positioning
- [ ] Playoff implications (high/medium/low/none)
- [ ] Division rival game (boolean)
- [ ] Conference opponent (boolean)

---

### CATEGORY 9: Game Script Features

#### Projected Game Dynamics
- [ ] Projected point spread
- [ ] Projected total (over/under)
- [ ] Blowout probability (>15 points)
- [ ] Close game probability (<5 points)
- [ ] Overtime probability

#### Competitive Context
- [ ] Playoff implications for our team
- [ ] Playoff implications for opponent
- [ ] Desperation index (must-win scenario)
- [ ] Tanking indicator (late season, low motivation)

---

### CATEGORY 10: Injury & Health Features

#### Player Health Status
- [ ] Current injury status (Healthy/Questionable/Probable/Out)
- [ ] Games since last missed game
- [ ] Days since injury return
- [ ] Minute restriction (boolean)
- [ ] Injury type (lower extremity = more impact)
- [ ] Injury severity history score

#### Recovery Patterns
- [ ] Performance in first game back from injury
- [ ] Performance in games 2-5 back from injury
- [ ] Performance in games 6+ back from injury
- [ ] Average PRA post-injury (if recent injury)

#### Injury Risk Indicators
- [ ] Games played streak (fatigue risk)
- [ ] Minutes played streak
- [ ] Physical activity index (rebounds + FGA)
- [ ] Contact rate (drives, post-ups)

---

### CATEGORY 11: Shot Distribution Features

#### Shot Types
- [ ] % of points from 2-pointers
- [ ] % of points from 3-pointers
- [ ] % of points from free throws
- [ ] Rim attempts percentage
- [ ] Mid-range attempts percentage
- [ ] Corner 3-point percentage
- [ ] Above-the-break 3-point percentage

#### Shot Creation
- [ ] Assisted field goals percentage
- [ ] Unassisted field goals percentage
- [ ] Isolation frequency
- [ ] Pick-and-roll ball handler frequency
- [ ] Transition frequency
- [ ] Post-up frequency

---

### CATEGORY 12: CleaningTheGlass Specific Features

#### Position-Adjusted Percentiles
- [ ] Points percentile (0-100)
- [ ] Rebounds percentile (0-100)
- [ ] Assists percentile (0-100)
- [ ] Usage percentile (0-100)
- [ ] Efficiency percentile (0-100)
- [ ] Shot quality percentile (0-100)

#### Role Indicators
- [ ] Primary ball handler (boolean)
- [ ] Secondary ball handler (boolean)
- [ ] Spot-up shooter role (boolean)
- [ ] Rim runner role (boolean)
- [ ] Post player role (boolean)

---

### CATEGORY 13: Derived Interaction Features

#### Cross-Feature Interactions
- [ ] Usage Rate * Projected Minutes
- [ ] Pace * Projected Minutes
- [ ] Rest Days * Usage Rate
- [ ] Opponent Defense * Player Usage
- [ ] Home Indicator * Rest Days
- [ ] Back-to-Back * Opponent Strength

#### Normalized Combinations
- [ ] PRA per 100 possessions * Projected possessions
- [ ] Usage Rate * Team Pace
- [ ] Rebound Rate * Opponent Missed Shots

---

## Feature Engineering Pipeline

### Step 1: Data Collection
```python
# Raw data sources
sources = {
    'player_stats': 'CleaningTheGlass CSV files',
    'team_stats': 'NBA API / CTG team data',
    'schedule': 'NBA API',
    'injuries': 'Official NBA Injury Report',
    'odds': 'Historical odds database (for backtesting)',
}
```

### Step 2: Basic Feature Creation
```python
# Start with season averages
features = calculate_season_averages(player_data)

# Add per-possession normalization
features = add_per_possession_stats(features, team_pace)

# Add percentiles from CTG
features = add_ctg_percentiles(features, ctg_data)
```

### Step 3: Temporal Feature Generation
```python
# Lag features
for lag in [1, 3, 5, 7, 10]:
    features[f'pra_lag_{lag}'] = create_lag(player_data, lag)

# Rolling averages
for window in [5, 10, 20, 30]:
    features[f'pra_sma_{window}'] = calculate_sma(player_data, window)

# EWMA
for alpha in [0.1, 0.2, 0.3]:
    features[f'pra_ewma_{alpha}'] = calculate_ewma(player_data, alpha)

# Volatility
features['pra_std_10'] = calculate_rolling_std(player_data, 10)
features['cv_10'] = features['pra_std_10'] / features['pra_sma_10']
```

### Step 4: Opponent Features
```python
# Merge opponent data
features = features.merge(
    opponent_defense_ratings,
    on=['game_id', 'opponent'],
    how='left'
)

# Historical matchup
features['vs_opponent_avg'] = calculate_historical_matchup(
    player_data,
    opponent_history
)
```

### Step 5: Rest & Schedule
```python
# Calculate rest days
features['days_rest'] = calculate_rest_days(schedule)
features['is_b2b'] = features['days_rest'] == 0
features['games_last_7'] = count_games_last_n_days(schedule, 7)
```

### Step 6: Minutes Projection
```python
# Train separate minutes model
minutes_model = train_minutes_predictor(historical_data)

# Add as feature
features['projected_minutes'] = minutes_model.predict(features)
```

### Step 7: Final Calculations
```python
# Derived features
features['usage_x_minutes'] = features['usage_rate'] * features['projected_minutes']
features['pace_adjusted_pra'] = features['pra_avg'] * (league_pace / team_pace)
features['opportunity_score'] = (
    features['projected_minutes'] *
    features['team_pace'] *
    features['usage_rate']
)
```

---

## Feature Selection Strategy

### Phase 1: Start with High-Priority Features
```python
# Must-have features (Tier 1)
tier1_features = [
    # Core stats
    'pra_avg_season', 'points_avg', 'rebounds_avg', 'assists_avg',

    # Temporal
    'pra_lag_1', 'pra_lag_3', 'pra_lag_5', 'pra_lag_10',
    'pra_sma_10', 'pra_sma_20', 'pra_ewma_0.2',

    # Opponent
    'opp_def_rating_vs_position', 'opp_pace',

    # Rest
    'days_rest', 'is_b2b', 'games_last_7',

    # Minutes
    'projected_minutes', 'avg_minutes_last_10',

    # Context
    'is_home', 'usage_rate', 'team_pace',
]
```

### Phase 2: Test Additional Features
```python
# Add if improving calibration
tier2_features = [
    # Advanced temporal
    'pra_std_10', 'cv_10', 'pra_ewma_0.1', 'pra_ewma_0.3',

    # Trend
    'form_vs_baseline', 'performance_slope',

    # Opponent
    'vs_opponent_avg', 'opp_def_percentile',

    # Minutes
    'key_teammate_out', 'minutes_volatility',

    # Location
    'home_pra_avg_last_10', 'away_pra_avg_last_10',
]
```

### Phase 3: Feature Selection Methods
```python
# 1. Correlation analysis (remove redundant)
correlation_matrix = features.corr()
high_corr_pairs = find_high_correlation(correlation_matrix, threshold=0.95)

# 2. L1 regularization (automatic selection)
from sklearn.linear_model import LassoCV
lasso = LassoCV(cv=5)
lasso.fit(X, y)
selected_features = X.columns[lasso.coef_ != 0]

# 3. Recursive Feature Elimination
from sklearn.feature_selection import RFECV
rfecv = RFECV(estimator=xgb_model, cv=5)
rfecv.fit(X, y)
selected_features = X.columns[rfecv.support_]

# 4. Feature importance from XGBoost
xgb_model.fit(X_train, y_train)
importance_df = pd.DataFrame({
    'feature': X.columns,
    'importance': xgb_model.feature_importances_
}).sort_values('importance', ascending=False)

# Keep top N features
top_features = importance_df.head(50)['feature'].tolist()
```

---

## Feature Engineering Best Practices

### DO
1. Start with simple features, add complexity gradually
2. Use domain knowledge (opponent defense, rest days)
3. Create features at multiple time scales (lag 1, 5, 10, 20)
4. Normalize stats to per-possession basis
5. Use EWMA for recency weighting instead of only last-N
6. Include both absolute and relative features
7. Test feature importance with SHAP
8. Remove highly correlated features (>0.95)
9. Use regularization to prevent overfitting
10. Validate new features with walk-forward CV

### DON'T
1. Add features without hypothesis/reasoning
2. Use only raw counting stats (no normalization)
3. Rely only on season averages (no temporal)
4. Ignore opponent adjustments
5. Skip minutes prediction
6. Create too many interaction terms
7. Use features with look-ahead bias
8. Keep redundant features
9. Add features that overfit to training data
10. Forget to test feature on out-of-sample data

---

## Feature Storage & Management

### Recommended Structure
```
/Users/diyagamah/Documents/nba_props_model/
├── data/
│   ├── raw/
│   │   ├── ctg_data_organized/          # Your CTG data
│   │   ├── ctg_team_data/               # Team stats
│   │   └── nba_api_data/                # Supplementary
│   ├── processed/
│   │   ├── player_features.parquet      # All player features
│   │   ├── opponent_features.parquet    # Opponent stats
│   │   └── game_features.parquet        # Game-level features
│   └── features/
│       ├── train_features_2021_2022.parquet
│       ├── train_features_2022_2023.parquet
│       └── test_features_2023_2024.parquet
```

### Feature Documentation
```python
# Create feature dictionary
feature_dict = {
    'pra_lag_1': {
        'description': 'PRA from previous game',
        'type': 'lag',
        'source': 'player_gamelog',
        'added_date': '2025-10-07',
        'importance_rank': 5,
    },
    'opp_def_rating_vs_position': {
        'description': 'Opponent defensive rating vs player position',
        'type': 'opponent_adjusted',
        'source': 'nba_api',
        'added_date': '2025-10-07',
        'importance_rank': 3,
    },
    # ... document all features
}

# Save for reference
import json
with open('feature_dictionary.json', 'w') as f:
    json.dump(feature_dict, f, indent=2)
```

---

## Testing New Features

### Validation Protocol
```python
def test_new_feature(feature_name, X_train, y_train, X_test, y_test, baseline_model):
    """
    Test if new feature improves model performance

    Returns: improvement metrics
    """
    # Baseline performance
    baseline_brier = brier_score_loss(y_test, baseline_model.predict_proba(X_test)[:, 1])

    # Add new feature
    X_train_new = X_train.copy()
    X_test_new = X_test.copy()

    # Train with new feature
    new_model = XGBClassifier(**params)
    new_model.fit(X_train_new, y_train)

    # Evaluate
    new_brier = brier_score_loss(y_test, new_model.predict_proba(X_test_new)[:, 1])

    improvement = baseline_brier - new_brier

    return {
        'feature': feature_name,
        'baseline_brier': baseline_brier,
        'new_brier': new_brier,
        'improvement': improvement,
        'keep_feature': improvement > 0.001,  # Meaningful improvement threshold
    }
```

---

## Recommended Implementation Order

### Week 1: Foundation
- [ ] Season averages (points, rebounds, assists, PRA)
- [ ] Per-game and per-100-possession stats
- [ ] Usage rate, minutes per game
- [ ] Basic home/away split
- [ ] Is starter indicator

### Week 2: Temporal Features
- [ ] Lag features (1, 3, 5, 7, 10)
- [ ] Simple moving averages (5, 10, 20, 30)
- [ ] EWMA (alpha = 0.2)
- [ ] Rolling standard deviations

### Week 3: Opponent & Context
- [ ] Opponent defensive rating
- [ ] Opponent pace
- [ ] Days rest
- [ ] Back-to-back indicator
- [ ] Games in last 7 days

### Week 4: Advanced Features
- [ ] Minutes projection model
- [ ] Opponent defense vs. position
- [ ] Historical vs. opponent averages
- [ ] Volatility metrics (CV, floor, ceiling)
- [ ] Form vs. baseline

### Week 5: Optimization
- [ ] Feature importance analysis
- [ ] Remove redundant features
- [ ] Add key interaction terms
- [ ] Calibration tuning
- [ ] Final feature selection

---

**Total Recommended Features**: Start with 30-50, expand to 100-150 with proper selection

**Critical Success Factor**: Use walk-forward validation when testing every new feature
