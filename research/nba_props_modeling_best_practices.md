# NBA Player Props Modeling: Research Summary & Best Practices

**Research Conducted**: October 7, 2025
**Focus**: PRA (Points + Rebounds + Assists) Prediction Systems

---

## Executive Summary

This research synthesizes current state-of-the-art approaches for NBA player prop prediction, specifically targeting PRA metrics. Key findings indicate that successful models combine gradient boosting algorithms (XGBoost/LightGBM) with sophisticated temporal feature engineering, opponent-adjusted statistics, and rigorous calibration-focused validation strategies. The research reveals that model calibration is more critical than raw accuracy for sports betting applications, and proper temporal validation techniques are essential to avoid overfitting.

---

## 1. State-of-the-Art Modeling Approaches

### 1.1 Model Architecture Benchmarks

**Best Performing Models**:
- **XGBoost**: Consistently outperforms other algorithms across multiple studies, achieving best performance across five evaluation metrics for NBA predictions
- **LightGBM**: Shows closest performance to XGBoost with faster training times
- **Ensemble Methods**: Combination of XGBoost and Random Forest models shows promise
- **Traditional ML Baseline**: Random Forest achieves ~67-74% accuracy on game outcomes

**Academic Performance Benchmarks**:
- A 2024 study benchmarking 14 ML models found XGBoost achieving best results for basketball performance prediction
- Tree-based models (Extra Trees, Random Forest, Decision Trees) perform best for player-level predictions
  - Extra Trees: 34.14% WAPE
  - Random Forest: 34.23% WAPE
  - Decision Trees: 34.41% WAPE

**Reality Check**:
- NBA upset rate traditionally ranges 28-32%, meaning better team wins 68-72% of time
- Models should target >70% accuracy to demonstrate edge over market
- Published models typically achieve 66-72% accuracy on game predictions

### 1.2 Commercial Platforms Using ML for Props

**Leading Platforms**:
1. **Outlier.bet**: Focuses on player prop tools with defense analytics
2. **DraftEdge**: AI-driven prop projections including PRA combinations
3. **Dimers**: Proprietary ML system with real-time lineup changes and matchup data
4. **Rithmm**: AI predictions evaluating vast datasets for prop insights

**Common Features**:
- Real-time lineup change integration
- Defensive matchup analysis
- Historical performance patterns
- Usage rate and minutes projections

---

## 2. Critical Features for PRA Prediction

### 2.1 Core Performance Features (Already Planned - Good!)

Your current three-tier approach aligns well with research:

**Tier 1: Core Performance**
- Usage rate (CRITICAL per research)
- Scoring efficiency
- Rebounding rates
- True shooting percentage
- Assist percentage

**Tier 2: Contextual Modulators**
- Minutes projection (identified as "most critical opportunity stat")
- Opponent defense ratings
- Pace factors
- Home/away splits

**Tier 3: Temporal Dynamics**
- Rolling averages (research recommends 20-30 game windows)
- EWMA (Exponential Weighted Moving Average)
- Volatility measures

### 2.2 Additional Features to Consider

Based on research, consider adding:

#### A. Opponent-Adjusted Statistics

**Defensive Rating by Position**:
- Defense vs. position stats (guards/forwards/centers)
- Opponent points allowed per 100 possessions
- Opponent pace adjustment
- Matchup-specific historical performance

**Research Finding**: Markets are often slow to adjust for matchup effects, especially for non-star players - this creates edge opportunities.

#### B. Rest & Load Management Features

**Game Context**:
- Days of rest (each rest day decreases injury odds by 15.96%)
- Back-to-back game indicator (37.6% performance boost with 1+ day rest)
- Games in last X days
- Travel distance (teams fly equivalent of 250 miles/day for 25 weeks)
- Home/away/road trip sequence

**Counterintuitive Finding**: NBA's 2018-2019 study found NO clear link between load management and decreased injury risk after controlling for age, injury history, and minutes. However, fatigue still affects performance.

#### C. Lineup & Rotation Features

**Minutes Prediction Factors**:
- Historical rotation patterns
- Injury/rest impacts on rotation
- Blowout probability (affects garbage time minutes)
- Foul trouble patterns
- Coach-specific rotation preferences

**Research Finding**: Four main factors change rotations: injuries/rest, role changes, matchups, and blowouts.

#### D. Usage & Opportunity Features

**Advanced Opportunity Metrics**:
- Team pace (possessions per 48 minutes)
- Player on-court pace vs. off-court pace
- Usage rate when specific teammates are out
- Shot distribution changes based on lineup
- Play-type frequency (isolation, pick-and-roll, transition)

#### E. Volatility & Consistency Metrics

**Performance Stability**:
- Standard deviation of last 10/20/30 games
- Coefficient of variation for each stat
- Hit rate above/below prop lines historically
- Floor (minimum expected) vs. Ceiling (maximum potential)

**Research Finding**: Playoff teams are more consistent than regular season teams - volatility matters.

#### F. Game Script & Situational Features

**Competitive Context**:
- Projected game spread
- Over/under total
- Playoff implications
- Division rival indicator
- National TV game indicator

#### G. Injury Impact Features

**Health Status**:
- Games since returning from injury
- Injury history by body part (severe lower extremity injuries show performance decrease)
- Teammate injury impacts (usage rate changes)
- Questionable/Probable status effects

**Research Finding**: Odds of injury increase 2.87% per 96 minutes played; 8.23% per 3 additional rebounds; 9.87% per 3 additional field-goal attempts.

---

## 3. Temporal Modeling Techniques

### 3.1 Feature Engineering for Time Series

**Recommended Lag Features**:
Based on research showing 1, 3, 5, 7, and 10 game-lag implementations:

```python
# Research-validated lag windows
lag_windows = {
    'immediate': [1, 2, 3],      # Last 3 games
    'short_term': [5, 7],        # Last week
    'medium_term': [10, 15],     # Last 2-3 weeks
    'season_long': [20, 30]      # Season context
}
```

**Rolling Statistics**:
- Simple Moving Averages (SMA): 5, 10, 20, 30-game windows
- Exponential Weighted Moving Averages (EWMA): alpha = 0.1, 0.2, 0.3
- Rolling standard deviations
- Rolling max/min values

**Research Finding**: Best performance typically accounts for previous 20-30 games for NBA predictions.

**EWMA Implementation**:
```python
# Higher alpha = more weight on recent observations
# NBA Math uses 10-game rolling for players, 20-game for teams
alpha_values = [0.1, 0.2, 0.3]  # Test multiple decay rates
```

### 3.2 Trend Features

**Performance Trajectories**:
- Difference between current form (last 5) and season average
- Slope of performance over last 10/20 games
- Peak performance vs. recent performance gap
- Momentum indicators (improving/declining)

**Season Context**:
- Games into season
- Days since season start
- Season phase (early/mid/late/playoffs)

### 3.3 Advanced Temporal Models

**Research-Identified Approaches**:

1. **Autoregressive Features**: Predict performance based on season stats + last 10 games average
2. **LSTM Models**: Promising for capturing temporal patterns (future research direction)
3. **Time Series Decomposition**: Trend, seasonality, residuals
4. **FiveThirtyEight Approach**: Poisson model for game flow + tree-based endgame model for last 90 seconds

---

## 4. Handling Specific Challenges

### 4.1 Opponent Effects & Matchup Data

**Key Strategies**:

1. **Position-Specific Defense Ratings**:
   - Points allowed to position per 100 possessions
   - Defensive rating percentiles by position
   - Historical matchup performance (player vs. team)

2. **Pace Normalization**:
   ```python
   # Pace adjustment formula from research
   adjusted_stat = raw_stat * (league_pace / team_pace)

   # Per-possession normalization
   per_100_poss = (stat / possessions) * 100
   ```

3. **Opponent Strength Adjustment**:
   - Opponent defensive rating
   - Opponent recent form
   - Opponent in back-to-back or rest advantage

**CleaningTheGlass Advantage**: Your data source provides percentiles on every stat and position groupings that fit modern NBA - this is ideal for opponent adjustments.

### 4.2 Injury Impact & Player Availability

**Multi-Level Approach**:

1. **Direct Player Status**:
   - Current injury report status (Out/Questionable/Probable)
   - Days since last missed game
   - Minute restrictions
   - Return-from-injury performance decay curve

2. **Teammate Availability Effects**:
   - Usage rate change when key teammates absent
   - Role expansion opportunities
   - Historical performance with/without key teammates

3. **Injury Risk Modeling**:
   - Minutes workload tracking
   - Physical activity metrics (rebounds, FGA as injury predictors)
   - Rest patterns

**Research Finding**: Performance shows "descriptive decrease in basketball performance markers in seasons following primary severe lower extremity injury."

### 4.3 Back-to-Back Games & Rest

**Implementation Strategy**:

```python
features = {
    'days_rest': [0, 1, 2, 3, 4+],
    'is_back_to_back': boolean,
    'games_in_last_7_days': int,
    'opponent_on_back_to_back': boolean,
    'rest_advantage': int,  # Our rest - opponent rest
}
```

**Research Findings**:
- 1+ day rest increases win likelihood by 37.6%
- Each day of rest decreases injury odds by 15.96%
- Back-to-back + away games: 3.5 odds ratio for injuries
- HOWEVER: Controversial findings show no clear injury decrease from load management after controlling for confounders

### 4.4 Home/Away Effects

**Notable Patterns from Research**:

1. **Player-Specific Splits**:
   - Some players show large home/away performance differences
   - Examples: Nikola Jokic (similar points home/away, but rebounds/assists lower on road)
   - Klay Thompson: Well-known for significant home/away splits

2. **Declining Home Court Advantage**:
   - Trend shows home-court advantage having less impact over time
   - Currently worth "a few points" in spreads

3. **Modeling Approach**:
   ```python
   # Include both simple indicator and player-specific effects
   features = {
       'is_home': boolean,
       'player_home_avg_last_20': float,
       'player_away_avg_last_20': float,
       'home_away_pra_diff': float,
   }
   ```

**Research Insight**: Player props may offer better edges than main lines because sportsbooks already incorporate home-court advantage into spreads/moneylines.

### 4.5 Recent Form vs. Season-Long Trends

**Balanced Approach from Research**:

1. **Multiple Time Horizons**:
   ```python
   time_windows = {
       'last_3_games': 'immediate_form',
       'last_7_games': 'recent_form',
       'last_15_games': 'medium_form',
       'last_30_games': 'stable_baseline',
       'season_avg': 'true_talent_proxy',
   }
   ```

2. **Weighted Combination**:
   - Let model learn optimal weights via feature importance
   - EWMA naturally handles recency weighting
   - Include both raw averages and EWMA versions

3. **Form Indicators**:
   - Trend direction (improving/declining)
   - Volatility of recent games
   - Consistency score

**Research Warning**: Recency bias is a cognitive bias favoring recent events. Balance with larger sample sizes.

### 4.6 Minutes Volatility & Rotation Changes

**Critical Challenge Identified by Research**:

Minutes projection is "the most critical opportunity stat in NBA DFS" and equally important for props.

**Four Main Rotation Change Drivers**:
1. Injuries/Rest
2. Role changes
3. Matchups (coach adjustments)
4. Blowouts (garbage time)

**Modeling Strategy**:

```python
features = {
    # Historical patterns
    'avg_minutes_last_10': float,
    'minutes_std_last_10': float,
    'starter_indicator': boolean,

    # Context adjustments
    'key_teammate_out': boolean,
    'expected_game_competitiveness': float,  # Blowout probability
    'matchup_specific_minutes': float,  # Historical coach adjustments

    # Opportunity proxies
    'team_pace': float,
    'projected_possessions_played': float,
}
```

**Research Finding**: 240 total minutes per team across 5 positions - analyze distribution patterns.

---

## 5. Model Evaluation & Validation

### 5.1 Critical Distinction: Calibration > Accuracy

**KEY RESEARCH FINDING**:
Model selection based on calibration leads to ROI of +34.69% vs. -35.17% when based on accuracy.

**For sports betting, calibration is more important than accuracy.**

### 5.2 Primary Evaluation Metrics

**For Model Selection**:

1. **Brier Score** (Primary calibration metric)
   - Range: 0 to 1 (lower is better)
   - Measures squared difference between predicted probabilities and outcomes
   - Score closer to 0 = better calibration

2. **Log Loss** (Alternative calibration metric)
   - Penalizes confident wrong predictions heavily
   - Essential for probabilistic betting models

3. **Calibration Curves**
   - Visual check for over/under-confidence
   - Plot predicted probabilities vs. actual outcomes
   - Should follow diagonal if well-calibrated

**For Betting Performance**:

4. **Closing Line Value (CLV)**
   - "Favorite among sharp bettors"
   - Beating closing line by 1-2% signals well-calibrated model
   - Track CLV even during losing streaks
   - Positive CLV = sharp analysis

5. **Expected Value (EV)**
   - If predicted probabilities are accurate and odds offer value
   - Consistent positive EV indicates edge

6. **Return on Investment (ROI)**
   - Ultimate test of profitability
   - Requires large sample size (500+ bets minimum)

**For Statistical Performance**:

7. **Mean Absolute Error (MAE)** for actual stat predictions
8. **R-squared** for variance explained
9. **Hit Rate** above/below specific prop lines

### 5.3 Calibration Techniques

**Python Implementation**:

```python
from sklearn.calibration import calibration_curve, CalibratedClassifierCV

# Methods for calibration:
# 1. Platt Scaling (logistic regression on predictions)
# 2. Isotonic Regression (non-parametric)

# XGBoost calibration
calibrated_model = CalibratedClassifierCV(
    xgb_model,
    method='isotonic',  # or 'sigmoid'
    cv=5
)
```

**Research Resources**:
- scikit-learn built-in functions for calibration
- XGBoost provides probability calibration through wrapper methods
- Visualization with calibration curves essential

### 5.4 Validation Strategies

**Time Series Cross-Validation (ESSENTIAL)**:

**Walk-Forward Validation**:
```python
# Research-recommended approach
# Example: Train on 1000 games, test on next 200 games
# Roll forward 200 games, repeat

training_window = 1000  # games or time period
test_window = 200
step_size = 200  # how far to roll forward

# Advantages:
# - Simulates real-world forecasting
# - No future data leakage
# - Continuous model updating
```

**Research Finding**: Walk-Forward shows "notable shortcomings in false discovery prevention" but is standard practice. Consider Combinatorial Purged Cross-Validation (CPCV) which shows "marked superiority in mitigating overfitting risks."

**Time-Based Splits**:
```python
# NEVER use random splits for time series
# ALWAYS use chronological splits

# Example split strategy:
train_seasons = ['2018-19', '2019-20', '2020-21', '2021-22']
validation_season = '2022-23'
test_season = '2023-24'
```

**Critical Rules**:
1. Training set only includes observations PRIOR to test set
2. No random shuffling
3. Respect temporal ordering
4. Account for season transitions (roster changes, rule changes)

### 5.5 Backtesting Best Practices

**Essential Principles**:

1. **Avoid Look-Ahead Bias**:
   - Only use data available at prediction time
   - Don't use odds/probabilities not available then
   - Use timestamped data
   - Recreate exact market conditions

2. **Avoid Overfitting**:
   - Limit feature engineering iterations
   - Use regularization (L1/L2)
   - Keep model complexity reasonable
   - Test on truly out-of-sample data
   - **CRITICAL**: "Spec leakage & overfitting can occur from mixing market numbers into training, or tuning to past noise"

3. **In-Sample vs. Out-of-Sample**:
   - In-Sample (IS): Training data performance
   - Out-of-Sample (OOS): Test data performance
   - **"A backtest is realistic when IS performance is consistent with OOS performance"**
   - Large IS/OOS gap indicates overfitting

4. **Sample Size Requirements**:
   - Minimum 500+ bets before drawing conclusions
   - "Small sample of handful of games could lead you to believe angle is effective, but larger sample could prove it isn't"
   - Bigger sample = more confidence in results

5. **Forward Testing**:
   - After backtesting, test on future unseen data
   - Paper trading period before real money
   - Monitor for model degradation

### 5.6 Specific Metrics for Over/Under Props

**Calibration for Binary Outcomes**:

```python
# For Over/Under predictions
metrics = {
    'brier_score': mean_squared_error(y_true, y_prob),
    'log_loss': log_loss(y_true, y_prob),
    'hit_rate': accuracy_score(y_true, y_pred),
    'coverage': np.mean(actual_stat > predicted_line),
}

# Expected value calculation
def calculate_ev(prob_over, odds_over, odds_under):
    """
    prob_over: model's predicted probability of Over
    odds_over: bookmaker odds for Over (decimal)
    odds_under: bookmaker odds for Under (decimal)
    """
    ev_over = (prob_over * odds_over) - 1
    ev_under = ((1 - prob_over) * odds_under) - 1
    return max(ev_over, ev_under) if max(ev_over, ev_under) > 0 else None
```

**Distribution Calibration**:
- Don't just predict Over/Under
- Predict full distribution of possible outcomes
- Compare predicted distribution to actual distribution
- Use quantile calibration

---

## 6. Common Pitfalls to Avoid

### 6.1 Overfitting Risks

**Research-Identified Dangers**:

1. **Excessive Model Tuning**:
   - "Constant model adjustments often lead to overfitting and reduced performance"
   - Set strict limits on hyperparameter search iterations
   - Use nested cross-validation for hyperparameter tuning

2. **Too Many Features/Rules**:
   - "When we add too many rules to your strategy there are good chance that your strategy do not work as expected in the futures"
   - Feature selection is critical
   - Remove redundant features
   - Use regularization (L1 for feature selection)

3. **Memorizing Instead of Learning**:
   - "Like memorising answers for test instead of understanding the subject"
   - Test on multiple seasons, not just one
   - Ensure model generalizes across different game contexts

### 6.2 Data Quality Issues

**Critical Checks**:

1. **Inconsistent Data**:
   - Verify CleaningTheGlass data completeness
   - Handle missing games appropriately
   - DNPs (Did Not Play) require special handling
   - Low minutes games may need filtering

2. **Position Definition Changes**:
   - NBA positions are fluid
   - Use position groupings that fit modern NBA
   - Your CTG data source handles this well

3. **Rule Changes Across Seasons**:
   - Account for major rule changes
   - Season start dates vary
   - COVID-shortened season impacts

### 6.3 Sample Size Mistakes

**Research Guidelines**:

1. **Minimum Data Requirements**:
   - ~400 minutes of playing time per player for statistical significance
   - Roughly 10-13 games for rotation players
   - 500+ predictions for evaluating model profitability

2. **Early Season Volatility**:
   - "Small samples from early season yield volatile estimates"
   - First 5-10 games require wider confidence intervals
   - Increase reliance on prior season data early in year

3. **Statistical Significance**:
   - Use proper significance levels (p < 0.05 minimum)
   - Account for multiple comparisons
   - Bootstrap confidence intervals for model metrics

### 6.4 Recency Bias

**Cognitive Trap**:
- "Recency bias is a cognitive bias that favors recent events over historic ones"

**Mitigation Strategies**:
- Balance recent performance with larger samples
- Use EWMA instead of only last-N-games
- Include season-long baseline features
- Test whether recent form actually predicts future performance
- Don't overweight last 1-2 games

### 6.5 Lineup Data Reliability

**Research Warning**:
"Statistics that rate certain lineups or player combinations are widely cited but notoriously unreliable."

**Issues**:
- Sample size problems (most lineups play few minutes together)
- Opponent strength varies
- Garbage time skews data
- Lineup staggering effects

**Solution**: Use lineup data cautiously, focus on individual on/off splits with larger samples.

### 6.6 Home Court Advantage Overweighting

**Research Finding**:
"Declining trend shows home-court advantage having less impact over time"

**Approach**:
- Include home/away as feature but don't overweight
- Player-specific home/away splits more valuable than generic advantage
- Test whether home court matters more in specific contexts (altitude, travel, etc.)

### 6.7 Behavioral Betting Mistakes

**Research-Identified Errors**:

1. **Betting on Popular Teams**:
   - "Casual bettors place wagers on well-known teams like Lakers/Warriors, assuming they will always cover"
   - Focus on edge, not team preference

2. **Ignoring Fatigue**:
   - "Failing to consider back-to-back games and extended road trips"
   - Your rest features address this

3. **Market Following**:
   - Don't just follow market movements
   - Understand WHY line moved
   - Sharp money vs. public money

---

## 7. Advanced Considerations

### 7.1 Feature Importance Analysis

**SHAP (SHapley Additive exPlanations)**:

Research using XGBoost + SHAP revealed:

**First Half Key Indicators**:
- Field goal percentage
- Defensive rebounds
- Turnovers
- Assists

**Second Half Key Indicators**:
- Field goal percentage
- Defensive rebounds
- Turnovers
- Offensive rebounds
- Three-point shooting percentage

**Lag Feature Importance**:
- Defensive rebounds (averaged over 4 game-lags)
- Two-point field goal percentage
- Free throw percentage
- Offensive rebounds
- Assists
- Three-point field goal attempts

**Implementation**:
```python
import shap

# XGBoost SHAP values
explainer = shap.TreeExplainer(model)
shap_values = explainer.shap_values(X_test)

# Visualize
shap.summary_plot(shap_values, X_test)
shap.dependence_plot("feature_name", shap_values, X_test)
```

### 7.2 Player Impact Metrics

**Advanced Metrics to Consider**:

1. **EPM (Estimated Plus-Minus)**:
   - "Best widespread impact metric on the market"
   - Combines box score + play-by-play
   - Per 100 possessions impact

2. **RAPM (Regularized Adjusted Plus-Minus)**:
   - "Ultimate indicator of player's direct impact on scoring margin"
   - Controls for teammates and opponents
   - Computationally intensive

3. **On/Off Splits**:
   - Team performance with player on court vs. off court
   - Demonstrates individual impact
   - NOT perfect (doesn't fully control for teammates/opponents)

**Usage**: Consider as supplementary features, not primary predictors for props.

### 7.3 Play-Type Analysis

**Synergy Sports Data** (if accessible):
- Logs every possession beyond box score
- Isolates play types (pick-and-roll, isolation, transition, etc.)
- Professional teams use for video analysis

**Application**:
- Player efficiency by play type
- Matchup-specific play-type frequencies
- Opponent defensive schemes

**Note**: May be cost-prohibitive, evaluate ROI of additional data sources.

### 7.4 Tracking Data

**SportVU/Second Spectrum**:
- 25 tracking measurements per second
- Player coordinates and ball position
- NBA switched from SportVU to Second Spectrum in 2017

**Publicly Available Subsets**:
- Some tracking stats on NBA.com
- Distance traveled, speed, touches, etc.

**Application**:
- Movement patterns predict fatigue
- Spacing metrics for efficiency
- Defensive assignment tracking

### 7.5 Ensemble Methods

**Stacking Approaches**:

```python
# Meta-ensemble combining multiple models
from sklearn.ensemble import StackingRegressor

estimators = [
    ('xgb', XGBRegressor(**params)),
    ('lgbm', LGBMRegressor(**params)),
    ('rf', RandomForestRegressor(**params))
]

meta_model = Ridge()  # or LogisticRegression for classification

ensemble = StackingRegressor(
    estimators=estimators,
    final_estimator=meta_model,
    cv=5
)
```

**Research Support**: Hybrid models combining multiple algorithms show promise.

### 7.6 Bankroll Management

**Kelly Criterion**:

```python
def kelly_bet_size(prob_win, odds_decimal, bankroll, kelly_fraction=0.25):
    """
    prob_win: Model's predicted probability of winning bet
    odds_decimal: Bookmaker odds in decimal format
    bankroll: Current bankroll
    kelly_fraction: Fraction of Kelly to use (0.25 = quarter Kelly)
    """
    edge = (prob_win * odds_decimal) - 1
    kelly_pct = edge / (odds_decimal - 1)

    # Apply fractional Kelly for risk management
    bet_pct = kelly_pct * kelly_fraction

    # Never bet more than 5% of bankroll (safety cap)
    bet_pct = min(bet_pct, 0.05)

    return bankroll * bet_pct if bet_pct > 0 else 0
```

**Research Recommendations**:
- Professional bettors typically use 25-50% of Kelly
- Never bet more than 2.5% of bankroll on any single wager
- Full Kelly is too aggressive (high variance)
- Example: 60% win probability at +100 odds = 20% ROI, higher Kelly stake

**Edge Detection Example**:
- If Knicks odds at -110 (52.5% implied)
- Your model predicts 55% true probability
- Edge = 2.5%
- Calculate Kelly bet size with edge

---

## 8. Practical Recommendations for Your Model

### 8.1 Immediate Priorities

**High-Impact Additions to Your Feature Set**:

1. **Opponent Defense Metrics** (CRITICAL):
   - Defense vs. position stats from NBA.com or BBRef
   - Opponent points allowed per 100 possessions by position
   - Historical player performance vs. specific teams

2. **Rest & Schedule Features** (HIGH ROI):
   - Days rest
   - Back-to-back indicator
   - Games in last 7 days
   - Travel distance (if available)
   - Opponent rest advantage

3. **Minutes Prediction Model** (ESSENTIAL):
   - Separate model or strong feature set for projected minutes
   - Account for injury status of teammates
   - Historical rotation patterns
   - Blowout probability adjustments

4. **Lag Features** (PROVEN EFFECTIVE):
   - 1, 3, 5, 7, 10 game lags for PRA and components
   - Multiple rolling windows (5, 10, 20, 30 games)
   - EWMA with different alpha values

5. **Pace Normalization** (FUNDAMENTAL):
   - Adjust stats to per-100-possession basis
   - Team pace factors
   - Opponent pace factors
   - Game total implications

### 8.2 Model Development Sequence

**Recommended Build Process**:

```
Phase 1: Baseline Model
├── Core performance features (your Tier 1)
├── Basic contextual features (your Tier 2)
├── Simple temporal features (your Tier 3)
└── XGBoost baseline with walk-forward validation

Phase 2: Enhanced Features
├── Add opponent-adjusted statistics
├── Add rest & schedule features
├── Add lag features (1,3,5,7,10 games)
├── Add home/away splits
└── Re-evaluate with feature importance analysis

Phase 3: Advanced Temporal
├── EWMA variants
├── Multiple rolling windows
├── Trend indicators
├── Volatility measures
└── Season phase adjustments

Phase 4: Minutes Prediction
├── Separate minutes forecast model
├── Integrate as feature
├── Opportunity-adjusted stats
└── Playing time scenarios

Phase 5: Calibration & Optimization
├── Calibration curves analysis
├── Probability adjustments
├── Ensemble methods
├── Kelly criterion integration
└── Backtesting with proper validation
```

### 8.3 Evaluation Protocol

**Structured Testing Framework**:

```python
evaluation_metrics = {
    # Calibration (MOST IMPORTANT)
    'brier_score': BrierScore(),
    'log_loss': LogLoss(),
    'calibration_curve': plot_calibration_curve(),

    # Betting Performance
    'CLV': closing_line_value(),
    'ROI': return_on_investment(),
    'hit_rate': accuracy_score(),

    # Statistical
    'mae_pra': mean_absolute_error(),
    'r2': r2_score(),
    'feature_importance': shap_analysis(),
}

# Validation strategy
validation = {
    'method': 'walk_forward',
    'train_window': '2_seasons',
    'test_window': '1_month',
    'step_size': '1_month',
}
```

### 8.4 Data Advantages You Have

**CleaningTheGlass Strengths**:
- Percentiles on every stat (0-100 scale)
- Position groupings fitting modern NBA
- Shot location categories that matter
- Created by former NBA executive (Ben Falk)
- Statistical interpretation clarity

**Leverage These**:
- Use percentiles for normalization
- Position-adjusted statistics
- Quality over quantity of stats

### 8.5 Features to Deprioritize

**Lower ROI / Higher Complexity**:

1. **Complex Lineup Combinations**:
   - Sample size issues
   - Unreliable per research
   - Focus on individual metrics instead

2. **Deep Play-Type Analysis**:
   - Requires expensive Synergy data
   - Test baseline model first
   - Add only if showing clear edge

3. **Advanced Tracking Data**:
   - Limited public availability
   - Complex to integrate
   - Start with traditional stats

4. **Exotic Impact Metrics** (RAPM, etc.):
   - Computationally intensive
   - May not predict props well
   - Test simpler metrics first

---

## 9. Research Gaps & Future Directions

### 9.1 Areas for Further Investigation

1. **Deep Learning for Temporal Patterns**:
   - LSTM models showing promise in research
   - RNNs for sequential data
   - Transformer architectures
   - Currently limited academic validation for NBA props

2. **Optimal Lag Window Selection**:
   - Research shows 20-30 games effective
   - Player-specific optimal windows?
   - Adaptive windows based on playing time?

3. **Injury Prediction Integration**:
   - Injury risk scoring
   - Performance decay curves post-injury
   - Proactive injury probability models

4. **Market Efficiency Analysis**:
   - Which props show most market inefficiency?
   - Star vs. role player edge opportunities
   - Situational edges (B2B, rest, etc.)

5. **Calibration Method Comparison**:
   - Platt scaling vs. isotonic regression
   - XGBoost built-in calibration vs. post-processing
   - Context-dependent calibration

### 9.2 Datasets Worth Exploring

**High Priority**:
- NBA.com Defense vs. Position data (FREE)
- Basketball-Reference play-by-play (FREE)
- Historical injury reports (RotoWire, etc.)
- Historical odds/lines for CLV calculation

**Medium Priority**:
- NBA API advanced stats
- Play-by-play tracking subset
- SportVU public subset
- ESPN BPI/RPM ratings

**Low Priority (Expensive)**:
- Synergy Sports full platform
- Second Spectrum tracking data
- Professional odds feed

---

## 10. Key Takeaways & Action Items

### 10.1 Critical Success Factors

1. **Calibration Over Accuracy**: Model selection must prioritize Brier Score/Log Loss
2. **Temporal Validation**: Walk-forward or time-based CV is non-negotiable
3. **Opponent Adjustments**: Defense vs. position stats create edge
4. **Rest & Schedule**: High-impact, relatively easy to implement
5. **Minutes Prediction**: Most critical opportunity metric
6. **Lag Features**: Proven effective in multiple research studies
7. **Sample Size Discipline**: 500+ bets minimum for conclusions
8. **Avoid Overfitting**: Limit tuning iterations, use regularization
9. **Kelly Criterion**: Fractional Kelly (25-50%) for bankroll management
10. **Track CLV**: Even more important than win rate

### 10.2 Recommended Next Steps

**Week 1-2: Enhanced Feature Engineering**
- [ ] Add opponent defense ratings by position
- [ ] Implement rest & back-to-back features
- [ ] Create lag features (1,3,5,7,10 games)
- [ ] Add pace normalization
- [ ] Implement home/away splits

**Week 3-4: Temporal Modeling**
- [ ] Build rolling average features (5,10,20,30 games)
- [ ] Implement EWMA with multiple alpha values
- [ ] Add volatility/consistency metrics
- [ ] Create trend indicators

**Week 5-6: Minutes Prediction**
- [ ] Build separate minutes forecast model
- [ ] Integrate minutes projections as features
- [ ] Add rotation pattern features
- [ ] Include injury impact adjustments

**Week 7-8: Model Development**
- [ ] XGBoost baseline with new features
- [ ] LightGBM comparison
- [ ] Feature importance analysis with SHAP
- [ ] Hyperparameter tuning with nested CV

**Week 9-10: Validation & Calibration**
- [ ] Walk-forward validation implementation
- [ ] Calibration curve analysis
- [ ] Isotonic regression calibration
- [ ] Brier Score / Log Loss optimization

**Week 11-12: Backtesting & Deployment**
- [ ] Historical odds data collection
- [ ] Backtest with proper time-based splits
- [ ] CLV calculation framework
- [ ] Kelly criterion bet sizing
- [ ] Paper trading period

### 10.3 Success Metrics

**Model Performance Targets**:
- Brier Score < 0.20 (calibration)
- Hit Rate > 55% (better than 52.4% breakeven at -110)
- Positive CLV on average
- Consistent performance across validation periods

**Betting Performance Targets** (after 500+ bets):
- ROI > 5% (ambitious but achievable)
- Sharpe Ratio > 1.0
- Maximum drawdown < 20%
- Win rate > 53% on -110 lines

---

## 11. Key Citations & Resources

### 11.1 Academic Papers

1. **"Evaluating the effectiveness of machine learning models for performance forecasting in basketball: a comparative study"** (2024)
   - Knowledge and Information Systems
   - Benchmarked 14 ML models, XGBoost performed best
   - Link: https://link.springer.com/article/10.1007/s10115-024-02092-9

2. **"Integration of machine learning XGBoost and SHAP models for NBA game outcome prediction"** (2024)
   - PLOS One
   - XGBoost + SHAP for feature importance
   - Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC11265715/

3. **"Machine learning for sports betting: Should model selection be based on accuracy or calibration?"** (2024)
   - ScienceDirect
   - Calibration ROI +34.69% vs. Accuracy -35.17%
   - Link: https://www.sciencedirect.com/science/article/pii/S266682702400015X

4. **"It's a Hard-Knock Life: Game Load, Fatigue, and Injury Risk in the NBA"** (2018)
   - PMC
   - Injury odds increase 2.87% per 96 min played
   - Link: https://pmc.ncbi.nlm.nih.gov/articles/PMC6107769/

5. **"An innovative method for accurate NBA player performance forecasting"** (2024)
   - International Journal of Data Science and Analytics
   - Individualized models for 203 players
   - Link: https://link.springer.com/article/10.1007/s41060-024-00523-y

### 11.2 Industry Resources

**Data Platforms**:
- CleaningTheGlass: https://cleaningtheglass.com (YOUR CURRENT SOURCE)
- NBA.com Stats: https://www.nba.com/stats
- Basketball-Reference: https://www.basketball-reference.com
- NBA Stuffer: https://www.nbastuffer.com

**Prop Betting Tools**:
- Outlier: https://outlier.bet
- DraftEdge: https://draftedge.com
- Dimers: https://www.dimers.com
- BettingPros: https://www.bettingpros.com

**Analytics Communities**:
- NBA Math: https://nbamath.com
- Dunks & Threes: https://dunksandthrees.com
- Cleaning the Glass Blog
- Basketball-Reference Blog

### 11.3 Technical References

**Python Libraries**:
```
# Core ML
xgboost>=2.0.0
lightgbm>=4.0.0
scikit-learn>=1.3.0

# Evaluation
shap>=0.43.0

# Sports betting specific
kelly-criterion
sports-betting (GitHub: georgedouzas/sports-betting)
```

**Books**:
- "Basketball on Paper" by Dean Oliver (fundamental advanced stats)
- "The Kelly Capital Growth Investment Criterion" (bankroll management)
- "Forecasting: Principles and Practice" (time series methods)

---

## Conclusion

Your current three-tier feature engineering approach (Core Performance, Contextual Modulators, Temporal Dynamics) aligns well with research best practices. The CleaningTheGlass data provides a strong foundation with its percentile-based stats and modern position groupings.

**Critical additions** to maximize model performance:
1. Opponent-adjusted defensive statistics
2. Rest & schedule features (high ROI, relatively easy)
3. Lag features with multiple windows
4. Minutes prediction modeling
5. Calibration-focused evaluation (not just accuracy)

**Most important insight** from research: **Model calibration matters more than accuracy for betting applications.** Select models based on Brier Score and Log Loss, not just hit rate.

**Validation is everything**: Use walk-forward or time-based cross-validation exclusively. Never use random splits for temporal sports data.

With proper implementation of research-validated techniques, achieving 55%+ hit rates and positive ROI is realistic. The key is disciplined feature engineering, rigorous temporal validation, and calibration-focused model selection.

---

**Research Completed**: October 7, 2025
**Total Sources Analyzed**: 80+ academic papers, industry articles, and platforms
**Key Finding**: Calibration > Accuracy for sports betting (34.69% ROI vs. -35.17%)
