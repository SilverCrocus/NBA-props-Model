# 12-Week NBA Props Model Development Roadmap

**Created:** 2025-10-14
**Target:** Transform from 51.98% win rate to 55-58% win rate with 4-5 MAE
**Timeline:** 12 weeks development + 3 months live testing
**Success Probability:** 70% (achievable with disciplined execution)

---

## Executive Summary

### Current State
- **Win Rate:** 51.98% (below 52.4% breakeven)
- **MAE:** 9.92 points (needs to be <5)
- **ROI:** +0.91% (needs to be 5-9%)
- **CLV:** 73.7% (elite - proves edge detection works)
- **Problem:** Strong predictive edge, poor model accuracy

### Target State
- **Win Rate:** 55-58% (sustainable profit margin)
- **MAE:** 4-5 points (elite accuracy)
- **ROI:** 5-9% (profitable after fees)
- **CLV:** Maintain >70%
- **Calibration:** Brier score <0.20

### Key Success Factors
1. **Minutes projection** - Single biggest MAE driver
2. **Opponent defense** - Untapped edge in current model
3. **Proper calibration** - Convert accuracy to wins
4. **Feature engineering** - Systematic, not random
5. **Validation rigor** - No temporal leakage, real-world simulation

---

## PHASE 1: Foundation (Weeks 1-4)
**Goal:** Implement core features and reduce MAE to 6-7 points

### Week 1: Infrastructure & Critical Fixes
**Priority:** HIGH | **Time:** 35 hours | **Risk:** LOW

#### Tasks
1. **Fix Walk-Forward Validation (8 hours)**
   - Audit current `walk_forward_validation_2024_25.py`
   - Ensure all features are passed to model training
   - Add feature completeness checks
   - Verify no features are accidentally dropped
   - Test with small date range first

2. **Implement Basic Efficiency Stats (10 hours)**
   - True Shooting % (TS% = PTS / (2 * (FGA + 0.44 * FTA)))
   - Effective FG% (eFG% = (FGM + 0.5 * 3PM) / FGA)
   - Free throw rate (FTr = FTA / FGA)
   - 3-point attempt rate (3PAr = 3PA / FGA)
   - Per-36 minute stats (all stats * 36 / MIN)
   - Per-100 possession stats (all stats * 100 / POSS)

3. **Feature Caching System (8 hours)**
   - Create `src/features/feature_cache.py`
   - Implement parquet-based feature storage
   - Add cache invalidation by date
   - Speed up iteration from hours to minutes
   - Hash function for feature set versioning

4. **Experiment Tracking Setup (6 hours)**
   - Install MLflow: `uv add mlflow`
   - Create `mlruns/` directory
   - Set up tracking for: features, hyperparameters, metrics
   - Create experiment comparison notebook
   - Log all validation metrics automatically

5. **Feature Registry (3 hours)**
   - Create `src/features/feature_registry.yaml`
   - Document each feature: name, formula, expected impact
   - Version control for feature sets
   - Enable easy feature toggling

#### Deliverables
- Fixed validation script with all features
- 12 new efficiency features implemented
- Feature cache reducing iteration time by 10x
- MLflow tracking operational
- Feature registry with 50+ features documented

#### Success Criteria
- Validation runs successfully with no errors
- Cache speeds up feature generation by >80%
- MLflow tracking all experiments
- Baseline MAE established with new features: **8-9 points**

#### Code Structure
```python
# src/features/efficiency_stats.py
def calculate_efficiency_stats(df):
    """Calculate shooting efficiency metrics"""
    df['ts_pct'] = df['PTS'] / (2 * (df['FGA'] + 0.44 * df['FTA']))
    df['efg_pct'] = (df['FGM'] + 0.5 * df['3PM']) / df['FGA']
    df['ftr'] = df['FTA'] / df['FGA']
    df['3par'] = df['3PA'] / df['FGA']

    # Per-36 normalization
    for stat in ['PTS', 'REB', 'AST', 'PRA']:
        df[f'{stat}_per36'] = df[stat] * 36 / df['MIN']

    return df

# src/features/feature_cache.py
class FeatureCache:
    def __init__(self, cache_dir='data/feature_cache'):
        self.cache_dir = cache_dir

    def get_or_create(self, date, feature_set_version, feature_fn):
        """Get cached features or create if not exists"""
        cache_key = self._hash(date, feature_set_version)
        cache_path = f"{self.cache_dir}/{cache_key}.parquet"

        if os.path.exists(cache_path):
            return pd.read_parquet(cache_path)

        features = feature_fn(date)
        features.to_parquet(cache_path)
        return features
```

---

### Week 2: Temporal Features Enhancement
**Priority:** HIGH | **Time:** 32 hours | **Risk:** MEDIUM

#### Tasks
1. **20-Game Rolling Windows (8 hours)**
   - Add L20 averages for PTS, REB, AST, PRA
   - Rolling std dev (volatility)
   - Rolling min/max (floor/ceiling)
   - Efficient pandas `.rolling()` implementation
   - Handle early-season edge cases (<20 games)

2. **EWMA with Multiple Spans (8 hours)**
   - EWMA with span=[5, 10, 20] games
   - Capture recent form at different timescales
   - Compare to simple moving averages
   - Decay parameter: alpha = 2/(span+1)
   - More responsive to recent performance

3. **Momentum Indicators (6 hours)**
   - Hot streak: PRA above mean in last N games
   - Cold streak: PRA below mean in last N games
   - Trend: linear regression slope over L10
   - Acceleration: difference in L5 vs L10 averages
   - Binary flags for extreme streaks

4. **Floor/Ceiling Projections (5 hours)**
   - 25th percentile (floor) over L20
   - 75th percentile (ceiling) over L20
   - Upside metric: (ceiling - mean) / std
   - Consistency metric: (mean - floor) / std
   - Risk-adjusted projections

5. **Coefficient of Variation (3 hours)**
   - CV = std / mean for PRA
   - Identifies boom/bust players
   - Lower CV = more consistent (safer bets)
   - Interaction with over/under lines
   - Position-specific CV benchmarks

6. **Testing & Validation (2 hours)**
   - Unit tests for each temporal feature
   - Verify no lookahead bias
   - Performance profiling (must be <2 min)
   - Integration into main pipeline

#### Deliverables
- 35 new temporal features
- Comprehensive unit tests
- Performance: <2 minutes for full dataset
- Documentation in feature registry
- Expected MAE improvement: -0.5 to -1.0 points

#### Success Criteria
- All temporal features pass lookahead tests
- Feature generation performance <2 min
- Initial model shows MAE reduction
- Target MAE: **7-8 points**

#### Code Structure
```python
# src/features/temporal_features.py
def calculate_temporal_features(player_df, window_sizes=[5, 10, 20]):
    """Calculate rolling and EWMA features"""
    features = []

    # Sort by date for proper temporal ordering
    player_df = player_df.sort_values('GAME_DATE')

    for window in window_sizes:
        # Rolling averages
        player_df[f'PRA_L{window}_mean'] = (
            player_df['PRA'].rolling(window, min_periods=5).mean()
        )
        player_df[f'PRA_L{window}_std'] = (
            player_df['PRA'].rolling(window, min_periods=5).std()
        )

        # EWMA
        player_df[f'PRA_EWMA{window}'] = (
            player_df['PRA'].ewm(span=window, min_periods=5).mean()
        )

        # Floor/ceiling
        player_df[f'PRA_L{window}_floor'] = (
            player_df['PRA'].rolling(window, min_periods=5).quantile(0.25)
        )
        player_df[f'PRA_L{window}_ceiling'] = (
            player_df['PRA'].rolling(window, min_periods=5).quantile(0.75)
        )

    # Momentum
    player_df['PRA_trend_L10'] = (
        player_df['PRA'].rolling(10, min_periods=5)
        .apply(lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) >= 5 else 0)
    )

    # Consistency
    player_df['PRA_cv_L20'] = (
        player_df['PRA_L20_std'] / player_df['PRA_L20_mean']
    )

    return player_df
```

---

### Week 3: Opponent Features
**Priority:** CRITICAL | **Time:** 40 hours | **Risk:** MEDIUM

#### Tasks
1. **Load CTG Team Defensive Data (8 hours)**
   - Access `data/ctg_team_data/` (270 files, 100% complete)
   - Parse team defensive ratings by position
   - Create opponent defense lookup table
   - Handle team name variations (BKN vs BRK)
   - Merge with game log data

2. **Opponent Defensive Rating by Position (10 hours)**
   - Extract defensive stats: DRtg, opponent FG%, opponent eFG%
   - Calculate position-specific defense (PG, SG, SF, PF, C)
   - Adjust for pace and league average
   - Rolling opponent defense (account for injuries)
   - Create opponent strength tiers (elite/average/poor)

3. **Historical Performance vs Opponent (8 hours)**
   - Player's last 5 games vs this opponent
   - Career average vs opponent
   - Last season vs opponent
   - Home/away splits vs opponent
   - Psychological factors (rival games)

4. **Pace-Adjusted Opponent Features (8 hours)**
   - Opponent team pace (possessions per 48 min)
   - Player's stats adjusted for opponent pace
   - Expected possessions in matchup
   - Pace x defensive rating interaction
   - Fast-break opportunities vs opponent

5. **Rest Days x Opponent Interaction (4 hours)**
   - Rest advantage (days off - opponent days off)
   - Back-to-back impact vs opponent strength
   - Travel distance to opponent city
   - Schedule spot analysis (trap game, revenge game)

6. **Testing & Integration (2 hours)**
   - Verify opponent data quality
   - Handle missing opponent data gracefully
   - Performance benchmarking
   - Add to feature cache

#### Deliverables
- 25 opponent-based features
- Opponent defense lookup system
- Historical matchup database
- Expected MAE improvement: -1.0 to -1.5 points

#### Success Criteria
- All opponent features merge correctly
- No missing data for >95% of games
- Feature importance shows opponent features in top 20
- Target MAE: **6-7 points**

#### Code Structure
```python
# src/features/opponent_features.py
class OpponentFeatureGenerator:
    def __init__(self, ctg_team_data_dir='data/ctg_team_data'):
        self.team_data = self._load_team_data(ctg_team_data_dir)
        self.team_name_map = self._create_name_mapping()

    def calculate_opponent_defense(self, opponent, date, position):
        """Get opponent's defensive rating for position"""
        # Get opponent's last 10 games before this date
        opp_recent = self.team_data[
            (self.team_data['TEAM'] == opponent) &
            (self.team_data['DATE'] < date)
        ].tail(10)

        # Position-specific defensive rating
        def_rating = opp_recent[f'{position}_DRtg'].mean()
        opp_fg_pct = opp_recent[f'OPP_{position}_FG_PCT'].mean()

        return {
            'opp_def_rating': def_rating,
            'opp_fg_pct_allowed': opp_fg_pct,
            'opp_def_tier': self._categorize_defense(def_rating)
        }

    def get_historical_matchup(self, player_id, opponent, date):
        """Player's historical performance vs opponent"""
        hist = self.player_game_logs[
            (self.player_game_logs['PLAYER_ID'] == player_id) &
            (self.player_game_logs['OPPONENT'] == opponent) &
            (self.player_game_logs['DATE'] < date)
        ]

        return {
            'vs_opp_L5_PRA': hist.tail(5)['PRA'].mean(),
            'vs_opp_career_PRA': hist['PRA'].mean(),
            'vs_opp_games_played': len(hist)
        }
```

---

### Week 4: Model Retraining & Validation
**Priority:** CRITICAL | **Time:** 30 hours | **Risk:** HIGH

#### Tasks
1. **Feature Selection & Importance (8 hours)**
   - Train XGBoost with all 100+ features
   - Calculate SHAP values for feature importance
   - Remove features with importance <0.01
   - Check for multicollinearity (VIF >10)
   - Create feature groups (temporal, opponent, efficiency)

2. **Hyperparameter Tuning (10 hours)**
   - Use Optuna for Bayesian optimization
   - Search space:
     - `max_depth`: [3, 5, 7, 9]
     - `learning_rate`: [0.01, 0.05, 0.1]
     - `n_estimators`: [100, 300, 500]
     - `min_child_weight`: [1, 3, 5]
     - `subsample`: [0.7, 0.8, 0.9]
     - `colsample_bytree`: [0.7, 0.8, 0.9]
   - 100 trials with 5-fold CV
   - Optimize for MAE

3. **Walk-Forward Validation 2023-24 (6 hours)**
   - Monthly retraining on 2023-24 season
   - Validate on held-out month
   - Calculate MAE, win rate, ROI per month
   - Analyze seasonal patterns
   - Compare to baseline (9.92 MAE)

4. **Walk-Forward Validation 2024-25 (4 hours)**
   - Apply same methodology to 2024-25
   - Test on most recent data
   - Ensure model generalizes to current season
   - Check for distribution shift

5. **Comprehensive Error Analysis (2 hours)**
   - Error by player tier (star/role/bench)
   - Error by position
   - Error by game situation (blowout, close)
   - Error by line type (over/under)
   - Identify systematic biases

#### Deliverables
- Optimized XGBoost model
- Feature importance report (top 30 features)
- Validation results on 2023-24 and 2024-25
- Error analysis by segments
- Feature set reduced to 60-80 most important

#### Success Criteria (CRITICAL CHECKPOINT)
- **MAE: 6-7 points** (must be <7.0 or extend Phase 1)
- **Win Rate: 53-54%**
- **ROI: 2-4%**
- Feature importance makes intuitive sense
- No overfitting (train MAE - val MAE <1 point)

#### Code Structure
```python
# src/models/walk_forward_validator.py
class WalkForwardValidator:
    def __init__(self, feature_generator, model_class=XGBRegressor):
        self.feature_gen = feature_generator
        self.model_class = model_class

    def validate(self, start_date, end_date, retrain_frequency='monthly'):
        """Perform walk-forward validation"""
        results = []

        date_ranges = self._create_date_ranges(
            start_date, end_date, retrain_frequency
        )

        for train_start, train_end, val_start, val_end in date_ranges:
            # Generate features
            X_train, y_train = self.feature_gen.generate(
                train_start, train_end
            )
            X_val, y_val = self.feature_gen.generate(
                val_start, val_end
            )

            # Train model
            model = self.model_class(**self.best_params)
            model.fit(X_train, y_train)

            # Validate
            y_pred = model.predict(X_val)

            metrics = {
                'period': f"{val_start} to {val_end}",
                'mae': mean_absolute_error(y_val, y_pred),
                'win_rate': self._calculate_win_rate(y_val, y_pred),
                'roi': self._calculate_roi(y_val, y_pred)
            }
            results.append(metrics)

        return pd.DataFrame(results)

    def _calculate_win_rate(self, y_true, y_pred, threshold=0.5):
        """Calculate win rate against closing lines"""
        # Simulate betting: if predicted > line, bet over
        # Win if actual > line
        # Simplified - real version needs actual odds
        correct = np.sum((y_pred > y_true + threshold) == (y_true > y_true + threshold))
        return correct / len(y_true)
```

---

## PHASE 2: Advanced Features (Weeks 5-8)
**Goal:** Reduce MAE to 5-6 points with sophisticated modeling

### Week 5: Minutes Projection Model
**Priority:** CRITICAL | **Time:** 38 hours | **Risk:** HIGH

#### Tasks
1. **Minutes Data Collection & Preprocessing (6 hours)**
   - Extract minutes played for all players
   - Label starters vs bench (first 5 on court)
   - Collect injury reports (nba_api.live.boxscore.BoxScore)
   - Coach rotation patterns (clustering by coach)
   - Blowout adjustments (score differential)

2. **Minutes Prediction Features (10 hours)**
   - Starter status (binary)
   - Position depth chart (starter/6th man/deep bench)
   - Recent minutes trend (L5, L10)
   - Teammate injury context (if star out, +minutes)
   - Coach tendency (avg minutes by role)
   - Game situation (home/away, back-to-back)
   - Historical minutes distribution (mean, std, 25th, 75th percentiles)
   - Season progression (early season = experimentation)

3. **Build Separate Minutes Model (12 hours)**
   - XGBoost regressor for minutes
   - Target: actual minutes played
   - Train on 3 years of data
   - Walk-forward validation
   - Handle edge cases (DNP-CD, ejections)
   - Expected accuracy: MAE 3-5 minutes

4. **Integrate into Main Model (6 hours)**
   - Add predicted_minutes as feature
   - Create per-minute stats: PRA / predicted_minutes
   - Projected stats: per_minute_stats * predicted_minutes
   - Compare to using actual minutes (should improve)
   - Weight uncertainty (low confidence minutes = penalty)

5. **Validation & Impact Assessment (4 hours)**
   - Compare MAE with/without minutes model
   - Expected improvement: -1.0 to -1.5 MAE
   - Analyze errors: Are minute predictions driving errors?
   - Calibration: Do minute projections match reality?

#### Deliverables
- Standalone minutes prediction model (MAE 3-5 min)
- 8 new minutes-based features
- Integration into main model
- Expected MAE: **5.5-6.5 points**

#### Success Criteria
- Minutes model MAE <5 minutes
- Main model MAE improves by >0.5 points
- Minutes predictions match actual distribution
- No false confidence (wide intervals when uncertain)

#### Code Structure
```python
# src/models/minutes_projector.py
class MinutesProjector:
    def __init__(self):
        self.model = XGBRegressor(
            objective='reg:squarederror',
            max_depth=5,
            learning_rate=0.05,
            n_estimators=300
        )

    def engineer_features(self, game_date, player_id):
        """Create features for minutes prediction"""
        features = {}

        # Starter status
        features['is_starter'] = self._get_starter_status(player_id, game_date)

        # Recent minutes trend
        recent_games = self._get_recent_games(player_id, game_date, n=10)
        features['minutes_L5_mean'] = recent_games.tail(5)['MIN'].mean()
        features['minutes_L10_mean'] = recent_games['MIN'].mean()
        features['minutes_L10_std'] = recent_games['MIN'].std()

        # Teammate injuries
        teammates = self._get_teammates(player_id, game_date)
        features['teammates_out'] = sum(
            1 for t in teammates if self._is_injured(t, game_date)
        )
        features['star_teammates_out'] = sum(
            1 for t in teammates
            if self._is_injured(t, game_date) and self._is_star(t)
        )

        # Coach tendency
        coach = self._get_coach(player_id, game_date)
        features['coach_avg_starter_min'] = (
            self._get_coach_tendency(coach, 'starter')
        )

        # Game situation
        features['is_home'] = self._is_home_game(player_id, game_date)
        features['is_back_to_back'] = self._is_back_to_back(player_id, game_date)
        features['days_rest'] = self._get_days_rest(player_id, game_date)

        return features

    def predict_with_uncertainty(self, features):
        """Predict minutes with confidence interval"""
        # Point prediction
        minutes_pred = self.model.predict(features)

        # Uncertainty from historical distribution
        similar_games = self._find_similar_situations(features)
        minutes_std = similar_games['MIN'].std()

        return {
            'minutes_pred': minutes_pred,
            'minutes_std': minutes_std,
            'minutes_lower': minutes_pred - 1.96 * minutes_std,
            'minutes_upper': minutes_pred + 1.96 * minutes_std
        }
```

---

### Week 6: Usage & Role Modeling
**Priority:** HIGH | **Time:** 35 hours | **Risk:** MEDIUM

#### Tasks
1. **Calculate Usage Rate (6 hours)**
   - USG% = 100 * ((FGA + 0.44 * FTA + TOV) * (Team_MIN / 5)) / (MIN * (Team_FGA + 0.44 * Team_FTA + Team_TOV))
   - Individual usage rate over L10, L20
   - Team usage rate (how ball-dominant is team?)
   - Position-adjusted usage (PGs naturally higher)
   - Usage rate vs starters out

2. **Teammate Injury Impact on Usage (12 hours)**
   - Identify key teammates (usage >20%)
   - When star teammate out, usage increases
   - Calculate historical usage lift when X is out
   - Create teammate injury indicators
   - Expected usage: f(baseline_usage, teammates_out)
   - Example: If Luka out, usage for other Mavs +5-8%

3. **Lineup-Dependent Usage (8 hours)**
   - Parse lineup data from nba_api.stats.endpoints.leaguedashlineups
   - Calculate usage rate by lineup combination
   - 5-man lineups may not have enough data, use 3-man core
   - On-court vs off-court splits
   - When does player dominate ball? (specific lineups)

4. **Role Indicators (5 hours)**
   - Primary option: USG >28%, high AST/TOV
   - Secondary option: USG 20-28%
   - Tertiary option: USG <20%
   - Role changes over season (rookie emergence)
   - Role vs opponent quality (stars get more vs weak teams)

5. **Usage x Minutes x Pace Interactions (4 hours)**
   - Combined feature: USG% * projected_min * opponent_pace
   - Represents total offensive opportunities
   - High usage + high minutes + fast pace = inflated stats
   - Non-linear interactions (XGBoost will learn)

#### Deliverables
- 18 usage and role features
- Teammate injury impact quantified
- Lineup-based usage adjustments
- Expected MAE improvement: -0.5 to -0.8 points

#### Success Criteria
- Usage features in top 15 by importance
- Teammate injury features correctly predict usage lift
- Model adjusts for role changes
- Target MAE: **5.0-6.0 points**

#### Code Structure
```python
# src/features/usage_features.py
class UsageFeatureGenerator:
    def calculate_usage_rate(self, player_stats, team_stats):
        """Calculate true usage rate"""
        player_usage = (
            player_stats['FGA'] +
            0.44 * player_stats['FTA'] +
            player_stats['TOV']
        ) * (team_stats['MIN'] / 5)

        team_usage = (
            team_stats['FGA'] +
            0.44 * team_stats['FTA'] +
            team_stats['TOV']
        )

        usage_rate = 100 * player_usage / (
            player_stats['MIN'] * team_usage
        )

        return usage_rate

    def get_usage_with_injuries(self, player_id, game_date, injured_players):
        """Predict usage given injuries"""
        baseline_usage = self._get_baseline_usage(player_id, game_date)

        # Historical usage when each teammate was out
        usage_lifts = []
        for injured in injured_players:
            hist_games = self._get_games_without_teammate(
                player_id, injured, game_date
            )
            if len(hist_games) > 5:
                lift = hist_games['USG'].mean() - baseline_usage
                usage_lifts.append(lift)

        # Conservative: take 80% of max lift
        expected_usage = baseline_usage + 0.8 * max(usage_lifts, default=0)

        return expected_usage

    def get_role_indicator(self, usage_rate, team_hierarchy):
        """Classify player role"""
        if usage_rate > 28 or team_hierarchy == 1:
            return 'primary'
        elif usage_rate > 20 or team_hierarchy <= 3:
            return 'secondary'
        else:
            return 'tertiary'
```

---

### Week 7: Shot Quality & Location
**Priority:** MEDIUM | **Time:** 32 hours | **Risk:** MEDIUM

#### Tasks
1. **Shot Location Data Collection (8 hours)**
   - Use nba_api.stats.endpoints.shotchartdetail
   - Categorize shots: rim, short-mid, long-mid, corner-3, above-break-3
   - Calculate frequency by zone over L20 games
   - Expected points per shot by zone
   - Shot distribution evolution (early season vs late)

2. **Rim Frequency x Defensive Rim Protection (8 hours)**
   - Player's rim attempt rate
   - Opponent's rim protection (blocks, opponent FG% at rim)
   - Expected points at rim vs this opponent
   - Adjustment: If elite rim protector (Gobert, Lopez), reduce rim points
   - Alternative: More mid-range attempts against rim protectors

3. **Three-Point Volume x Perimeter Defense (8 hours)**
   - Player's 3PA rate
   - Opponent's 3P% defense
   - Corner 3-point defense (often weaker)
   - Expected 3-point makes vs opponent
   - Adjustment for defensive schemes (switch vs drop coverage)

4. **Shot Distribution Shifts (4 hours)**
   - Compare current shot distribution to season average
   - Reasons for shift: injury, role change, hot streak
   - Regression to mean: Extreme distributions regress
   - Impact on PRA (more 3s = fewer rebounds/assists)

5. **Integration & Testing (4 hours)**
   - Add 15 shot quality features
   - Test on historical data
   - Verify features make intuitive sense
   - Performance: Should improve points prediction most

#### Deliverables
- 15 shot quality and location features
- Shot chart visualizations
- Expected impact: -0.3 to -0.5 MAE (modest)
- Target MAE: **5.0-5.5 points**

#### Success Criteria
- Shot features improve points prediction
- Opponent rim protection correctly adjusts rim attempts
- No performance degradation from extra API calls
- Cache shot chart data to avoid rate limits

#### Code Structure
```python
# src/features/shot_quality.py
class ShotQualityAnalyzer:
    def __init__(self):
        self.shot_zones = {
            'rim': {'distance': (0, 5), 'expected_fg': 0.65},
            'short_mid': {'distance': (5, 16), 'expected_fg': 0.42},
            'long_mid': {'distance': (16, 24), 'expected_fg': 0.38},
            'three': {'distance': (24, 50), 'expected_fg': 0.36}
        }

    def get_shot_distribution(self, player_id, start_date, end_date):
        """Get shot distribution by zone"""
        from nba_api.stats.endpoints import shotchartdetail

        shot_chart = shotchartdetail.ShotChartDetail(
            player_id=player_id,
            season_nullable='2024-25',
            date_from_nullable=start_date,
            date_to_nullable=end_date
        ).get_data_frames()[0]

        distribution = {}
        for zone, params in self.shot_zones.items():
            zone_shots = shot_chart[
                (shot_chart['SHOT_DISTANCE'] >= params['distance'][0]) &
                (shot_chart['SHOT_DISTANCE'] < params['distance'][1])
            ]

            distribution[f'{zone}_attempts'] = len(zone_shots)
            distribution[f'{zone}_freq'] = len(zone_shots) / len(shot_chart)
            distribution[f'{zone}_fg_pct'] = zone_shots['SHOT_MADE_FLAG'].mean()

        return distribution

    def adjust_for_opponent_defense(self, shot_dist, opponent_defense):
        """Adjust expected points for opponent defense"""
        expected_points = 0

        # Rim attempts vs rim protection
        rim_points = (
            shot_dist['rim_freq'] *
            self.shot_zones['rim']['expected_fg'] *
            2 *
            (1 - opponent_defense['rim_protection_strength'])
        )

        # Three-point attempts vs perimeter defense
        three_points = (
            shot_dist['three_freq'] *
            self.shot_zones['three']['expected_fg'] *
            3 *
            (1 - opponent_defense['perimeter_defense_strength'])
        )

        expected_points = rim_points + three_points + ...

        return expected_points
```

---

### Week 8: Validation & Refinement
**Priority:** CRITICAL | **Time:** 36 hours | **Risk:** HIGH

#### Tasks
1. **Multi-Season Walk-Forward Validation (16 hours)**
   - 2021-22 season (post-COVID normalization)
   - 2022-23 season
   - 2023-24 season
   - 2024-25 season (current)
   - Monthly retrain and validate for each
   - Total: 48 months of validation

2. **Performance Consistency Analysis (6 hours)**
   - MAE by month (detect seasonal patterns)
   - MAE by player tier (stars vs role players)
   - MAE by position (guards vs bigs)
   - MAE by game type (home/away, B2B)
   - Identify when model performs worst

3. **Edge Case Handling (8 hours)**
   - **Low-minute players** (<20 min): Higher uncertainty
   - **Blowouts** (>20 point diff): Starters benched early
   - **Injuries** (return from injury): Minute restrictions
   - **Trade deadline** (new team): Role changes
   - **Season start** (<10 games): Limited data
   - Create special handling for each case

4. **Feature Importance Deep Dive (4 hours)**
   - SHAP summary plot for top 30 features
   - SHAP dependence plots for key features
   - Feature interactions (SHAP interaction values)
   - Validate feature importance makes sense
   - Remove any unintuitive high-importance features

5. **Error Analysis & Bias Detection (2 hours)**
   - Residual plots (predicted vs actual)
   - Error distribution (should be normal)
   - Systematic biases (always over/under predicting certain players)
   - Calibration by prediction bins
   - Heteroskedasticity check (error variance constant?)

#### Deliverables
- 48-month validation report (4 seasons)
- Performance consistency metrics
- Edge case handling system
- Feature importance report with SHAP
- Bias detection report

#### Success Criteria (CRITICAL CHECKPOINT)
- **MAE: 5-6 points** (must be <6.0 or debug features)
- **Win Rate: 54-55%**
- **ROI: 3-5%**
- Consistent performance across all 4 seasons
- No major biases detected
- Model ready for calibration phase

#### Code Structure
```python
# src/validation/multi_season_validator.py
class MultiSeasonValidator:
    def __init__(self, seasons=['2021-22', '2022-23', '2023-24', '2024-25']):
        self.seasons = seasons
        self.validator = WalkForwardValidator()

    def run_full_validation(self):
        """Run walk-forward validation across multiple seasons"""
        all_results = []

        for season in self.seasons:
            print(f"Validating {season}...")
            start_date = self._get_season_start(season)
            end_date = self._get_season_end(season)

            results = self.validator.validate(
                start_date, end_date, retrain_frequency='monthly'
            )
            results['season'] = season
            all_results.append(results)

        combined = pd.concat(all_results, ignore_index=True)
        return self._generate_report(combined)

    def analyze_consistency(self, results):
        """Analyze performance consistency"""
        consistency_metrics = {
            'mae_mean': results['mae'].mean(),
            'mae_std': results['mae'].std(),
            'mae_cv': results['mae'].std() / results['mae'].mean(),
            'worst_month': results.loc[results['mae'].idxmax()],
            'best_month': results.loc[results['mae'].idxmin()],
            'months_above_6_mae': (results['mae'] > 6).sum(),
            'win_rate_mean': results['win_rate'].mean(),
            'win_rate_std': results['win_rate'].std()
        }
        return consistency_metrics

    def handle_edge_cases(self, prediction_context):
        """Apply special handling for edge cases"""
        adjustments = {}

        # Low-minute players: Increase uncertainty
        if prediction_context['predicted_minutes'] < 20:
            adjustments['confidence_penalty'] = 0.7
            adjustments['note'] = 'Low minutes expected'

        # Blowout risk: Reduce predictions
        if prediction_context['blowout_probability'] > 0.3:
            adjustments['stat_multiplier'] = 0.85
            adjustments['note'] = 'Blowout risk'

        # Return from injury: Reduce predictions
        if prediction_context['games_since_injury'] < 3:
            adjustments['stat_multiplier'] = 0.90
            adjustments['note'] = 'Return from injury'

        # New team (traded): High uncertainty
        if prediction_context['games_with_team'] < 10:
            adjustments['confidence_penalty'] = 0.6
            adjustments['note'] = 'New team adjustment'

        return adjustments
```

---

## PHASE 3: Calibration & Optimization (Weeks 9-10)
**Goal:** Improve win rate from 54% to 55-57% through proper calibration

### Week 9: Probability Calibration
**Priority:** CRITICAL | **Time:** 35 hours | **Risk:** HIGH

#### Tasks
1. **Calibration Theory & Setup (4 hours)**
   - Problem: Predictions may be accurate but not well-calibrated
   - Goal: Predicted probability should match observed frequency
   - Methods: Isotonic regression, Platt scaling, Beta calibration
   - Read: "Predicting Good Probabilities With Supervised Learning" (Niculescu-Mizil)
   - Install: `uv add scikit-learn netcal`

2. **Convert Regression to Classification (8 hours)**
   - Current: Regression model (predicts PRA value)
   - Need: Probability model (P(PRA > line))
   - Method 1: Quantile regression (predict distribution)
   - Method 2: Residual distribution (fit normal to residuals)
   - Method 3: Train classifier directly (binary: over/under)
   - Test all three methods, choose best

3. **Implement Isotonic Regression Calibration (8 hours)**
   - Train isotonic regression on validation set
   - Maps predicted probabilities to calibrated probabilities
   - Non-parametric: No assumptions about calibration curve
   - Requires large dataset (we have 48 months)
   - sklearn.calibration.CalibratedClassifierCV

4. **Implement Platt Scaling (4 hours)**
   - Parametric alternative to isotonic regression
   - Fits logistic regression: P_calibrated = 1 / (1 + exp(A * P_pred + B))
   - Works better with smaller datasets
   - Faster inference than isotonic

5. **Calibration Curve Analysis (6 hours)**
   - Plot predicted probability vs observed frequency
   - Perfect calibration: y = x line
   - Reliability diagram by bins
   - Calculate Brier score (lower = better)
   - Expected Calibration Error (ECE)
   - Compare before/after calibration

6. **Brier Score Optimization (3 hours)**
   - Brier score = mean((predicted_prob - actual)^2)
   - Target: <0.20 (well-calibrated)
   - Optimize calibration method hyperparameters
   - Validate on multiple seasons

7. **Integration & Testing (2 hours)**
   - Apply calibration to walk-forward validation
   - Measure impact on win rate
   - Expected improvement: +1-2 percentage points
   - Test on 2024-25 holdout set

#### Deliverables
- Calibrated probability model
- Calibration curves showing improvement
- Brier score <0.20
- Expected win rate improvement: +1-2 points
- Target win rate: **55-56%**

#### Success Criteria
- Calibration curve close to diagonal (y=x)
- Brier score <0.20
- Win rate improves by at least 1 percentage point
- ECE (Expected Calibration Error) <0.05
- Predictions trustworthy for bet sizing

#### Code Structure
```python
# src/models/calibration.py
from sklearn.calibration import calibration_curve, CalibratedClassifierCV
from sklearn.isotonic import IsotonicRegression
import numpy as np

class ProbabilityCalibrator:
    def __init__(self, method='isotonic'):
        """
        method: 'isotonic', 'platt', or 'beta'
        """
        self.method = method
        self.calibrator = None

    def fit(self, y_pred, y_true, lines):
        """
        Fit calibration model

        Args:
            y_pred: Predicted PRA values
            y_true: Actual PRA values
            lines: Betting lines (over/under threshold)
        """
        # Convert to binary classification
        y_binary = (y_true > lines).astype(int)

        # Convert predictions to probabilities
        # Assume normal distribution of errors
        residuals = y_true - y_pred
        residual_std = np.std(residuals)

        # P(actual > line) given predicted value
        from scipy.stats import norm
        z_scores = (y_pred - lines) / residual_std
        probs_uncalibrated = norm.cdf(z_scores)

        # Fit calibration
        if self.method == 'isotonic':
            self.calibrator = IsotonicRegression(out_of_bounds='clip')
            self.calibrator.fit(probs_uncalibrated, y_binary)

        elif self.method == 'platt':
            # Fit logistic regression
            from sklearn.linear_model import LogisticRegression
            self.calibrator = LogisticRegression()
            self.calibrator.fit(
                probs_uncalibrated.reshape(-1, 1),
                y_binary
            )

        return self

    def predict_proba(self, y_pred, lines):
        """Return calibrated probabilities"""
        # Same process as fit
        residual_std = self.residual_std_  # stored during fit
        z_scores = (y_pred - lines) / residual_std
        from scipy.stats import norm
        probs_uncalibrated = norm.cdf(z_scores)

        if self.method == 'isotonic':
            probs_calibrated = self.calibrator.transform(probs_uncalibrated)
        elif self.method == 'platt':
            probs_calibrated = self.calibrator.predict_proba(
                probs_uncalibrated.reshape(-1, 1)
            )[:, 1]

        return probs_calibrated

    def evaluate_calibration(self, probs_pred, y_true):
        """Calculate calibration metrics"""
        # Calibration curve
        prob_true, prob_pred = calibration_curve(
            y_true, probs_pred, n_bins=10
        )

        # Brier score
        brier = np.mean((probs_pred - y_true) ** 2)

        # Expected Calibration Error
        ece = np.mean(np.abs(prob_true - prob_pred))

        return {
            'brier_score': brier,
            'ece': ece,
            'calibration_curve': (prob_true, prob_pred)
        }

    def plot_calibration(self, probs_pred, y_true, save_path=None):
        """Plot calibration curve"""
        import matplotlib.pyplot as plt

        prob_true, prob_pred = calibration_curve(
            y_true, probs_pred, n_bins=10
        )

        plt.figure(figsize=(8, 8))
        plt.plot([0, 1], [0, 1], 'k--', label='Perfect calibration')
        plt.plot(prob_pred, prob_true, 's-', label='Model')
        plt.xlabel('Predicted probability')
        plt.ylabel('Observed frequency')
        plt.title('Calibration Curve')
        plt.legend()
        plt.grid(True)

        if save_path:
            plt.savefig(save_path)
        plt.show()
```

**Calibration Impact Example:**
```
Before Calibration:
- When model predicts 60% probability: Actual win rate = 55%
- When model predicts 70% probability: Actual win rate = 65%
- Brier score: 0.24 (poor)

After Calibration:
- When model predicts 60% probability: Actual win rate = 60%
- When model predicts 70% probability: Actual win rate = 70%
- Brier score: 0.18 (good)

Impact: More accurate probability estimates → Better bet sizing → Higher ROI
```

---

### Week 10: Model Ensemble & Optimization
**Priority:** HIGH | **Time:** 38 hours | **Risk:** MEDIUM

#### Tasks
1. **Train LightGBM as Second Model (8 hours)**
   - Install: `uv add lightgbm`
   - Different architecture from XGBoost
   - Hyperparameters:
     - `num_leaves`: [31, 63, 127]
     - `learning_rate`: [0.01, 0.05, 0.1]
     - `n_estimators`: [100, 300, 500]
     - `min_child_samples`: [20, 50, 100]
   - Train on same features as XGBoost
   - Compare individual performance

2. **Create Weighted Ensemble (8 hours)**
   - Simple average: (XGB + LGBM) / 2
   - Weighted average: Optimize weights via validation
   - Stacking: Train meta-model on predictions
   - Test all three approaches
   - Choose based on validation MAE

3. **Hyperparameter Optimization with Optuna (12 hours)**
   - Install: `uv add optuna`
   - Optimize both models jointly
   - Search space: 200+ combinations
   - Multi-objective: Minimize MAE, maximize win rate
   - Pareto frontier analysis
   - 300 trials (6-8 hours runtime)

4. **Cross-Validation Strategy Refinement (4 hours)**
   - Current: Walk-forward with monthly retrain
   - Alternative: Weekly retrain (more adaptive)
   - Alternative: Expanding window vs rolling window
   - Test different window sizes (90 days, 180 days, 365 days)
   - Choose optimal based on validation performance

5. **Feature Selection Refinement (4 hours)**
   - Remove features with near-zero importance
   - Test removing correlated features
   - Forward feature selection (iterative)
   - Backward feature elimination
   - Recursive feature elimination (RFE)
   - Final feature set: 50-70 features

6. **Final Validation (2 hours)**
   - Run full 48-month validation
   - Ensemble vs individual models
   - MAE, win rate, ROI, Sharpe ratio
   - Consistency across seasons

#### Deliverables
- LightGBM model (MAE ~5-6 points)
- Ensemble model (MAE 4.5-5.5 points)
- Optimized hyperparameters
- Final feature set (50-70 features)
- Validation report across 4 seasons

#### Success Criteria (CRITICAL CHECKPOINT)
- **MAE: 4.5-5.5 points**
- **Win Rate: 55-57%**
- **ROI: 4-7%**
- **Brier Score: <0.20**
- Ensemble outperforms individual models
- Ready for production deployment

#### Code Structure
```python
# src/models/ensemble.py
import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import Ridge
import optuna

class ModelEnsemble:
    def __init__(self):
        self.xgb_model = None
        self.lgb_model = None
        self.ensemble_weights = None
        self.meta_model = None

    def train_base_models(self, X_train, y_train, X_val, y_val):
        """Train XGBoost and LightGBM"""
        # XGBoost
        self.xgb_model = xgb.XGBRegressor(**self.xgb_params)
        self.xgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            early_stopping_rounds=50,
            verbose=False
        )

        # LightGBM
        self.lgb_model = lgb.LGBMRegressor(**self.lgb_params)
        self.lgb_model.fit(
            X_train, y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)]
        )

    def optimize_ensemble_weights(self, X_val, y_val):
        """Find optimal weighted average"""
        xgb_preds = self.xgb_model.predict(X_val)
        lgb_preds = self.lgb_model.predict(X_val)

        def objective(trial):
            w_xgb = trial.suggest_float('w_xgb', 0, 1)
            w_lgb = 1 - w_xgb

            ensemble_preds = w_xgb * xgb_preds + w_lgb * lgb_preds
            mae = np.mean(np.abs(ensemble_preds - y_val))
            return mae

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=100)

        self.ensemble_weights = {
            'xgb': study.best_params['w_xgb'],
            'lgb': 1 - study.best_params['w_xgb']
        }

    def train_stacking_meta_model(self, X_train, y_train):
        """Train meta-model on base model predictions"""
        # Generate out-of-fold predictions
        xgb_oof = self._get_oof_predictions(self.xgb_model, X_train, y_train)
        lgb_oof = self._get_oof_predictions(self.lgb_model, X_train, y_train)

        # Stack predictions
        meta_features = np.column_stack([xgb_oof, lgb_oof])

        # Train meta-model (simple Ridge regression)
        self.meta_model = Ridge(alpha=1.0)
        self.meta_model.fit(meta_features, y_train)

    def predict(self, X, method='weighted'):
        """Generate ensemble prediction"""
        xgb_pred = self.xgb_model.predict(X)
        lgb_pred = self.lgb_model.predict(X)

        if method == 'simple':
            return (xgb_pred + lgb_pred) / 2

        elif method == 'weighted':
            return (
                self.ensemble_weights['xgb'] * xgb_pred +
                self.ensemble_weights['lgb'] * lgb_pred
            )

        elif method == 'stacking':
            meta_features = np.column_stack([xgb_pred, lgb_pred])
            return self.meta_model.predict(meta_features)

    def hyperparameter_optimization(self, X_train, y_train, X_val, y_val):
        """Joint hyperparameter optimization"""
        def objective(trial):
            # XGBoost params
            xgb_params = {
                'max_depth': trial.suggest_int('xgb_max_depth', 3, 9),
                'learning_rate': trial.suggest_float('xgb_lr', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('xgb_n_est', 100, 500),
                'subsample': trial.suggest_float('xgb_subsample', 0.7, 0.9),
                'colsample_bytree': trial.suggest_float('xgb_colsample', 0.7, 0.9)
            }

            # LightGBM params
            lgb_params = {
                'num_leaves': trial.suggest_int('lgb_num_leaves', 31, 127),
                'learning_rate': trial.suggest_float('lgb_lr', 0.01, 0.1, log=True),
                'n_estimators': trial.suggest_int('lgb_n_est', 100, 500),
                'min_child_samples': trial.suggest_int('lgb_min_child', 20, 100)
            }

            # Train models
            xgb_model = xgb.XGBRegressor(**xgb_params)
            xgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], early_stopping_rounds=50, verbose=False)

            lgb_model = lgb.LGBMRegressor(**lgb_params)
            lgb_model.fit(X_train, y_train, eval_set=[(X_val, y_val)], callbacks=[lgb.early_stopping(50), lgb.log_evaluation(False)])

            # Ensemble
            xgb_pred = xgb_model.predict(X_val)
            lgb_pred = lgb_model.predict(X_val)
            ensemble_pred = (xgb_pred + lgb_pred) / 2

            mae = np.mean(np.abs(ensemble_pred - y_val))
            return mae

        study = optuna.create_study(direction='minimize')
        study.optimize(objective, n_trials=300, show_progress_bar=True)

        self.xgb_params = {k: v for k, v in study.best_params.items() if k.startswith('xgb_')}
        self.lgb_params = {k: v for k, v in study.best_params.items() if k.startswith('lgb_')}

        return study.best_value
```

---

## PHASE 4: Production Readiness (Weeks 11-12)
**Goal:** Deploy production system ready for live testing

### Week 11: Risk Management & Edge Calculation
**Priority:** CRITICAL | **Time:** 36 hours | **Risk:** MEDIUM

#### Tasks
1. **Implement Kelly Criterion Bet Sizing (8 hours)**
   - Kelly formula: f* = (bp - q) / b
     - f* = fraction of bankroll to bet
     - b = odds (decimal odds - 1)
     - p = probability of win (from calibrated model)
     - q = probability of loss (1 - p)
   - Fractional Kelly (25-50% of full Kelly for safety)
   - Minimum edge threshold (only bet if edge >3%)
   - Maximum bet size (cap at 5% of bankroll)

2. **Confidence Intervals for Predictions (8 hours)**
   - Quantile regression (10th, 50th, 90th percentiles)
   - Prediction interval from residual distribution
   - Confidence as function of:
     - Feature availability (missing data = lower confidence)
     - Model agreement (XGB and LGB disagree = lower confidence)
     - Historical variance (volatile players = wider intervals)
   - Only bet when confidence >70%

3. **Edge Threshold Optimization (6 hours)**
   - Calculate edge: calibrated_prob - implied_prob_from_odds
   - Simulate betting at different edge thresholds (1%, 2%, 3%, 5%)
   - Optimize for Sharpe ratio: (mean_return / std_return)
   - Account for reduced volume at higher thresholds
   - Recommended: 3-5% edge threshold

4. **Bankroll Management Strategies (6 hours)**
   - Starting bankroll: $10,000 (example)
   - Bet sizing: Fractional Kelly (25%)
   - Compounding: Recalculate bankroll after each bet
   - Drawdown limits: Stop trading if down 20%
   - Maximum concurrent bets: 10 per day
   - Track: ROI, Sharpe, max drawdown, win rate

5. **Line Shopping Integration (4 hours)**
   - Get odds from multiple books (TheOddsAPI)
   - Calculate edge for each book
   - Choose best odds available
   - Expected improvement: +0.5-1% ROI from line shopping
   - Track CLV (closing line value) for each book

6. **Risk Dashboard (4 hours)**
   - Streamlit or Plotly Dash
   - Real-time display:
     - Current bankroll
     - Today's PnL
     - Win rate (last 30 days)
     - ROI (last 30 days)
     - Largest drawdown
     - Top/bottom performers
   - Alerts for anomalies

#### Deliverables
- Kelly criterion bet sizer
- Confidence interval estimator
- Edge threshold optimizer (optimal: 3-5%)
- Bankroll management system
- Risk dashboard

#### Success Criteria
- Kelly criterion properly implemented
- Confidence intervals cover 90% of actuals
- Edge threshold optimized for Sharpe ratio
- Dashboard displays real-time metrics
- System ready for paper trading

#### Code Structure
```python
# src/betting/kelly_criterion.py
class KellyCriterion:
    def __init__(self, fractional_kelly=0.25, max_bet_pct=0.05):
        self.fractional_kelly = fractional_kelly
        self.max_bet_pct = max_bet_pct

    def calculate_bet_size(self, prob_win, decimal_odds, bankroll):
        """
        Calculate optimal bet size using Kelly Criterion

        Args:
            prob_win: Calibrated probability of winning (0-1)
            decimal_odds: Decimal odds (e.g., 1.95 for -105)
            bankroll: Current bankroll

        Returns:
            Bet size in dollars
        """
        # Kelly formula
        b = decimal_odds - 1  # Net odds
        q = 1 - prob_win

        kelly_fraction = (b * prob_win - q) / b

        # Apply fractional Kelly
        adjusted_fraction = kelly_fraction * self.fractional_kelly

        # Cap at max bet percentage
        adjusted_fraction = min(adjusted_fraction, self.max_bet_pct)

        # No negative bets
        adjusted_fraction = max(adjusted_fraction, 0)

        bet_size = bankroll * adjusted_fraction

        return bet_size

    def calculate_edge(self, prob_win, decimal_odds):
        """Calculate betting edge"""
        implied_prob = 1 / decimal_odds
        edge = prob_win - implied_prob
        return edge

    def should_bet(self, prob_win, decimal_odds, min_edge=0.03, min_confidence=0.70):
        """Determine if bet meets thresholds"""
        edge = self.calculate_edge(prob_win, decimal_odds)

        # Must have minimum edge
        if edge < min_edge:
            return False, f"Edge too low: {edge:.2%}"

        # Could add confidence check here if available

        return True, f"Edge: {edge:.2%}"

# src/betting/bankroll_manager.py
class BankrollManager:
    def __init__(self, starting_bankroll=10000):
        self.starting_bankroll = starting_bankroll
        self.current_bankroll = starting_bankroll
        self.bet_history = []
        self.max_drawdown = 0

    def place_bet(self, bet_size, prob_win, decimal_odds, outcome=None):
        """Record a bet"""
        bet = {
            'timestamp': datetime.now(),
            'bet_size': bet_size,
            'prob_win': prob_win,
            'decimal_odds': decimal_odds,
            'potential_win': bet_size * (decimal_odds - 1),
            'outcome': outcome  # Will be updated later
        }
        self.bet_history.append(bet)

        # Update bankroll if outcome known
        if outcome is not None:
            self._update_bankroll(bet)

    def _update_bankroll(self, bet):
        """Update bankroll after bet settles"""
        if bet['outcome'] == 'win':
            self.current_bankroll += bet['potential_win']
        elif bet['outcome'] == 'loss':
            self.current_bankroll -= bet['bet_size']

        # Track drawdown
        drawdown = (self.starting_bankroll - self.current_bankroll) / self.starting_bankroll
        self.max_drawdown = max(self.max_drawdown, drawdown)

        # Stop trading if down 20%
        if drawdown > 0.20:
            print("WARNING: 20% drawdown reached. Stop trading.")

    def get_statistics(self):
        """Calculate performance statistics"""
        settled_bets = [b for b in self.bet_history if b['outcome'] is not None]

        if not settled_bets:
            return {}

        wins = sum(1 for b in settled_bets if b['outcome'] == 'win')
        total = len(settled_bets)

        returns = []
        for bet in settled_bets:
            if bet['outcome'] == 'win':
                returns.append(bet['potential_win'])
            else:
                returns.append(-bet['bet_size'])

        roi = sum(returns) / self.starting_bankroll
        sharpe = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0

        return {
            'current_bankroll': self.current_bankroll,
            'total_bets': total,
            'win_rate': wins / total,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': self.max_drawdown
        }

# src/betting/line_shopping.py
class LineShoppingOptimizer:
    def __init__(self, odds_api_key):
        self.odds_api_key = odds_api_key
        self.books = ['fanduel', 'draftkings', 'betmgm', 'caesars']

    def get_best_odds(self, player_name, prop_type, line):
        """Find best odds across multiple books"""
        all_odds = self._fetch_odds(player_name, prop_type, line)

        best_over = max(all_odds, key=lambda x: x['over_odds'])
        best_under = max(all_odds, key=lambda x: x['under_odds'])

        return {
            'best_over': best_over,
            'best_under': best_under,
            'edge_improvement': self._calculate_improvement(all_odds)
        }
```

---

### Week 12: Production Infrastructure
**Priority:** CRITICAL | **Time:** 40 hours | **Risk:** HIGH

#### Tasks
1. **Automated Daily Prediction Pipeline (12 hours)**
   - Cron job to run daily at 10 AM ET
   - Steps:
     1. Fetch today's games (nba_api)
     2. Get latest player data
     3. Get opponent data
     4. Generate features
     5. Load trained model
     6. Generate predictions
     7. Get odds from TheOddsAPI
     8. Calculate edges
     9. Generate bet recommendations
     10. Send report via email/Slack
   - Error handling: Retry logic, fallback to previous data
   - Logging: All steps logged to file

2. **Monitoring Dashboards (10 hours)**
   - Streamlit dashboard with tabs:
     - **Today's Predictions:** All bets for today
     - **Performance:** Win rate, ROI, Sharpe over time
     - **Model Health:** Feature drift, prediction drift
     - **Bankroll:** Current bankroll, max drawdown, PnL chart
   - Refresh every 5 minutes
   - Deploy on Streamlit Cloud or local server

3. **Alert System for Anomalies (6 hours)**
   - Alerts via email/SMS/Slack:
     - Model prediction outside historical range
     - Feature value outside expected range
     - Unexpected edge (>15% - likely data error)
     - Bankroll drawdown >10%
     - API failures (nba_api, TheOddsAPI)
   - Use Twilio for SMS, SendGrid for email

4. **Performance Tracking System (6 hours)**
   - SQLite database to store:
     - Predictions: player, date, predicted_PRA, line, edge
     - Actuals: player, date, actual_PRA, result
     - Bets: date, player, bet_size, odds, outcome
   - Daily aggregation: win rate, ROI, Sharpe
   - Weekly report generation

5. **Documentation & Handoff (4 hours)**
   - README: How to run pipeline
   - Architecture diagram
   - Feature documentation (feature_registry.yaml)
   - Model documentation: Training process, hyperparameters
   - API documentation: Endpoints, data formats
   - Troubleshooting guide

6. **Final Validation & Production Test (2 hours)**
   - Run full pipeline end-to-end
   - Verify all components work
   - Test error handling (simulate API failures)
   - Paper trade for 1 week before live money

#### Deliverables
- Automated daily pipeline (runs at 10 AM ET)
- Real-time monitoring dashboard
- Alert system for anomalies
- Performance tracking database
- Complete documentation
- Production-ready system

#### Success Criteria (FINAL CHECKPOINT)
- Pipeline runs successfully every day
- Dashboard displays accurate real-time data
- Alerts trigger correctly
- All documentation complete
- **Final validation:**
  - **MAE: <5 points**
  - **Win Rate: 55-58%**
  - **ROI: 5-9%**
  - **Brier Score: <0.20**
  - **Sharpe Ratio: >1.5**
- System ready for 3-month live testing

#### Code Structure
```python
# src/production/daily_pipeline.py
import schedule
import time
from datetime import datetime
import logging

class DailyPredictionPipeline:
    def __init__(self):
        self.logger = self._setup_logging()
        self.feature_generator = FeatureGenerator()
        self.model = self._load_model()
        self.calibrator = self._load_calibrator()
        self.kelly = KellyCriterion()
        self.bankroll_manager = BankrollManager()

    def run_daily_pipeline(self):
        """Main pipeline execution"""
        try:
            self.logger.info(f"Starting pipeline: {datetime.now()}")

            # 1. Fetch today's games
            games = self._fetch_todays_games()
            self.logger.info(f"Found {len(games)} games today")

            # 2. Generate features for each player
            predictions = []
            for game in games:
                players = self._get_players_in_game(game)

                for player in players:
                    features = self.feature_generator.generate(player, game)
                    pred = self._make_prediction(player, features, game)
                    predictions.append(pred)

            # 3. Get odds and calculate edges
            predictions_with_edges = self._add_odds_and_edges(predictions)

            # 4. Filter for positive edge bets
            bets = [p for p in predictions_with_edges if p['edge'] > 0.03]

            # 5. Calculate bet sizes
            for bet in bets:
                bet['bet_size'] = self.kelly.calculate_bet_size(
                    bet['prob_win'],
                    bet['decimal_odds'],
                    self.bankroll_manager.current_bankroll
                )

            # 6. Generate report
            self._generate_report(bets)

            # 7. Send alerts
            self._send_notifications(bets)

            self.logger.info(f"Pipeline completed: {len(bets)} bets recommended")

        except Exception as e:
            self.logger.error(f"Pipeline failed: {e}")
            self._send_error_alert(e)

    def _fetch_todays_games(self):
        """Fetch today's NBA games"""
        from nba_api.stats.static import teams
        from nba_api.live.nba.endpoints import scoreboard

        board = scoreboard.ScoreBoard()
        games = board.games.get_dict()
        return games

    def _make_prediction(self, player, features, game):
        """Generate prediction with confidence"""
        # Model prediction
        X = pd.DataFrame([features])
        y_pred = self.model.predict(X)[0]

        # Confidence interval
        y_lower, y_upper = self._get_confidence_interval(X)

        # Calibrated probability
        line = self._get_betting_line(player, game)
        prob_over = self.calibrator.predict_proba(y_pred, line)

        return {
            'player': player['name'],
            'game': f"{game['home_team']} vs {game['away_team']}",
            'predicted_PRA': y_pred,
            'confidence_lower': y_lower,
            'confidence_upper': y_upper,
            'line': line,
            'prob_over': prob_over,
            'prob_under': 1 - prob_over
        }

    def _send_notifications(self, bets):
        """Send daily betting recommendations"""
        if not bets:
            return

        # Email report
        subject = f"NBA Props Bets - {datetime.now().strftime('%Y-%m-%d')}"
        body = self._format_report(bets)
        self._send_email(subject, body)

        # Slack notification
        slack_message = f"Found {len(bets)} bets today with average edge {np.mean([b['edge'] for b in bets]):.2%}"
        self._send_slack(slack_message)

def main():
    """Run daily pipeline on schedule"""
    pipeline = DailyPredictionPipeline()

    # Schedule to run daily at 10 AM ET
    schedule.every().day.at("10:00").do(pipeline.run_daily_pipeline)

    print("Daily pipeline scheduled. Running...")
    while True:
        schedule.run_pending()
        time.sleep(60)  # Check every minute

if __name__ == "__main__":
    main()

# src/production/monitoring_dashboard.py
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go

class MonitoringDashboard:
    def __init__(self):
        self.db = self._connect_db()

    def run(self):
        """Run Streamlit dashboard"""
        st.set_page_config(page_title="NBA Props Model Monitor", layout="wide")
        st.title("NBA Props Betting Model - Live Monitor")

        # Tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "Today's Predictions",
            "Performance",
            "Model Health",
            "Bankroll"
        ])

        with tab1:
            self._show_todays_predictions()

        with tab2:
            self._show_performance()

        with tab3:
            self._show_model_health()

        with tab4:
            self._show_bankroll()

    def _show_todays_predictions(self):
        """Display today's bet recommendations"""
        st.header("Today's Recommended Bets")

        predictions = self._get_todays_predictions()

        if predictions.empty:
            st.info("No bets found for today")
            return

        # Filter for positive edge only
        positive_edge = predictions[predictions['edge'] > 0.03]

        st.metric("Total Bets", len(positive_edge))
        st.metric("Average Edge", f"{positive_edge['edge'].mean():.2%}")

        # Display table
        st.dataframe(
            positive_edge[['player', 'game', 'predicted_PRA', 'line',
                          'edge', 'prob_over', 'bet_size', 'book']],
            use_container_width=True
        )

        # Visualization
        fig = px.bar(positive_edge, x='player', y='edge',
                    title='Edge by Player',
                    labels={'edge': 'Betting Edge (%)'})
        st.plotly_chart(fig, use_container_width=True)

    def _show_performance(self):
        """Display historical performance"""
        st.header("Model Performance")

        # Last 30 days
        performance = self._get_performance_last_30_days()

        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Win Rate", f"{performance['win_rate']:.1%}")
        col2.metric("ROI", f"{performance['roi']:.1%}")
        col3.metric("Sharpe Ratio", f"{performance['sharpe']:.2f}")
        col4.metric("Total Bets", performance['total_bets'])

        # Performance over time
        daily_performance = self._get_daily_performance()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=daily_performance['date'],
            y=daily_performance['cumulative_roi'],
            name='Cumulative ROI',
            mode='lines'
        ))
        fig.update_layout(
            title='Cumulative ROI Over Time',
            xaxis_title='Date',
            yaxis_title='ROI (%)'
        )
        st.plotly_chart(fig, use_container_width=True)

    def _show_model_health(self):
        """Monitor model health and data quality"""
        st.header("Model Health")

        # Feature drift detection
        st.subheader("Feature Drift")
        drift = self._calculate_feature_drift()

        if drift['has_drift']:
            st.warning(f"Feature drift detected in: {', '.join(drift['drifted_features'])}")
        else:
            st.success("No significant feature drift detected")

        # Prediction distribution
        st.subheader("Prediction Distribution")
        predictions = self._get_recent_predictions(days=30)

        fig = px.histogram(predictions, x='predicted_PRA',
                          title='Distribution of Predictions (Last 30 Days)',
                          nbins=50)
        st.plotly_chart(fig, use_container_width=True)

    def _show_bankroll(self):
        """Display bankroll and betting history"""
        st.header("Bankroll Management")

        bankroll = self._get_current_bankroll()

        col1, col2, col3 = st.columns(3)
        col1.metric("Current Bankroll", f"${bankroll['current']:,.0f}")
        col2.metric("Starting Bankroll", f"${bankroll['starting']:,.0f}")
        col3.metric("Max Drawdown", f"{bankroll['max_drawdown']:.1%}")

        # Bankroll over time
        history = self._get_bankroll_history()

        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=history['date'],
            y=history['bankroll'],
            name='Bankroll',
            mode='lines',
            fill='tozeroy'
        ))
        fig.update_layout(
            title='Bankroll Over Time',
            xaxis_title='Date',
            yaxis_title='Bankroll ($)'
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    dashboard = MonitoringDashboard()
    dashboard.run()
```

---

## Timeline Summary

| Phase | Weeks | Key Deliverable | Target MAE | Target Win Rate |
|-------|-------|----------------|------------|----------------|
| **Phase 1: Foundation** | 1-4 | Core features implemented | 6-7 pts | 53-54% |
| **Phase 2: Advanced** | 5-8 | Minutes model + usage features | 5-6 pts | 54-55% |
| **Phase 3: Calibration** | 9-10 | Calibrated probabilities + ensemble | 4.5-5.5 pts | 55-57% |
| **Phase 4: Production** | 11-12 | Automated pipeline deployed | <5 pts | 55-58% |

---

## Critical Path Dependencies

```
Week 1 (Infrastructure) → All subsequent weeks
Week 2 (Temporal) → Week 4 (Validation)
Week 3 (Opponent) → Week 4 (Validation)
Week 4 (Validation) → CHECKPOINT → Phase 2
Week 5 (Minutes) → Week 6 (Usage)
Week 8 (Validation) → CHECKPOINT → Phase 3
Week 9 (Calibration) → Week 11 (Kelly)
Week 10 (Ensemble) → CHECKPOINT → Phase 4
Week 11 (Risk) → Week 12 (Production)
```

---

## Time Estimates by Phase

| Phase | Total Hours | Weeks | Hours/Week |
|-------|-------------|-------|------------|
| Phase 1 | 137 | 4 | 34 |
| Phase 2 | 141 | 4 | 35 |
| Phase 3 | 73 | 2 | 37 |
| Phase 4 | 76 | 2 | 38 |
| **TOTAL** | **427** | **12** | **36** |

**Realistic estimate:** 36 hours/week = full-time commitment

---

## Validation Checkpoints

### Checkpoint 1 (End of Week 4):
- **MAE: Must be <7.0 points**
- **Win Rate: 53-54%**
- **Action if fail:** Extend Phase 1, debug features

### Checkpoint 2 (End of Week 8):
- **MAE: Must be <6.0 points**
- **Win Rate: 54-55%**
- **Action if fail:** Revisit feature engineering, remove weak features

### Checkpoint 3 (End of Week 10):
- **MAE: Must be <5.5 points**
- **Win Rate: >54%**
- **Brier Score: <0.20**
- **Action if fail:** Recalibrate, adjust ensemble weights

### Final Checkpoint (End of Week 12):
- **MAE: <5.0 points**
- **Win Rate: 55-58%**
- **ROI: 5-9%**
- **Sharpe: >1.5**
- **Action if fail:** Do NOT deploy to live betting, extend development

---

## Risk Mitigation Strategies

### Risk 1: Features Don't Improve MAE as Expected
**Mitigation:**
- Validate each feature individually before integration
- Use SHAP to ensure features make intuitive sense
- A/B test feature groups (add/remove and measure impact)
- Fallback: Focus on fewer high-impact features

### Risk 2: Overfitting
**Mitigation:**
- Use walk-forward validation (no lookahead)
- Early stopping in XGBoost/LightGBM
- L1/L2 regularization
- Monitor train vs validation gap (should be <1 point MAE)
- Test on multiple seasons (2021-22, 2022-23, 2023-24, 2024-25)

### Risk 3: Live Performance Lower Than Validation
**Mitigation:**
- 3-month paper trading before real money
- Conservative bet sizing (fractional Kelly at 25%)
- Strict edge thresholds (3-5%)
- Monitor CLV (should stay >70%)
- Circuit breaker: Stop trading if 20% drawdown

### Risk 4: Data Quality Issues
**Mitigation:**
- Automated data quality checks daily
- Alert on missing data or outliers
- Fallback to previous day's data if API fails
- Manual review of predictions before betting
- Graceful degradation (skip bets if data incomplete)

### Risk 5: API Rate Limits or Failures
**Mitigation:**
- Cache all data (feature_cache system)
- Respect rate limits (delays between requests)
- Multiple data sources (nba_api, TheOddsAPI, CTG)
- Retry logic with exponential backoff
- Email alerts on API failures

### Risk 6: Model Drift Over Time
**Mitigation:**
- Monthly retraining on new data
- Feature drift detection (compare distributions)
- Performance monitoring (alert if win rate <52%)
- Adaptive learning: Retrain if performance degrades
- Version control for models (rollback if needed)

---

## Required Tools & Infrastructure

### Development Tools
- **Package Manager:** uv (already installed)
- **ML Libraries:** `uv add xgboost lightgbm scikit-learn optuna`
- **Feature Engineering:** pandas, numpy
- **Visualization:** matplotlib, seaborn, plotly
- **Experiment Tracking:** `uv add mlflow`
- **Calibration:** `uv add netcal`

### Data Sources
- **NBA Data:** nba_api (already used)
- **Odds Data:** TheOddsAPI (requires paid subscription ~$50-100/month)
- **Advanced Stats:** CleaningTheGlass.com (already have 614/660 files)

### Production Infrastructure
- **Pipeline:** Python scripts + cron jobs
- **Database:** SQLite (local) or PostgreSQL (cloud)
- **Dashboard:** Streamlit (deploy on Streamlit Cloud or local)
- **Alerts:** Twilio (SMS), SendGrid (email), Slack webhooks
- **Hosting:** Local machine or AWS EC2 (t3.medium ~$30/month)

### Monitoring & Alerts
- **Model Monitoring:** Custom dashboard (Streamlit)
- **Error Tracking:** Python logging + email alerts
- **Performance Tracking:** SQLite database with daily aggregation

### Estimated Monthly Costs
- TheOddsAPI: $50-100
- AWS (if used): $30
- **Total: ~$100/month**

---

## Success Metrics

### Model Performance
- **MAE:** <5 points (current: 9.92)
- **Win Rate:** 55-58% (current: 51.98%)
- **ROI:** 5-9% (current: 0.91%)
- **Brier Score:** <0.20 (calibration)
- **CLV:** >70% (maintain, current: 73.7%)

### Risk-Adjusted Performance
- **Sharpe Ratio:** >1.5 (return / volatility)
- **Max Drawdown:** <20% (stop trading if hit)
- **Win Rate Consistency:** Std dev <5% across months

### Operational Metrics
- **Pipeline Uptime:** >99% (run daily without failures)
- **Data Quality:** >95% complete features
- **Alert Response:** <1 hour to address anomalies
- **Paper Trading:** 3 months with no major issues

---

## Post-Development: 3-Month Live Testing Plan

### Month 1: Paper Trading
- Run full pipeline daily
- Record hypothetical bets
- Compare predictions to actuals
- Monitor all metrics
- No real money

### Month 2: Small Stakes
- Start with 10% of bankroll ($1,000 if $10k total)
- Bet minimum sizes ($10-20 per bet)
- Full Kelly criterion active
- Daily performance review

### Month 3: Scale to Target
- If Month 1-2 successful (ROI >5%, Win Rate >55%):
  - Scale to full bankroll
  - Standard bet sizes (Kelly criterion)
  - Full automation
- If not successful:
  - Extend paper trading
  - Debug issues
  - Retrain model

### Go/No-Go Decision Criteria
**GO (proceed to full betting):**
- Win rate >55% over 3 months
- ROI >5% over 3 months
- Sharpe ratio >1.5
- Max drawdown <10%
- CLV remains >70%
- No systematic biases detected

**NO-GO (extend testing or pause):**
- Any of the above metrics not met
- Unexplained performance degradation
- Data quality issues
- Model drift detected

---

## Conclusion

This 12-week plan provides a systematic, rigorous path from your current 51.98% win rate to a profitable 55-58% win rate. The key success factors are:

1. **Disciplined execution:** Follow the plan sequentially, don't skip steps
2. **Validation rigor:** Walk-forward validation with no temporal leakage
3. **Feature engineering:** Focus on high-impact features (minutes, opponent defense, usage)
4. **Proper calibration:** Convert accuracy into winning bets
5. **Risk management:** Conservative bet sizing, strict edge thresholds

With a 70% success probability and your current elite CLV (73.7%), this model has strong potential to become profitable. The infrastructure you've built (614 player files, 270 team files) provides the data foundation needed for success.

**Next immediate steps:**
1. Review this plan and adjust timeline if needed
2. Set up development environment (MLflow, feature cache)
3. Start Week 1 tasks (fix validation, add efficiency stats)
4. Track progress weekly and adjust as needed

Good luck! This is a professional-grade development plan that, if executed properly, should produce a profitable betting model.
