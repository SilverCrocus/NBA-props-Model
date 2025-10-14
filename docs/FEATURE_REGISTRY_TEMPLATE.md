# Feature Registry

**Purpose**: Central documentation for all features used in the NBA Props Model.

**Why This Matters**: Features are the foundation of the model. Proper documentation ensures:
- No temporal leakage
- Reproducibility
- Debugging capability
- Knowledge transfer

---

## How to Use This Template

When adding a new feature:

1. Copy the template section below
2. Fill in all required fields
3. Run correlation analysis
4. Add tests to verify correctness
5. Update this registry

---

## Feature Template

### [Feature Name]

**Status**: [Active / Experimental / Deprecated]

**Definition**:
[Clear mathematical or logical definition]

**Formula**:
```python
# Exact implementation or pseudocode
feature = formula_here
```

**Source**:
- Data source: [e.g., NBA API game logs, CleaningTheGlass.com]
- Update frequency: [e.g., Daily, Season-level]
- Data availability: [Date range, missing data patterns]

**Type**:
[Core Performance / Contextual Modulator / Temporal Dynamic]

**Rationale**:
[Why we believe this feature will improve predictions]

**Implementation**:
- Module: `src/features/[module_name].py`
- Function: `[function_name]`
- Tests: `tests/unit/test_features/test_[module_name].py`

**Leakage Risk**: [HIGH / MEDIUM / LOW]
[Explanation of why and how leakage is prevented]

**Performance**:
- Correlation with PRA: [e.g., 0.82 (training set), 0.78 (validation set)]
- Feature importance: [e.g., 0.12 in XGBoost model]
- Incremental MAE improvement: [e.g., -0.3 MAE when added to baseline]

**Edge Cases**:
- Missing data: [How handled, e.g., "Use league average"]
- First N games: [Behavior for new players or season start]
- Traded players: [How handled across team changes]

**Dependencies**:
- Required features: [List any features this depends on]
- Required data: [List data files/sources needed]

**Validation**:
- [x] Unit tests pass
- [x] No temporal leakage detected
- [x] Correlation measured
- [x] Feature importance tracked

**Notes**:
[Any additional context, research references, or observations]

**Change Log**:
- 2024-10-14: Initial implementation
- [Date]: [Change description]

---

## Example: CTG Usage Rate

### CTG_USG (Usage Rate)

**Status**: Active

**Definition**:
Percentage of team plays used by a player while on the court. Measures offensive involvement.

**Formula**:
```python
CTG_USG = (FGA + 0.44 * FTA + TOV) / (Team Possessions While Player On Court)
```
Source: CleaningTheGlass.com's proprietary calculation

**Source**:
- Data source: CleaningTheGlass.com premium data
- Update frequency: Season-level (updated throughout season)
- Data availability: 2003-04 season to present
- Missing data: ~7% of players (low minutes players filtered by CTG)

**Type**: Core Performance

**Rationale**:
Usage rate is the strongest predictor of counting stats. Players with high usage get more
opportunities to score, rebound, and assist. Research shows 0.80+ correlation with PRA.

**Implementation**:
- Module: `src/features/core.py`
- Function: `get_ctg_usage_rate(player_name, season)`
- Tests: `tests/unit/test_features/test_ctg_features.py`

**Leakage Risk**: LOW

Leakage prevention:
1. Season-level stat (not game-level) - provides context but no future info
2. For 2024-25 predictions, we use 2023-24 USG% (prior season)
3. Never use current season USG% to predict current season games

**Performance**:
- Correlation with PRA: 0.82 (training), 0.80 (validation)
- Feature importance: 0.18 (highest in XGBoost model)
- Incremental MAE improvement: -1.2 MAE vs. baseline without CTG features

**Edge Cases**:
- Missing data: Use league average (0.20) if player not found
- Rookies: No prior season data, use college usage if available, else league avg
- Traded players: Use weighted average based on games played with each team

**Dependencies**:
- Required data: `data/ctg_data_organized/players/{season}/regular_season/offensive_overview/offensive_overview.csv`
- Required features: None (standalone feature)

**Validation**:
- [x] Unit tests pass (`test_ctg_usage_rate_loading`)
- [x] No temporal leakage (uses prior season only)
- [x] Correlation measured (0.82 on training set)
- [x] Feature importance tracked (0.18 in prod model)

**Notes**:
- CTG's USG% calculation differs slightly from NBA.com (includes more play types)
- Strong predictor but can be misleading for very low minute players
- Consider interaction with pace (high usage + fast pace = more opportunities)

**References**:
- CleaningTheGlass glossary: https://cleaningtheglass.com/stats/guide/glossary
- Research: [Internal analysis doc: `docs/research/ctg_feature_analysis.md`]

**Change Log**:
- 2024-09-15: Initial implementation
- 2024-10-01: Added fallback to league average for missing players

---

## Example: Last 5 Games PRA Average

### PRA_L5_mean (Rolling 5-Game PRA Average)

**Status**: Active

**Definition**:
Average PRA over the player's last 5 games, excluding the current game.

**Formula**:
```python
PRA_L5_mean = mean(PRA for last 5 games) with .shift(1) to exclude current game

# Implementation:
df['PRA_L5_mean'] = (
    df.groupby('PLAYER_ID')['PRA']
    .shift(1)  # CRITICAL: Exclude current game
    .rolling(window=5, min_periods=1)
    .mean()
    .reset_index(level=0, drop=True)
)
```

**Source**:
- Data source: NBA API game logs
- Update frequency: Daily (after each game)
- Data availability: Complete for all games
- Missing data: None (calculated from existing game logs)

**Type**: Temporal Dynamic

**Rationale**:
Recent performance is highly predictive of near-term performance. L5 average captures
"current form" while smoothing out single-game variance.

**Implementation**:
- Module: `src/features/temporal.py`
- Function: `create_rolling_features(df, stats=['PRA'], windows=[5])`
- Tests: `tests/unit/test_features/test_temporal.py`

**Leakage Risk**: LOW (if implemented correctly)

Leakage prevention:
1. Uses `.shift(1)` to exclude current game from average
2. First 5 games have NaN or fewer games in average (min_periods=1)
3. Unit tests verify no leakage (`test_rolling_avg_excludes_current_game`)

**Performance**:
- Correlation with PRA: 0.65 (training), 0.62 (validation)
- Feature importance: 0.12 (2nd highest in XGBoost model)
- Incremental MAE improvement: -0.8 MAE when added to CTG features

**Edge Cases**:
- First game of season: NaN (no previous games)
- Games 2-4: Average of available games (min_periods=1)
- Game 5+: Average of last 5 games
- Injury returns: May have large gap in games (still uses last 5)

**Dependencies**:
- Required features: PRA (calculated from PTS + REB + AST)
- Required data: Game logs with PLAYER_ID, GAME_DATE, PRA

**Validation**:
- [x] Unit tests pass (`test_lag1_excludes_current_game`)
- [x] No temporal leakage (shift verified)
- [x] Correlation measured (0.65)
- [x] Feature importance tracked (0.12)

**Notes**:
- L5 window chosen empirically (L3 too noisy, L10 too slow to adapt)
- Consider EWMA as alternative (gives more weight to recent games)
- High correlation with actual PRA, but careful not to overfit to recent performance

**Change Log**:
- 2024-09-15: Initial implementation
- 2024-09-20: Fixed leakage bug (added .shift(1))
- 2024-10-05: Changed min_periods from 3 to 1 to get predictions earlier in season

---

## Quick Reference: All Active Features

| Feature | Type | Correlation | Importance | Leakage Risk | Status |
|---------|------|-------------|------------|--------------|--------|
| CTG_USG | Core | 0.82 | 0.18 | LOW | Active |
| PRA_L5_mean | Temporal | 0.65 | 0.12 | LOW | Active |
| CTG_PSA | Core | 0.75 | 0.10 | LOW | Active |
| MIN_L10_mean | Temporal | 0.58 | 0.09 | LOW | Active |
| days_rest | Contextual | 0.15 | 0.05 | LOW | Active |
| is_b2b | Contextual | -0.12 | 0.04 | LOW | Active |
| CTG_eFG | Core | 0.45 | 0.08 | LOW | Active |
| PRA_L5_std | Temporal | -0.08 | 0.03 | LOW | Active |
| opp_def_rating | Contextual | -0.18 | 0.06 | MEDIUM | Active |
| [Add more...] | | | | | |

---

## Feature Categories

### Tier 1: Core Performance (Season-Level Context)
Features that capture a player's fundamental abilities and role.

**Characteristics:**
- Season-level statistics
- Low leakage risk (use prior season)
- High correlation with PRA
- Stable over short periods

**Examples:**
- CTG_USG: Usage rate
- CTG_PSA: Points per shot attempt
- CTG_AST_PCT: Assist percentage
- CTG_REB_PCT: Rebounding percentage

### Tier 2: Temporal Dynamics (Recent Performance)
Features that capture current form and trends.

**Characteristics:**
- Game-level rolling averages
- CRITICAL: Must use .shift(1) to prevent leakage
- Captures hot/cold streaks
- Adapts to changing roles

**Examples:**
- PRA_L5_mean: 5-game rolling average
- PRA_L10_std: 10-game volatility
- PRA_ewma5: Exponentially weighted average
- PRA_trend: (L5 - L20) / L20 performance trend

### Tier 3: Contextual Modulators (Situational)
Features that adjust predictions based on circumstances.

**Characteristics:**
- Game-specific context
- Moderate correlation
- Interaction effects with other features
- Higher leakage risk (be careful!)

**Examples:**
- days_rest: Days since last game
- is_b2b: Back-to-back game indicator
- IS_HOME: Home/away indicator
- opp_def_rating: Opponent defensive strength
- pace: Game pace factor

---

## Feature Engineering Best Practices

### 1. Always Document Leakage Prevention
```python
# GOOD: Clear documentation
def create_lag_feature(df, stat, lag=1):
    """
    Create lag feature.

    CRITICAL: Uses .shift() to prevent temporal leakage.
    lag=1 means previous game, NOT current game.
    """
    return df.groupby('PLAYER_ID')[stat].shift(lag)


# BAD: No documentation
def create_lag_feature(df, stat, lag=1):
    return df.groupby('PLAYER_ID')[stat].shift(lag)
```

### 2. Test Feature Correlation
```python
# Always measure correlation before adding to model
feature_corr = df['new_feature'].corr(df['PRA'])
print(f"Correlation with PRA: {feature_corr:.3f}")

# Flag suspiciously high correlation (possible leakage)
if abs(feature_corr) > 0.95:
    logger.warning(f"Very high correlation ({feature_corr:.3f}) - check for leakage!")
```

### 3. Handle Missing Data Explicitly
```python
# GOOD: Explicit handling with documentation
def get_feature(player, season):
    """Get feature with fallback to league average."""
    try:
        return load_feature(player, season)
    except FileNotFoundError:
        logger.warning(f"Feature not found, using league average")
        return LEAGUE_AVERAGE


# BAD: Silent failure
def get_feature(player, season):
    try:
        return load_feature(player, season)
    except:
        return 0  # Why 0? What does this mean?
```

### 4. Version Features
When changing feature calculation:
1. Create new feature with version suffix (e.g., `PRA_L5_mean_v2`)
2. Run both versions in parallel for comparison
3. Deprecate old version after validation
4. Update feature registry with change log

---

## Feature Review Process

Before adding a feature to production:

- [ ] Feature documented in registry
- [ ] Leakage risk assessed
- [ ] Unit tests written and passing
- [ ] Correlation measured on train/val sets
- [ ] Feature importance tracked
- [ ] Edge cases identified and handled
- [ ] Code reviewed
- [ ] Incremental performance improvement validated

---

## Deprecated Features

Document deprecated features for historical reference.

### [Deprecated Feature Name]

**Deprecated Date**: YYYY-MM-DD

**Reason for Deprecation**:
[Why feature was removed]

**Replacement**:
[What feature replaced it, if any]

**Performance**:
[Historical performance metrics]

---

## Research Pipeline

Features currently under research (not in production):

### [Experimental Feature Name]

**Status**: Experimental

**Hypothesis**: [What we're testing]

**Preliminary Results**: [Early findings]

**Next Steps**: [What needs to be done before production]

---

## Maintenance Schedule

**Weekly**:
- Review feature importance in latest model
- Check for data quality issues
- Update correlations if significant drift

**Monthly**:
- Analyze feature stability over time
- Review deprecated features for cleanup
- Research new feature opportunities

**Seasonally**:
- Comprehensive feature audit
- Update league averages
- Revalidate all features on new season data
