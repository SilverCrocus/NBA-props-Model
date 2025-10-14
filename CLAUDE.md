# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

NBA Props Model - Machine learning system for NBA player prop predictions (PRA - Points + Rebounds + Assists). Uses premium CleaningTheGlass.com analytics combined with NBA API game logs for prediction.

**Current Status (October 2025):**
- Model performance: 52% win rate, 0.91% ROI (walk-forward validated on 2024-25)
- MAE: 8.83 points (target: <5 points)
- Phase: Feature engineering improvement to reduce MAE

## Package Management

This project uses `uv` for package management:
```bash
# Install dependencies
uv sync

# Run Python scripts
uv run <script.py>

# Add new packages
uv add <package-name>
```

## Common Development Commands

### Data Collection
```bash
# DEPRECATED: Data collection complete (614/660 player files, 270/270 team files)
# Legacy scrapers kept for reference only
```

### Model Training & Validation
```bash
# Run walk-forward validation (proper temporal isolation)
uv run walk_forward_validation_enhanced.py

# Run backtest with betting simulation
uv run backtest_walkforward_2024_25.py

# Build training dataset from game logs + CTG stats
uv run build_2024_25_dataset.py
```

### Data Analysis
```bash
# Analyze data quality
uv run analyze_data_quality.py

# Feature inventory
uv run feature_inventory.py

# Validate dataset integrity
uv run validate_dataset.py
```

### Testing
```bash
# Run tests
uv run pytest tests/

# Run specific test file
uv run pytest tests/unit/test_features.py
```

## Critical Architecture Concepts

### Walk-Forward Validation (MANDATORY for Time Series)

**The Problem:** Standard train/test splits leak future information in time series data.

**The Solution:** Walk-forward validation predicts each day using only past data.

**Implementation Pattern:**
```python
# For each prediction date:
for pred_date in unique_dates:
    # Games to predict TODAY
    games_today = df[df['GAME_DATE'] == pred_date]

    # Historical data BEFORE today (for features)
    past_games = df[df['GAME_DATE'] < pred_date]

    # Calculate features using ONLY past_games
    features = calculate_lag_features(past_games)  # Uses .shift(1)

    # Make prediction
    prediction = model.predict(features)
```

**Files Using This Pattern:**
- `walk_forward_validation_enhanced.py` - Enhanced with CTG stats
- `walk_forward_validation_2024_25.py` - Baseline implementation
- `src/data/game_log_builder.py` - Feature calculation with .shift(1)

### Data Processing Pipeline

```
1. Raw Game Logs (561K games, 2003-2024)
   ↓ [GameLogDatasetBuilder]
   ↓
2. CTG Season Stats (614 files, premium analytics)
   ↓ [Merge with deduplication - CRITICAL FIX]
   ↓
3. Temporal Features (lag, rolling, EWMA)
   ↓ [ALL use .shift(1) to prevent leakage]
   ↓
4. Contextual Features (opponent, rest, schedule)
   ↓
5. Training Dataset (game-level with features)
   ↓
6. XGBoost Model Training
   ↓
7. Walk-Forward Validation → Backtest → Betting Simulation
```

### Feature Engineering - Three-Tier Architecture

**Tier 1: Core Performance (Player Baseline)**
- Usage Rate (USG%), True Shooting % (TS%)
- Points per Shot Attempt (PSA)
- Assist Rate (AST%), Rebounding % (REB%)
- Source: CTG season stats merged to each game

**Tier 2: Contextual Modulators (Game-Specific)**
- Minutes projection (L5 average)
- Opponent defensive rating (DRtg)
- Rest days, back-to-backs
- Pace differentials

**Tier 3: Temporal Dynamics (Recent Form)**
- Lag features (PRA_lag1, PRA_lag3, etc.)
- Rolling averages (L5, L10, L20)
- EWMA (exponentially weighted moving average)
- Trend detection (L5 vs L20)

**Implementation:**
- `src/features/engineering.py` - NBAFeatureEngineer class
- `src/data/game_log_builder.py` - GameLogDatasetBuilder class
- `utils/ctg_feature_builder.py` - CTGFeatureBuilder class

### Critical Bug Fixes (DO NOT REVERT)

**CTG Duplicate Bug (Fixed Oct 2025):**
- Problem: CTG merge created 8 duplicate rows per player-game
- Fix: 3-level deduplication in `game_log_builder.py:180-205`
  1. Dedupe CTG categories before merge
  2. Dedupe CTG combined before merge to game logs
  3. Final safety check after all features added
- Verification: Check `len(df)` before/after merge operations

**Temporal Leakage (Fixed Oct 2025):**
- Problem: Lag features used future games
- Fix: All rolling/lag features use `.shift(1)` before calculation
- See: `game_log_builder.py:254`, `game_log_builder.py:292`
- Test: Predictions should only use data from `GAME_DATE < pred_date`

## Key Data Files & Locations

### Input Data
```
data/game_logs/
  └── all_game_logs_combined.csv        # 561K games, 2003-2024 (PTS, REB, AST, etc.)

data/ctg_data_organized/players/
  ├── 2024-25/regular_season/
  │   ├── offensive_overview/           # USG%, PSA, AST%, TOV%
  │   ├── shooting_accuracy/            # eFG%, shooting splits
  │   ├── defense_rebounding/           # REB%, defensive stats
  │   └── on_off/                       # On/off court impact
  └── [21 more seasons]

data/ctg_team_data/
  └── [270 team stat files]
```

### Processed Data
```
data/processed/
  ├── train.parquet                     # Training set (2003-2023)
  ├── val.parquet                       # Validation set (2023-24)
  └── game_level_training_data.parquet  # Full dataset with features
```

### Results
```
data/results/
  ├── walkforward_predictions_2024-25.csv      # Walk-forward predictions
  └── backtest_walkforward_2024_25.csv         # Betting simulation results
```

## Important Implementation Notes

### Working with CTG Data

CTG stats are **season-level** (one row per player per season), not game-level:
```python
# Correct: Merge CTG as context for each game
ctg_stats = load_ctg_season_stats("2024-25", "regular_season")
game_logs = game_logs.merge(
    ctg_stats,
    left_on=['PLAYER_NAME', 'SEASON', 'SEASON_TYPE'],
    right_on=['Player', 'SEASON', 'SEASON_TYPE'],
    how='left'
)

# MUST deduplicate CTG before merge (see game_log_builder.py:180-205)
```

### Calculating Temporal Features

ALWAYS use `.shift(1)` to prevent data leakage:
```python
# Lag features
df['PRA_lag1'] = df.groupby('PLAYER_ID')['PRA'].shift(1)

# Rolling averages
df['PRA_L10_mean'] = (
    df.groupby('PLAYER_ID')['PRA']
    .shift(1)  # Exclude current game
    .rolling(window=10, min_periods=1)
    .mean()
)

# EWMA
df['PRA_ewma5'] = (
    df.groupby('PLAYER_ID')['PRA']
    .shift(1)  # Exclude current game
    .ewm(span=5, min_periods=1)
    .mean()
)
```

### Train/Test Splitting

NEVER use random splits for time series:
```python
# Correct: Chronological split
train = df[df['GAME_DATE'] <= '2023-06-30']
val = df[(df['GAME_DATE'] > '2023-06-30') & (df['GAME_DATE'] <= '2024-06-30')]
test = df[df['GAME_DATE'] > '2024-06-30']

# Wrong: DO NOT USE
X_train, X_test = train_test_split(X, y, test_size=0.2)  # ❌ LEAKS DATA
```

## Current Development Focus

**Priority 1: Reduce MAE from 8.83 to <5 points**

Missing features causing high MAE:
1. True Shooting % (TS%) - game-level calculation
2. CTG advanced stats in walk-forward (USG%, AST%, REB%)
3. Opponent defensive features (DRtg, pace)
4. L3 recent form features (strongest temporal signal)

See: `NEXT_STEPS.md` and `FEATURE_ENGINEERING_RECOMMENDATIONS.md`

**Priority 2: Model Calibration**

Issue: Large edges (10+ pts) underperform vs small edges (3-5 pts)
Fix: Implement isotonic regression or Platt scaling

## Validation Standards

When making changes to prediction pipeline:

1. **Verify no temporal leakage:** Predictions only use `past_games`
2. **Check for duplicates:** Count rows before/after merge operations
3. **Validate MAE:** Should be 7-9 points (baseline) or better
4. **Test on 2024-25:** Walk-forward validation on out-of-sample data
5. **Backtest betting:** Win rate should be 52-58% (not 99%)

## Documentation References

- `FINAL_VALIDATION_REPORT.md` - True model performance (52% win rate)
- `NEXT_STEPS.md` - Feature improvement roadmap
- `FEATURE_ENGINEERING_RECOMMENDATIONS.md` - Research-backed feature recommendations
- `TEMPORAL_LEAKAGE_PROOF.md` - Proof of no leakage in walk-forward
- `docs/features_plan.md` - Three-tier feature architecture

## Notes for Future Development

- Data collection is complete (93% CTG coverage)
- Scrapers are legacy code (keep for reference)
- Focus is on feature engineering and model improvement
- Target: 55%+ win rate, 5-10% ROI, <5 points MAE
- Timeline to production: 3-6 months (as of Oct 2025)
