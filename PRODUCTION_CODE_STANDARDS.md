# Production Code Standards for NBA Props Model

**Project Context**: 12-week development timeline | Real money betting | Team of 1-2 developers

**Philosophy**: Balance speed with quality. We need production-ready code that won't blow up with real money on the line, but we can't spend 4 weeks building testing infrastructure.

---

## Table of Contents

1. [Code Organization](#1-code-organization)
2. [Testing Strategy](#2-testing-strategy)
3. [Code Quality Standards](#3-code-quality-standards)
4. [Error Handling](#4-error-handling)
5. [Documentation Requirements](#5-documentation-requirements)
6. [Performance Guidelines](#6-performance-guidelines)
7. [Production Readiness Checklist](#7-production-readiness-checklist)

---

## 1. Code Organization

### Current State Analysis

**Issues Identified:**
- 3 competing feature systems: `game_log_builder.py`, `src/features/engineering.py`, `ctg_feature_builder.py`
- Root-level scripts mixed with organized `src/` structure
- No clear separation between research code and production code

### Recommended Structure

```
nba_props_model/
├── src/                          # Production code only
│   ├── data/
│   │   ├── __init__.py
│   │   ├── loaders.py           # Data loading utilities
│   │   ├── game_log_builder.py  # KEEP (Primary dataset builder)
│   │   └── validation.py        # Data quality checks
│   ├── features/
│   │   ├── __init__.py
│   │   ├── core.py              # Core CTG features (USG%, PSA, etc.)
│   │   ├── temporal.py          # Lag, rolling, EWMA features
│   │   ├── contextual.py        # Rest, opponent, situational
│   │   └── builder.py           # Feature pipeline orchestrator
│   ├── models/
│   │   ├── __init__.py
│   │   ├── xgboost_model.py     # XGBoost implementation
│   │   ├── lightgbm_model.py    # LightGBM implementation
│   │   ├── ensemble.py          # Ensemble methods
│   │   └── calibration.py       # Probability calibration
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── walk_forward.py      # Walk-forward validation
│   │   ├── metrics.py           # Custom metrics (MAE, edge, ROI)
│   │   └── leakage_detection.py # Temporal leakage tests
│   ├── betting/
│   │   ├── __init__.py
│   │   ├── edge_calculator.py   # Edge calculation & Kelly sizing
│   │   ├── kelly.py             # Position sizing
│   │   └── risk_management.py   # Risk limits & safeguards
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── logging_config.py    # Centralized logging
│   │   ├── config.py            # Configuration management
│   │   └── validators.py        # Input validation
│   └── pipelines/
│       ├── __init__.py
│       ├── training_pipeline.py  # End-to-end training
│       └── prediction_pipeline.py # Production predictions
├── scripts/                      # One-off scripts & data collection
│   ├── collect_odds.py
│   ├── build_training_data.py
│   └── backtest_historical.py
├── experiments/                  # Research notebooks & analysis
│   ├── eda/
│   ├── feature_research/
│   └── model_tuning/
├── tests/
│   ├── unit/                     # Fast unit tests (<1s each)
│   ├── integration/              # Integration tests (<10s each)
│   └── system/                   # End-to-end tests (<60s)
├── configs/                      # Configuration files
│   ├── model_config.yaml
│   ├── features_config.yaml
│   └── production.yaml
├── data/                         # Data storage (gitignored)
└── docs/                         # Technical documentation
```

### Refactoring Strategy

**Phase 1 (Week 1-2): Foundation**
- Keep `game_log_builder.py` as primary dataset builder
- Keep `ctg_feature_builder.py` in utils/ for now
- Move all backtest/analysis scripts to `experiments/` or `scripts/`

**Phase 2 (Week 3-4): Consolidation**
- Extract feature logic from `game_log_builder.py` into `src/features/` modules
- Create unified feature pipeline in `src/features/builder.py`
- Deprecate old feature engineering scripts

**Phase 3 (Week 5-6): Production Ready**
- Build production prediction pipeline
- Implement edge calculation and Kelly sizing
- Add comprehensive logging and monitoring

**Incremental Refactoring Rules:**
1. Never refactor without tests
2. Refactor only when adding new features
3. Maintain backward compatibility during transition
4. Use deprecation warnings for old interfaces

---

## 2. Testing Strategy

### Pragmatic Testing Philosophy

**Goal**: 60-70% coverage of critical paths, not 100% coverage everywhere.

**Priority Tiers:**

#### Tier 1: MUST TEST (Critical Path)
- **Temporal leakage detection**: Any feature using future data breaks the model
- **Data quality checks**: Missing data, duplicates, invalid values
- **Feature calculation correctness**: CTG features, rolling averages, lags
- **Walk-forward validation logic**: Ensure proper train/test splits
- **Edge calculation**: Betting logic must be bulletproof
- **Kelly sizing**: Position sizing errors = bankruptcy

#### Tier 2: SHOULD TEST (High Value)
- Model prediction logic
- Feature pipeline integration
- Data loading and merging
- Configuration validation

#### Tier 3: NICE TO TEST (Lower Priority)
- Utility functions
- Logging helpers
- Formatting functions

### Test Coverage Targets

- **Critical modules**: 85%+ coverage (features, evaluation, betting)
- **Core modules**: 70%+ coverage (data, models, pipelines)
- **Utilities**: 50%+ coverage (utils, scripts)
- **Overall project**: 65%+ coverage

### Test Types & Time Budgets

**Unit Tests** (Fast, <1 second each)
- Test individual functions in isolation
- Mock external dependencies
- Run on every commit (pre-commit hook)

**Integration Tests** (Medium, <10 seconds each)
- Test module interactions
- Use small test datasets
- Run before push

**System Tests** (Slow, <60 seconds total)
- End-to-end pipeline tests
- Run nightly or before releases
- Use realistic but small datasets

### Example Test Structure

```python
# tests/unit/test_features/test_temporal.py
import pytest
import pandas as pd
import numpy as np
from src.features.temporal import create_lag_features

class TestLagFeatures:
    """Test lag feature creation for temporal leakage prevention."""

    @pytest.fixture
    def sample_game_logs(self):
        """Create sample game logs for testing."""
        return pd.DataFrame({
            'PLAYER_ID': [1, 1, 1, 2, 2, 2],
            'GAME_DATE': pd.date_range('2024-01-01', periods=6),
            'PRA': [25, 30, 28, 22, 26, 24],
            'MIN': [32, 35, 33, 28, 30, 29]
        })

    def test_lag1_uses_previous_game_only(self, sample_game_logs):
        """Critical: lag=1 must use previous game, not current."""
        result = create_lag_features(sample_game_logs, stats=['PRA'], lags=[1])

        # First game should have NaN (no previous game)
        assert pd.isna(result.loc[0, 'PRA_lag1'])

        # Second game should have first game's PRA
        assert result.loc[1, 'PRA_lag1'] == 25

        # Third game should have second game's PRA
        assert result.loc[2, 'PRA_lag1'] == 30

    def test_no_data_leakage_across_players(self, sample_game_logs):
        """Critical: lag features must not leak across players."""
        result = create_lag_features(sample_game_logs, stats=['PRA'], lags=[1])

        # Player 2's first game (index 3) should have NaN, not Player 1's data
        assert pd.isna(result.loc[3, 'PRA_lag1'])

    def test_rolling_excludes_current_game(self, sample_game_logs):
        """Critical: rolling averages must shift(1) to exclude current game."""
        from src.features.temporal import create_rolling_features

        result = create_rolling_features(
            sample_game_logs,
            stats=['PRA'],
            windows=[2]
        )

        # Check that current game is not included in rolling avg
        # This is hard to test directly, but we can verify the shift
        # by ensuring first value is NaN
        assert pd.isna(result.loc[0, 'PRA_L2_mean'])


# tests/integration/test_dataset_building.py
import pytest
from src.data.game_log_builder import GameLogDatasetBuilder

class TestDatasetBuilding:
    """Integration tests for complete dataset building."""

    @pytest.fixture
    def small_test_data(self):
        """Use 2024-25 season only for fast tests."""
        return "tests/fixtures/small_game_logs.csv"

    def test_complete_pipeline_no_duplicates(self, small_test_data):
        """Ensure dataset building produces unique player-game combinations."""
        builder = GameLogDatasetBuilder(game_logs_path=small_test_data)

        df = builder.build_complete_dataset(merge_ctg=False)

        # Check for duplicates
        duplicates = df.duplicated(subset=['PLAYER_ID', 'GAME_DATE']).sum()
        assert duplicates == 0, f"Found {duplicates} duplicate player-game combinations"

    def test_temporal_ordering_preserved(self, small_test_data):
        """Ensure data is sorted by player and date."""
        builder = GameLogDatasetBuilder(game_logs_path=small_test_data)
        df = builder.build_complete_dataset(merge_ctg=False)

        # Check sorting within each player
        for player_id in df['PLAYER_ID'].unique()[:5]:  # Check first 5 players
            player_df = df[df['PLAYER_ID'] == player_id]
            assert player_df['GAME_DATE'].is_monotonic_increasing


# tests/system/test_walk_forward_validation.py
import pytest
from src.evaluation.walk_forward import WalkForwardValidator
from src.models.xgboost_model import XGBoostModel

class TestWalkForwardValidation:
    """System test for walk-forward validation."""

    def test_walk_forward_no_leakage(self):
        """
        Critical: Ensure walk-forward validation never trains on future data.
        This is a slow test but absolutely critical for production.
        """
        validator = WalkForwardValidator(
            train_start='2023-01-01',
            train_end='2023-12-31',
            val_start='2024-01-01',
            val_end='2024-03-31',
            window_size=60  # 60-day windows
        )

        model = XGBoostModel()

        results = validator.validate(model, check_leakage=True)

        # Should complete without raising TemporalLeakageError
        assert results['mae'] > 0  # Has valid results
        assert results['leakage_detected'] == False
```

### Critical Tests to Implement First (Week 1)

1. **Temporal leakage test**: `tests/unit/test_features/test_temporal_leakage.py`
2. **Feature correctness test**: `tests/unit/test_features/test_ctg_features.py`
3. **Data quality test**: `tests/unit/test_data/test_validation.py`
4. **Walk-forward test**: `tests/integration/test_walk_forward.py`
5. **Edge calculation test**: `tests/unit/test_betting/test_edge_calculator.py`

### Test Data Fixtures

Create small, realistic test datasets in `tests/fixtures/`:
- `small_game_logs.csv` (1000 rows, 3 players, 1 season)
- `small_ctg_stats.csv` (50 players)
- `test_odds.csv` (100 props with real odds)

---

## 3. Code Quality Standards

### Type Hints

**Gradual Adoption Strategy:**
- **Phase 1 (Week 1-3)**: Type hints for all new functions
- **Phase 2 (Week 4-6)**: Add types to critical functions (features, evaluation, betting)
- **Phase 3 (Week 7+)**: Fill in remaining gaps

**Where to use type hints:**
```python
from typing import Dict, List, Optional, Tuple, Union
import pandas as pd
import numpy as np
from numpy.typing import NDArray

# GOOD: Clear function signature
def create_lag_features(
    df: pd.DataFrame,
    stats: List[str],
    lags: List[int]
) -> pd.DataFrame:
    """Create lag features from game logs."""
    pass

# GOOD: Type hints for return values
def calculate_edge(
    prediction: float,
    odds: float,
    threshold: float = 0.03
) -> Tuple[bool, float]:
    """
    Calculate betting edge.

    Returns:
        (should_bet, edge_amount)
    """
    pass

# ACCEPTABLE: Complex types can use Union
def load_config(path: str) -> Dict[str, Union[int, float, str, List]]:
    pass

# DON'T: Avoid over-engineering with TypedDict for now
# We can add these later if needed
```

**Type checking:**
- Use `mypy` in strict mode for `src/betting/` and `src/evaluation/`
- Use `mypy` in normal mode for rest of codebase
- Run mypy in CI but don't block on warnings (only errors)

### Docstrings

**Required for:**
- All public functions and classes
- Any function with non-obvious behavior
- Any function used across modules

**Format:** Google style docstrings

```python
def create_rolling_features(
    df: pd.DataFrame,
    stats: List[str] = ['PRA', 'PTS', 'REB'],
    windows: List[int] = [5, 10, 20]
) -> pd.DataFrame:
    """
    Create rolling average features for temporal modeling.

    CRITICAL: Uses .shift(1) before rolling to prevent data leakage.

    Args:
        df: Game logs DataFrame, must be sorted by (PLAYER_ID, GAME_DATE)
        stats: List of statistics to calculate rolling averages for
        windows: Window sizes for rolling calculations

    Returns:
        DataFrame with rolling features added (2 features per stat per window:
        mean and std)

    Example:
        >>> df = create_rolling_features(game_logs, stats=['PRA'], windows=[5])
        >>> assert 'PRA_L5_mean' in df.columns
        >>> assert 'PRA_L5_std' in df.columns

    Note:
        - First N-1 games will have NaN values for window size N
        - Uses min_periods=1 for mean, min_periods=2 for std
        - Rolling features are player-specific (grouped by PLAYER_ID)
    """
    pass
```

**Optional for:**
- Simple getters/setters
- Private functions with obvious behavior
- One-liner utilities

### Linting & Formatting

**Tools:**
- **black**: Code formatter (line length: 100)
- **flake8**: Linter with relaxed rules
- **isort**: Import sorting

**Configuration:** Create `pyproject.toml` additions:

```toml
[tool.black]
line-length = 100
target-version = ['py312']
include = '\.pyi?$'
extend-exclude = '''
/(
  # directories
  \.eggs
  | \.git
  | \.venv
  | build
  | dist
  | experiments
)/
'''

[tool.isort]
profile = "black"
line_length = 100
skip = [".venv", "experiments", "notebooks"]

[tool.flake8]
max-line-length = 100
extend-ignore = ["E203", "W503"]
exclude = [".venv", "experiments", "notebooks"]
```

**Pragmatic Rules:**
- Run `black` on all code in `src/` before committing
- Run `flake8` and fix critical errors (undefined names, syntax errors)
- Ignore formatting warnings in experiments and notebooks
- Use `# noqa` sparingly for edge cases

### Code Review Process

**For team of 1-2 developers:**

**Option 1: Solo Developer**
- Use checklist-based self-review (see section 7)
- Run pre-commit hooks before every commit
- Do weekly code reviews of your own work

**Option 2: 2 Developers**
- Pair program on critical modules (betting, evaluation)
- Async reviews for feature engineering and data processing
- Weekly sync to review architecture decisions

**Review Checklist:**
- [ ] No temporal leakage (future data in features)
- [ ] No duplicate player-game combinations
- [ ] Proper error handling for edge cases
- [ ] Type hints on public functions
- [ ] Docstrings for complex logic
- [ ] Tests for critical paths
- [ ] No hardcoded paths or credentials
- [ ] Logging for important operations

### Git Workflow

**Branch Strategy:**
- `main`: Production-ready code (always deployable)
- `dev`: Integration branch for features
- `feature/feature-name`: Individual features

**Commit Guidelines:**
```bash
# GOOD commits
git commit -m "feat: add lag features with leakage prevention"
git commit -m "fix: remove duplicate player-game combinations in merge"
git commit -m "test: add temporal leakage detection tests"
git commit -m "refactor: extract CTG feature logic to features/core.py"

# BAD commits (too vague)
git commit -m "update code"
git commit -m "fix bug"
```

**Commit Message Format:**
- `feat:` New feature
- `fix:` Bug fix
- `refactor:` Code restructuring (no behavior change)
- `test:` Add or update tests
- `docs:` Documentation changes
- `perf:` Performance improvement

**Before merging to main:**
1. All tests pass
2. Code formatted with black
3. No flake8 errors
4. Manual testing on small dataset

---

## 4. Error Handling

### Error Handling Strategy

**Philosophy:** Fail fast, fail loudly. For betting, silent failures = lost money.

### When to Raise Exceptions

**Critical Errors (raise immediately):**
- Temporal leakage detected
- Missing required data (game logs, CTG stats)
- Invalid betting parameters (negative odds, invalid edge)
- Duplicate player-game combinations
- Model prediction failures

**Warnings (log and continue):**
- Missing optional data (e.g., CTG stats for some players)
- Low confidence predictions
- Feature calculation issues for small number of rows

**Silent handling (use defaults):**
- Missing non-critical configuration values
- Cosmetic issues (plot formatting, etc.)

### Custom Exceptions

Create `src/utils/exceptions.py`:

```python
class NBAPropsException(Exception):
    """Base exception for NBA Props Model."""
    pass


class TemporalLeakageError(NBAPropsException):
    """Raised when temporal leakage is detected in features or validation."""
    pass


class DataQualityError(NBAPropsException):
    """Raised when data quality checks fail."""
    pass


class InvalidBettingParametersError(NBAPropsException):
    """Raised when betting parameters are invalid."""
    pass


class ModelPredictionError(NBAPropsException):
    """Raised when model fails to generate predictions."""
    pass


class InsufficientDataError(NBAPropsException):
    """Raised when not enough data is available for reliable predictions."""
    pass
```

### Error Handling Patterns

**Pattern 1: Fail Fast (Critical Errors)**
```python
from src.utils.exceptions import TemporalLeakageError

def validate_no_leakage(train_dates, test_dates):
    """Ensure no temporal leakage in train/test split."""
    if train_dates.max() >= test_dates.min():
        raise TemporalLeakageError(
            f"Training data ({train_dates.max()}) overlaps with "
            f"test data ({test_dates.min()}). This causes temporal leakage!"
        )
```

**Pattern 2: Graceful Degradation (Missing Optional Data)**
```python
import logging

logger = logging.getLogger(__name__)

def load_ctg_stats(player_name: str, season: str) -> Dict[str, float]:
    """Load CTG stats, fall back to league averages if not found."""
    try:
        stats = _load_from_file(player_name, season)
        return stats
    except FileNotFoundError:
        logger.warning(
            f"CTG stats not found for {player_name} ({season}). "
            f"Using league averages."
        )
        return DEFAULT_LEAGUE_AVERAGES
```

**Pattern 3: Retry with Exponential Backoff (External APIs)**
```python
import time
import requests

def fetch_odds_with_retry(prop_id: str, max_retries: int = 3) -> Dict:
    """Fetch odds with exponential backoff."""
    for attempt in range(max_retries):
        try:
            response = requests.get(f"https://api.theoddsapi.com/props/{prop_id}")
            response.raise_for_status()
            return response.json()
        except requests.RequestException as e:
            if attempt == max_retries - 1:
                raise

            wait_time = 2 ** attempt
            logger.warning(f"API request failed (attempt {attempt+1}), retrying in {wait_time}s")
            time.sleep(wait_time)
```

**Pattern 4: Context Managers (Resource Cleanup)**
```python
from contextlib import contextmanager

@contextmanager
def model_session(model_path: str):
    """Load model and ensure cleanup."""
    model = load_model(model_path)
    try:
        yield model
    finally:
        model.cleanup()  # Release resources
```

### Input Validation

**Validate all external inputs:**
```python
from src.utils.validators import validate_game_logs, validate_odds

def predict(game_logs: pd.DataFrame, odds: float) -> float:
    """Generate prediction for prop bet."""
    # Validate inputs
    validate_game_logs(game_logs)

    if odds <= 1.0:
        raise InvalidBettingParametersError(
            f"Invalid odds: {odds}. Odds must be > 1.0"
        )

    # ... rest of prediction logic
```

**Create validators in `src/utils/validators.py`:**
```python
import pandas as pd
from src.utils.exceptions import DataQualityError

def validate_game_logs(df: pd.DataFrame) -> None:
    """Validate game logs DataFrame has required columns and valid data."""
    required_cols = ['PLAYER_ID', 'GAME_DATE', 'PRA', 'MIN']

    missing_cols = set(required_cols) - set(df.columns)
    if missing_cols:
        raise DataQualityError(f"Missing required columns: {missing_cols}")

    # Check for duplicates
    duplicates = df.duplicated(subset=['PLAYER_ID', 'GAME_DATE']).sum()
    if duplicates > 0:
        raise DataQualityError(
            f"Found {duplicates} duplicate player-game combinations"
        )

    # Check for nulls in critical columns
    null_counts = df[required_cols].isnull().sum()
    if null_counts.any():
        raise DataQualityError(f"Null values in required columns: {null_counts[null_counts > 0]}")
```

---

## 5. Documentation Requirements

### Documentation Hierarchy

**Tier 1: Critical (Must Document)**
1. Feature definitions and formulas
2. Model architecture and decisions
3. Betting logic and Kelly sizing
4. Production deployment process

**Tier 2: Important (Should Document)**
5. Data pipeline architecture
6. Walk-forward validation methodology
7. Configuration management
8. API contracts (if building APIs)

**Tier 3: Nice-to-Have (Optional)**
9. Exploratory data analysis
10. Hyperparameter tuning experiments
11. Code examples and tutorials

### Required Documentation Files

**1. Feature Registry** (`docs/FEATURE_REGISTRY.md`)

Document all features with formulas:

```markdown
# Feature Registry

## Core Performance Features (Tier 1)

### CTG_USG (Usage Rate)
- **Source**: CleaningTheGlass.com
- **Formula**: `(FGA + 0.44 * FTA + TOV) / (Team's total possessions while player is on court)`
- **Type**: Season-level contextual feature
- **Correlation with PRA**: 0.82
- **Leakage Risk**: LOW (season stats, no future data)
- **Default Value**: 0.20 (league average)

### PRA_L5_mean (Last 5 Games PRA Average)
- **Source**: NBA API game logs
- **Formula**: `mean(PRA for last 5 games), excluding current game`
- **Type**: Temporal feature
- **Leakage Risk**: LOW (uses .shift(1) to exclude current game)
- **Default Value**: NaN for first 5 games

[... continue for all features ...]
```

**2. Model Architecture** (`docs/MODEL_ARCHITECTURE.md`)

```markdown
# Model Architecture

## Current Model: XGBoost Regressor

### Why XGBoost?
- Handles non-linear relationships (rest days, usage rate interactions)
- Built-in feature importance for debugging
- Proven in sports betting (see references)

### Hyperparameters (2024-25 Season)
```python
xgb_params = {
    'objective': 'reg:squarederror',
    'max_depth': 6,
    'learning_rate': 0.05,
    'n_estimators': 500,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'min_child_weight': 3,
    'random_state': 42
}
```

### Feature Importance (Top 10)
1. CTG_USG: 0.18
2. PRA_L5_mean: 0.12
3. MIN_L10_mean: 0.10
[...]

### Walk-Forward Validation Results
- MAE: 4.2
- Edge Detection: 54% accuracy at 3%+ edge
- ROI: 8.3% (2023-24 season)

[... continue with full architecture details ...]
```

**3. Production Runbook** (`docs/PRODUCTION_RUNBOOK.md`)

```markdown
# Production Runbook

## Daily Prediction Workflow

### 1. Data Collection (9:00 AM ET)
```bash
# Fetch today's props from TheOddsAPI
uv run scripts/fetch_todays_props.py

# Update player game logs
uv run scripts/update_game_logs.py --date today
```

### 2. Generate Predictions (10:00 AM ET)
```bash
# Run production pipeline
uv run src/pipelines/prediction_pipeline.py \
    --date today \
    --model models/prod/xgboost_v2.pkl \
    --output predictions/2024-10-14.csv
```

### 3. Calculate Edges & Sizing
```bash
# Calculate betting edges
uv run src/betting/edge_calculator.py \
    --predictions predictions/2024-10-14.csv \
    --odds data/odds/2024-10-14.csv \
    --min-edge 0.03
```

### 4. Review & Place Bets (11:00 AM ET)
- Manually review all bets with edge > 5%
- Check for news (injuries, lineup changes)
- Place bets via sportsbook

## Error Recovery

### Problem: Model predictions fail
**Symptoms**: `ModelPredictionError` raised
**Resolution**:
1. Check if game logs are up to date
2. Verify CTG features loaded correctly
3. Check model file integrity: `uv run scripts/validate_model.py`
4. If still failing, use backup model: `models/backup/xgboost_v1.pkl`

[... continue with all common issues ...]
```

**4. Inline Code Documentation**

**Document decision rationale:**
```python
def calculate_kelly_fraction(edge: float, odds: float, kelly_fraction: float = 0.25) -> float:
    """
    Calculate Kelly Criterion bet sizing.

    IMPORTANT: We use 1/4 Kelly to reduce variance.
    Full Kelly (kelly_fraction=1.0) is too aggressive for our bankroll.

    Math:
        Full Kelly = (edge * odds - 1) / (odds - 1)
        Fractional Kelly = Full Kelly * kelly_fraction

    Example:
        If we have 3% edge on +100 odds:
        Full Kelly = (0.03 * 2 - 1) / 1 = -0.94 (NO BET, negative expected value)

        If we have 10% edge on +100 odds:
        Full Kelly = (0.10 * 2 - 1) / 1 = 0.20 (bet 20% of bankroll)
        1/4 Kelly = 0.05 (bet 5% of bankroll)
    """
    pass
```

### Documentation Templates

**Template: New Feature Addition**

When adding a new feature, create a GitHub issue or doc section:

```markdown
## Feature: [Feature Name]

**Definition**: [Clear mathematical definition]
**Source**: [Where data comes from]
**Rationale**: [Why we think this will help]
**Leakage Risk**: [HIGH/MEDIUM/LOW with explanation]
**Implementation**: [Code location]
**Tests**: [Link to test file]
**Correlation with PRA**: [Expected or measured correlation]
```

---

## 6. Performance Guidelines

### Build Time Targets

**Acceptable performance for 854K row dataset:**

| Operation | Target Time | Max Acceptable |
|-----------|-------------|----------------|
| Load game logs | <5 seconds | 10 seconds |
| Create lag features | <30 seconds | 60 seconds |
| Create rolling features | <60 seconds | 120 seconds |
| Merge CTG stats | <10 seconds | 20 seconds |
| Complete dataset build | <3 minutes | 5 minutes |
| Model training | <5 minutes | 10 minutes |
| Walk-forward validation | <30 minutes | 60 minutes |

**If exceeding targets:**
1. Profile the code to find bottlenecks
2. Optimize only the slowest 20% of operations
3. Consider parallel processing for independent operations

### When to Optimize

**DON'T optimize unless:**
1. Build time > 5 minutes (blocking development)
2. Prediction time > 30 seconds (blocking production)
3. Memory usage > 8GB (crashing on typical machines)

**Premature optimization is evil.** Get it working first, then optimize if needed.

### Optimization Strategies

**Strategy 1: Vectorization (First Resort)**
```python
# SLOW: Row-by-row iteration
for i in range(len(df)):
    df.loc[i, 'feature'] = df.loc[i, 'A'] * df.loc[i, 'B']

# FAST: Vectorized operations
df['feature'] = df['A'] * df['B']
```

**Strategy 2: Efficient Groupby**
```python
# SLOW: Multiple groupby operations
df['lag1'] = df.groupby('PLAYER_ID')['PRA'].shift(1)
df['lag2'] = df.groupby('PLAYER_ID')['PRA'].shift(2)
df['lag3'] = df.groupby('PLAYER_ID')['PRA'].shift(3)

# FASTER: Single groupby with multiple operations
grouped = df.groupby('PLAYER_ID')['PRA']
df['lag1'] = grouped.shift(1)
df['lag2'] = grouped.shift(2)
df['lag3'] = grouped.shift(3)
```

**Strategy 3: Parquet Instead of CSV**
```python
# SLOW: CSV reading (5-10 seconds)
df = pd.read_csv('game_logs.csv')

# FAST: Parquet reading (1-2 seconds)
df = pd.read_parquet('game_logs.parquet')
```

**Strategy 4: Parallel Processing (Last Resort)**
```python
from joblib import Parallel, delayed

def process_player(player_df):
    # Heavy computation per player
    return player_df.with_features()

# Process players in parallel
results = Parallel(n_jobs=-1)(
    delayed(process_player)(group)
    for name, group in df.groupby('PLAYER_ID')
)
```

### Caching Strategy

**Cache expensive operations:**
```python
from functools import lru_cache
import hashlib

@lru_cache(maxsize=128)
def load_ctg_stats(season: str) -> pd.DataFrame:
    """Load CTG stats with caching."""
    return pd.read_parquet(f'data/ctg/{season}.parquet')


class FeatureBuilder:
    def __init__(self):
        self._cache = {}

    def get_rolling_features(self, df: pd.DataFrame, cache_key: str = None):
        """Get rolling features with optional caching."""
        if cache_key and cache_key in self._cache:
            return self._cache[cache_key]

        result = self._compute_rolling_features(df)

        if cache_key:
            self._cache[cache_key] = result

        return result
```

### Memory Management

**For large datasets:**
```python
# Read in chunks if needed
chunk_size = 100000
chunks = []

for chunk in pd.read_csv('huge_file.csv', chunksize=chunk_size):
    processed = process_chunk(chunk)
    chunks.append(processed)

df = pd.concat(chunks)
```

**Use appropriate data types:**
```python
# MEMORY INEFFICIENT: Default types
df['PLAYER_ID'] = df['PLAYER_ID']  # int64 (8 bytes)
df['IS_HOME'] = df['IS_HOME']      # int64 (8 bytes)

# MEMORY EFFICIENT: Optimized types
df['PLAYER_ID'] = df['PLAYER_ID'].astype('int32')  # 4 bytes
df['IS_HOME'] = df['IS_HOME'].astype('int8')       # 1 byte
```

---

## 7. Production Readiness Checklist

### Pre-Live Testing Checklist

**Must Complete Before Live Betting:**

#### Data Quality
- [ ] No duplicate player-game combinations in training data
- [ ] All game logs sorted by (PLAYER_ID, GAME_DATE)
- [ ] CTG stats loaded for 90%+ of players
- [ ] No missing values in target (PRA)
- [ ] Date ranges verified (training < validation < test)

#### Temporal Leakage Prevention
- [ ] All lag features use `.shift()` to exclude current game
- [ ] Rolling averages use `.shift(1)` before `.rolling()`
- [ ] Walk-forward validation uses strict chronological splits
- [ ] No future data in any features (verified by tests)
- [ ] Feature creation order documented and validated

#### Model Validation
- [ ] Walk-forward validation completed on 2023-24 season
- [ ] MAE < 5.0 on out-of-sample data
- [ ] Model predictions correlate with actual PRA (r > 0.60)
- [ ] Edge detection accuracy > 52% at 3%+ threshold
- [ ] No catastrophic predictions (PRA < 0 or PRA > 100)

#### Betting Logic
- [ ] Edge calculation verified against known examples
- [ ] Kelly sizing produces reasonable bet sizes (< 5% of bankroll)
- [ ] Risk limits implemented (max bet, max daily exposure)
- [ ] Negative expected value bets filtered out
- [ ] Position sizing accounts for correlation across props

#### Error Handling
- [ ] All critical exceptions defined and raised appropriately
- [ ] Graceful degradation for missing CTG stats
- [ ] Logging configured for production (file + console)
- [ ] Error alerts configured (email or Slack)
- [ ] Retry logic for API failures

#### Testing
- [ ] Temporal leakage tests pass
- [ ] Feature calculation tests pass
- [ ] Data validation tests pass
- [ ] Walk-forward validation tests pass
- [ ] Edge calculation tests pass
- [ ] Unit test coverage > 60% for critical modules

#### Documentation
- [ ] Feature registry complete with formulas
- [ ] Model architecture documented
- [ ] Production runbook complete
- [ ] Error recovery procedures documented
- [ ] Configuration management documented

#### Configuration
- [ ] All credentials in environment variables (not hardcoded)
- [ ] Model paths configurable
- [ ] Feature configurations externalized
- [ ] Risk parameters documented and reviewed

### Code Quality Gates

**Before merging to main:**
- [ ] All tests pass (`pytest tests/`)
- [ ] Code formatted (`black src/`)
- [ ] No critical flake8 errors
- [ ] Type hints on public functions
- [ ] Docstrings for complex functions
- [ ] No `print()` statements (use `logging`)
- [ ] No hardcoded paths or credentials
- [ ] Code reviewed (self or peer)

### Weekly Review Checklist

**Every Friday:**
- [ ] Review model performance on past week
- [ ] Check for data quality issues
- [ ] Review bet sizing and exposure
- [ ] Update feature importance analysis
- [ ] Review error logs
- [ ] Update documentation if needed

---

## Pre-Commit Hook Setup

Create `.pre-commit-config.yaml`:

```yaml
repos:
  - repo: https://github.com/psf/black
    rev: 24.4.2
    hooks:
      - id: black
        args: [--line-length=100]
        exclude: ^(experiments/|notebooks/)

  - repo: https://github.com/pycqa/flake8
    rev: 7.0.0
    hooks:
      - id: flake8
        args: [--max-line-length=100, --extend-ignore=E203,W503]
        exclude: ^(experiments/|notebooks/)

  - repo: https://github.com/pycqa/isort
    rev: 5.13.2
    hooks:
      - id: isort
        args: [--profile=black, --line-length=100]
        exclude: ^(experiments/|notebooks/)

  - repo: local
    hooks:
      - id: pytest-check
        name: pytest-check
        entry: pytest tests/unit -x
        language: system
        pass_filenames: false
        always_run: true
```

Install pre-commit:
```bash
uv add pre-commit --dev
pre-commit install
```

---

## Quick Reference

### Daily Development Workflow

```bash
# 1. Create feature branch
git checkout -b feature/add-momentum-features

# 2. Make changes
# ... edit code ...

# 3. Run tests
pytest tests/unit -v

# 4. Format code
black src/
isort src/

# 5. Commit (pre-commit hooks run automatically)
git add .
git commit -m "feat: add momentum features based on L5 vs L20 comparison"

# 6. Push and merge
git push origin feature/add-momentum-features
# (merge to dev after review)
```

### When Adding a New Feature

1. [ ] Document in Feature Registry
2. [ ] Write test first (test-driven development)
3. [ ] Implement feature
4. [ ] Verify no temporal leakage
5. [ ] Add to feature pipeline
6. [ ] Update configuration
7. [ ] Commit with clear message

### When Fixing a Bug

1. [ ] Write test that reproduces bug
2. [ ] Fix the bug
3. [ ] Verify test passes
4. [ ] Check for similar bugs elsewhere
5. [ ] Update documentation if needed
6. [ ] Commit with "fix:" prefix

---

## Appendix: Tool Installation

```bash
# Add development dependencies
uv add pytest pytest-cov black flake8 isort mypy pre-commit --dev

# Install pre-commit hooks
pre-commit install

# Run full quality check
black src/
isort src/
flake8 src/
mypy src/betting src/evaluation
pytest tests/ -v --cov=src --cov-report=html
```

---

## Summary: Pragmatic Priorities for 12-Week Timeline

**Weeks 1-2 (Foundation)**
- Set up pre-commit hooks (black, flake8, isort)
- Write 5 critical tests (temporal leakage, data quality, walk-forward, edge calc, features)
- Document feature registry for existing features
- Organize code structure (separate research from production)

**Weeks 3-4 (Consolidation)**
- Achieve 60% test coverage on critical modules
- Add type hints to betting/ and evaluation/ modules
- Create production runbook
- Refactor feature pipeline

**Weeks 5-6 (Production Ready)**
- Complete all pre-live testing checklist items
- Add comprehensive error handling
- Set up production logging and monitoring
- Build prediction pipeline

**Weeks 7-8 (Live Testing)**
- Paper trade with real odds
- Monitor performance daily
- Fix any issues discovered
- Refine betting logic

**Weeks 9-12 (Optimization & Scale)**
- Optimize performance if needed
- Add advanced features
- Build ensemble models
- Scale to multiple props

**Key Success Metrics:**
- 60%+ test coverage on critical modules by Week 4
- Zero temporal leakage (verified by tests) by Week 6
- Production-ready pipeline by Week 6
- Positive ROI in paper trading by Week 8
